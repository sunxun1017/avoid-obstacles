"""Source task composition independent of mjlab data structures.

Source methods are adapted under Apache-2.0; explicit batched state and caching.
"""

import math

import torch

from embodied_learning.tasks.cat_command import compute_cat_command
from embodied_learning.tasks.cat_gait import CatGaitState
from embodied_learning.tasks.cat_observation import build_cat_observations
from embodied_learning.tasks.cat_reward import (
    SOURCE_SCALES,
    CatRewardParameters,
    aggregate_cat_reward,
    cat_reward_terms,
)


class CatTaskState:
    # Reporting only: these thresholds do not change rewards or termination.
    metric_definitions = {
        "settled_stop": {
            "linear_speed_below_m_s": 0.1,
            "angular_speed_below_rad_s": 0.1,
            "continuous_seconds": 0.5,
            "control_dt_seconds": 0.02,
        },
        "success": (
            "Reached within 0.1 m at least once, with no observed obstacle or "
            "monitored self contact or source collision during the episode, "
            "and no fall/invalid termination. Uses the source 20 s horizon."
        ),
        "physical_collision": "Any observed obstacle contact during the episode.",
        "stop_command": "Source command vector norm below 0.2; not actual speed.",
    }

    def __init__(self, manifest, bank, allowed, num_envs, device, options):
        if bank.source_profile != {"version": "cat_source_v1"}:
            raise ValueError("Source task requires source-profile cache.")
        self.manifest, self.bank, self.allowed = manifest, bank, allowed
        defaults = dict(
            noise_level=1.0,
            random_pd=True,
            random_rfi=True,
            push=True,
            ground=1.0,
            lateral=1.0,
            overhead=1.0,
            term_collision_threshold=0.04,
        )
        if set(options) - set(defaults):
            raise ValueError("Unknown source task option.")
        self.options = {**defaults, **options}
        for k in (
            "noise_level",
            "ground",
            "lateral",
            "overhead",
            "term_collision_threshold",
        ):
            if not math.isfinite(self.options[k]) or self.options[k] < 0:
                raise ValueError("Invalid source task scale.")
        for k in ("random_pd", "random_rfi", "push"):
            if type(self.options[k]) is not bool:
                raise ValueError("Source randomization flags must be boolean.")
        self.scene_ids = torch.full(
            (num_envs,), allowed[0], device=device, dtype=torch.long
        )
        self.goal_ids = torch.zeros_like(self.scene_ids)
        self.goals = torch.zeros(num_envs, 3, device=device)
        self.gait = CatGaitState(num_envs, device)
        self.foot_height = torch.full((num_envs,), 0.07, device=device)
        self.previous_vel = torch.zeros(num_envs, 29, device=device)
        self.last_positions = torch.zeros(num_envs, 11, 3, device=device)
        self.odom_position = torch.zeros(num_envs, 3, device=device)
        self.odom_rotation = torch.eye(3, device=device).repeat(num_envs, 1, 1)
        self.initial = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.ever_goal = torch.zeros_like(self.initial)
        self.ever_physical_collision = torch.zeros_like(self.initial)
        self.ever_collision = torch.zeros_like(self.initial)
        self.settled_steps = torch.zeros_like(self.scene_ids)
        self.min_goal_distance = torch.full((num_envs,), math.inf, device=device)
        self.push_interval = torch.ones_like(self.scene_ids) * 250
        self.generation = 0
        self._key = None
        self._last_common = None
        self._obs = None
        self.goal_distance = torch.zeros(num_envs, device=device)
        self.saved_outcomes = {}
        self.saved_terms = {}
        self.saved_reward_diagnostics = {}
        self.last_metrics = {}
        self.scales = SOURCE_SCALES.copy()
        for names, coef in [
            (("feetgf", "feetdf"), "ground"),
            (("handsgf", "handsdf", "kneesdf", "shldsdf"), "lateral"),
            (("headgf", "headdf"), "overhead"),
        ]:
            for name in names:
                self.scales[name] = self.options[coef]

    def reset(self, env_ids):
        n = len(env_ids)
        freq = 1.3 + 0.2 * torch.rand(n, device=env_ids.device)
        phase = torch.tensor(
            [[0.0, math.pi], [math.pi, 0.0]], device=env_ids.device
        )[torch.randint(2, (n,), device=env_ids.device)]
        self.gait.reset(env_ids, freq, phase)
        self.previous_vel[env_ids] = 0
        self.initial[env_ids] = True
        self.ever_goal[env_ids] = False
        self.ever_physical_collision[env_ids] = False
        self.ever_collision[env_ids] = False
        self.settled_steps[env_ids] = 0
        self.min_goal_distance[env_ids] = math.inf
        self.push_interval[env_ids] = torch.round(
            (5 + 5 * torch.rand(n, device=env_ids.device)) / 0.02
        ).long()
        self.generation += 1
        self._key = None

    @staticmethod
    def classify(*, up_z, head_z, self_contact, sdf, invalid, steps, threshold):
        collision = (self_contact | (sdf < -threshold).any(-1)) & (steps >= 50)
        return dict(
            fall=(up_z < 0) | (head_z < 0.7),
            collision=collision,
            invalid=invalid,
            timeout=steps >= 1000,
        )

    def record_goal(self, distance, safe):
        self.ever_goal |= (distance < 0.1) & safe

    def advance_before_physics(self, actions):
        # Actions/PD are handled by a separate controller, not this state.
        self._key = None

    def refresh(self, payload, controller, steps, common_step):
        key = (common_step, self.generation)
        if self._key == key:
            return
        same_step = self._last_common == common_step
        previous_obs = self._obs
        continuing = ~self.initial
        ids = torch.nonzero(self.initial, as_tuple=True)[0]
        positions = payload["body_positions"]
        rotation = payload["root_rotation"]
        root = payload["root_position"]
        invalid = (
            payload["invalid"]
            | ~torch.isfinite(positions).all((1, 2))
            | ~torch.isfinite(root).all(1)
            | ~torch.isfinite(rotation).all((1, 2))
        )
        # Nonfinite physics is a terminal state, not a valid mapping query.
        # Finite placeholders protect indexing only; original kinematics and
        # the explicit invalid flag still reach rewards/termination/observations.
        query_positions = torch.where(invalid[:, None, None], 0.0, positions)
        root = torch.where(invalid[:, None], 0.0, root)
        rotation = torch.where(
            invalid[:, None, None], torch.eye(3, device=root.device)[None], rotation
        )
        if len(ids):
            self.last_positions[ids] = positions[ids]
            self.odom_position[ids] = root[ids]
            self.odom_rotation[ids] = rotation[ids]
        velocity = (positions - self.last_positions) / 0.02
        sample = self.bank.sample_source(
            query_positions, self.scene_ids, self.goal_ids
        )
        bf = sample.sdf_gradient / (
            sample.sdf_gradient.norm(dim=-1, keepdim=True) + 1e-6
        )
        command = compute_cat_command(
            sample.guidance[:, 1],
            sample.guidance[:, [0, 3, 4, 5, 6]],
            sample.sdf_gradient[:, [0, 3, 4, 5, 6]],
        )
        normalized_gf = sample.guidance / (
            sample.guidance.norm(dim=-1, keepdim=True) + 1e-6
        )
        initial_command = compute_cat_command(
            normalized_gf[:, 1],
            normalized_gf[:, [0, 3, 4, 5, 6]],
            bf[:, [0, 3, 4, 5, 6]],
        )
        old_phase = self.gait.phase.clone()
        old_mask = self.gait.gait_mask.clone()
        old_last = self.gait.last_command.clone()
        old_stop = self.gait.stop_timestep.clone()
        command = self.gait.advance(command)
        # Reset observations do not advance the source phase or stop state.
        restored = (
            torch.arange(len(steps), device=steps.device) if same_step else ids
        )
        self.gait.phase[restored] = old_phase[restored]
        self.gait.gait_mask[restored] = old_mask[restored]
        self.gait.last_command[restored] = old_last[restored]
        self.gait.stop_timestep[restored] = old_stop[restored]
        command[ids] = initial_command[ids]
        self.gait.last_command[ids] = initial_command[ids]
        gf = (
            sample.guidance
            * (command[:, 0, None, None] > 0.5)
            / (sample.guidance.norm(dim=-1, keepdim=True) + 1e-6)
        )
        gf[ids] = normalized_gf[ids]
        update = ((steps - 1) % 5 == 0) | self.initial
        self.odom_position[update] = root[update]
        self.odom_rotation[update] = rotation[update]
        local = (
            rotation.transpose(-1, -2)[:, None]
            @ (query_positions - root[:, None])[..., None]
        ).squeeze(-1)
        delayed = self.odom_position[:, None] + (
            self.odom_rotation[:, None] @ local[..., None]
        ).squeeze(-1)
        delayed_sample = self.bank.sample_source(
            delayed, self.scene_ids, self.goal_ids
        )
        delayed_bf = delayed_sample.sdf_gradient / (
            delayed_sample.sdf_gradient.norm(dim=-1, keepdim=True) + 1e-6
        )
        delayed_gf = (
            delayed_sample.guidance
            * (command[:, 0, None, None] > 0.5)
            / (delayed_sample.guidance.norm(dim=-1, keepdim=True) + 1e-6)
        )
        delayed_gf[ids] = (
            delayed_sample.guidance
            / (delayed_sample.guidance.norm(dim=-1, keepdim=True) + 1e-6)
        )[ids]
        delayed_command = compute_cat_command(
            delayed_gf[:, 1],
            delayed_gf[:, [0, 3, 4, 5, 6]],
            delayed_bf[:, [0, 3, 4, 5, 6]],
        )
        if getattr(self.bank, "deployment", False):
            # The actor uses delayed odometry, while the critic uses the current
            # pose. Either query can leave the measured grid independently.
            # Gate both together after source gait processing (which may restore
            # move_flag=1), so delayed guidance cannot restart an unavailable field.
            available = (
                sample.guidance_valid.all(-1)
                & delayed_sample.guidance_valid.all(-1)
                & ~invalid
            )
            unavailable = ~available
            command[unavailable] = 0
            delayed_command[unavailable] = 0
            self.gait.last_command[unavailable] = 0
            gf[unavailable] = bf[unavailable] = 0
            delayed_gf[unavailable] = delayed_bf[unavailable] = 0
            sample.sdf[unavailable] = delayed_sample.sdf[unavailable] = 0.5
            self.bank.query_available.copy_(available)
        x = payload["reward_inputs"]
        x.gf = gf
        x.bf = bf
        x.sdf = sample.sdf
        x.move_flag = command[:, 0]
        x.cmd_vel = command[:, 1:]
        x.gait_mask = self.gait.gait_mask
        x.foot_height = self.foot_height
        x.previous_vel = self.previous_vel
        x.actions = controller.actions
        x.last_actions = controller.last_actions
        x.last_last_actions = controller.last_last_actions
        x.body_velocities = velocity
        params = CatRewardParameters(controller.lower, controller.upper)
        diagnostics = {}
        self.terms = cat_reward_terms(x, parameters=params, diagnostics=diagnostics)
        self._reward = aggregate_cat_reward(self.terms, self.scales, dt=0.02)
        flags = self.classify(
            up_z=payload["up_z"],
            head_z=positions[:, 0, 2],
            self_contact=payload["self_contact"],
            sdf=sample.sdf,
            invalid=invalid,
            steps=steps,
            threshold=self.options["term_collision_threshold"],
        )
        if getattr(self.bank, "deployment", False):
            # Partial measured occupancy cannot certify physical collision.
            # Keep native collision reporting separate from deployment fields.
            flags["collision"] = (
                payload["physical_collision"] | payload["self_contact"]
            ) & (steps >= 50)
        self.goal_distance = (root[:, :2] - self.goals[:, :2]).norm(dim=-1)
        self.goal_distance = torch.where(invalid, torch.nan, self.goal_distance)
        self.record_goal(
            self.goal_distance,
            ~(flags["fall"] | flags["collision"] | flags["invalid"]),
        )
        self.min_goal_distance = torch.minimum(
            self.min_goal_distance,
            torch.where(
                flags["fall"] | flags["collision"] | invalid,
                torch.inf,
                self.goal_distance,
            ),
        )
        self._outcomes = flags
        if not same_step and not self.initial.all():
            # Accumulate only executed control steps, before auto-reset. A second
            # observation refresh after reset must preserve the terminal report.
            self.ever_physical_collision |= (
                payload["physical_collision"] & continuing
            )
            self.ever_collision |= (
                payload["physical_collision"]
                | payload["self_contact"]
                | flags["collision"]
            ) & continuing
            stop = self.metric_definitions["settled_stop"]
            root_speed = payload["root_local_vel"].norm(dim=-1)
            angular_speed = payload["gyro"].norm(dim=-1)
            slow = (
                (root_speed < stop["linear_speed_below_m_s"])
                & (angular_speed < stop["angular_speed_below_rad_s"])
                & ~flags["fall"]
                & ~invalid
                & continuing
            )
            self.settled_steps = torch.where(slow, self.settled_steps + 1, 0)
            required_steps = math.ceil(
                stop["continuous_seconds"] / stop["control_dt_seconds"]
            )
            self.saved_reward_diagnostics = {
                k: v.clone() for k, v in diagnostics.items()
            }
            self.saved_terms = {k: v.clone() for k, v in self.terms.items()}
            self.saved_outcomes = {
                **{k: v.clone() for k, v in flags.items()},
                "goal": self.ever_goal.clone(),
                "physical_collision": self.ever_physical_collision.clone(),
                "success": (
                    self.ever_goal
                    & ~self.ever_collision
                    & ~flags["fall"]
                    & ~flags["invalid"]
                ),
            }
            self.last_metrics = dict(
                goal_distance=self.goal_distance.clone(),
                min_goal_distance=self.min_goal_distance.clone(),
                stop_command=(command[:, 1:].norm(dim=-1) < 0.2),
                settled_stop=(self.settled_steps >= required_steps),
                root_speed=root_speed,
                root_angular_speed=angular_speed,
            )
        self._obs = build_cat_observations(
            gyro=payload["gyro"],
            gravity=payload["gravity"],
            joint_pos=x.joint_pos,
            joint_vel=x.joint_vel,
            default_joints=controller.default,
            last_actions=controller.last_actions,
            targets=controller.targets,
            command=command,
            foot_height=self.foot_height,
            phase=self.gait.phase,
            root_local_vel=payload["root_local_vel"],
            gf=gf,
            bf=bf,
            sdf=sample.sdf,
            body_positions=positions,
            body_velocities=velocity,
            torso_rpy=x.torso_rpy,
            gait_mask=self.gait.gait_mask,
            feet_contact=x.feet_contact,
            kp_scale=controller.kp_scale,
            kd_scale=controller.kd_scale,
            rfi_scale=controller.rfi_scale,
            navigation_rotation=x.navigation_rotation,
            gf_delayed=delayed_gf,
            bf_delayed=delayed_bf,
            sdf_delayed=delayed_sample.sdf,
            command_delayed=delayed_command,
            noise_level=self.options["noise_level"],
        )
        if same_step and previous_obs is not None:
            for group in ("actor", "critic"):
                self._obs[group][continuing] = previous_obs[group][continuing]
        self._last_common = common_step
        self.previous_vel = x.joint_vel.clone()
        self.last_positions = positions.clone()
        self.initial.fill_(False)
        self._key = key

    def observations(self):
        return self._obs

    def reward(self):
        return self._reward

    def outcomes(self):
        return self._outcomes
