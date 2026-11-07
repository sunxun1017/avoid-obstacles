"""Source task composition independent of mjlab data structures.

Source methods are adapted under Apache-2.0; explicit batched state and caching.
"""

import math

import torch

from embodied_learning.tasks.cat_command import (
    compute_cat_command,
    retime_cat_arrival,
)
from embodied_learning.tasks.cat_foot_progress import RootProgressWindow
from embodied_learning.tasks.cat_gait import CatGaitState, CatPlaybackGaitState
from embodied_learning.tasks.cat_observation import build_cat_observations
from embodied_learning.tasks.cat_recovery import (
    RecoveryConfig,
    RecoveryNavigator,
    aggregate_recovery_reward,
    research_reward_terms,
)
from embodied_learning.tasks.cat_reward import (
    SOURCE_SCALES,
    CatRewardParameters,
    aggregate_cat_reward,
    cat_reward_terms,
    standing_stability_bonus,
)


UNITREE_G1_BRIDGE = "unitree_g1_bridge_v1"
UNITREE_PUSH_VELOCITY_RANGE = {
    "x": (-0.50, 0.50),
    "y": (-0.50, 0.50),
    "z": (-0.40, 0.40),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


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

    def __init__(
        self,
        manifest,
        bank,
        allowed,
        num_envs,
        device,
        options,
        *,
        task_profile="cat_source_v1",
        recovery_options=None,
        execution_profile="source_train",
        risk_guidance=None,
        unitree_bridge=False,
    ):
        if bank.source_profile != {"version": "cat_source_v1"}:
            raise ValueError("Source task requires source-profile cache.")
        if task_profile not in ("cat_source_v1", "cat_recovery_v1"):
            raise ValueError("Unsupported CAT task profile.")
        if execution_profile not in ("source_train", "author_playback"):
            raise ValueError("Unknown CAT execution profile.")
        if (
            execution_profile == "author_playback"
            and task_profile != "cat_source_v1"
        ):
            raise ValueError("Author playback only supports the source task.")
        from embodied_learning.tasks.cat_risk_guidance import validate_schema

        self.risk_guidance = validate_schema(risk_guidance)
        if self.risk_guidance is not None and (
            task_profile != "cat_source_v1" or getattr(bank, "deployment", False)
        ):
            raise ValueError(
                "Risk guidance pilot requires offline source task fields."
            )
        self.risk_alpha = self.risk_alpha_delayed = None
        self.execution_profile = execution_profile
        self.unitree_bridge = unitree_bridge
        if task_profile == "cat_source_v1" and recovery_options:
            raise ValueError("Recovery options require the research task profile.")
        self.task_profile = task_profile
        self.recovery = (
            RecoveryNavigator(
                num_envs, device, RecoveryConfig(**(recovery_options or {}))
            )
            if task_profile == "cat_recovery_v1"
            else None
        )
        if self.recovery is not None and any(
            value > torch.finfo(torch.float32).max
            for value in vars(self.recovery.config).values()
        ):
            raise ValueError("Recovery parameters must be representable in float32.")
        if self.recovery is not None and getattr(bank, "deployment", False):
            raise ValueError(
                "Recovery profile requires explicit offline scene geometry."
            )
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
        if (
            set(options)
            - set(defaults)
            - {
                "command_deadband_m_s",
                "arrival_control",
                "root_tracking_weight",
                "feetgf_progress_gate",
                "training_initialization",
                "stop_transition",
                "standing_stability",
            }
        ):
            raise ValueError("Unknown source task option.")
        self.options = {**defaults, **options}
        self.standing_stability = self.options.get("standing_stability")
        if self.standing_stability is not None and (
            self.standing_stability != "goal_double_support_v1"
            or self.recovery is not None
            or getattr(bank, "deployment", False)
            or any(manifest.scenes[index].boxes for index in allowed)
        ):
            raise ValueError("Standing stability requires a known empty CAT scene.")
        progress_profile = self.options.get("feetgf_progress_gate")
        if progress_profile not in (None, "root_displacement_0p5s_v1"):
            raise ValueError("Unknown feet guidance progress gate profile.")
        if progress_profile and self.recovery is not None:
            raise ValueError("Feet progress gate requires the source task profile.")
        self.feet_progress = (
            RootProgressWindow(num_envs, device) if progress_profile else None
        )
        self.training_initialization = self.options.get(
            "training_initialization", "fixed_v1"
        )
        if self.training_initialization not in ("fixed_v1", "author_random_v1"):
            raise ValueError("Unknown training initialization profile.")
        if (
            self.training_initialization == "author_random_v1"
            and task_profile != "cat_source_v1"
        ):
            raise ValueError("Author initialization requires the source task.")
        tracking_weight = self.options.get("root_tracking_weight", 1.0)
        if (
            type(tracking_weight) not in (int, float)
            or not math.isfinite(tracking_weight)
            or not 0 < tracking_weight <= torch.finfo(torch.float32).max
        ):
            raise ValueError("Root tracking weight must be positive and finite.")
        if "root_tracking_weight" in options and self.recovery is not None:
            raise ValueError(
                "Root tracking weight requires the source task profile."
            )
        self.arrival_control = self.options.get("arrival_control")
        if self.arrival_control not in (
            None,
            "root_linear_v1",
            "root_planar_blend_v2",
        ):
            raise ValueError("Unknown arrival control profile.")
        if self.arrival_control and self.recovery is not None:
            raise ValueError("Arrival control requires the source task profile.")
        from embodied_learning.tasks.cat_stop_transition import VERSION

        self.stop_transition = self.options.get("stop_transition")
        if self.stop_transition not in (None, VERSION):
            raise ValueError("Unknown stop transition profile.")
        if self.stop_transition and (
            self.arrival_control != "root_planar_blend_v2"
            or self.recovery is not None
            or execution_profile != "source_train"
            or getattr(bank, "deployment", False)
        ):
            raise ValueError(
                "Stop transition requires offline root-planar source training."
            )
        self.command_deadband = self.options.get("command_deadband_m_s", 0.2)
        if (
            type(self.command_deadband) not in (int, float)
            or not math.isfinite(self.command_deadband)
            or self.command_deadband <= 0
        ):
            raise ValueError("Task command deadband must be positive and finite.")
        if self.arrival_control and self.command_deadband >= 0.08:
            raise ValueError("Arrival control requires command deadband < 0.08 m/s.")
        if "command_deadband_m_s" in options:
            self.metric_definitions = {
                **self.metric_definitions,
                "stop_command": (
                    f"Command vector norm below {self.command_deadband} m/s; "
                    "not actual speed. Explicit arrival-threshold experiment."
                ),
            }
        if execution_profile == "author_playback":
            self.options.update(
                noise_level=0.0, random_pd=False, random_rfi=False, push=False
            )
        for k in (
            "noise_level",
            "ground",
            "lateral",
            "overhead",
            "term_collision_threshold",
        ):
            if not math.isfinite(self.options[k]) or self.options[k] < 0:
                raise ValueError("Invalid source task scale.")
            if (
                self.recovery is not None
                and self.options[k] > torch.finfo(torch.float32).max
            ):
                raise ValueError(
                    "Research task scales must be representable in float32."
                )
        for k in ("random_pd", "random_rfi", "push"):
            if type(self.options[k]) is not bool:
                raise ValueError("Source randomization flags must be boolean.")
        self.scene_ids = torch.full(
            (num_envs,), allowed[0], device=device, dtype=torch.long
        )
        self.goal_ids = torch.zeros_like(self.scene_ids)
        self.goals = torch.zeros(num_envs, 3, device=device)
        gait_type = (
            CatPlaybackGaitState
            if execution_profile == "author_playback" or self.arrival_control
            else CatGaitState
        )
        self.gait = (
            gait_type(num_envs, device, command_deadband=self.command_deadband)
            if gait_type is CatPlaybackGaitState
            else gait_type(num_envs, device)
        )
        if self.stop_transition:
            from embodied_learning.tasks.cat_stop_transition import (
                CatStopTransitionGait,
            )

            self.gait = CatStopTransitionGait(
                num_envs, device, command_deadband=self.command_deadband
            )
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
        if self.standing_stability is not None:
            self.scales["standing_stability"] = 2.0
        # Explicit experiment: prioritize following the requested velocity,
        # including zero velocity, keeping original posture/contact coefficients.
        self.scales["tracking_root_field"] = tracking_weight
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
        if self.execution_profile == "author_playback":
            freq.fill_(1.5)
            phase[:] = torch.tensor([0.0, math.pi], device=env_ids.device)
        self.gait.reset(env_ids, freq, phase)
        if self.recovery is not None:
            self.recovery.reset(env_ids)
        self.previous_vel[env_ids] = 0
        self.initial[env_ids] = True
        self.ever_goal[env_ids] = False
        self.ever_physical_collision[env_ids] = False
        self.ever_collision[env_ids] = False
        self.settled_steps[env_ids] = 0
        self.min_goal_distance[env_ids] = math.inf
        self.push_interval[env_ids] = self.sample_push_interval(n)
        self.generation += 1
        self._key = None

    def sample_push_interval(self, count):
        random = torch.rand(count, device=self.push_interval.device)
        if self.unitree_bridge:
            return torch.ceil((5 + random) / 0.02).long()
        return torch.round((5 + 5 * random) / 0.02).long()

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
        if self.recovery is not None:
            envelope = payload["body_envelope"]
            invalid |= (
                ~envelope.valid
                | ~torch.isfinite(payload["reward_inputs"].root_vel).all(-1)
                | ~torch.isfinite(payload["gyro"]).all(-1)
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
        transition_snapshot = self.gait.snapshot() if self.stop_transition else None
        command = compute_cat_command(
            sample.guidance[:, 1],
            sample.guidance[:, [0, 3, 4, 5, 6]],
            sample.sdf_gradient[:, [0, 3, 4, 5, 6]],
            deadband=1e-6 if self.arrival_control else self.command_deadband,
        )
        if self.arrival_control == "root_planar_blend_v2":
            from embodied_learning.tasks.cat_command import (
                corrected_navigation_command,
            )

            # Root navigation must query the root, not the pelvis IMU site.
            # FootGuidanceBank overrides only foot columns; query its base bank
            # directly so a one-point root query keeps the same interpolation.
            base_bank = getattr(self.bank, "base", self.bank)
            root_sample = base_bank.sample_source(
                root[:, None], self.scene_ids, self.goal_ids
            )
            command = corrected_navigation_command(
                root_sample.guidance[:, 0],
                sample.guidance[:, [0, 3, 4, 5, 6]],
                sample.sdf_gradient[:, [0, 3, 4, 5, 6]],
                root,
                self.goals,
                sample.sdf,
            )
        elif self.arrival_control:
            command = retime_cat_arrival(command, root, self.goals, sample.sdf)
        if self.arrival_control:
            command = torch.where(
                (command[:, 1:].norm(dim=-1) < self.command_deadband)[:, None],
                0.0,
                command,
            )
        if self.stop_transition:
            command = self.gait.filter_command(
                command,
                root,
                self.goals,
                payload["reward_inputs"].root_vel,
                payload["gyro"],
            )
        normalized_gf = sample.guidance / (
            sample.guidance.norm(dim=-1, keepdim=True) + 1e-6
        )
        initial_command = compute_cat_command(
            normalized_gf[:, 1],
            normalized_gf[:, [0, 3, 4, 5, 6]],
            bf[:, [0, 3, 4, 5, 6]],
            deadband=self.command_deadband,
        )
        if self.arrival_control:
            initial_command = command.clone()
        if self.recovery is None:
            old_phase = self.gait.phase.clone()
            old_mask = self.gait.gait_mask.clone()
            old_last = self.gait.last_command.clone()
            old_stop = self.gait.stop_timestep.clone()
            playback = isinstance(self.gait, CatPlaybackGaitState)
            if playback:
                old_step = self.gait.step.clone()
                old_has_velocity = self.gait.has_velocity.clone()
                initial_command = command.clone()
                initial_command[:, 0] = 0.0  # Source reset last_flags[1].
            command = self.gait.advance(command)
            # Reset observations do not advance the source phase or stop state.
            restored = (
                torch.arange(len(steps), device=steps.device) if same_step else ids
            )
            self.gait.phase[restored] = old_phase[restored]
            self.gait.gait_mask[restored] = old_mask[restored]
            self.gait.last_command[restored] = old_last[restored]
            self.gait.stop_timestep[restored] = old_stop[restored]
            if playback:
                self.gait.step[restored] = old_step[restored]
                self.gait.has_velocity[restored] = old_has_velocity[restored]
            if self.stop_transition:
                self.gait.restore(transition_snapshot, restored)
                if same_step:
                    command[continuing] = old_last[continuing]
                self.gait.seed(
                    initial_command,
                    (root[:, :2] - self.goals[:, :2]).norm(dim=-1),
                    ids,
                )
            command[ids] = initial_command[ids]
            self.gait.last_command[ids] = initial_command[ids]
        else:
            active = continuing & ~invalid & (not same_step)
            # Preserve field direction below the source command deadband. The
            # navigator applies its own continuous arrival speed cap.
            deadband = command[:, 1:3].norm(dim=-1) < 1e-6
            command[deadband, 1:3] = 0.525 * sample.guidance[deadband, 1, :2]
            command, restart = self.recovery.advance(
                goal_delta=self.goals[:, :2] - root[:, :2],
                velocity=payload["reward_inputs"].root_vel,
                angular_velocity=payload["gyro"],
                guidance_command=command,
                feet_contact=payload["reward_inputs"].feet_contact,
                active=active,
            )
            self.gait.phase[restart] = torch.tensor(
                [0.0, math.pi], device=root.device
            )
            next_phase = (
                torch.remainder(
                    self.gait.phase + self.gait.phase_dt[:, None] + math.pi,
                    2 * math.pi,
                )
                - math.pi
            )
            next_phase = torch.where(command[:, :1] > 0.5, next_phase, 0.0)
            self.gait.phase[active] = next_phase[active]
            cosine = torch.cos(self.gait.phase)
            mask = torch.where(
                cosine > 0.6, 1.0, torch.where(cosine < -0.6, -1.0, 0.0)
            )
            self.gait.gait_mask[active] = mask[active]
            self.gait.stop_timestep[active] = torch.where(
                command[active, 0] > 0.5, 100, 0
            )
            command[invalid] = 0
            self.gait.last_command = command.clone()
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
            deadband=self.command_deadband,
        )
        observation_sdf, delayed_sdf = sample.sdf, delayed_sample.sdf
        if self.execution_profile == "author_playback":
            # Player queries current pose and retains command from raw GF.
            # Reset uses raw fields; subsequent player steps normalize them.
            gf[ids] = sample.guidance[ids]
            bf[ids] = sample.sdf_gradient[ids]
            delayed_gf, delayed_bf, delayed_sdf = gf, bf, sample.sdf
            delayed_command = command
        if self.arrival_control:
            # Match the Actor's desired speed to the command used by rewards.
            # Delayed point SDF/BF/GF observations retain their existing semantics.
            delayed_command = command
        if self.recovery is not None:
            # Exact known-box envelope is an offline research input. It is not
            # presented as delayed/noisy perception or proof of native contact.
            observation_sdf = torch.nan_to_num(
                envelope.clearance, posinf=10.0, neginf=0.0
            )
            bf = torch.nan_to_num(envelope.gradient)
            delayed_sdf, delayed_bf = observation_sdf, bf
            gf = normalized_gf
            delayed_gf = delayed_sample.guidance / (
                delayed_sample.guidance.norm(dim=-1, keepdim=True) + 1e-6
            )
            delayed_command = command
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
        observation_gf, observation_gf_delayed = gf, delayed_gf
        if self.risk_guidance is not None:
            from embodied_learning.tasks.cat_risk_guidance import (
                blend_guidance,
                risk_weights,
            )

            previous_risk = self.risk_alpha
            previous_delayed_risk = self.risk_alpha_delayed
            horizon = self.risk_guidance["horizon_s"]
            finite_velocity = torch.nan_to_num(
                velocity, nan=0.0, posinf=0.0, neginf=0.0
            )
            predicted = self.bank.sample_source(
                query_positions + horizon * finite_velocity,
                self.scene_ids,
                self.goal_ids,
            )
            predicted_delayed = self.bank.sample_source(
                delayed + horizon * finite_velocity, self.scene_ids, self.goal_ids
            )
            self.risk_alpha = risk_weights(
                sample.sdf,
                sample.sdf_gradient,
                velocity,
                predicted.sdf,
                self.risk_guidance,
            )
            self.risk_alpha_delayed = risk_weights(
                delayed_sample.sdf,
                delayed_sample.sdf_gradient,
                velocity,
                predicted_delayed.sdf,
                self.risk_guidance,
            )
            if same_step and previous_risk is not None:
                # Partial resets rebuild observations on the same physics step.
                # Continuing worlds retain their previous observation snapshot,
                # so their associated risk must retain that same snapshot too.
                self.risk_alpha[continuing] = previous_risk[continuing]
                self.risk_alpha_delayed[continuing] = previous_delayed_risk[
                    continuing
                ]
            observation_gf = blend_guidance(
                gf,
                command,
                self.risk_alpha,
                self.risk_guidance,
                walk_direction=self.gait.walk_vector
                if self.stop_transition
                else None,
            )
            observation_gf_delayed = blend_guidance(
                delayed_gf,
                delayed_command,
                self.risk_alpha_delayed,
                self.risk_guidance,
                walk_direction=self.gait.walk_vector
                if self.stop_transition
                else None,
            )
        x = payload["reward_inputs"]
        x.gf = gf
        x.bf = bf
        x.sdf = observation_sdf
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
        progress_gate = (
            self.feet_progress.update(
                root[:, :2],
                command[:, 1:3],
                step=common_step,
                initial=self.initial,
            )
            if self.feet_progress is not None
            else None
        )
        self.terms = cat_reward_terms(
            x,
            parameters=params,
            diagnostics=diagnostics,
            feet_progress_gate=progress_gate,
            contact_force_rate=(
                controller.contact_force.cost_rate
                if getattr(controller, "contact_force", None) is not None
                else None
            ),
            landing_speed_rate=(
                controller.landing.cost_rate
                if getattr(controller, "landing", None) is not None
                else None
            ),
            risk_alpha=self.risk_alpha,
            standing_contact_slip=getattr(self, "standing_contact_slip", False),
            standing_balance_free=getattr(self, "standing_balance_free", False),
            whole_body_joint_velocity=getattr(
                self, "whole_body_joint_velocity", False
            ),
        )
        goal_distance = (root[:, :2] - self.goals[:, :2]).norm(dim=-1)
        if self.standing_stability is not None:
            self.terms["standing_stability"] = standing_stability_bonus(
                torch.where(invalid, torch.inf, goal_distance),
                x.cmd_vel,
                x.root_vel,
                payload["gyro"],
                x.feet_contact,
            )
        if self.recovery is None:
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
        if self.recovery is not None:
            # The helper receives only source scales; research additions are
            # regenerated every step, rather than recursively accumulating.
            source_scales = {k: self.scales[k] for k in SOURCE_SCALES}
            self.terms, self.scales = research_reward_terms(
                self.terms,
                inputs=x,
                goal_delta=self.goals[:, :2] - root[:, :2],
                angular_velocity=payload["gyro"],
                mode=self.recovery.mode,
                source_scales=source_scales,
                config=self.recovery.config,
            )
            # Finite physics may still overflow squared reward errors. Such a
            # transition is invalid, rather than an infinite PPO target.
            invalid |= ~torch.stack(
                [torch.isfinite(term) for term in self.terms.values()]
            ).all(0)
            invalid |= ~torch.isfinite(
                aggregate_recovery_reward(self.terms, self.scales, dt=0.02)
            )
            command[invalid] = 0
            self.gait.last_command[invalid] = 0
            self.terms = {
                name: torch.where(invalid, 0.0, term)
                for name, term in self.terms.items()
            }
            failed = flags["fall"] | flags["collision"] | invalid
            # One terminal transition costs 10 reward units at 50 Hz. Keeping
            # signed rewards must not make immediate falling a cheap shortcut.
            self.terms["recovery_failure"] = -500.0 * failed.to(root.dtype)
            self.scales["recovery_failure"] = 1.0
            # Native contact can be real while all source point SDFs stay
            # positive. Penalize its first occurrence without changing the
            # historical episode horizon or treating cover overlap as contact.
            first_contact = (
                payload["physical_collision"]
                & ~self.ever_physical_collision
                & continuing
                & ~invalid
                & (not same_step)
            )
            self.terms["recovery_first_contact"] = -250.0 * first_contact.to(
                root.dtype
            )
            self.scales["recovery_first_contact"] = 1.0
            self._reward = aggregate_recovery_reward(
                self.terms, self.scales, dt=0.02
            )
        self.goal_distance = torch.where(invalid, torch.nan, goal_distance)
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
                "prohibited_contact": self.ever_collision.clone(),
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
                stop_command=(command[:, 1:].norm(dim=-1) < self.command_deadband),
                settled_stop=(self.settled_steps >= required_steps),
                root_speed=root_speed,
                root_angular_speed=angular_speed,
            )
            if self.recovery is not None:
                self.last_metrics.update(
                    goal_hold=(self.goal_distance < 0.1)
                    & (self.settled_steps >= required_steps),
                    envelope_clearance=envelope.minimum.clone(),
                    navigation_mode=self.recovery.mode.clone(),
                )
        self._obs = build_cat_observations(
            gyro=payload["gyro"],
            gravity=payload["gravity"],
            joint_pos=x.joint_pos,
            joint_vel=x.joint_vel,
            default_joints=controller.default,
            # The next decision observes the control just executed. Reward
            # history is separate and retains the two preceding controls.
            last_actions=controller.actions,
            targets=controller.targets,
            command=command,
            foot_height=self.foot_height,
            phase=self.gait.phase,
            phase_features=self.gait.phase_features
            if self.stop_transition
            else None,
            root_local_vel=payload["root_local_vel"],
            gf=observation_gf,
            bf=bf,
            sdf=observation_sdf,
            body_positions=positions,
            body_velocities=velocity,
            torso_rpy=x.torso_rpy,
            gait_mask=self.gait.gait_mask,
            feet_contact=x.feet_contact,
            kp_scale=controller.kp_scale,
            kd_scale=controller.kd_scale,
            rfi_scale=controller.rfi_scale,
            navigation_rotation=x.navigation_rotation,
            gf_delayed=observation_gf_delayed,
            bf_delayed=delayed_bf,
            sdf_delayed=delayed_sdf,
            command_delayed=delayed_command,
            noise_level=self.options["noise_level"],
            joint_pos_noise=0.01 if self.unitree_bridge else 0.03,
        )
        if self.recovery is not None:
            delta = torch.cat(
                (self.goals[:, :2] - root[:, :2], torch.zeros_like(root[:, :1])), -1
            )
            goal_local = (rotation.transpose(-1, -2) @ delta[..., None]).squeeze(-1)[
                :, :2
            ]
            mode = torch.nn.functional.one_hot(self.recovery.mode, 4).to(root)
            extra = torch.cat((goal_local, mode), -1)
            self._obs["actor"] = torch.nan_to_num(
                torch.cat((self._obs["actor"], payload["root_local_vel"], extra), -1)
            )
            self._obs["critic"] = torch.nan_to_num(
                torch.cat((self._obs["critic"], extra), -1)
            )
            self._obs["actor"][invalid] = 0
            self._obs["critic"][invalid] = 0
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
