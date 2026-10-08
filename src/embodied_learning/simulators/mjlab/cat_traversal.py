"""Native mjlab assembly for the source G1Cat task.

Robot model/contact geometry remain mjlab assets; source-name mappings are explicit.
"""

from dataclasses import replace

import torch
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat

from embodied_learning.robots.g1 import (
    CAT_ACTION_NAMES,
    CAT_DEFAULT_JOINTS,
    CAT_JOINT_NAMES,
    CAT_LEG_BODIES,
    CAT_OBS_NAMES,
    G1_ADDED_SITES,
    G1_BODY_SCHEMA,
)
from embodied_learning.simulators.mjlab.cat_control import CatActionCfg
from embodied_learning.simulators.mjlab.field_traversal import (
    _box_spec,
    _reset_field,
)
from embodied_learning.tasks.cat_reward import CatRewardInputs
from embodied_learning.tasks.cat_traversal import CatTaskState


def build_cat_cfg(
    *, num_envs: int, seed: int, random_pd: bool = True, random_rfi: bool = True
):
    cfg = unitree_g1_flat_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 512
    cfg.seed = seed
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    cfg.episode_length_s = 20.0
    robot = cfg.scene.entities["robot"]
    cfg.scene.entities["robot"] = replace(
        robot,
        articulation=replace(
            robot.articulation,
            actuators=(
                BuiltinMotorActuatorCfg(
                    target_names_expr=CAT_JOINT_NAMES, effort_limit=139.0
                ),
            ),
            soft_joint_pos_limit_factor=0.95,
        ),
        init_state=replace(
            robot.init_state,
            joint_pos=dict(zip(CAT_JOINT_NAMES, CAT_DEFAULT_JOINTS, strict=True)),
            joint_vel={".*": 0.0},
            pos=(0.0, 0.0, 0.8),
        ),
    )
    original = robot.spec_fn

    def spec_fn():
        spec = original()
        for name, (body, pos) in G1_ADDED_SITES.items():
            spec.body(body).add_site(name=name, pos=pos, size=(0.01, 0, 0))
        return spec

    cfg.scene.entities["robot"] = replace(
        cfg.scene.entities["robot"], spec_fn=spec_fn
    )
    cfg.actions = {
        "joint_pos": CatActionCfg(
            entity_name="robot", random_pd=random_pd, random_rfi=random_rfi
        )
    }
    return cfg


def rpy(rotation):
    # Signed atan2 roll/pitch/yaw, equivalent to jaxlie SO3.as_rpy_radians.
    return torch.stack(
        (
            torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2]),
            torch.atan2(
                -rotation[:, 2, 0],
                torch.sqrt(rotation[:, 2, 1].square() + rotation[:, 2, 2].square()),
            ),
            torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0]),
        ),
        1,
    )


def read_cat_reward_inputs(env, state):
    robot = env.scene["robot"]
    data = robot.data
    c = env.cat_controller
    site = [robot.site_names.index(name) for name in G1_BODY_SCHEMA]
    positions = data.site_pose_w[:, site, :3] - env.scene.env_origins[:, None]
    site_rotation = matrix_from_quat(data.site_pose_w[:, site, 3:7])
    root_rotation = matrix_from_quat(data.root_link_quat_w)
    pelvis = site_rotation[:, 1]
    torso = site_rotation[:, 2]
    yaw = torch.atan2(pelvis[:, 1, 0], pelvis[:, 0, 0])
    cs = torch.cos(yaw)
    sn = torch.sin(yaw)
    nav = torch.zeros_like(pelvis)
    nav[:, 0, 0] = cs
    nav[:, 1, 0] = sn
    nav[:, 0, 1] = -sn
    nav[:, 1, 1] = cs
    nav[:, 2, 2] = 1
    native_body_vel = data.body_link_vel_w
    torso_id = robot.body_names.index("torso_link")
    torso_ang = (
        nav.transpose(-1, -2) @ native_body_vel[:, torso_id, 3:, None]
    ).squeeze(-1)
    root_vel = data.root_link_vel_w[:, :3]
    gyro = (pelvis.transpose(-1, -2) @ data.root_link_vel_w[:, 3:, None]).squeeze(-1)
    gravity = (
        pelvis.transpose(-1, -2)
        @ positions.new_tensor([0.0, 0.0, -1.0]).expand(len(positions), 3)[..., None]
    ).squeeze(-1)
    feet = torch.stack(
        [
            env.scene[n].data.found.flatten(1).any(1)
            for n in ("cat_left_ground", "cat_right_ground")
        ],
        1,
    ).to(positions)
    self_contact = torch.stack(
        [
            env.scene[n].data.found.flatten(1).any(1)
            for n in ("cat_feet_self", "cat_left_foot_shin", "cat_right_foot_shin")
        ],
        1,
    ).any(1)
    physical = torch.zeros(len(positions), device=env.device, dtype=torch.bool)
    for n in env.cat_obstacle_contacts:
        physical |= env.scene[n].data.found.flatten(1).any(1)
    legs = [robot.body_names.index(n) for n in CAT_LEG_BODIES]
    global_root = robot.indexing.root_body_id
    com = env.sim.data.subtree_com[:, global_root] - env.scene.env_origins
    q = data.joint_pos[:, c.indices]
    v = data.joint_vel[:, c.indices]
    zeros = torch.zeros_like(positions)
    x = CatRewardInputs(
        pelvis_rpy=rpy(nav.transpose(-1, -2) @ pelvis),
        torso_rpy=rpy(nav.transpose(-1, -2) @ torso),
        head_pos=positions[:, 0],
        cmd_vel=zeros[:, 0],
        root_vel=root_vel,
        torso_ang_vel=torso_ang,
        leg_rotation=matrix_from_quat(data.body_link_pose_w[:, legs, 3:]),
        foot_pos=positions[:, 3:5],
        foot_vel=data.site_vel_w[:, site[3:5], :3],
        hands_pos=positions[:, 5:7],
        pelvis_subtree_com=com,
        feet_contact=feet,
        gait_mask=state.gait.gait_mask,
        move_flag=state.gait.last_command[:, 0],
        foot_height=state.foot_height,
        joint_pos=q,
        joint_vel=v,
        previous_vel=state.previous_vel,
        torques=data.qfrc_actuator[:, c.indices],
        actions=c.actions,
        last_actions=c.last_actions,
        last_last_actions=c.last_last_actions,
        gf=zeros,
        bf=zeros,
        sdf=zeros[:, :, 0],
        navigation_rotation=nav,
        body_velocities=zeros,
    )
    return dict(
        reward_inputs=x,
        body_positions=positions,
        root_position=data.root_link_pos_w - env.scene.env_origins,
        root_rotation=root_rotation,
        gyro=gyro,
        gravity=gravity,
        up_z=pelvis[:, 2, 2],
        self_contact=self_contact,
        physical_collision=physical,
        root_local_vel=(pelvis.transpose(-1, -2) @ root_vel[..., None]).squeeze(-1),
        invalid=(
            ~torch.isfinite(q).all(-1)
            | ~torch.isfinite(v).all(-1)
            | ~torch.isfinite(data.root_link_vel_w).all(-1)
        ),
    )


def _refresh(env):
    state = env.cat_state
    if state._key != (env.common_step_counter, state.generation):
        env.sim.forward()
        env.sim.sense()
        state.refresh(
            read_cat_reward_inputs(env, state),
            env.cat_controller,
            env.episode_length_buf,
            env.common_step_counter,
        )


def _source_obs(env, group):
    _refresh(env)
    return env.cat_state.observations()[group]


def _source_reward(env):
    _refresh(env)
    # Native RewardManager multiplies this rate by dt, yielding the source total.
    return env.cat_state.reward() / env.step_dt


def _source_done(env, name):
    _refresh(env)
    return env.cat_state.outcomes()[name]


def _source_reset(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    _reset_field(env, env_ids)
    env.cat_state.reset(env_ids)


def make_cat_env(manifest, bank, *, scene_ids, num_envs, device, seed, options):
    names = tuple(s.id for s in manifest.scenes)
    if not scene_ids or any(n not in names for n in scene_ids):
        raise ValueError("Unknown source scene.")
    state = CatTaskState(
        manifest,
        bank,
        tuple(names.index(n) for n in scene_ids),
        num_envs,
        device,
        options,
    )
    cfg = build_cat_cfg(
        num_envs=num_envs,
        seed=seed,
        random_pd=state.options["random_pd"],
        random_rfi=state.options["random_rfi"],
    )
    cfg.commands = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.events = {"cat_reset": EventTermCfg(func=_source_reset, mode="reset")}
    cfg.observations = {
        g: ObservationGroupCfg(
            terms={
                "source": ObservationTermCfg(func=_source_obs, params={"group": g})
            },
            enable_corruption=False,
        )
        for g in ("actor", "critic")
    }
    cfg.rewards = {"source": RewardTermCfg(func=_source_reward, weight=1.0)}
    cfg.terminations = {
        n: TerminationTermCfg(
            func=_source_done, params={"name": n}, time_out=n == "timeout"
        )
        for n in ("fall", "collision", "invalid", "timeout")
    }
    sensors = []

    def contact(name, primary, secondary):
        sensors.append(
            ContactSensorCfg(
                name=name,
                primary=primary,
                secondary=secondary,
                fields=("found",),
                reduce="netforce",
                num_slots=1,
            )
        )

    for side in ("left", "right"):
        contact(
            f"cat_{side}_ground",
            ContactMatch(
                mode="geom", pattern=f"{side}_foot.*_collision", entity="robot"
            ),
            ContactMatch(mode="body", pattern="terrain"),
        )
    contact(
        "cat_feet_self",
        ContactMatch(mode="geom", pattern="left_foot.*_collision", entity="robot"),
        ContactMatch(mode="geom", pattern="right_foot.*_collision", entity="robot"),
    )
    for side, other in [("left", "right"), ("right", "left")]:
        contact(
            f"cat_{side}_foot_shin",
            ContactMatch(
                mode="geom", pattern=f"{side}_foot.*_collision", entity="robot"
            ),
            ContactMatch(
                mode="geom", pattern=f"{other}_shin_collision", entity="robot"
            ),
        )
    obstacle_contacts = []
    for i, scene in enumerate(manifest.scenes):
        for j, box in enumerate(scene.boxes):
            entity = f"obstacle_{i}_{j}"
            cfg.scene.entities[entity] = EntityCfg(
                spec_fn=lambda half=box.half_size: _box_spec(half),
                init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
            )
            name = f"cat_obstacle_{i}_{j}"
            obstacle_contacts.append(name)
            contact(
                name,
                ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                ContactMatch(mode="body", pattern="box", entity=entity),
            )
    cfg.scene.sensors = tuple(sensors)

    class CatNativeEnv(ManagerBasedRlEnv):
        def __init__(self):
            self.cat_state = self.field_state = state
            self.cat_obstacle_contacts = obstacle_contacts
            super().__init__(cfg=cfg, device=device)

        def step(self, actions):
            state.advance_before_physics(actions)
            if state.options["push"]:
                push = ((self.episode_length_buf + 1) % state.push_interval) == 0
                if push.any():
                    ids = torch.nonzero(push, as_tuple=True)[0]
                    root = torch.cat(
                        (
                            self.scene["robot"].data.root_link_pose_w[ids],
                            self.scene["robot"].data.root_link_vel_w[ids],
                        ),
                        1,
                    ).clone()
                    angle = 2 * torch.pi * torch.rand(len(ids), device=device)
                    magnitude = 0.1 + 0.9 * torch.rand(len(ids), device=device)
                    root[:, 7:9] += (
                        torch.stack((torch.cos(angle), torch.sin(angle)), 1)
                        * magnitude[:, None]
                    )
                    self.scene["robot"].write_root_state_to_sim(root, env_ids=ids)
            return super().step(actions)

    return CatEnv(CatNativeEnv())


class CatEnv(RslRlVecEnvWrapper):
    def __init__(self, native):
        super().__init__(native, clip_actions=None)
        self.cat_state = self.field_state = native.cat_state
        self.scene_ids = self.cat_state.scene_ids
        state = self.cat_state
        self.schema = dict(
            task_profile="cat_source_v1",
            source_commit="866ba392f1c1e84b92ad75fa66550f26e8af8e48",
            scene_hash=state.manifest.identity,
            scene_names=[s.id for s in state.manifest.scenes],
            body_sites=list(G1_BODY_SCHEMA),
            joint_names=list(CAT_JOINT_NAMES),
            action_names=list(CAT_ACTION_NAMES),
            obs_joint_names=list(CAT_OBS_NAMES),
            action_scale=0.5,
            task=state.options,
            source_profile=state.bank.source_profile,
            physics={"timestep": 0.002, "decimation": 10},
            observation="cat_source_actor162_critic250_v1",
            normalization=False,
        )

    @property
    def training_step(self):
        return self.unwrapped.common_step_counter

    @training_step.setter
    def training_step(self, value):
        self.unwrapped.common_step_counter = value
        self.cat_state._key = None

    def reset_slots(self, slots):
        self.unwrapped.reset(env_ids=slots)

    def step(self, actions):
        result = super().step(actions)
        state = self.cat_state
        result[3]["field_outcomes"] = state.saved_outcomes
        result[3]["cat_metrics"] = state.last_metrics
        log = result[3].setdefault("log", {})
        for name, v in state.saved_terms.items():
            log["CatReward/" + name] = v.mean()
            log["CatWeighted/" + name] = (v * state.scales[name]).mean()
        for name, v in state.saved_reward_diagnostics.items():
            log["CatRewardDiagnostic/" + name] = v.float().mean()
        for name, v in state.last_metrics.items():
            finite = torch.isfinite(v)
            safe = torch.where(finite, v.float(), 0.0)
            log["Cat/" + name] = safe.sum() / finite.sum().clamp_min(1)
        log["Cat/clipped_reward"] = result[1].mean()
        return result
