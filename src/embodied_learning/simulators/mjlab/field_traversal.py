"""Native mjlab G1 known-geometry field task; no sensors or online mapping.

Contact sensors classify physical obstacle contact only. Mocap obstacle
entities permit distinct scenes in batched worlds sharing one MuJoCo model.
"""

from __future__ import annotations

from dataclasses import replace

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from embodied_learning.data.field_scenes import SceneManifest
from embodied_learning.robots.g1 import G1_ADDED_SITES, G1_BODY_SCHEMA
from embodied_learning.tasks.field_assets import FieldBank
from embodied_learning.tasks.field_traversal import FieldTaskState


def _field_obs(env):
    # Native forward refreshes kinematics after reward/reset at the same step.
    env.field_state._key = None
    return env.field_state.observation(env)


def _field_reward(env):
    return env.field_state.reward(env)


def _termination(env, name: str):
    return env.field_state.outcomes(env)[name]


def _reset_field(env, env_ids):
    state = env.field_state
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    count = len(env_ids)
    allowed = torch.tensor(state.allowed, device=env.device)
    selected = allowed[torch.randint(len(allowed), (count,), device=env.device)]
    state.scene_ids[env_ids] = selected
    robot = env.scene["robot"]
    roots = robot.data.default_root_state[env_ids].clone()
    roots[:, 3:7] = roots.new_tensor([1.0, 0.0, 0.0, 0.0])
    roots[:, 7:] = 0.0
    for scene_id, scene in enumerate(state.manifest.scenes):
        slots = env_ids[selected == scene_id]
        if slots.numel() == 0:
            continue
        start_ids = torch.randint(
            len(scene.starts), (len(slots),), device=env.device
        )
        goal_ids = torch.randint(len(scene.goals), (len(slots),), device=env.device)
        starts = roots.new_tensor(scene.starts)[start_ids]
        roots[selected == scene_id, :3] = starts + env.scene.env_origins[slots]
        state.goal_ids[slots] = goal_ids
        state.goals[slots] = roots.new_tensor(scene.goals)[goal_ids]
    robot.write_root_state_to_sim(roots, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[env_ids],
        robot.data.default_joint_vel[env_ids],
        env_ids=env_ids,
    )
    for scene_id, scene in enumerate(state.manifest.scenes):
        for box_id, box in enumerate(scene.boxes):
            pose = torch.zeros(count, 7, device=env.device)
            pose[:, 3] = 1.0
            pose[:, :3] = pose.new_tensor(box.center)
            pose[selected != scene_id, 2] = -100.0
            pose[:, :3] += env.scene.env_origins[env_ids]
            env.scene[f"obstacle_{scene_id}_{box_id}"].write_mocap_pose_to_sim(
                pose,
                env_ids=env_ids,
            )
    state.generation += 1


def _box_spec(half_size):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box")
    body.add_geom(
        name="collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half_size,
        rgba=(0.6, 0.4, 0.2, 1),
        friction=(1.0, 0.005, 0.001),
    )
    return spec


class FieldEnv(RslRlVecEnvWrapper):
    """Training adapter exposing independent scene IDs and ordered metadata."""

    def __init__(self, native: ManagerBasedRlEnv) -> None:
        super().__init__(native, clip_actions=5.0)
        self.field_state = native.field_state
        self.scene_ids = self.field_state.scene_ids
        self.schema = {
            "body_sites": list(G1_BODY_SCHEMA),
            "scene_hash": self.field_state.manifest.identity,
            "scene_names": [s.id for s in self.field_state.manifest.scenes],
            "joint_names": list(native.scene["robot"].joint_names),
            "action_scale": native.action_manager.get_term("joint_pos")
            ._scale.flatten()
            .tolist(),
            "action_offset": native.scene["robot"]
            .data.default_joint_pos[0]
            .tolist(),
            "action_clip": 5.0,
            "observation": "proprio+field_guidance+body_velocity+sdf_v1",
        }

    def reset_slots(self, slots: torch.Tensor) -> None:
        """Reset selected slots, then refresh native kinematics and sensors."""
        self.unwrapped._reset_idx(slots)
        self.unwrapped.sim.forward()
        self.field_state.generation += 1

    def step(self, actions: torch.Tensor):
        result = super().step(actions)
        result[3]["field_outcomes"] = self.field_state.step_outcomes
        return result


def make_field_env(
    manifest: SceneManifest,
    bank: FieldBank,
    *,
    scene_ids: tuple[str, ...],
    num_envs: int,
    device: str,
    seed: int,
) -> FieldEnv:
    """Build one real G1 per slot with reset-routed known scene geometry."""
    names = tuple(s.id for s in manifest.scenes)
    if num_envs < 1 or not scene_ids or any(n not in names for n in scene_ids):
        raise ValueError("Invalid environment count or scene selection.")
    state = FieldTaskState(
        manifest, bank, tuple(names.index(n) for n in scene_ids), num_envs, device
    )
    cfg = unitree_g1_flat_env_cfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.commands, cfg.curriculum, cfg.metrics = {}, {}, {}
    cfg.events = {"field_reset": EventTermCfg(func=_reset_field, mode="reset")}
    cfg.scene.sensors = ()
    original = cfg.scene.entities["robot"].spec_fn

    def robot_spec():
        spec = original()
        for name, (body_name, pos) in G1_ADDED_SITES.items():
            spec.body(body_name).add_site(name=name, pos=pos, size=(0.01, 0, 0))
        return spec

    cfg.scene.entities["robot"] = replace(
        cfg.scene.entities["robot"], spec_fn=robot_spec
    )
    contacts = []
    for scene_id, scene in enumerate(manifest.scenes):
        for box_id, box in enumerate(scene.boxes):
            entity_name = f"obstacle_{scene_id}_{box_id}"
            cfg.scene.entities[entity_name] = EntityCfg(
                spec_fn=lambda half=box.half_size: _box_spec(half),
                init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
            )
            contacts.append(
                ContactSensorCfg(
                    name=f"contact_{scene_id}_{box_id}",
                    primary=ContactMatch(
                        mode="subtree", pattern="pelvis", entity="robot"
                    ),
                    secondary=ContactMatch(
                        mode="body", pattern="box", entity=entity_name
                    ),
                    fields=("found",),
                    reduce="netforce",
                    num_slots=1,
                )
            )
    cfg.scene.sensors = tuple(contacts)
    for group in cfg.observations.values():
        group.terms = {
            k: v
            for k, v in cfg.observations["actor"].terms.items()
            if k != "command"
        }
        group.terms["body_fields"] = ObservationTermCfg(func=_field_obs)
        group.enable_corruption = False
    cfg.rewards = {k: cfg.rewards[k] for k in ("dof_pos_limits", "action_rate_l2")}
    cfg.rewards["field"] = RewardTermCfg(func=_field_reward, weight=1.0)
    cfg.terminations = {
        name: TerminationTermCfg(
            func=_termination,
            params={"name": name},
            time_out=(name == "timeout"),
        )
        for name in ("goal", "collision", "fall", "invalid", "timeout")
    }

    class NativeFieldEnv(ManagerBasedRlEnv):
        def __init__(self):
            self.field_state = state
            self.field_contact_names = tuple(s.name for s in contacts)
            super().__init__(cfg=cfg, device=device)

    return FieldEnv(NativeFieldEnv())
