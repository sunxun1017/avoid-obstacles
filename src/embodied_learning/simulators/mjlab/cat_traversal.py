"""Native mjlab assembly for the source G1Cat task.

Robot model/contact geometry remain mjlab assets; source-name mappings are explicit.
"""

from dataclasses import replace

from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from embodied_learning.robots.g1 import (
    CAT_DEFAULT_JOINTS,
    CAT_JOINT_NAMES,
    G1_ADDED_SITES,
)
from embodied_learning.simulators.mjlab.cat_control import CatActionCfg


def build_cat_cfg(
    *, num_envs: int, seed: int, random_pd: bool = True, random_rfi: bool = True
):
    cfg = unitree_g1_flat_env_cfg()
    cfg.scene.num_envs = num_envs
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
                    target_names_expr=(".*",), effort_limit=139.0
                ),
            ),
            soft_joint_pos_limit_factor=0.95,
        ),
        init_state=replace(
            robot.init_state,
            joint_pos=dict(zip(CAT_JOINT_NAMES, CAT_DEFAULT_JOINTS)),
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
