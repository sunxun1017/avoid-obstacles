import pytest
import torch

from embodied_learning.robots.g1 import CAT_DEFAULT_JOINTS, CAT_JOINT_NAMES
from embodied_learning.simulators.mjlab.cat_control import CatJointController


def controller():
    return CatJointController(
        CAT_JOINT_NAMES, torch.full((29,), -2.0), torch.full((29,), 2.0), 2, "cpu"
    )


def test_source_joint_order_and_incremental_targets():
    ctrl = controller()
    before = ctrl.targets.clone()
    ctrl.process_action(torch.ones(2, 12) * 0.2)
    ctrl.process_action(torch.ones(2, 12) * 0.2)
    torch.testing.assert_close(ctrl.targets[:, :12], before[:, :12] + 0.2)
    ctrl.reset(torch.tensor([0]))
    torch.testing.assert_close(ctrl.targets[1, :12], before[1, :12] + 0.2)


def test_source_pd_torque_clip_each_substep():
    ctrl = controller()
    pos = ctrl.targets.clone()
    vel = torch.zeros_like(pos)
    torch.testing.assert_close(
        ctrl.compute_torques(pos, vel, rfi=torch.zeros_like(pos)),
        torch.zeros_like(pos),
    )
    torque = ctrl.compute_torques(pos - 100, vel, rfi=torch.zeros_like(pos))
    torch.testing.assert_close(torque, ctrl.limits.expand_as(torque))


def test_upper_body_default_hold():
    ctrl = controller()
    ctrl.process_action(torch.ones(2, 12))
    torch.testing.assert_close(
        ctrl.targets[:, 12:], torch.tensor(CAT_DEFAULT_JOINTS[12:]).expand(2, 17)
    )


def test_required_contacts_fail_fast():
    with pytest.raises(ValueError, match="joint"):
        CatJointController(
            CAT_JOINT_NAMES[:-1], torch.zeros(28), torch.ones(28), 1, "cpu"
        )


def test_motor_conversion_preserves_native_rotor_inertia():
    import re

    from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

    from embodied_learning.simulators.mjlab.cat_traversal import build_cat_cfg

    original = (
        unitree_g1_flat_env_cfg().scene.entities["robot"].articulation.actuators
    )
    converted = (
        build_cat_cfg(num_envs=1, seed=42)
        .scene.entities["robot"]
        .articulation.actuators
    )
    for name in CAT_JOINT_NAMES:
        native = [
            a
            for a in original
            if any(re.fullmatch(p, name) for p in a.target_names_expr)
        ]
        motor = [
            a
            for a in converted
            if any(re.fullmatch(p, name) for p in a.target_names_expr)
        ]
        assert len(native) == len(motor) == 1
        assert motor[0].armature == native[0].armature


def test_constraint_capacity_matches_native_g1_flat_default():
    from embodied_learning.simulators.mjlab.cat_traversal import build_cat_cfg

    cfg = build_cat_cfg(num_envs=4096, seed=42)
    assert cfg.sim.njmax == 300


def test_constraint_overflow_stops_before_reward_is_used():
    from embodied_learning.simulators.mjlab.cat_traversal import (
        check_physics_buffers,
    )

    with pytest.raises(RuntimeError, match="njmax=300.*1725"):
        check_physics_buffers(
            torch.tensor([0, 1]), torch.tensor([20, 1725]), njmax=300
        )
    check_physics_buffers(
        torch.zeros(2, dtype=torch.int32), torch.tensor([20, 48]), njmax=300
    )
