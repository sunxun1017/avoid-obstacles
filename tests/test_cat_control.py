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
