"""Joint control must distinguish motion, preserve margins and charge failures."""

import pytest
import torch


def test_relative_motion_and_full_body_radius():
    from embodied_learning.tasks.cat_dynamic import predict_sphere_clearance
    p = torch.tensor([[[0., 0., 1.]], [[0., 0., 1.]]])
    obstacle = torch.tensor([[1., 0., 1.], [1., 0., 1.]])
    velocity = torch.tensor([[-1., 0., 0.], [1., 0., 0.]])
    d = predict_sphere_clearance(p, torch.tensor([.1]), torch.zeros_like(p),
                                 obstacle, velocity, torch.tensor([.1, .1]))
    torch.testing.assert_close(d, torch.tensor([0., .8]), atol=1e-6, rtol=0)


def test_chest_sphere_does_not_release_head_posture():
    from embodied_learning.tasks.cat_dynamic import head_allowance
    head = torch.tensor([[[0., 0., 1.35]], [[0., 0., 1.35]]])
    ball = torch.tensor([[0., 0., .9], [0., 0., 1.35]])
    alpha = head_allowance(head, torch.tensor([.1]), torch.zeros_like(head),
                           ball, torch.zeros_like(ball), torch.tensor([.1, .1]))
    torch.testing.assert_close(alpha, torch.tensor([0., 1.]))


def test_waiting_does_not_remove_nominal_forward_risk():
    from embodied_learning.tasks.cat_dynamic import predict_sphere_clearance, nominal_risk
    p = torch.tensor([[[0., 0., 1.]]])
    d = predict_sphere_clearance(p, torch.tensor([.1]),
                                 torch.tensor([[[.5, 0., 0.]]]),
                                 torch.tensor([[.5, 0., 1.]]),
                                 torch.zeros(1, 3), torch.tensor([.1]))
    assert nominal_risk(d, torch.tensor([True])).item() == 1
    assert nominal_risk(d, torch.tensor([False])).item() == 0


def test_progress_cannot_be_farmed_and_partial_reset_is_local():
    from embodied_learning.tasks.cat_dynamic import JointMemory
    m = JointMemory(2, 'cpu')
    delta = m.update(torch.tensor([.1, .2]), torch.zeros(2), torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(delta, torch.tensor([.1, .2]))
    assert m.update(torch.tensor([.05, .1]), torch.zeros(2), torch.ones(2, dtype=torch.bool)).sum() == 0
    assert m.update(torch.tensor([.1, .2]), torch.zeros(2), torch.ones(2, dtype=torch.bool)).sum() == 0
    m.reset(torch.tensor([0]))
    torch.testing.assert_close(m.best, torch.tensor([0., .2]))


def test_safe_stall_is_penalized_but_dangerous_wait_is_not_stall():
    from embodied_learning.tasks.cat_dynamic import JointMemory
    m = JointMemory(2, 'cpu')
    for _ in range(102):
        m.update(torch.zeros(2), torch.tensor([0., 1.]), torch.ones(2, dtype=torch.bool))
    assert m.stalled.tolist() == [True, False]
    assert m.features(torch.ones(2)).shape == (2, 4)


def test_failure_and_timeout_are_signed_outside_rate_limit():
    from embodied_learning.tasks.cat_dynamic_reward import joint_reward
    r, _ = joint_reward(delta_progress=torch.zeros(3), gait_cost=torch.ones(3)*100,
        clearance=torch.ones(3), margins=torch.ones(3, 2), velocity_xy=torch.zeros(3, 2),
        nominal_xy=torch.zeros(3, 2), risk=torch.zeros(3), stalled=torch.zeros(3, dtype=torch.bool),
        failed=torch.tensor([True, False, False]), success=torch.tensor([True, True, False]),
        timeout=torch.tensor([True, False, True]))
    torch.testing.assert_close(r, torch.tensor([-25.1, 9.9, -10.1]))


def test_gait_cost_keeps_posture_without_constant_standing_bonus():
    from embodied_learning.tasks.cat_dynamic_reward import gait_cost
    terms = dict(tracking_orientation=torch.tensor([1., .5]), body_rotation=torch.ones(2),
                 headgf=torch.ones(2)*4, smoothness_action=torch.zeros(2))
    scales = dict(tracking_orientation=2., body_rotation=1., headgf=1., smoothness_action=-.001)
    raw, scaled = gait_cost(terms, scales, scale=1.)
    torch.testing.assert_close(raw, torch.tensor([0., 1.]))
    torch.testing.assert_close(scaled, torch.tensor([0., 1.]))


def test_geometry_rejects_wrong_batch_and_negative_radius():
    from embodied_learning.tasks.cat_dynamic import predict_sphere_clearance
    with pytest.raises(ValueError):
        predict_sphere_clearance(torch.zeros(2, 1, 3), torch.tensor([-.1]), torch.zeros(2, 1, 3),
                                 torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2)*.1)
