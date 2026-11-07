"""Walking/standing handoff hysteresis and control continuity."""
import torch

from embodied_learning.policies.cat_stand_handoff import StandHandoff


def test_walk_until_supported_then_stand_and_return_only_after_recovery():
    handoff = StandHandoff(2, "cpu")
    reset = torch.zeros(2, dtype=torch.bool)
    support = torch.tensor([True, False])
    near = torch.tensor([.04, .04])
    still = torch.tensor([.05, .05])
    for _ in range(4):
        handoff.update(reset, near, still, still, support)
    assert not handoff.standing.any()
    for _ in range(10):
        handoff.update(reset, near, still, still, support)
    assert handoff.standing.tolist() == [True, False]
    torch.testing.assert_close(handoff.weight, torch.tensor([1., 0.]))
    displaced = torch.tensor([.15, .15])
    unsettled = torch.tensor([.9, .9])
    for _ in range(8):
        handoff.update(reset, displaced, still, unsettled, support)
    assert handoff.standing[0]
    for _ in range(5):
        handoff.update(reset, displaced, still, still, support)
    assert not handoff.standing.any()
    assert 0 < handoff.weight[0] < 1
    handoff.update(torch.tensor([True, False]), near, still, still, support)
    assert not handoff.weight.any()


def test_mix_limits_only_the_transition_and_reaches_both_controllers():
    handoff = StandHandoff(3, "cpu")
    handoff.weight.copy_(torch.tensor([0., .1, 1.]))
    handoff.transitioning.copy_(torch.tensor([False, True, False]))
    previous = torch.zeros(3, 2)
    walk = torch.full_like(previous, .3)
    stand = torch.full_like(previous, -.3)
    expected = torch.tensor([[.3, .3], [.06, .06], [-.3, -.3]])
    torch.testing.assert_close(handoff.mix_targets(previous, walk, stand), expected)


def test_large_displacement_releases_standing_without_waiting():
    handoff = StandHandoff(1, "cpu")
    for _ in range(14):
        handoff.update(
            torch.tensor([False]), torch.tensor([.04]), torch.tensor([.05]),
            torch.tensor([.05]), torch.tensor([True]),
        )
    handoff.update(
        torch.tensor([False]), torch.tensor([.32]), torch.tensor([.6]),
        torch.tensor([1.5]), torch.tensor([False]),
    )
    assert not handoff.standing[0]
    assert handoff.transitioning[0]
    torch.testing.assert_close(handoff.weight, torch.tensor([.9]))


def test_unitree_goal_commands_stand_through_impulse_then_recenter_with_hysteresis():
    from embodied_learning.policies.cat_stand_handoff import UnitreeGoalCommands

    correction = UnitreeGoalCommands(2, "cpu")
    no = torch.tensor([False, False])
    yes = torch.tensor([True, True])
    zero = torch.zeros(2)
    step = torch.tensor([8, 8])
    error = torch.tensor([[.20, 0.], [0., -.20]])
    command = correction.update(no, yes, error, torch.tensor([.55, .05]), zero, yes, step)
    assert command[0].eq(0).all()
    torch.testing.assert_close(command[1, :3], torch.tensor([0., -.3, 0.]))
    assert command[1, 3:].norm() > .99
    error[1, 1] = -.06
    command = correction.update(no, yes, error, zero, zero, yes, step)
    torch.testing.assert_close(command[1, :3], torch.tensor([0., -.12, 0.]))
    error[1, 1] = -.03
    command = correction.update(no, yes, error, zero, zero, yes, step)
    assert command[1].eq(0).all()
    command = correction.update(yes, yes, error, zero, zero, yes, step)
    assert command.eq(0).all()
