import torch

from embodied_learning.tasks.cat_reward import standing_stability_bonus


def test_stability_bonus_penalizes_motion_on_every_base_axis_when_stopped():
    n = 7
    zeros = torch.zeros(n, 3)
    angular = zeros.clone()
    angular[1:4] = torch.eye(3) * .35
    angular[6, 0] = torch.nan
    linear = zeros.clone()
    linear[4, 1] = .20
    linear[5, 2] = .40
    result = standing_stability_bonus(
        torch.full((n,), .08), zeros, linear, angular, torch.ones(n, 2),
    )
    torch.testing.assert_close(result, torch.tensor([1., .5, .5, .5, .5, .2, 0.]))


def test_no_extra_reward_away_from_goal_or_during_walking_and_recovery_steps():
    zeros = torch.zeros(4, 3)
    command = zeros.clone()
    command[2, 0] = .04
    feet_contact = torch.ones(4, 2)
    feet_contact[3, 0] = 0
    result = standing_stability_bonus(
        torch.tensor([.10, .11, .08, .08]), command, zeros, zeros, feet_contact,
    )
    assert result.tolist() == [1., 0., 0., 0.]
