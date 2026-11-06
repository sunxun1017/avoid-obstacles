import torch

from embodied_learning.tasks.cat_phase_push_curriculum import S0PhasePushCurriculum


def test_selects_lateral_walking_and_braking_and_sagittal_stopping_once_per_episode():
    curriculum = S0PhasePushCurriculum(16, "cpu")
    distance = torch.tensor([1., 1., 1., 1., .65, .25, .07, .07] * 2)
    command = torch.tensor([0., 0., 0., 0., .4, .18, 0., 0.] * 2)
    speed = torch.tensor([0., 0., 0., 0., .30, .20, .05, .05] * 2)
    interval = torch.full((16,), 400)

    for tick in range(190, 205):
        ids = curriculum.select(torch.full((16,), tick), interval, distance, command, speed)
        expected = {190: [4, 5, 12, 13], 204: [6, 7, 14, 15]}.get(tick, [])
        assert ids.tolist() == expected
    assert curriculum.counts.tolist() == [2, 2, 4]

    curriculum.select(torch.ones(16, dtype=torch.long), interval, distance, command, speed)
    ids = curriculum.select(torch.full((16,), 30), interval, distance, command, speed)
    assert ids.tolist() == [4, 5, 12, 13]
    delta = curriculum.impulse(ids, 48000)
    assert (delta[:, 0] == 0).all()
    assert (delta[:2, 1] > 0).all() and (delta[2:, 1] < 0).all()
    assert (delta.norm(dim=1) <= .45).all()


def test_keeps_original_push_separate_and_increases_impulse_cap():
    curriculum = S0PhasePushCurriculum(8, "cpu")
    interval = torch.full((8,), 250)
    distance = torch.tensor([1., 1., 1., 1., .65, .25, .07, .07])
    command = torch.tensor([0., 0., 0., 0., .4, .18, 0., 0.])
    speed = torch.tensor([0., 0., 0., 0., .30, .20, .05, .05])
    for tick in range(230, 265):
        assert curriculum.select(torch.full((8,), tick), interval, distance, command, speed).numel() == 0
    ids = curriculum.select(torch.full((8,), 301), interval, distance, command, speed)
    assert ids.tolist() == [4, 5, 6, 7]
    torch.manual_seed(17)
    gentle = curriculum.impulse(ids, 0)
    torch.manual_seed(17)
    difficult = curriculum.impulse(ids, 48000)
    assert (gentle.norm(dim=1) <= .20).all()
    assert (difficult.norm(dim=1) > gentle.norm(dim=1)).all()
    assert (gentle[:2, 0] == 0).all() and (gentle[2:, 1] == 0).all()
    assert gentle[2, 0] > 0 and gentle[3, 0] < 0


def test_near_goal_starts_only_use_half_the_stopping_slots_and_count_partial_resets():
    ids = torch.tensor([1, 4, 6, 7, 12, 14, 15, 22, 23, 28, 30, 31])
    default = S0PhasePushCurriculum(32, "cpu")
    assert default.near_goal_reset_slots(ids).tolist() == []

    curriculum = S0PhasePushCurriculum(32, "cpu", near_goal_starts=True)
    reset_ids = curriculum.near_goal_reset_slots(ids)
    assert reset_ids.tolist() == [6, 7, 22, 23]
    assert (curriculum.phase[reset_ids] == 2).all()
    assert curriculum.direction[reset_ids, 0].tolist() == [1, -1, 1, -1]
    assert curriculum.near_goal_reset_count.item() == 4
    assert curriculum.near_goal_reset_slots(torch.tensor([6, 20])).tolist() == [6]
    assert curriculum.near_goal_reset_count.item() == 5
