import pytest

from embodied_learning.evaluation.cat_route_probe import route_goal_ids


def test_route_keeps_original_lateral_target_and_field_index():
    goals = [(1.8, -0.4, 0.8), (1.2, 0.4, 0.8), (1.2, -0.4, 0.8), (1.8, 0.4, 0.8)]
    assert route_goal_ids(goals, far=False) == [2, 1, 2, 1]
    assert route_goal_ids(goals, far=True) == [0, 3, 0, 3]


def test_route_rejects_missing_same_lane_distance_control():
    with pytest.raises(ValueError, match="two distinct"):
        route_goal_ids([(1.2, 0, 0.8), (1.8, 0.4, 0.8)], far=True)


def test_post_landing_stop_uses_command_transition_and_keeps_censoring():
    from embodied_learning.evaluation.cat_route_probe import first_stop_after_landing

    assert first_stop_after_landing([True, True, False, True], 0.021, 50) == pytest.approx(0.04)
    assert first_stop_after_landing([True, True, True], 0.021, 50) is None
    assert first_stop_after_landing([True, False, False], 0.051, 50) == pytest.approx(0.06)
