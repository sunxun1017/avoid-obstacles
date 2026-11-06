import pytest

from embodied_learning.evaluation.cat_terminal_gate import (
    classify_terminal,
    promotion_passed,
    update_hold,
)


def test_position_and_speeds_must_hold_together():
    ordinary = strict = 0
    for _ in range(24):
        ordinary, strict = update_hold(ordinary, strict, 0.1, 0.09, 0.09)
    assert (ordinary, strict) == (24, 24)
    assert update_hold(ordinary, strict, 0.101, 0, 0) == (0, 0)
    assert update_hold(ordinary, strict, 0.1, 0.1, 0.19) == (25, 0)
    assert update_hold(ordinary, strict, 0, 0.2, 0) == (0, 0)


@pytest.mark.parametrize(
    "steps,hold,strict,invalid,fall,contact,reached,expected",
    [
        (1000, 25, 25, False, False, False, True, (True, True, None)),
        (1000, 25, 24, False, False, False, True, (True, False, None)),
        (1000, 24, 24, False, False, False, True, (False, False, "not_settled")),
        (1000, 0, 0, False, False, False, False, (False, False, "not_reached")),
        (1000, 30, 30, False, False, True, True, (False, False, "prohibited_contact")),
        (500, 30, 30, False, True, True, True, (False, False, "fall")),
        (500, 0, 0, True, True, True, False, (False, False, "invalid")),
    ],
)
def test_terminal_classification(steps, hold, strict, invalid, fall, contact, reached, expected):
    assert classify_terminal(
        steps, hold, strict, invalid=invalid, fall=fall, contact=contact, reached=reached
    ) == expected


def test_early_clean_end_cannot_count_as_fixed_horizon_success():
    with pytest.raises(ValueError, match="fixed horizon"):
        classify_terminal(500, 25, 25, invalid=False, fall=False, contact=False, reached=True)


def test_gate_requires_update_600_and_two_distinct_128_episode_cohorts():
    assert not promotion_passed(600, {7: (112, 128), 8: (113, 128)})
    assert promotion_passed(600, {7: (113, 128), 8: (113, 128)})
    for update, cohorts in [(599, {7: (128, 128), 8: (128, 128)}),
                            (600, {7: (226, 256)}),
                            (600, {7: (129, 128), 8: (128, 128)})]:
        with pytest.raises(ValueError):
            promotion_passed(update, cohorts)
