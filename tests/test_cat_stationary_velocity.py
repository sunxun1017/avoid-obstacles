import numpy as np

from embodied_learning.evaluation.cat_stationary_velocity import velocity_ema_step


def test_velocity_filter_waits_for_dwell_and_bypasses_on_first_moving_frame():
    count = np.array([0, 0])
    previous = np.zeros((2, 1))
    for tick in range(1, 11):
        raw = np.full((2, 1), tick, dtype=float)
        count, previous, enabled = velocity_ema_step(
            raw, previous, count, np.array([True, False]), where=np.where,
        )
        assert enabled.tolist() == [tick == 10, False]
        assert previous[1, 0] == raw[1, 0]
        if tick < 10:
            assert np.array_equal(previous, raw)
    assert previous[0, 0] == 9.45
    raw = np.array([[15.0], [5.0]])
    count, value, enabled = velocity_ema_step(
        raw, previous, count, np.array([False, True]), where=np.where,
    )
    assert not enabled.any()
    assert np.array_equal(value, raw)
    assert count.tolist() == [0, 1]
