import numpy as np

from embodied_learning.evaluation.cat_hurdle_events import crossing_events


def test_event_trigger_requires_swing_over_bar_and_grounded_landing_beyond_bar():
    foot = np.array([[.50, .10, .20], [.10, .10, .04]])
    grounded = np.array([False, True])
    seen, swing, landing = crossing_events(foot, grounded, np.array([False, False]), .4, .64, 0, .5, .12)
    assert seen.tolist() == swing.tolist() == [True, False]
    assert not landing.any()
    foot[0] = [.71, .10, .03]
    _, _, landing = crossing_events(foot, np.array([True, True]), seen, .4, .64, 0, .5, .12)
    assert landing.tolist() == [True, False]
    _, _, missed = crossing_events(foot, np.array([True, True]), ~seen, .4, .64, 0, .5, .12)
    assert not missed.any()
