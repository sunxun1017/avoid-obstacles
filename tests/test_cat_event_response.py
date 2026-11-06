from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/experiments"))
from summarize_cat_hurdle_events import recovery_seconds


def test_recovery_requires_a_full_matched_walking_window():
    hz = 50
    reference = np.tile([0.7, 0.3, 1.0], (100, 1))
    pushed = reference.copy()
    pushed[:10, 0] = -0.3
    pushed[:10, 1] = 2.0
    assert recovery_seconds(pushed, reference, hz) == .50
    pushed[:, 2] = 0
    assert recovery_seconds(pushed, reference, hz) is None
    assert recovery_seconds(reference[:10], reference, hz) is None
