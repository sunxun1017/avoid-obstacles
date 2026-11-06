from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/experiments"))
from summarize_cat_stance_disturbance import stationary_spectrum


def test_stationary_spectrum_distinguishes_fast_motion_from_slow_drift():
    time = np.arange(50) / 50
    slow = np.sin(2 * np.pi * 2 * time)
    fast = np.sin(2 * np.pi * 16 * time)
    assert stationary_spectrum(fast[:, None], 50)[0] == pytest.approx(16)
    assert stationary_spectrum(fast[:, None], 50)[1] > 0.99
    assert stationary_spectrum(slow[:, None], 50)[0] == pytest.approx(2)
    assert stationary_spectrum(slow[:, None], 50)[1] < 0.01
    assert stationary_spectrum(np.ones((50, 3)), 50) is None
