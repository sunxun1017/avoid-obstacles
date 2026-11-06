from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/experiments"))
from summarize_cat_landing_stance import landing_force_metrics


def test_impulse_and_peak_time_use_physical_sampling_rate():
    force_bw = np.r_[np.array([4., 5., 4., 3., 2.]), np.ones(20)]
    metrics = landing_force_metrics(force_bw, 500)
    assert metrics["landing_peak_bw"] == 5
    assert metrics["landing_impulse_10ms_bw_s"] == pytest.approx(18 * 0.002)
    assert metrics["landing_impulse_50ms_bw_s"] == pytest.approx(38 * 0.002)
    assert metrics["landing_peak_after_contact_ms"] == 2
