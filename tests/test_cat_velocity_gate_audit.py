from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/experiments"))
from summarize_cat_stationary_velocity import eligible_control_ticks


def test_gate_audit_resets_at_support_loss_and_ignores_terminal_observation():
    trace = np.zeros((10, 8))
    trace[:, 4:7] = 1
    trace[2, 4] = 0
    trace[6, 7] = 1
    # Ready runs are samples 1–2, 4–6 and 8–10. The last sample
    # is terminal, so it cannot drive a subsequent policy decision.
    assert eligible_control_ticks(trace, 3).tolist() == [6]
    assert eligible_control_ticks(trace[:2], 3).size == 0
