from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/experiments"))
from summarize_cat_paired_uncertainty import paired_effect


def test_paired_effect_matches_slot_and_keeps_seed_breakdown():
    left = [
        {"seed": seed, "slot": slot, "value": x}
        for seed in ("10", "11") for slot, x in (("0", "1"), ("1", "10"))
    ]
    right = [
        {"seed": row["seed"], "slot": row["slot"], "value": str(float(row["value"]) + 2)}
        for row in reversed(left)
    ]
    result = paired_effect(left, right, "value", seed_key="seed")
    assert result["pairs"] == 4
    assert result["median"] == 2
    assert result["interval_95"] == [2, 2]
    assert result["by_seed"]["11"]["pairs"] == 2
    right[0]["value"] = ""
    assert paired_effect(left, right, "value", seed_key="seed")["pairs"] == 3
