from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/experiments"))
from summarize_cat_speed_matched import closest_speed_match


def test_speed_matching_rejects_out_of_band_and_keeps_nearest_candidate():
    contacts = [{"speed": 0.31}, {"speed": 0.59}, {"speed": 0.77}]
    assert closest_speed_match(contacts, 0.65) == contacts[1]
    assert closest_speed_match(contacts, 1.0) is None
    assert closest_speed_match([], 0.65) is None
