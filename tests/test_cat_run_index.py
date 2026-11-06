import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "cat_run_index", Path(__file__).parents[1] / "tools/experiments/update_cat_run_index.py",
)
indexer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indexer)


def test_index_preserves_other_batches_and_rejects_wrong_episode_total(tmp_path):
    root = tmp_path / "outputs/batch-a"
    cohort = root / "quiet/replay"
    cohort.mkdir(parents=True)
    manifest = dict(
        route="empty-stand", disturbance="quiet", evaluation_seed=9, evaluation_episodes=1,
        code_commit="a" * 40, checkpoint_sha256="b" * 64, config_sha256="c" * 64,
        protocol_sha256="d" * 64,
    )
    (cohort / "manifest.json").write_text(json.dumps(manifest))
    evaluation = dict(terminal_v1=dict(episodes=1, success_count=1, strict_success_count=0))
    (cohort / "evaluation.json").write_text(json.dumps(evaluation))
    (cohort / "episodes.jsonl").write_text('{"terminal_v1":{"success":true,"strict_success":false}}\n')
    p = tmp_path / "index.json"
    p.write_text('[{"change_id":"older","run_id":"unrelated"}]\n')
    rows = indexer.update(p, root, "change-a", 42, tmp_path)
    assert len(rows) == 2 and rows[0]["run_id"] == "unrelated"
    assert rows[1]["run_id"] == "batch-a/quiet/replay"
    assert rows[1]["training_seed"] == 42 and rows[1]["evaluation_seed"] == 9
    assert indexer.update(p, root, "change-a", 42, tmp_path) == rows
    original = p.read_bytes()
    evaluation["terminal_v1"]["success_count"] = 0
    (cohort / "evaluation.json").write_text(json.dumps(evaluation))
    with pytest.raises(ValueError, match="episode"):
        indexer.update(p, root, "change-a", 42, tmp_path)
    assert p.read_bytes() == original
