"""Invalid physics must remain reportable by the diagnostic instrumentation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_pose_does_not_abort_diagnostic_field_query(monkeypatch, bad):
    tool_dir = Path(__file__).parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tool_dir))
    spec = importlib.util.spec_from_file_location(
        "cat_diagnostic", tool_dir / "diagnose_cat_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    positions = torch.ones(3, 11, 3)
    positions[1, 0, 0] = bad
    original = positions.clone()

    class FiniteOnlyBank:
        def sample_source(self, query, scenes, goals):
            assert torch.isfinite(query).all(), "Field query rejects invalid poses"
            assert torch.equal(query[0], positions[0])
            assert torch.count_nonzero(query[1:]) == 0
            return SimpleNamespace(sdf=query[..., 0])

    sample, valid = module.sample_diagnostic_source(
        FiniteOnlyBank(),
        positions,
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
        torch.tensor([False, True, True]),
    )
    assert valid.tolist() == [True, False, False]
    assert torch.equal(sample.sdf[0], torch.ones(11))
    torch.testing.assert_close(positions, original, equal_nan=True)
