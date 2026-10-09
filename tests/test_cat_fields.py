from pathlib import Path

import numpy as np
import pytest
import torch

from embodied_learning.data.field_scenes import load_scene_manifest
from embodied_learning.perception.voxel_field import build_source_voxel_field
from embodied_learning.tasks.field_assets import (
    FieldBank,
    load_field_cache,
    prepare_fields,
)


def test_source_field_matches_author_grid_fixture():
    x = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    field = build_source_voxel_field(
        x["grid_occupied"],
        goal=tuple(x["grid_goal"]),
        origin=(0.0, 0.0, 0.0),
        voxel_size=0.1,
    )
    for name in ("sdf", "guidance", "geodesic"):
        torch.testing.assert_close(
            getattr(field, name),
            torch.from_numpy(x["grid_" + name]),
            rtol=1e-5,
            atol=1e-6,
        )


def test_source_query_clamps_without_cross_env_shift():
    x = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    f = build_source_voxel_field(
        x["grid_occupied"],
        goal=tuple(x["grid_goal"]),
        origin=(0.0, 0.0, 0.0),
        voxel_size=0.1,
    )
    bank = FieldBank(((f,),), "test", ("s",))
    positions = torch.tensor([[[-4.0, 0.25, 0.33]], [[0.21, 0.35, 0.19]]])
    result = bank.sample_source(
        positions, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)
    )
    torch.testing.assert_close(
        result.guidance, torch.from_numpy(x["query_guidance"]), rtol=1e-5, atol=1e-6
    )


def test_cache_rejects_source_profile_drift(tmp_path):
    manifest = load_scene_manifest(Path("configs/experiments/g1_field.yaml"))
    prepare_fields(manifest, tmp_path, source_profile={"version": "cat_source_v1"})
    with pytest.raises(ValueError, match="profile"):
        load_field_cache(tmp_path / "manifest.json", manifest, device="cpu")


def test_unknown_cache_not_silently_filled(tmp_path):
    manifest = load_scene_manifest(Path("configs/experiments/g1_field.yaml"))
    with pytest.raises(ValueError):
        prepare_fields(manifest, tmp_path, source_profile={"version": "unknown"})
