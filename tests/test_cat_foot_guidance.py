"""Catch global-field leakage and loss of original point-distance semantics."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

from embodied_learning.perception.voxel_field import build_source_voxel_field
from embodied_learning.tasks.field_assets import FieldBank


def fixture_field():
    x = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    goal = tuple(x["grid_goal"])
    return build_source_voxel_field(
        x["grid_occupied"], goal=goal, origin=(0.0, 0.0, 0.0), voxel_size=0.1
    ), goal


def test_original_radius_reproduces_author_guidance():
    from embodied_learning.tasks.cat_foot_guidance import rebuild_guidance

    field, goal = fixture_field()
    rebuilt = rebuild_guidance(field, goal, influence_radius=0.5)
    torch.testing.assert_close(rebuilt, field.guidance, atol=1e-7, rtol=2e-6)


def test_foot_override_keeps_other_sites_and_all_scalar_fields_exact():
    from embodied_learning.tasks.cat_foot_guidance import (
        FootGuidanceBank,
        rebuild_guidance,
    )

    field, goal = fixture_field()
    base = FieldBank(
        ((field,),),
        "scene",
        ("scene",),
        {"version": "cat_source_v1"},
        "original-cache",
    )
    override = rebuild_guidance(field, goal, influence_radius=0.7)
    bank = FootGuidanceBank(
        base, ((override,),), radius=0.7, manifest_identity="foot-cache"
    )
    xyz = torch.tensor([[[0.21, 0.35, 0.19]] * 11, [[0.31, 0.25, 0.33]] * 11])
    ids = torch.zeros(2, dtype=torch.long)
    expected = base.sample_source(xyz, ids, ids)
    got = bank.sample_source(xyz, ids, ids)
    for member in fields(expected):
        if member.name != "guidance":
            assert torch.equal(
                getattr(expected, member.name), getattr(got, member.name)
            )
    assert torch.equal(
        got.guidance[:, [0, 1, 2, 5, 6, 7, 8, 9, 10]],
        expected.guidance[:, [0, 1, 2, 5, 6, 7, 8, 9, 10]],
    )
    assert not torch.allclose(got.guidance[:, 3:5], expected.guidance[:, 3:5])
    assert torch.equal(field.guidance, base.fields[0][0].guidance)
    assert bank.cache_identity != base.cache_identity
    # A partial query cannot silently misidentify its two points as body indices.
    with pytest.raises(ValueError, match="11"):
        bank.sample_source(xyz[:, :2], ids, ids)


@pytest.mark.parametrize("distance,radius", [(0.2, 0.25), (0.27, 0.30)])
def test_larger_radius_turns_guidance_earlier_without_moving_obstacles(
    distance, radius
):
    from dataclasses import replace

    from embodied_learning.tasks.cat_foot_guidance import rebuild_guidance

    field, goal = fixture_field()
    n = field.sdf.shape
    field = replace(
        field,
        sdf=torch.full(n, distance),
        occupied=torch.zeros(n, dtype=torch.bool),
        sdf_gradient=torch.tensor([-1.0, 0.0, 0.0]).expand(*n, 3).clone(),
        geodesic_gradient=torch.tensor([-1.0, 0.0, -1.0]).expand(*n, 3).clone(),
    )
    a = rebuild_guidance(field, goal, influence_radius=0.2)
    b = rebuild_guidance(field, goal, influence_radius=radius)
    valid = a.norm(dim=-1) > 0.001
    assert torch.all(b[..., 2][valid] > a[..., 2][valid])
    assert torch.all(b[..., 0][valid] < a[..., 0][valid])
    torch.testing.assert_close(a.norm(dim=-1), b.norm(dim=-1), atol=1e-7, rtol=1e-6)
    assert torch.equal(field.sdf, torch.full(n, distance))


def test_saved_cache_rejects_wrong_radius_and_modified_arrays(tmp_path):
    from types import SimpleNamespace

    from embodied_learning.tasks.cat_foot_guidance import (
        load_foot_bank,
        prepare_foot_cache,
    )

    field, goal = fixture_field()
    base = FieldBank(
        ((field,),),
        "scene",
        ("scene",),
        {"version": "cat_source_v1"},
        "original-cache",
    )
    scene = SimpleNamespace(id="scene", goals=[goal], voxel_size=0.1)
    manifest = SimpleNamespace(identity="scene", scenes=[scene])
    bank = prepare_foot_cache(base, manifest, tmp_path, radius=0.7)
    loaded = load_foot_bank(base, tmp_path, radius=0.7)
    assert bank.cache_identity == loaded.cache_identity
    with pytest.raises(ValueError, match="mismatch"):
        load_foot_bank(base, tmp_path, radius=0.6)
    path = tmp_path / "scene-goal-0.npy"
    grid = np.load(path)
    grid[0, 0, 0, 0] += 1
    np.save(path, grid)
    with pytest.raises(ValueError, match="identity"):
        load_foot_bank(base, tmp_path, radius=0.7)
