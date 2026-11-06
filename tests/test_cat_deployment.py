from pathlib import Path

import numpy as np
import pytest
import torch

from embodied_learning.perception.cat_deployment import (
    build_cat_deployment_field,
    prepare_cat_occupancy,
)


def reference():
    return np.load(Path(__file__).parent / "fixtures/cat_online_reference.npz")


def test_preprocessing_matches_pinned_octomap_bridge():
    x = reference()
    actual = prepare_cat_occupancy(torch.from_numpy(x["raw"])[None])
    assert torch.equal(actual[0], torch.from_numpy(x["processed"]))
    assert np.array_equal(x["raw"], reference()["raw"])


def test_deployment_field_matches_pinned_author_functions():
    x = reference()
    field = build_cat_deployment_field(
        torch.from_numpy(x["processed"]),
        origin=tuple(x["origin"]),
        voxel_size=float(x["voxel"]),
        goal=tuple(x["goal"]),
    )
    for key, actual in (
        ("sdf", field.sdf),
        ("bf", field.sdf_gradient),
        ("gf", field.guidance),
    ):
        torch.testing.assert_close(
            actual, torch.from_numpy(x[key]), atol=2e-5, rtol=2e-4
        )


def test_empty_grid_source_corner_extension_and_invalid_goal():
    occupied = torch.zeros(20, 20, 25, dtype=torch.bool)
    field = build_cat_deployment_field(
        occupied, origin=(0.0, 0.0, 0.0), voxel_size=0.04, goal=(0.4, 0.4, 0.6)
    )
    assert field.occupied[0, 0, -1] and field.occupied[0, -1, -1]
    assert not occupied.any()
    with pytest.raises(ValueError, match="goal"):
        build_cat_deployment_field(
            occupied, origin=(0.0, 0.0, 0.0), voxel_size=0.04, goal=(100.0, 0.0, 0.0)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_preprocessing_and_sampling_match_cpu():
    x = reference()
    raw = torch.from_numpy(x["raw"])[None]
    torch.testing.assert_close(
        prepare_cat_occupancy(raw.cuda()).cpu(), prepare_cat_occupancy(raw)
    )
    kw = dict(
        origin=tuple(x["origin"]),
        voxel_size=float(x["voxel"]),
        goal=tuple(x["goal"]),
    )
    cpu = build_cat_deployment_field(torch.from_numpy(x["processed"]), **kw)
    gpu = build_cat_deployment_field(torch.from_numpy(x["processed"]).cuda(), **kw)
    torch.testing.assert_close(cpu.guidance, gpu.guidance.cpu())
