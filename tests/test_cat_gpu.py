"""GPU Eikonal solver is compared with the CPU source, never graph distance."""

from pathlib import Path

import numpy as np
import pytest
import skfmm
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_gpu_eikonal_planar_interface_and_disconnected_region():
    from embodied_learning.perception.eikonal_warp import CudaEikonal

    solver = CudaEikonal((32, 24, 20), "cuda:0")
    phi = torch.ones(32, 24, 20, device="cuda:0")
    phi[:5] = -1
    blocked = torch.zeros_like(phi, dtype=torch.bool)
    result = solver.solve(phi, blocked, spacing=0.04)
    expected = torch.from_numpy(skfmm.distance(phi.cpu().numpy(), dx=0.04)).to(
        result
    )
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
    blocked[12] = True
    result = solver.solve(phi, blocked, spacing=0.04)
    assert torch.isfinite(result[:12]).all()
    assert torch.isinf(result[12:]).all()


def test_gpu_fields_match_source_with_documented_discretization_error():
    from embodied_learning.perception.cat_gpu import CudaCatFieldBuilder

    reference = np.load(Path(__file__).parent / "fixtures/cat_online_reference.npz")
    occupied = torch.from_numpy(reference["processed"]).cuda()
    builder = CudaCatFieldBuilder(tuple(occupied.shape), "cuda:0")
    field = builder.build(
        occupied,
        origin=tuple(reference["origin"]),
        voxel_size=float(reference["voxel"]),
        goal=tuple(reference["goal"]),
    )
    # Sub-voxel scalar agreement, and small direction error away from zero vectors
    # and source gradient discontinuities. These do not measure navigation success.
    sdf_error = np.abs(field.sdf.cpu().numpy() - reference["sdf"])
    assert np.quantile(sdf_error, 0.99) < 0.04
    ref = torch.from_numpy(reference["gf"]).cuda()
    mask = (ref.norm(dim=-1) > 0.1) & (field.guidance.norm(dim=-1) > 0.1) & ~occupied
    cosine = torch.nn.functional.cosine_similarity(field.guidance[mask], ref[mask])
    assert torch.quantile(cosine, 0.05) > 0.95
    assert torch.isfinite(field.guidance).all()


def test_gpu_solver_rebuild_handles_obstacle_removal_and_reports_nonconvergence():
    from embodied_learning.perception.eikonal_warp import CudaEikonal

    shape = (32, 24, 20)
    solver = CudaEikonal(shape, "cuda:0")
    phi = torch.ones(shape, device="cuda:0")
    phi[3:6, 10:14, 8:12] = -1
    blocked = torch.zeros(shape, device="cuda:0", dtype=torch.bool)
    blocked[14, :20] = True
    detour = solver.solve(phi, blocked, spacing=0.04).clone()
    free = solver.solve(phi, torch.zeros_like(blocked), spacing=0.04).clone()
    assert detour[25, 10, 10] > free[25, 10, 10] + 0.1
    with pytest.raises(ValueError, match="converge"):
        solver.solve(phi, blocked, spacing=0.04, max_iterations=1)
