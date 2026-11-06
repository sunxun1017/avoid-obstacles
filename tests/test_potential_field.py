import numpy as np
import pytest
import torch

from embodied_learning.perception.potential_field import (
    paper_potential,
    prioritize_guidance,
)
from embodied_learning.perception.voxel_field import (
    VoxelPotentialField,
    build_voxel_field,
)


def test_paper_force_matches_potential_derivative():
    distance = torch.tensor([[0.25]], dtype=torch.float64, requires_grad=True)
    normal = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    result = paper_potential(
        distance, normal, distance, normal, influence_radius=0.5
    )
    derivative = torch.autograd.grad(
        (result.attractive + result.repulsive).sum(), distance
    )[0]
    torch.testing.assert_close(result.force[..., 0], -derivative)
    assert result.force[0, 0, 0] > 0


def test_contact_extension_is_finite_and_still_pushes_outward():
    d = torch.tensor([[0.0, -0.1]], requires_grad=True)
    normal = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    result = paper_potential(
        torch.zeros_like(d), torch.zeros_like(normal), d, normal, min_distance=0.02
    )
    derivative = torch.autograd.grad(result.repulsive.sum(), d)[0]
    assert torch.isfinite(result.repulsive).all()
    assert (result.force[..., 0] > 0).all()
    torch.testing.assert_close(result.force[..., 0], -derivative)


def test_repulsion_stops_at_influence_boundary():
    d = torch.tensor([[0.5, 1.0]])
    normal = torch.ones(1, 2, 3)
    result = paper_potential(torch.ones_like(d), normal, d, normal)
    assert torch.count_nonzero(result.repulsive) == 0
    torch.testing.assert_close(result.force, -normal)


def test_priority_depends_on_approach_speed_and_handles_zero_force():
    force = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    velocity = torch.tensor([[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    normal = torch.tensor([[[1.0, 0.0, 0.0]]]).expand_as(force)
    result = prioritize_guidance(
        force, torch.zeros(1, 3), normal, velocity, torch.tensor([1.0, 0.5, 0.5])
    )
    torch.testing.assert_close(result[0, :, 0], torch.tensor([2.0, 0.25, 0.0]))


def test_invalid_potential_parameters_are_rejected():
    d, grad = torch.ones(1, 1), torch.ones(1, 1, 3)
    with pytest.raises(ValueError):
        paper_potential(d, grad, d, grad, min_distance=0.5)
    with pytest.raises(ValueError):
        paper_potential(d, grad, d, grad, eta=float("nan"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_potential_and_field_dtypes_are_rejected(dtype):
    distance = torch.tensor([[0.02]], dtype=dtype)
    gradient = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=dtype)
    with pytest.raises(ValueError, match="float32 or float64"):
        paper_potential(distance, gradient, distance, gradient)
    with pytest.raises(ValueError, match="float32 or float64"):
        prioritize_guidance(
            gradient, distance, gradient, gradient, torch.ones(1, dtype=dtype)
        )
    with pytest.raises(ValueError, match="float32 or float64"):
        affine_field().to("cpu", dtype=dtype)


def affine_field():
    axes = torch.arange(4, dtype=torch.float64)
    x, y, z = torch.meshgrid(axes, axes, axes, indexing="ij")
    sdf = x + 10 * y + 100 * z
    gradient = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64).expand(
        4, 4, 4, 3
    )
    valid = torch.ones(4, 4, 4, dtype=torch.bool)
    return VoxelPotentialField(
        sdf=sdf,
        geodesic=sdf,
        sdf_gradient=gradient,
        geodesic_gradient=gradient,
        guidance=-gradient,
        known=valid,
        reachable=valid,
        occupied=~valid,
        origin=(1.0, 2.0, 3.0),
        voxel_size=0.5,
    )


def test_trilinear_query_axis_order_batch_and_outer_boundary():
    field = affine_field()
    points = torch.tensor(
        [[[1.125, 2.25, 3.375]], [[2.5, 3.5, 4.5]]], dtype=torch.float64
    )
    sample = field.sample(points)
    torch.testing.assert_close(
        sample.sdf, torch.tensor([[80.25], [333.0]], dtype=torch.float64)
    )
    assert sample.distance_valid.all() and sample.geodesic_valid.all()
    torch.testing.assert_close(
        sample.guidance[0, 0],
        torch.tensor([-1.0, -10.0, -100.0], dtype=torch.float64),
    )


def test_unknown_outside_and_nan_are_invalid_not_clamped_to_free_space():
    field = affine_field()
    field.known[1, 1, 1] = False
    points = torch.tensor(
        [[[1.25, 2.25, 3.25], [0.0, 0.0, 0.0], [float("nan"), 2.0, 3.0]]],
        dtype=torch.float64,
    )
    sample = field.sample(points)
    assert not sample.distance_valid.any()
    assert not sample.guidance_valid.any()
    assert torch.count_nonzero(sample.sdf) == 0
    assert torch.isfinite(sample.guidance).all()


def test_field_device_dtype_conversion_and_query_contract():
    field = affine_field().to("cpu", dtype=torch.float32)
    assert field.sdf.dtype == torch.float32 and field.known.dtype == torch.bool
    with pytest.raises(ValueError):
        field.sample(torch.ones(1, 3))
    with pytest.raises(ValueError):
        field.sample(torch.ones(1, 1, 3, dtype=torch.float64))


def test_fast_marching_routes_around_wall():
    occupied = np.zeros((31, 31, 5), dtype=bool)
    occupied[15, :24, :] = True
    field = build_voxel_field(
        occupied, goal=(2.5, 1.0, 0.2), voxel_size=0.1, goal_radius=0.11
    )
    # The target is directly across the wall; the opening is toward +Y.
    assert field.geodesic[10, 10, 2] > 2.0
    assert field.guidance[10, 10, 2, 1] > 0
    assert field.sdf[15, 10, 2] < 0
    assert field.sdf[10, 10, 2] > 0


def test_disconnected_region_and_unknown_space_keep_validity_masks():
    occupied = np.zeros((21, 15, 5), dtype=bool)
    occupied[10, :, :] = True
    known = np.ones_like(occupied)
    known[0, :, :] = False
    field = build_voxel_field(
        occupied, goal=(1.7, 0.7, 0.2), voxel_size=0.1, known_mask=known
    )
    assert not field.reachable[5, 7, 2]
    assert torch.isinf(field.geodesic[5, 7, 2])
    sample = field.sample(torch.tensor([[[0.5, 0.7, 0.2]]]))
    assert sample.distance_valid.item()
    assert not sample.geodesic_valid.item()
    assert not sample.guidance_valid.item()


def test_goal_and_empty_obstacle_scene_have_finite_guidance():
    occupied = np.zeros((15, 15, 5), dtype=bool)
    field = build_voxel_field(occupied, goal=(0.7, 0.7, 0.2), voxel_size=0.1)
    assert torch.isfinite(field.guidance).all()
    assert torch.isinf(field.sdf).all()
    assert torch.count_nonzero(field.guidance[7, 7, 2]) == 0
    sample = field.sample(torch.tensor([[[0.2, 0.2, 0.2]]]))
    assert sample.distance_valid.item() and sample.guidance_valid.item()
    assert torch.isinf(sample.sdf).all()


def test_near_wall_guidance_becomes_tangential_and_inside_points_escape():
    occupied = np.zeros((31, 31, 5), dtype=bool)
    occupied[15:18, :24, :] = True
    field = build_voxel_field(occupied, goal=(2.5, 1.0, 0.2), voxel_size=0.1)
    direction = field.guidance[14, 10, 2]
    assert direction[1] > 0
    assert abs(direction[0]) < 0.1 * abs(direction[1])
    assert field.guidance[15, 10, 2, 0] < 0
    sample = field.sample(torch.tensor([[[1.5, 1.0, 0.2]]]))
    assert sample.distance_valid.item() and sample.guidance_valid.item()
    assert not sample.geodesic_valid.item()
    assert sample.sdf.item() < 0


def test_goal_inside_obstacle_is_rejected():
    occupied = np.zeros((9, 9, 5), dtype=bool)
    occupied[4, 4, 2] = True
    with pytest.raises(ValueError, match="known free"):
        build_voxel_field(occupied, goal=(0.4, 0.4, 0.2), voxel_size=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_query_agrees_with_cpu():
    field = affine_field()
    points = torch.tensor([[[1.125, 2.25, 3.375]]], dtype=torch.float64)
    cpu = field.sample(points)
    gpu = field.to("cuda").sample(points.cuda())
    torch.testing.assert_close(cpu.sdf, gpu.sdf.cpu())
    torch.testing.assert_close(cpu.guidance, gpu.guidance.cpu())
    assert torch.equal(cpu.distance_valid, gpu.distance_valid.cpu())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voxel_size": 0.0},
        {"goal": (10.0, 10.0, 10.0)},
        {"known_mask": np.zeros((9, 9, 5), dtype=bool)},
    ],
)
def test_invalid_map_config_is_rejected(kwargs):
    config = {"goal": (0.4, 0.4, 0.2), "voxel_size": 0.1}
    config.update(kwargs)
    with pytest.raises(ValueError):
        build_voxel_field(np.zeros((9, 9, 5), dtype=bool), **config)
