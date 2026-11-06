import numpy as np
import pytest
import torch

from embodied_learning.perception.occupancy import OccupancyMapper
from embodied_learning.perception.point_cloud import PointCloud
from embodied_learning.perception.rays import (
    RayMeasurements,
    rays_from_depth,
    rays_from_point_cloud,
)


def rays(endpoints, *, origins=None, hit=None, valid=None, dtype=torch.float32):
    endpoints = torch.tensor(endpoints, dtype=dtype)
    if origins is None:
        origins = torch.zeros_like(endpoints)
    else:
        origins = torch.tensor(origins, dtype=dtype)
    shape = endpoints.shape[:2]
    return RayMeasurements(
        origins,
        endpoints,
        torch.ones(shape, dtype=torch.bool)
        if valid is None
        else torch.tensor(valid),
        torch.ones(shape, dtype=torch.bool) if hit is None else torch.tensor(hit),
    )


def mapper(batch=1):
    return OccupancyMapper(batch, (7, 5, 5), voxel_size=1.0)


def test_hit_clears_ray_but_not_space_behind_surface():
    m = mapper()
    m.update(rays([[[3.0, 0.0, 0.0]]]))
    state = m.snapshot()
    assert state.known_mask[0, :4, 0, 0].all()
    assert not state.occupied[0, :3, 0, 0].any()
    assert state.occupied[0, 3, 0, 0]
    assert not state.known_mask[0, 4:, 0, 0].any()


def test_no_hit_clears_to_endpoint_without_occupying_it():
    m = mapper()
    m.update(rays([[[3.0, 0.0, 0.0]]], hit=[[False]]))
    state = m.snapshot()
    assert state.known_mask[0, :4, 0, 0].all()
    assert not state.occupied.any()


def test_invalid_nonfinite_and_zero_length_rays_do_not_update():
    m = mapper()
    m.update(
        rays(
            [[[3.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]],
            valid=[[False, True, True]],
        )
    )
    assert not m.snapshot().observed.any()


def test_duplicates_update_once_and_hits_win_within_frame():
    m = mapper()
    m.update(
        rays(
            [[[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]]],
            hit=[[True, True, False]],
        )
    )
    state = m.snapshot()
    assert state.log_odds[0, 3, 0, 0] == pytest.approx(0.85)
    assert state.log_odds[0, 1, 0, 0] == pytest.approx(-0.4)


def test_multiframe_fusion_uncertainty_and_log_odds_clamping():
    m = mapper()
    m.update(rays([[[3.0, 0.0, 0.0]]]))
    m.update(rays([[[3.0, 0.0, 0.0]]], hit=[[False]]))
    m.update(rays([[[3.0, 0.0, 0.0]]], hit=[[False]]))
    state = m.snapshot()
    assert state.observed[0, 3, 0, 0]
    assert not state.known_mask[0, 3, 0, 0]
    m.update(rays([[[3.0, 0.0, 0.0]]], hit=[[False]]))
    assert m.snapshot().known_mask[0, 3, 0, 0]
    for _ in range(20):
        m.update(rays([[[3.0, 0.0, 0.0]]]))
    assert m.snapshot().log_odds.max() <= 3.5
    assert m.snapshot().log_odds.min() >= -2.0


def test_rays_crossing_map_are_clipped_and_outside_hits_are_not_inserted():
    m = mapper()
    m.update(rays([[[10.0, 0.0, 0.0]]], origins=[[[-3.0, 0.0, 0.0]]]))
    state = m.snapshot()
    assert state.known_mask[0, :, 0, 0].all()
    assert not state.occupied.any()
    m.reset()
    m.update(rays([[[10.0, 7.0, 0.0]]], origins=[[[-3.0, 7.0, 0.0]]]))
    assert not m.snapshot().observed.any()


def test_diagonal_traversal_and_negative_direction():
    m = mapper()
    m.update(rays([[[0.0, 0.0, 0.0]]], origins=[[[3.0, 3.0, 3.0]]]))
    state = m.snapshot()
    assert state.occupied[0, 0, 0, 0]
    for i in range(1, 4):
        assert state.known_mask[0, i, i, i]
    assert not state.observed[0, 2, 1, 1]


def test_cell_center_origin_and_endpoint_on_upper_boundary():
    m = OccupancyMapper(1, (7, 5, 5), voxel_size=1.0, origin=(10.0, 20.0, 30.0))
    m.update(rays([[[16.5, 20.0, 30.0]]], origins=[[[10.0, 20.0, 30.0]]]))
    state = m.snapshot()
    assert state.known_mask[0, :, 0, 0].all()
    assert not state.occupied.any()
    torch.testing.assert_close(state.origin, torch.tensor([[10.0, 20.0, 30.0]]))


def test_maps_are_isolated_and_reset_can_select_environment():
    m = OccupancyMapper(
        2,
        (7, 5, 5),
        voxel_size=1.0,
        origin=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
    )
    m.update(
        rays(
            [[[3.0, 0.0, 0.0]], [[14.0, 0.0, 0.0]]],
            origins=[[[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]],
        )
    )
    assert m.snapshot().occupied[0, 3, 0, 0]
    assert m.snapshot().occupied[1, 4, 0, 0]
    m.reset(torch.tensor([0]))
    state = m.snapshot()
    assert not state.observed[0].any() and state.occupied[1, 4, 0, 0]
    state.log_odds.zero_()
    state.origin.zero_()
    assert m.snapshot().occupied[1, 4, 0, 0]
    assert m.snapshot().origin[1, 0] == 10


def test_point_cloud_adapter_preserves_invalid_as_unknown():
    cloud = PointCloud(
        torch.tensor([[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        torch.tensor([[True, False]]),
        "world",
    )
    measurement = rays_from_point_cloud(cloud, torch.zeros(1, 3))
    assert torch.equal(measurement.valid_mask, cloud.valid_mask)
    assert torch.equal(measurement.hit_mask, cloud.valid_mask)
    with pytest.raises(ValueError):
        rays_from_point_cloud(
            PointCloud(cloud.points, cloud.valid_mask, "optical"), torch.zeros(1, 3)
        )


def test_depth_adapter_uses_optical_depth_and_explicit_no_returns():
    depth = torch.tensor([[[2.0, float("inf"), float("nan"), 8.0]]])
    k = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    measurement = rays_from_depth(
        depth,
        k,
        rotation=torch.eye(3)[None],
        position=torch.zeros(1, 3),
        max_depth=4.0,
        no_hit_mask=torch.tensor([[[False, True, False, False]]]),
    )
    assert measurement.valid_mask.tolist() == [[True, True, False, True]]
    assert measurement.hit_mask.tolist() == [[True, False, False, False]]
    torch.testing.assert_close(
        measurement.endpoints[0, 0], torch.tensor([0.0, 0.0, 2.0])
    )
    torch.testing.assert_close(
        measurement.endpoints[0, 1], torch.tensor([4.0, 0.0, 4.0])
    )


def test_invalid_shapes_parameters_and_half_precision_are_rejected():
    with pytest.raises(ValueError):
        OccupancyMapper(1, (7, 5, 5), voxel_size=0.0)
    with pytest.raises(ValueError):
        OccupancyMapper(1, (7, 5, 5), voxel_size=1.0, dtype=torch.float16)
    m = mapper()
    with pytest.raises(ValueError):
        m.update(rays([[[3.0, 0.0, 0.0]]], dtype=torch.float64))
    with pytest.raises(ValueError):
        m.reset(torch.tensor([2]))


def test_dda_agrees_with_independent_cell_segment_intersections():
    rng = np.random.default_rng(2026)
    origins = rng.uniform(-3.0, 8.0, (24, 3))
    endpoints = rng.uniform(-3.0, 8.0, (24, 3))
    hits = rng.random(24) > 0.5
    measurement = rays(
        [endpoints.tolist()],
        origins=[origins.tolist()],
        hit=[hits.tolist()],
        dtype=torch.float64,
    )
    m = OccupancyMapper(1, (7, 5, 5), voxel_size=1.0, dtype=torch.float64)
    m.update(measurement)
    expected_free = np.zeros((7, 5, 5), dtype=bool)
    expected_hit = np.zeros_like(expected_free)
    for o, e, hit in zip(origins, endpoints, hits, strict=True):
        endpoint_cell = np.floor(e + 0.5).astype(int)
        hit_inside = (
            hit
            and np.all(endpoint_cell >= 0)
            and np.all(endpoint_cell < np.array(expected_free.shape))
        )
        if hit_inside:
            expected_hit[tuple(endpoint_cell)] = True
        direction = e - o
        for cell in np.ndindex(expected_free.shape):
            if hit_inside and np.array_equal(cell, endpoint_cell):
                continue
            lower, upper = np.array(cell) - 0.5, np.array(cell) + 0.5
            a, b = (lower - o) / direction, (upper - o) / direction
            entry = max(0.0, np.minimum(a, b).max())
            exit_time = min(1.0, np.maximum(a, b).min())
            if exit_time - entry > 1e-10:
                expected_free[cell] = True
    state = m.snapshot()
    assert np.array_equal(state.occupied[0].numpy(), expected_hit)
    assert np.array_equal(state.known_mask[0].numpy(), expected_hit | expected_free)


@pytest.mark.parametrize(
    "dtype,shape,end,untouched",
    [
        (torch.float32, (13, 13, 3), [11.5, 3.5, 0.0], [(9, 2, 0), (12, 3, 0)]),
        (torch.float64, (10, 10, 3), [8.5, 2.5, 0.0], [(5, 2, 0)]),
    ],
)
def test_corner_and_endpoint_touches_do_not_clear_cells(
    dtype, shape, end, untouched
):
    m = OccupancyMapper(1, shape, voxel_size=1.0, dtype=dtype)
    m.update(
        rays([[end]], origins=[[[-0.5, -0.5, 0.0]]], hit=[[False]], dtype=dtype)
    )
    state = m.snapshot()
    for cell in untouched:
        assert not state.observed[(0, *cell)]
    assert state.known_mask[0, 0, 0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_mapping_agrees_with_cpu():
    measurement = rays(
        [[[3.0, 3.0, 3.0], [10.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
        hit=[[True, False, True]],
    )
    cpu = mapper()
    gpu = OccupancyMapper(1, (7, 5, 5), voxel_size=1.0, device="cuda")
    cpu.update(measurement)
    gpu.update(
        RayMeasurements(
            *(
                getattr(measurement, name).cuda()
                for name in ("origins", "endpoints", "valid_mask", "hit_mask")
            )
        )
    )
    torch.testing.assert_close(
        cpu.snapshot().log_odds, gpu.snapshot().log_odds.cpu()
    )
    assert torch.equal(cpu.snapshot().known_mask, gpu.snapshot().known_mask.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_dense_depth_cpu_cuda_maps_agree():
    cpu = OccupancyMapper(1, (21, 21, 31), voxel_size=0.1, origin=(-1.0, -1.0, 0.0))
    gpu = OccupancyMapper(
        1, (21, 21, 31), voxel_size=0.1, origin=(-1.0, -1.0, 0.0), device="cuda"
    )
    depth = torch.full((1, 21, 21), 2.0)
    k = torch.tensor([[10.0, 0.0, 10.5], [0.0, 10.0, 10.5], [0.0, 0.0, 1.0]])
    for x in (0.0, 0.1):
        measurement = rays_from_depth(
            depth,
            k,
            rotation=torch.eye(3)[None],
            position=torch.tensor([[x, 0.0, 0.0]]),
            max_depth=3.0,
        )
        cpu.update(measurement)
        gpu.update(
            RayMeasurements(
                *(
                    getattr(measurement, name).cuda()
                    for name in ("origins", "endpoints", "valid_mask", "hit_mask")
                )
            )
        )
    assert torch.equal(cpu.snapshot().observed, gpu.snapshot().observed.cpu())
    torch.testing.assert_close(
        cpu.snapshot().log_odds, gpu.snapshot().log_odds.cpu()
    )
