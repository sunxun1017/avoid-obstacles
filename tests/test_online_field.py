import pytest
import torch

from embodied_learning.perception.occupancy import OccupancyMapper
from embodied_learning.perception.online_field import OnlineFieldMap
from embodied_learning.perception.rays import RayMeasurements


def inputs(batch=2):
    # Off-grid coordinates avoid the source sampler's known corner permutation
    # discontinuity when float32 translation changes floor at an exact node.
    origins = torch.tensor([[[0.213, 0.217, 0.413]]]).repeat(batch, 1, 1)
    ends = origins.clone()
    ends[..., 0] = 0.69
    masks = torch.ones(batch, 1, dtype=torch.bool)
    return RayMeasurements(origins, ends, masks, masks)


def pipeline():
    mapper = OccupancyMapper(2, (20, 20, 25), voxel_size=0.04)
    return OnlineFieldMap(mapper, preprocess=False)


def goals():
    return torch.tensor([[0.413, 0.427, 0.613], [0.413, 0.427, 0.613]])


def test_initial_defaults_then_measurements_build_source_fields():
    p = pipeline()
    points = torch.tensor([[[0.2, 0.2, 0.4]], [[0.2, 0.2, 0.4]]])
    empty = p.sample(points, timestamp=0.0)
    assert not empty.sample.guidance_valid.any()
    assert (empty.sample.sdf == 0.5).all()
    p.update(inputs(), goals(), timestamp=0.0)
    live = p.sample(points, timestamp=0.02)
    assert live.sample.guidance_valid.all()
    assert p.versions.tolist() == [1, 1]
    # CAT allows unobserved space; diagnostics must not claim it was measured.
    unseen = p.sample(
        torch.tensor([[[0.4, 0.4, 0.6]], [[0.4, 0.4, 0.6]]]), timestamp=0.02
    )
    assert unseen.sample.guidance_valid.all() and not unseen.observed.any()


def test_cadence_stale_invalid_scan_and_partial_reset():
    p = pipeline()
    rays = inputs()
    p.update(rays, goals(), timestamp=0.0)
    p.update(rays, goals(), timestamp=0.02)
    assert p.versions.tolist() == [1, 1]
    p.update(rays, goals(), timestamp=0.1)
    assert p.versions.tolist() == [2, 2]
    rays.valid_mask.zero_()
    p.update(rays, goals(), timestamp=0.5)
    assert not p.sample(inputs().origins, timestamp=0.5).sample.guidance_valid.any()
    p.reset(torch.tensor([0]))
    assert p.versions.tolist() == [0, 2]
    assert not p.mapper.snapshot().observed[0].any()
    assert p.mapper.snapshot().observed[1].any()


def test_goal_change_invalidates_old_field_immediately():
    p = pipeline()
    p.update(inputs(), goals(), timestamp=0.0)
    target = goals()
    target[0, 0] = 10
    p.update(inputs(), target, timestamp=0.02)
    sample = p.sample(inputs().origins, timestamp=0.02).sample
    assert not sample.guidance_valid[0].any()
    assert sample.guidance_valid[1].all()
    assert "goal" in p.errors[0]
    with pytest.raises(ValueError, match="timestamp"):
        p.update(inputs(), target, timestamp=0.01)


def test_outside_and_nonfinite_queries_do_not_clamp_to_valid_observations():
    p = pipeline()
    p.update(inputs(), goals(), timestamp=0.0)
    points = inputs().origins.clone()
    points[0, 0, 0] = -100
    points[1, 0, 0] = float("nan")
    sample = p.sample(points, timestamp=0.02).sample
    assert not sample.guidance_valid.any()
    assert torch.isfinite(sample.guidance).all()


def test_different_batch_map_origins_do_not_shift_fields():
    shift = torch.tensor([[0.0, 0.0, 0.0], [5.0, -4.0, 0.0]])
    mapper = OccupancyMapper(2, (20, 20, 25), voxel_size=0.04, origin=shift)
    p = OnlineFieldMap(mapper, preprocess=False)
    rays = inputs()
    rays = RayMeasurements(
        rays.origins + shift[:, None],
        rays.endpoints + shift[:, None],
        rays.valid_mask,
        rays.hit_mask,
    )
    p.update(rays, goals() + shift, timestamp=0.0)
    sampled = p.sample(rays.origins, timestamp=0.0).sample
    torch.testing.assert_close(
        sampled.guidance[0], sampled.guidance[1], atol=1e-5, rtol=1e-4
    )


def test_scans_outside_grid_do_not_publish_or_refresh_fields():
    p = pipeline()
    rays = inputs()
    outside = RayMeasurements(
        rays.origins + 100, rays.endpoints + 100, rays.valid_mask, rays.hit_mask
    )
    p.update(outside, goals(), timestamp=0.0)
    assert not p.versions.any()
    p.update(rays, goals(), timestamp=0.1)
    p.update(outside, goals(), timestamp=0.5)
    assert p.versions.tolist() == [1, 1]
    assert not p.sample(rays.origins, timestamp=0.5).sample.guidance_valid.any()


def test_empty_snapshot_selection_is_noop():
    p = pipeline()
    p.publish_snapshot(
        p.mapper.snapshot(),
        goals(),
        timestamp=0.0,
        env_ids=torch.tensor([], dtype=torch.long),
    )
    assert not p.versions.any()


def test_fresh_unchanged_measurement_reuses_field_but_reset_and_goal_do_not():
    p = pipeline()
    p.update(inputs(), goals(), timestamp=0.0)
    first = p._fields[0]
    p.update(inputs(), goals(), timestamp=0.1)
    assert p._fields[0] is first
    assert p.build_counts.tolist() == [1, 1]
    assert p.versions.tolist() == [2, 2]
    assert p.sample(inputs().origins, timestamp=0.15).age[0] == pytest.approx(0.05)
    changed = goals()
    changed[0, 0] += 0.04
    p.update(inputs(), changed, timestamp=0.2)
    assert p.build_counts.tolist() == [2, 1]
    p.reset(torch.tensor([0]))
    p.update(inputs(), changed, timestamp=0.3)
    assert p.build_counts.tolist() == [1, 1]
