"""Range conventions and filtering before occupancy integration."""

import pytest
import torch

from embodied_learning.perception.rays import RayMeasurements, rays_from_lidar
from embodied_learning.simulators.mjlab.cat_sensors import (
    LidarPatternCfg,
    OnlineSensorConfig,
    filter_mapping_rays,
)


def test_lidar_metric_range_and_confirmed_misses():
    directions = torch.tensor([[[3.0, 4.0, 0.0]]]).expand(1, 6, 3)
    origins = torch.tensor([[[1.0, 2.0, 3.0]]]).expand_as(directions)
    distance = torch.tensor([[2.0, -1.0, float("nan"), 9.0, 0.0, float("inf")]])
    rays = rays_from_lidar(
        origins, directions, distance, max_range=5.0, no_hit_mask=distance == -1
    )
    assert rays.valid_mask.tolist() == [[True, True, False, True, False, False]]
    assert rays.hit_mask.tolist() == [[True, False, False, False, False, False]]
    assert torch.allclose(
        rays.endpoints[0, :2], torch.tensor([[2.2, 3.6, 3.0], [4.0, 6.0, 3.0]])
    )
    assert torch.isfinite(rays.endpoints).all()


def test_ground_clears_only_to_surface_and_self_does_not_clear():
    points = torch.randn(1, 4, 3)
    rays = RayMeasurements(
        torch.zeros_like(points),
        points,
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[True, True, True, False]]),
    )
    filtered = filter_mapping_rays(
        rays,
        torch.tensor([[False, True, False, False]]),
        torch.tensor([[False, False, True, False]]),
    )
    assert filtered.valid_mask.tolist() == [[True, False, True, True]]
    assert filtered.hit_mask.tolist() == [[True, False, False, False]]
    assert torch.equal(filtered.endpoints, points)
    assert rays.valid_mask.all()  # Do not mutate shared sensor data.


def test_lidar_pattern_is_spherical_and_configuration_is_validated():
    origins, rays = LidarPatternCfg(16, 5).generate_rays(None, "cpu")
    assert origins.shape == rays.shape == (80, 3)
    assert torch.allclose(rays.norm(dim=-1), torch.ones(80))
    assert rays[:, 2].min() < -0.5 and rays[:, 2].max() > 0.5
    assert not origins.any()
    with pytest.raises(ValueError):
        OnlineSensorConfig(source="offline")
    with pytest.raises(ValueError):
        OnlineSensorConfig(max_range=0)
