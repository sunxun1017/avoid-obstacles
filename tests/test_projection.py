import math

import pytest
import torch

from embodied_learning.perception.DualProjector import DualProjector
from embodied_learning.perception.geometry import world_to_yaw_local
from embodied_learning.perception.point_cloud import PointCloud


def test_yaw_local_batch_and_invalid_points():
    points = torch.tensor(
        [
            [[1.0, 3.0, 3.0], [float("nan"), 0.0, 0.0]],
            [[3.0, 2.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, False], [True, False]])
    cloud = PointCloud(points, mask, "world")
    position = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    q = torch.tensor(
        [[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], [1.0, 0.0, 0.0, 0.0]]
    )
    local = world_to_yaw_local(cloud, position, q)
    torch.testing.assert_close(
        local.points[:, 0],
        torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, -2.0]]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert local.frame == "yaw_local"
    assert torch.isfinite(local.points).all()
    assert torch.all(local.points[:, 1] == 0)


def test_roll_does_not_tilt_height_axis():
    cloud = PointCloud(
        torch.tensor([[[1.0, 0.0, 2.0]]]), torch.tensor([[True]]), "world"
    )
    q = torch.tensor([[math.cos(0.2), math.sin(0.2), 0.0, 0.0]])
    local = world_to_yaw_local(cloud, torch.zeros(1, 3), q)
    torch.testing.assert_close(local.points, cloud.points)


def projector(batch=1):
    return DualProjector(
        batch,
        elevation_shape=(2, 2),
        distance_shape=(2, 2),
        height_range=(-1.0, 1.0),
        elevation_height_range=(-1.0, -0.25),
        forward_range=(0.0, 1.0),
        lateral_range=(-1.0, 1.0),
    )


def test_ground_and_overhead_aggregation():
    cloud = PointCloud(
        torch.tensor([[[0.5, 0.1, -0.9], [0.5, 0.1, -0.8], [0.3, 0.1, 0.2]]]),
        torch.ones(1, 3, dtype=torch.bool),
        "yaw_local",
    )
    result = projector().project(cloud, radius=2.0)
    assert result.elevation_valid.sum() == 1
    assert result.distance_valid.sum() == 2
    torch.testing.assert_close(result.elevation[0, 0, 1], torch.tensor(-0.8))
    torch.testing.assert_close(
        result.distance[0, 1, 1], torch.tensor(math.hypot(0.3, 0.1))
    )
    assert result.flattened().elevation.shape == (1, 4)


def test_empty_maps_and_no_stale_frame():
    p = projector()
    cloud = PointCloud(
        torch.tensor([[[0.5, 0.0, -0.5]]]), torch.tensor([[True]]), "yaw_local"
    )
    first = p.project(cloud, 2.0)
    empty = PointCloud(
        torch.zeros(1, 0, 3), torch.zeros(1, 0, dtype=torch.bool), "yaw_local"
    )
    second = p.project(empty, 3.0)
    assert first.distance_valid.any()
    assert not second.distance_valid.any()
    assert not second.elevation_valid.any()
    assert torch.all(second.distance == 3.0)
    assert torch.all(second.elevation == 0.0)


def test_per_environment_radius_and_invalid_point_filtering():
    p = projector(2)
    points = torch.tensor(
        [
            [[1.5, 0.0, -0.5], [float("nan"), 0.0, 0.0]],
            [[1.5, 0.0, -0.5], [0.0, 0.0, 0.0]],
        ]
    )
    cloud = PointCloud(
        points, torch.tensor([[True, True], [True, False]]), "yaw_local"
    )
    result = p.project(cloud, torch.tensor([1.0, 2.0]))
    assert not result.distance_valid[0].any()
    assert result.distance_valid[1].any()
    assert torch.isfinite(result.elevation).all()


def test_rejects_wrong_frame_and_bad_radius():
    cloud = PointCloud(
        torch.zeros(1, 1, 3), torch.ones(1, 1, dtype=torch.bool), "world"
    )
    with pytest.raises(ValueError, match="yaw_local"):
        projector().project(cloud, 2.0)
    cloud = PointCloud(cloud.points, cloud.valid_mask, "yaw_local")
    with pytest.raises(ValueError, match="radius"):
        projector().project(cloud, 0.0)


def test_upper_height_boundary_and_nearest_distance():
    cloud = PointCloud(
        torch.tensor([[[0.8, 0.0, 0.2], [0.4, 0.0, 0.2], [0.2, 0.0, 1.0]]]),
        torch.ones(1, 3, dtype=torch.bool),
        "yaw_local",
    )
    result = projector().project(cloud, 2.0)
    assert result.distance_valid.sum() == 1
    torch.testing.assert_close(result.distance[0, 1, 1], torch.tensor(0.4))


def test_robot_pose_is_an_independent_snapshot():
    from types import SimpleNamespace

    from embodied_learning.simulators.mjlab.sensors import read_robot_pose

    data = SimpleNamespace(
        root_link_pos_w=torch.tensor([[1.0, 2.0, 3.0]]),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    position, quaternion = read_robot_pose(env)
    data.root_link_pos_w.zero_()
    data.root_link_quat_w.zero_()
    assert position.tolist() == [[1.0, 2.0, 3.0]]
    assert quaternion.tolist() == [[1.0, 0.0, 0.0, 0.0]]
