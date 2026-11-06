import pytest
import torch

from embodied_learning.perception.depth import backproject_depth


def test_backprojection_and_invalid_pixels():
    depth = torch.tensor([[[2.0, 0.0, float("nan")]]])
    k = torch.tensor([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    cloud = backproject_depth(depth, k)
    torch.testing.assert_close(cloud.points[0, 0], torch.tensor([0.0, 0.0, 2.0]))
    assert cloud.valid_mask.tolist() == [[True, False, False]]
    assert torch.isfinite(cloud.points).all()


def test_world_transform_and_batch_intrinsics():
    depth = torch.ones(2, 1, 1) * 2
    k = torch.eye(3).repeat(2, 1, 1)
    k[:, :2, 2] = 0.5
    rotation = torch.diag(torch.tensor([1.0, -1.0, -1.0])).repeat(2, 1, 1)
    position = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cloud = backproject_depth(depth, k, rotation=rotation, position=position)
    torch.testing.assert_close(
        cloud.points[:, 0], torch.tensor([[1.0, 2.0, 1.0], [4.0, 5.0, 4.0]])
    )
    assert cloud.frame == "world"


def test_off_axis_point_and_range_filter():
    k = torch.tensor([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    cloud = backproject_depth(torch.tensor([[[2.0, 2.0]]]), k)
    torch.testing.assert_close(cloud.points[0, 1], torch.tensor([1.0, 0.0, 2.0]))
    filtered = backproject_depth(torch.tensor([[[2.0, 3.0]]]), k, max_depth=2.5)
    assert filtered.valid_mask.tolist() == [[True, False]]


def test_reject_partial_transform():
    with pytest.raises(ValueError, match="together"):
        backproject_depth(
            torch.ones(1, 1, 1), torch.eye(3), position=torch.zeros(1, 3)
        )
