import pytest
import torch

from embodied_learning.perception.occupancy import OccupancyMapper
from embodied_learning.perception.rays import RayMeasurements

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_warp_mapping_matches_reference_for_random_rays_and_reset():
    torch.manual_seed(17)
    cpu = OccupancyMapper(
        3, (21, 17, 15), voxel_size=0.04, origin=(-0.4, -0.3, -0.2)
    )
    gpu = OccupancyMapper(
        3,
        (21, 17, 15),
        voxel_size=0.04,
        origin=(-0.4, -0.3, -0.2),
        device="cuda:0",
        backend="warp",
    )
    for _ in range(3):
        o = torch.rand(3, 500, 3) * 1.2 - 0.6
        e = torch.rand(3, 500, 3) * 1.2 - 0.6
        valid = torch.rand(3, 500) > 0.1
        hit = torch.rand(3, 500) > 0.5
        o[:, 0] = float("nan")
        e[:, 1] = o[:, 1]
        cpu.update(RayMeasurements(o, e, valid, hit))
        gpu.update(RayMeasurements(o.cuda(), e.cuda(), valid.cuda(), hit.cuda()))
        torch.testing.assert_close(
            cpu.snapshot().log_odds, gpu.snapshot().log_odds.cpu()
        )
    cpu.reset(torch.tensor([1]))
    gpu.reset(torch.tensor([1], device="cuda:0"))
    assert torch.equal(cpu.snapshot().observed, gpu.snapshot().observed.cpu())


def test_warp_ties_clipping_and_duplicate_hit_precedence():
    cpu = OccupancyMapper(1, (13, 13, 3), voxel_size=1.0)
    gpu = OccupancyMapper(
        1, (13, 13, 3), voxel_size=1.0, device="cuda:0", backend="warp"
    )
    o = torch.tensor(
        [
            [
                [-0.5, -0.5, 0.0],
                [-0.5, -0.5, 0.0],
                [8.0, 8.0, 0.0],
                [-3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )
    e = torch.tensor(
        [
            [
                [11.5, 3.5, 0.0],
                [11.5, 3.5, 0.0],
                [0.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ]
    )
    valid = torch.ones(1, 6, dtype=torch.bool)
    hit = torch.tensor([[False, False, True, False, True, True]])
    cpu.update(RayMeasurements(o, e, valid, hit))
    gpu.update(RayMeasurements(o.cuda(), e.cuda(), valid.cuda(), hit.cuda()))
    torch.testing.assert_close(
        cpu.snapshot().log_odds, gpu.snapshot().log_odds.cpu()
    )
    assert not gpu.snapshot().observed[0, 9, 2, 0]
