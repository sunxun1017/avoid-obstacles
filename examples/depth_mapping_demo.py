"""Fuse synthetic depth frames, build a field, and query body observations."""

import argparse

import torch

from embodied_learning.perception.occupancy import OccupancyMapper
from embodied_learning.perception.rays import rays_from_depth
from embodied_learning.perception.voxel_field import build_voxel_field


def main() -> None:
    """Use a planar depth scene without importing a simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    mapper = OccupancyMapper(
        1,
        (21, 21, 31),
        voxel_size=0.1,
        origin=(-1.0, -1.0, 0.0),
        device=args.device,
    )
    depth = torch.full((1, 21, 21), 2.0, device=args.device)
    intrinsics = depth.new_tensor(
        [[10.0, 0.0, 10.5], [0.0, 10.0, 10.5], [0.0, 0.0, 1.0]]
    )
    rotation = torch.eye(3, device=args.device)[None]
    # Two views of the same plane z=2, with known optical-to-world poses.
    for camera_x in (0.0, 0.1):
        position = depth.new_tensor([[camera_x, 0.0, 0.0]])
        mapper.update(
            rays_from_depth(
                depth,
                intrinsics,
                rotation=rotation,
                position=position,
                max_depth=3.0,
            )
        )
    state = mapper.snapshot()
    assert state.occupied.any()
    assert not state.observed[0, :, :, 21:].any(), "Behind the plane is unknown."
    # Field construction is explicit and CPU-based; mapping itself stays on device.
    field = build_voxel_field(
        state.occupied[0].cpu().numpy(),
        known_mask=state.known_mask[0].cpu().numpy(),
        origin=tuple(state.origin[0].cpu().tolist()),
        voxel_size=state.voxel_size,
        goal=(0.0, 0.0, 1.0),
    ).to(args.device)
    body_positions = depth.new_tensor([[[0.0, 0.0, 0.8], [0.1, 0.1, 0.9]]])
    sample = field.sample(body_positions)
    assert sample.guidance_valid.all()
    print("Observed cells:", state.observed.sum().item())
    print("Occupied cells:", state.occupied.sum().item())
    print("Known cells:", state.known_mask.sum().item())
    print("Body guidance:", sample.guidance.cpu().tolist())
    print("Observation shape:", tuple(sample.guidance.flatten(1).shape))


if __name__ == "__main__":
    main()
