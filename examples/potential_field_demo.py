"""Compare released-code guidance with paper potentials in a small wall map."""

import argparse

import numpy as np
import torch

from embodied_learning.perception.potential_field import (
    paper_potential,
    prioritize_guidance,
)
from embodied_learning.perception.voxel_field import build_voxel_field


def main() -> None:
    """Build a CPU map and query configurable Torch CPU/CUDA observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    occupied = np.zeros((31, 31, 5), dtype=bool)
    occupied[15, :24, :] = True
    field = build_voxel_field(
        occupied,
        goal=(2.5, 1.0, 0.2),
        voxel_size=0.1,
    ).to(args.device)
    positions = torch.tensor(
        [[[1.0, 1.0, 0.2], [1.0, 1.5, 0.2], [1.0, 2.0, 0.2]]],
        device=args.device,
    )
    sample = field.sample(positions)
    assert sample.distance_valid.all() and sample.geodesic_valid.all()
    potential = paper_potential(
        sample.geodesic,
        sample.geodesic_gradient,
        sample.sdf,
        sample.sdf_gradient,
    )
    prioritized = prioritize_guidance(
        potential.force,
        sample.sdf,
        sample.sdf_gradient,
        torch.zeros_like(positions),
        positions.new_tensor([1.0, 0.5, 0.5]),
    )
    assert sample.guidance[0, 0, 1] > 0, "The wall opening is toward +Y."
    print("Obstacle distances (m):", sample.sdf.cpu().tolist())
    print("Goal distances (m):", sample.geodesic.cpu().tolist())
    print("Official-style guidance:", sample.guidance.cpu().tolist())
    print("Paper priority guidance:", prioritized.cpu().tolist())
    print("Flattened observation shape:", tuple(sample.guidance.flatten(1).shape))
    print("Guidance validity:", sample.guidance_valid.cpu().tolist())


if __name__ == "__main__":
    main()
