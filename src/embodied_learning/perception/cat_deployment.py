"""CAT deployment field preprocessing, adapted under Apache-2.0.

Source: Click-and-Traverse 866ba392, octomap_bridge.py and g1_cat/pf.py.
Torch replaces SciPy morphology; existing source field math supplies CPU FMM.
Origins denote voxel centers, converting the author's corner-origin convention.
The source treats unobserved cells as non-obstacles; coverage belongs to the
mapper, not to this occupancy-only builder. This is not a SLAM implementation.
"""

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from embodied_learning.perception.voxel_field import (
    VoxelPotentialField,
    build_source_voxel_field,
)


@torch.no_grad()
def prepare_cat_occupancy(occupied: Tensor) -> Tensor:
    """Apply the author's fixed-index morphology to bool [B,X,Y,Z] grids.

    The source grid is 0.04 m: upper filling begins at index 20, lower filling
    covers indices 0..4. Other resolutions explicitly change these distances.
    Boundary values match scipy.ndimage's zero-padded binary closing.
    """
    if occupied.ndim != 4 or occupied.dtype != torch.bool:
        raise ValueError("occupied must be bool [B,X,Y,Z].")
    if min(occupied.shape) < 1:
        raise ValueError("occupied axes must be nonempty.")
    values = occupied[:, None].float()
    for _ in range(2):
        values = F.max_pool3d(F.pad(values, (2,) * 6), 5, stride=1)
    for _ in range(2):
        values = 1 - F.max_pool3d(F.pad(1 - values, (2,) * 6, value=1), 5, stride=1)
    neighbors = F.conv3d(values, values.new_ones(1, 1, 3, 3, 3), padding=1)
    result = ((values > 0) & (neighbors > 1))[:, 0]
    result[..., 20:] = result[..., 20:].int().cumsum(-1) > 0
    result[..., :5] = result[..., :5].flip(-1).int().cumsum(-1).flip(-1) > 0
    return result


def build_cat_deployment_field(
    occupied: Tensor,
    *,
    origin: tuple[float, float, float],
    voxel_size: float,
    goal: tuple[float, float, float],
) -> VoxelPotentialField:
    """Build source SDF/BF/GF on CPU, returning tensors on the input device.

    Args:
        occupied: Processed bool [X,Y,Z] obstacle grid, independent of coverage.
        origin: World position of the first voxel center, in meters.
        voxel_size: Isotropic spacing in meters.
        goal: World goal inside a non-obstacle voxel of the grid.

    Raises:
        ValueError: Malformed grid, goal outside the grid or inside an obstacle.

    Empty maps receive the same two artificial corner obstacles as the author's
    OctomapHandler so FMM has a zero interface. Input tensors are never mutated.
    """
    if occupied.ndim != 3 or occupied.dtype != torch.bool:
        raise ValueError("occupied must be bool [X,Y,Z].")
    if min(occupied.shape) < 3:
        raise ValueError("Grid axes must contain at least three cells.")
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive.")
    if (
        len(origin) != 3
        or len(goal) != 3
        or not all(math.isfinite(v) for v in (*origin, *goal))
    ):
        raise ValueError("origin and goal must contain three finite coordinates.")
    grid = occupied.detach().cpu().numpy().copy()
    import numpy as np

    index = np.rint((np.asarray(goal) - origin) / voxel_size).astype(int)
    if (index < 0).any() or (index >= grid.shape).any():
        raise ValueError("goal is outside the online grid.")
    if grid[tuple(index)]:
        raise ValueError("goal is occupied in the online grid.")
    if not grid.any():
        grid[0, 0, -1] = grid[0, -1, -1] = True
    return build_source_voxel_field(
        grid,
        goal=goal,
        origin=origin,
        voxel_size=voxel_size,
        max_speed=0.6,
        goal_radius=0.12,
    ).to(occupied.device)
