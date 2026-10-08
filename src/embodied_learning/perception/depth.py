"""Back-project metric optical-axis depth using pixel-center intrinsics."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from embodied_learning.perception.point_cloud import PointCloud


def backproject_depth(
    depth: Tensor,
    intrinsics: Tensor,
    *,
    rotation: Tensor | None = None,
    position: Tensor | None = None,
    min_depth: float = 0.0,
    max_depth: float = math.inf,
) -> PointCloud:
    """Convert depth [B, H, W] into a fixed-size point cloud.

    Args:
        depth: Floating-point optical-axis depth in meters, not ray range.
        intrinsics: Pinhole K shaped [3, 3] or [B, 3, 3], with zero skew.
            Pixel centers are (u + 0.5, v + 0.5).
        rotation: Optional optical-to-world rotation [B, 3, 3].
        position: Optional camera origin in world coordinates [B, 3].
        min_depth: Exclusive minimum valid depth in meters.
        max_depth: Inclusive maximum valid depth in meters.

    Returns:
        Points [B, H * W, 3] and a mask, on depth's device and dtype.
        No simulator imports or CPU transfers are performed.

    Raises:
        ValueError: If shapes, intrinsics, or range limits are invalid.
    """
    if depth.ndim != 3 or not depth.is_floating_point():
        raise ValueError("depth must be a floating-point tensor [B, H, W].")
    batch, height, width = depth.shape
    if min(depth.shape) <= 0:
        raise ValueError("depth dimensions must be positive.")
    if not math.isfinite(min_depth) or not 0 <= min_depth < max_depth:
        raise ValueError("Expected 0 <= min_depth < max_depth.")
    k = intrinsics.to(device=depth.device, dtype=depth.dtype)
    if k.shape == (3, 3):
        k = k.unsqueeze(0).expand(batch, -1, -1)
    if k.shape != (batch, 3, 3):
        raise ValueError("intrinsics must have shape [3, 3] or [B, 3, 3].")
    if not torch.isfinite(k).all() or not (k[:, (0, 1), (0, 1)] > 0).all():
        raise ValueError("Intrinsics must be finite with positive focal lengths.")
    expected = torch.tensor([0.0, 0.0, 1.0], device=k.device, dtype=k.dtype)
    if not torch.allclose(k[:, 2], expected.expand(batch, -1)):
        raise ValueError("Expected intrinsics bottom row [0, 0, 1].")
    if (k[:, 0, 1] != 0).any() or (k[:, 1, 0] != 0).any():
        raise ValueError("Only zero-skew pinhole intrinsics are supported.")
    if (rotation is None) != (position is None):
        raise ValueError("rotation and position must be supplied together.")

    v, u = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype) + 0.5,
        torch.arange(width, device=depth.device, dtype=depth.dtype) + 0.5,
        indexing="ij",
    )
    valid = torch.isfinite(depth) & (depth > min_depth) & (depth <= max_depth)
    z = torch.where(valid, depth, 0.0)
    x = (u - k[:, 0, 2, None, None]) * z / k[:, 0, 0, None, None]
    y = (v - k[:, 1, 2, None, None]) * z / k[:, 1, 1, None, None]
    points = torch.stack((x, y, z), dim=-1).reshape(batch, -1, 3)
    frame = "optical"
    if rotation is not None and position is not None:
        if rotation.shape != (batch, 3, 3) or position.shape != (batch, 3):
            raise ValueError("Expected rotation [B, 3, 3] and position [B, 3].")
        r = rotation.to(device=depth.device, dtype=depth.dtype)
        p = position.to(device=depth.device, dtype=depth.dtype)
        if not torch.isfinite(r).all() or not torch.isfinite(p).all():
            raise ValueError("Camera pose must be finite.")
        points = points @ r.transpose(-1, -2) + p[:, None, :]
        frame = "world"
    mask = valid.reshape(batch, -1)
    points = torch.where(mask[..., None], points, 0.0)
    return PointCloud(points=points, valid_mask=mask, frame=frame)
