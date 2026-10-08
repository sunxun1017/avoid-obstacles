"""Simulator-independent ray measurements in a stable world/map frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from embodied_learning.perception.depth import backproject_depth
from embodied_learning.perception.geometry import validate_cloud
from embodied_learning.perception.point_cloud import PointCloud


@dataclass(frozen=True)
class RayMeasurements:
    """Origins/endpoints [B,N,3] in meters, masks [B,N], in world coordinates.

    valid_mask means that a ray provides information. hit_mask means its
    endpoint is an observed surface; valid non-hit rays only clear free space.
    Invalid rays provide no information. Masked or nonfinite coordinates are
    ignored by mapping. Tensors remain mutable despite frozen fields.
    """

    origins: Tensor
    endpoints: Tensor
    valid_mask: Tensor
    hit_mask: Tensor
    frame: str = "world"


def validate_rays(rays: RayMeasurements) -> None:
    """Validate tensor shapes, precision, device and stable world frame."""
    points = rays.endpoints
    if (
        points.ndim != 3
        or points.shape[-1] != 3
        or points.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("Ray coordinates must be float32 or float64 [B,N,3].")
    if rays.frame != "world":
        raise ValueError("Ray mapping requires a stable world frame.")
    if (
        rays.origins.shape != points.shape
        or rays.origins.dtype != points.dtype
        or rays.origins.device != points.device
    ):
        raise ValueError("Origins must match endpoint shape, dtype and device.")
    for mask in (rays.valid_mask, rays.hit_mask):
        if (
            mask.shape != points.shape[:2]
            or mask.dtype != torch.bool
            or mask.device != points.device
        ):
            raise ValueError("Ray masks must be boolean [B,N] on the same device.")


def rays_from_point_cloud(cloud: PointCloud, origins: Tensor) -> RayMeasurements:
    """Convert world-frame surface hits to rays without inferring missing rays.

    Args:
        cloud: Surface endpoints in world coordinates, [B,N,3].
        origins: Sensor origins [B,3] or per-ray origins [B,N,3], in meters.

    Returns:
        Independent ray snapshots with hit_mask equal to valid_mask. A cloud
        with missing points cannot certify free space along missing beams.

    Raises:
        ValueError: Incompatible shapes, coordinates or unsupported precision.
    """
    validate_cloud(cloud)
    if cloud.frame != "world":
        raise ValueError("Expected a world-frame surface cloud.")
    points = cloud.points
    if origins.shape == (points.shape[0], 3):
        origins = origins[:, None, :].expand_as(points)
    rays = RayMeasurements(
        origins.clone(),
        points.clone(),
        cloud.valid_mask.clone(),
        cloud.valid_mask.clone(),
    )
    validate_rays(rays)
    return rays


def rays_from_depth(
    depth: Tensor,
    intrinsics: Tensor,
    *,
    rotation: Tensor,
    position: Tensor,
    max_depth: float,
    no_hit_mask: Tensor | None = None,
) -> RayMeasurements:
    """Convert optical-axis depth and optical-to-world pose into mapping rays.

    Args:
        depth: Floating optical-axis depth [B,H,W], in meters.
        intrinsics: Pinhole K [3,3] or [B,3,3], using pixel centers.
        rotation: Optical-to-world rotation [B,3,3].
        position: Camera origins [B,3] in world coordinates.
        max_depth: Finite optical Z truncation, not spherical ray range.
            Finite positive returns beyond this limit clear to the limit,
            without adding a surface at the truncated endpoint.
        no_hit_mask: Optional boolean [B,H,W] confirmed no-return pixels.
            These clear to max_depth even if the depth buffer is inf/NaN.
            Unmarked inf, NaN and nonpositive depths provide no information.

    Returns:
        World rays [B,H*W,3] and independent information/hit masks.

    Raises:
        ValueError: Invalid shapes, limits, intrinsics, pose or precision.
            Classify sensor-specific background values before this call.
    """
    if depth.ndim != 3 or depth.dtype not in (torch.float32, torch.float64):
        raise ValueError("depth must be float32 or float64 [B,H,W].")
    if not math.isfinite(max_depth) or max_depth <= 0:
        raise ValueError("max_depth must be finite and positive.")
    if no_hit_mask is None:
        no_hit_mask = torch.zeros_like(depth, dtype=torch.bool)
    if (
        no_hit_mask.shape != depth.shape
        or no_hit_mask.dtype != torch.bool
        or no_hit_mask.device != depth.device
    ):
        raise ValueError("no_hit_mask must be boolean and match depth shape/device.")
    measured = torch.isfinite(depth) & (depth > 0)
    valid = measured | no_hit_mask
    hit = measured & ~no_hit_mask & (depth <= max_depth)
    effective_depth = torch.where(hit, depth, max_depth)
    effective_depth = torch.where(valid, effective_depth, 0.0)
    cloud = backproject_depth(
        effective_depth,
        intrinsics,
        rotation=rotation,
        position=position,
        max_depth=max_depth,
    )
    origins = position.to(depth)[:, None, :].expand_as(cloud.points).clone()
    rays = RayMeasurements(origins, cloud.points, valid.flatten(1), hit.flatten(1))
    validate_rays(rays)
    return rays
