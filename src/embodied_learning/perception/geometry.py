"""Transform world points to a gravity-aligned robot heading frame."""

from __future__ import annotations

import torch
from torch import Tensor

from embodied_learning.perception.point_cloud import PointCloud


def validate_cloud(cloud: PointCloud) -> None:
    """Validate the batched point and mask contract without changing values."""
    points, mask = cloud.points, cloud.valid_mask
    if points.ndim != 3 or points.shape[-1] != 3 or not points.is_floating_point():
        raise ValueError("points must be floating-point [B, N, 3].")
    if mask.shape != points.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean [B, N].")
    if mask.device != points.device:
        raise ValueError("points and valid_mask must share a device.")


def world_to_yaw_local(
    cloud: PointCloud,
    position: Tensor,
    quaternion: Tensor,
) -> PointCloud:
    """Express world points relative to a robot's position and yaw.

    Args:
        cloud: World points in meters, with a boolean valid mask.
        position: Robot root position in world coordinates [B, 3].
        quaternion: Robot root orientation [B, 4], ordered wxyz.

    Returns:
        A cloud in yaw_local coordinates: X forward, Y left, Z up.
        Z is relative to the root height, not an absolute ground height.
        Roll and pitch do not tilt this frame. Invalid points contain zeros.

    Raises:
        ValueError: For mismatched shapes, frames, or invalid robot poses.
    """
    validate_cloud(cloud)
    if cloud.frame != "world":
        raise ValueError("Expected a world-frame cloud.")
    points = cloud.points
    batch = points.shape[0]
    if position.shape != (batch, 3) or quaternion.shape != (batch, 4):
        raise ValueError("Expected position [B, 3] and quaternion [B, 4].")
    pos = position.to(points)
    quat = quaternion.to(points)
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    if not torch.isfinite(pos).all() or not torch.isfinite(quat).all():
        raise ValueError("Robot pose must be finite.")
    if (norm <= torch.finfo(points.dtype).eps).any():
        raise ValueError("Robot quaternion must have nonzero norm.")
    w, x, y, z = (quat / norm).unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    valid = cloud.valid_mask & torch.isfinite(points).all(-1)
    delta = torch.where(valid[..., None], points, 0.0) - pos[:, None, :]
    local = torch.stack(
        (
            c * delta[..., 0] + s * delta[..., 1],
            -s * delta[..., 0] + c * delta[..., 1],
            delta[..., 2],
        ),
        dim=-1,
    )
    local = torch.where(valid[..., None], local, 0.0)
    return PointCloud(local, valid, "yaw_local")
