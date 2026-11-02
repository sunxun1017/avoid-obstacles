"""Batched dual projections with explicit, configurable geometry.

This is an engineering interpretation of ADAPT, not a verified reproduction:
distance rows are height bins and columns are azimuth bins. The paper's
radial-bin wording conflicts with its vertical cylindrical illustration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from embodied_learning.perception.geometry import validate_cloud
from embodied_learning.perception.point_cloud import PointCloud


@dataclass(frozen=True)
class ProjectionResult:
    """Metric maps and boolean observed-cell masks with matching shapes.

    Unknown elevation is filled with zero; unknown distance with radius.
    These fill values never establish free space. Consumers need the masks.
    """

    elevation: Tensor
    distance: Tensor
    elevation_valid: Tensor
    distance_valid: Tensor

    def flattened(self) -> ProjectionResult:
        """Flatten spatial axes while preserving the environment axis."""
        return ProjectionResult(
            self.elevation.flatten(1),
            self.distance.flatten(1),
            self.elevation_valid.flatten(1),
            self.distance_valid.flatten(1),
        )


def _check_range(name: str, bounds: tuple[float, float]) -> None:
    if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds):
        raise ValueError(f"{name} must contain two finite limits.")
    if bounds[0] >= bounds[1]:
        raise ValueError(f"{name} must be strictly increasing.")


class DualProjector:
    """Rasterize yaw-local points into elevation and obstacle-distance maps.

    Elevation rows index X and columns Y. Extents are multiples of radius.
    Distance rows index height and columns azimuth, storing minimum XY range.
    All bin intervals are lower-inclusive and upper-exclusive. Horizontal
    range is additionally limited to radius. Height limits are root-relative.
    No temporal fusion or ground segmentation is performed.
    """

    def __init__(
        self,
        num_envs: int,
        elevation_shape: tuple[int, int] = (21, 17),
        distance_shape: tuple[int, int] = (13, 17),
        *,
        height_range: tuple[float, float],
        elevation_height_range: tuple[float, float],
        forward_range: tuple[float, float] = (-0.2, 1.0),
        lateral_range: tuple[float, float] = (-0.5, 0.5),
        azimuth_range: tuple[float, float] = (-math.pi / 4, math.pi / 4),
    ) -> None:
        """Set grid geometry; choose height ranges for the robot and terrain.

        The height filter excludes overhead surfaces from elevation. It is
        only a simple filter, not a guarantee of correct ground selection.
        Angular limits are in radians and height limits in meters.
        """
        if (
            not isinstance(num_envs, int)
            or isinstance(num_envs, bool)
            or num_envs <= 0
        ):
            raise ValueError("num_envs must be a positive integer.")
        for shape in (elevation_shape, distance_shape):
            if len(shape) != 2 or any(
                not isinstance(x, int) or isinstance(x, bool) or x <= 0
                for x in shape
            ):
                raise ValueError("Grid shapes must contain two positive integers.")
        for name, limits in (
            ("height_range", height_range),
            ("elevation_height_range", elevation_height_range),
            ("forward_range", forward_range),
            ("lateral_range", lateral_range),
            ("azimuth_range", azimuth_range),
        ):
            _check_range(name, limits)
        if not -math.pi <= azimuth_range[0] < azimuth_range[1] <= math.pi:
            raise ValueError("azimuth_range must lie within [-pi, pi].")
        self.num_envs = num_envs
        self.elevation_shape = elevation_shape
        self.distance_shape = distance_shape
        self.height_range = height_range
        self.elevation_height_range = elevation_height_range
        self.forward_range = forward_range
        self.lateral_range = lateral_range
        self.azimuth_range = azimuth_range
        self._last_result: ProjectionResult | None = None

    def project(self, cloud: PointCloud, radius: float | Tensor) -> ProjectionResult:
        """Project one frame, using a scalar radius or radii [B] in meters.

        Finite masked points are used; unknown cells retain explicit masks.
        Each call creates fresh maps and never reuses previous observations.
        Outputs preserve the point tensor's device and floating-point dtype.
        """
        validate_cloud(cloud)
        if cloud.frame != "yaw_local":
            raise ValueError("Projection requires a yaw_local point cloud.")
        points = cloud.points
        if points.shape[0] != self.num_envs:
            raise ValueError("Cloud batch size must match num_envs.")
        r = torch.as_tensor(radius, device=points.device, dtype=points.dtype)
        if r.ndim == 0:
            r = r.expand(self.num_envs)
        if (
            r.shape != (self.num_envs,)
            or not torch.isfinite(r).all()
            or (r <= 0).any()
        ):
            raise ValueError("radius must be finite, positive, and scalar or [B].")
        valid = cloud.valid_mask & torch.isfinite(points).all(-1)
        safe = torch.where(valid[..., None], points, 0.0)
        x, y, z = safe.unbind(-1)
        horizontal = torch.hypot(x, y)
        valid = valid & (horizontal <= r[:, None])

        def bins(
            values: Tensor, bounds: tuple[float, float], count: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            scaled = (values - bounds[0]) / (bounds[1] - bounds[0])
            inside = (scaled >= 0) & (scaled < 1)
            indices = (scaled * count).floor().long().clamp(0, count - 1)
            return indices, inside

        ex, vx = bins(x / r[:, None], self.forward_range, self.elevation_shape[0])
        ey, vy = bins(y / r[:, None], self.lateral_range, self.elevation_shape[1])
        emin, emax = self.elevation_height_range
        elevation_valid = valid & vx & vy & (z >= emin) & (z < emax)
        elevation, elevation_mask = self._reduce(
            z,
            ex * self.elevation_shape[1] + ey,
            elevation_valid,
            self.elevation_shape,
            "amax",
        )
        elevation = torch.where(elevation_mask, elevation, 0.0)

        dz, vz = bins(z, self.height_range, self.distance_shape[0])
        da, va = bins(torch.atan2(y, x), self.azimuth_range, self.distance_shape[1])
        distance_valid = valid & vz & va & (horizontal > 0)
        distance, distance_mask = self._reduce(
            horizontal,
            dz * self.distance_shape[1] + da,
            distance_valid,
            self.distance_shape,
            "amin",
        )
        distance = torch.where(distance_mask, distance, r[:, None, None])
        result = ProjectionResult(elevation, distance, elevation_mask, distance_mask)
        self._last_result = result
        return result

    def _reduce(
        self,
        values: Tensor,
        indices: Tensor,
        valid: Tensor,
        shape: tuple[int, int],
        reduction: str,
    ) -> tuple[Tensor, Tensor]:
        neutral = -math.inf if reduction == "amax" else math.inf
        output = values.new_full((self.num_envs, math.prod(shape)), neutral)
        source = torch.where(valid, values, neutral)
        output.scatter_reduce_(
            1, indices, source, reduce=reduction, include_self=True
        )
        observed = torch.isfinite(output)
        return output.reshape(self.num_envs, *shape), observed.reshape(
            self.num_envs, *shape
        )

    def flattened(self) -> ProjectionResult:
        """Return flattened maps from the most recent project call."""
        if self._last_result is None:
            raise RuntimeError("Call project() before flattened().")
        return self._last_result.flattened()
