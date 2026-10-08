"""Voxel fields with CPU Fast Marching construction and Torch body queries.

Adapted from Click-and-Traverse's pf_modular.py at commit
866ba392f1c1e84b92ad75fa66550f26e8af8e48 (Apache-2.0).
Changes: explicit grid metadata, unknown/reachable masks, validated inputs,
safe masked derivatives, batched Torch interpolation and no plotting/config
dependencies. See docs/potential_fields.md and third_party/licenses/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from embodied_learning.perception.potential_field import _positive


@dataclass(frozen=True)
class FieldSample:
    """Metric scalars [B,K], vectors [B,K,3], and distinct validity masks.

    Out-of-map, unknown and nonfinite queries are invalid, with zero fills.
    In empty maps sdf is +inf (distance_valid=True). Invalid goal distances
    are zero-filled, and must be interpreted using geodesic_valid.
    """

    sdf: Tensor
    geodesic: Tensor
    sdf_gradient: Tensor
    geodesic_gradient: Tensor
    guidance: Tensor
    distance_valid: Tensor
    geodesic_valid: Tensor
    guidance_valid: Tensor


@dataclass(frozen=True)
class VoxelPotentialField:
    """One map shared across batches, with XYZ axis order and metric metadata.

    Grid vertex [i,j,k] has position origin + voxel_size * [i,j,k]. Vector
    grids have shape [X,Y,Z,3]; scalar/mask grids have shape [X,Y,Z].
    Reachability excludes unknown and occupied cells. Known occupied cells
    still provide distance and outward guidance, but no goal distance.
    """

    sdf: Tensor
    geodesic: Tensor
    sdf_gradient: Tensor
    geodesic_gradient: Tensor
    guidance: Tensor
    known: Tensor
    reachable: Tensor
    occupied: Tensor
    origin: tuple[float, float, float]
    voxel_size: float

    def __post_init__(self) -> None:
        if self.sdf.dtype not in (torch.float32, torch.float64):
            raise ValueError("Voxel fields require float32 or float64.")
        _positive("voxel_size", self.voxel_size)
        if len(self.origin) != 3 or not all(math.isfinite(x) for x in self.origin):
            raise ValueError("origin must contain three finite coordinates.")
        shape = self.sdf.shape
        if len(shape) != 3 or min(shape) < 3 or not self.sdf.is_floating_point():
            raise ValueError("Scalar grids must be floating XYZ with size >=3.")
        for name in (
            "geodesic",
            "sdf_gradient",
            "geodesic_gradient",
            "guidance",
            "known",
            "reachable",
            "occupied",
        ):
            tensor = getattr(self, name)
            is_mask = name in ("known", "reachable", "occupied")
            expected = (
                (*shape, 3) if "gradient" in name or name == "guidance" else shape
            )
            dtype = torch.bool if is_mask else self.sdf.dtype
            if (
                tensor.shape != expected
                or tensor.dtype != dtype
                or tensor.device != self.sdf.device
            ):
                raise ValueError(f"{name} has incompatible shape, dtype or device.")
        if torch.isnan(self.sdf).any() or torch.isneginf(self.sdf).any():
            raise ValueError("sdf must not contain NaN or negative infinity.")
        if (self.reachable & (~self.known | self.occupied)).any():
            raise ValueError("Reachable cells must be known and unoccupied.")
        if not torch.isfinite(self.geodesic[self.reachable]).all():
            raise ValueError("Reachable goal distances must be finite.")
        for name in ("sdf_gradient", "geodesic_gradient", "guidance"):
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must be finite.")

    def to(
        self, device: str | torch.device, *, dtype: torch.dtype | None = None
    ) -> VoxelPotentialField:
        """Move the map once before repeated queries, retaining boolean masks."""
        dtype = self.sdf.dtype if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Voxel fields require float32 or float64.")
        updates = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Tensor):
                target_dtype = torch.bool if value.dtype == torch.bool else dtype
                updates[item.name] = value.to(device=device, dtype=target_dtype)
        return replace(self, **updates)

    def sample(self, positions: Tensor) -> FieldSample:
        """Trilinearly query [B,K,3] positions in the map's coordinate frame.

        Queries must match map dtype/device. Each mask requires all eight
        interpolation corners to be valid; outside points are never silently
        clamped to the map boundary. Both boundary vertices are queryable.
        """
        if (
            positions.ndim != 3
            or positions.shape[-1] != 3
            or positions.device != self.sdf.device
            or positions.dtype != self.sdf.dtype
        ):
            raise ValueError("positions must be [B,K,3] with map dtype/device.")
        origin = positions.new_tensor(self.origin)
        index = (positions - origin) / self.voxel_size
        maximum = positions.new_tensor(self.sdf.shape) - 1
        # Small roundoff tolerance only; genuinely outside points stay invalid.
        tolerance = 8 * torch.finfo(positions.dtype).eps * maximum
        inside = (
            torch.isfinite(index).all(-1)
            & (index >= -tolerance).all(-1)
            & (index <= maximum + tolerance).all(-1)
        )
        index = torch.where(inside[..., None], index, 0.0).clamp_min(0)
        index = torch.minimum(index, maximum)
        base = torch.minimum(index.floor().long(), maximum.long() - 1)
        fraction = index - base
        offsets = torch.tensor(
            [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
            device=positions.device,
        )
        corners = base[..., None, :] + offsets
        weights = torch.where(
            offsets.bool(), fraction[..., None, :], 1 - fraction[..., None, :]
        ).prod(-1)
        ix, iy, iz = corners.unbind(-1)

        def mask(grid: Tensor) -> Tensor:
            return inside & grid[ix, iy, iz].all(-1)

        def interpolate(grid: Tensor, valid: Tensor) -> Tensor:
            values = grid[ix, iy, iz]
            scalar = grid.ndim == 3
            w = weights if scalar else weights[..., None]
            values_finite = torch.where(torch.isfinite(values), values, 0.0)
            result = (values_finite * w).sum(-1 if scalar else -2)
            if scalar:
                infinite = (torch.isposinf(values) & (w > 0)).any(-1)
                result = torch.where(infinite, math.inf, result)
            else:
                valid = valid[..., None]
            return torch.where(valid, result, 0.0)

        distance_valid = mask(self.known)
        geodesic_valid = mask(self.known & self.reachable)
        guidance_valid = mask(self.known & (self.reachable | self.occupied))
        return FieldSample(
            interpolate(self.sdf, distance_valid),
            interpolate(self.geodesic, geodesic_valid),
            interpolate(self.sdf_gradient, distance_valid),
            interpolate(self.geodesic_gradient, geodesic_valid),
            interpolate(self.guidance, guidance_valid),
            distance_valid,
            geodesic_valid,
            guidance_valid,
        )


def _masked_gradient(values: NDArray, valid: NDArray, spacing: float) -> NDArray:
    """Differentiate within reachable space without importing obstacle fills."""
    safe = np.where(valid, values, 0.0)
    derivatives = []
    for axis in range(3):
        lo = np.maximum(np.arange(values.shape[axis]) - 1, 0)
        hi = np.minimum(np.arange(values.shape[axis]) + 1, values.shape[axis] - 1)
        lower, upper = np.take(safe, lo, axis), np.take(safe, hi, axis)
        vl, vu = np.take(valid, lo, axis), np.take(valid, hi, axis)
        dims = [1, 1, 1]
        dims[axis] = values.shape[axis]
        has_lower = vl & (np.arange(values.shape[axis]) > 0).reshape(dims)
        has_upper = vu & (
            np.arange(values.shape[axis]) < values.shape[axis] - 1
        ).reshape(dims)
        numerator = np.where(has_upper, upper, safe) - np.where(
            has_lower, lower, safe
        )
        count = has_lower.astype(float) + has_upper.astype(float)
        derivatives.append(
            np.where(valid, numerator / (np.maximum(count, 1) * spacing), 0.0)
        )
    return np.stack(derivatives, -1).astype(np.float32)


def build_voxel_field(
    occupied: NDArray[np.bool_],
    *,
    goal: tuple[float, float, float],
    voxel_size: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    known_mask: NDArray[np.bool_] | None = None,
    goal_radius: float | None = None,
    influence_radius: float | None = None,
    max_speed: float = 0.6,
    slowdown_radius: float = 0.3,
) -> VoxelPotentialField:
    """Construct SDF, Fast Marching goal distance and official-style guidance.

    Args:
        occupied: Boolean XYZ grid, True for forbidden occupied geometry.
        goal: Target in the same frame as origin, in meters.
        voxel_size: Uniform vertex spacing in meters.
        origin: Position of grid vertex [0,0,0].
        known_mask: Observed cells. Unknown cells block traversal and SDF
            clearance, but cannot be queried as known geometry. None assumes
            a completely known map, suitable for synthetic/simulator maps.
        goal_radius: Negative goal seed sphere radius; default 1.2 voxels.
        influence_radius: Tangential projection support; default 5 voxels.
        max_speed: Guidance magnitude far from the goal, in m/s.
        slowdown_radius: XY target distance where cubic slowdown begins.

    Returns:
        A float32 CPU map. Call .to(device) once for batched GPU queries.
        SDF is measured to the implicit zero surface between voxel vertices,
        not directly to mesh triangles. Unknown/unreachable goal distances
        remain +inf. Offline CPU construction is not a per-step GPU mapper.

    Raises:
        ValueError: Invalid grid, target, parameters or goal seed.
        ImportError: Install the optional scikit-fmm dependency.
    """
    _positive("voxel_size", voxel_size)
    occupied = np.asarray(occupied)
    if occupied.dtype != np.bool_ or occupied.ndim != 3 or min(occupied.shape) < 3:
        raise ValueError("occupied must be a bool XYZ grid with axis sizes >=3.")
    if known_mask is None:
        known = np.ones_like(occupied)
    else:
        known = np.asarray(known_mask)
        if known.shape != occupied.shape or known.dtype != np.bool_:
            raise ValueError("known_mask must be bool with the occupied grid shape.")
    for name, point in (("goal", goal), ("origin", origin)):
        if len(point) != 3 or not all(math.isfinite(x) for x in point):
            raise ValueError(f"{name} must have three finite coordinates.")
    goal_index = (np.asarray(goal) - origin) / voxel_size
    if np.any(goal_index < 0) or np.any(goal_index > np.asarray(occupied.shape) - 1):
        raise ValueError("goal must lie inside the grid.")
    blocked = occupied | ~known
    if blocked[tuple(np.rint(goal_index).astype(int))]:
        raise ValueError("goal must lie in known free space.")
    goal_radius = 1.2 * voxel_size if goal_radius is None else goal_radius
    influence_radius = (
        5 * voxel_size if influence_radius is None else influence_radius
    )
    for name, value in (
        ("goal_radius", goal_radius),
        ("influence_radius", influence_radius),
        ("max_speed", max_speed),
        ("slowdown_radius", slowdown_radius),
    ):
        _positive(name, value)
    try:
        import skfmm
    except ImportError as error:
        raise ImportError(
            "Install embodied-learning[potential-fields] for scikit-fmm."
        ) from error

    coordinates = np.stack(
        np.meshgrid(
            *(
                np.arange(n) * voxel_size + origin[i]
                for i, n in enumerate(occupied.shape)
            ),
            indexing="ij",
        ),
        -1,
    )
    delta = coordinates - goal
    seed = (np.linalg.norm(delta, axis=-1) <= goal_radius) & ~blocked
    if not seed.any() or not ((~blocked) & ~seed).any():
        raise ValueError("Goal sphere needs both seeded and unseeded free vertices.")
    # Use only the target-connected seed component, even if the sphere cuts walls.
    from scipy.ndimage import label

    components, _ = label(~blocked)
    target_component = components[tuple(np.rint(goal_index).astype(int))]
    seed &= components == target_component
    if not seed.any():
        raise ValueError("Goal seed contains no target-connected grid vertices.")
    phi = np.ma.MaskedArray(np.where(seed, -1.0, 1.0), mask=blocked)
    distance = np.ma.asarray(skfmm.distance(phi, dx=voxel_size))
    reachable = ~np.ma.getmaskarray(distance) & ~blocked
    geodesic = np.where(reachable, np.maximum(distance.filled(np.inf), 0.0), np.inf)
    goal_gradient = _masked_gradient(geodesic, reachable, voxel_size)

    if blocked.any():
        sdf = np.asarray(skfmm.distance(np.where(blocked, -1.0, 1.0), dx=voxel_size))
        normal = np.stack(np.gradient(sdf, voxel_size, edge_order=2), -1)
    else:
        sdf = np.full(occupied.shape, np.inf)
        normal = np.zeros((*occupied.shape, 3))
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    unit_normal = normal / np.maximum(norm, 1e-9)
    attraction = -goal_gradient
    tangent = (
        attraction
        - np.sum(attraction * unit_normal, -1, keepdims=True) * unit_normal
    )
    t = np.clip(np.maximum(sdf, 0.0) / influence_radius, 0.0, 1.0)
    weight = (1 - t * t * (3 - 2 * t))[..., None]
    mixed = (1 - weight) * attraction + weight * tangent
    mixed[occupied & known] = unit_normal[occupied & known]
    mixed[~known] = 0.0
    mixed_norm = np.linalg.norm(mixed, axis=-1, keepdims=True)
    direction = mixed / np.maximum(mixed_norm, 1e-9)
    xy_distance = np.linalg.norm(delta[..., :2], axis=-1)
    speed = max_speed * np.minimum(xy_distance / slowdown_radius, 1.0) ** 3
    guidance = direction * speed[..., None]
    return VoxelPotentialField(
        sdf=torch.tensor(sdf, dtype=torch.float32),
        geodesic=torch.tensor(geodesic, dtype=torch.float32),
        sdf_gradient=torch.tensor(normal, dtype=torch.float32),
        geodesic_gradient=torch.tensor(goal_gradient, dtype=torch.float32),
        guidance=torch.tensor(guidance, dtype=torch.float32),
        known=torch.tensor(known),
        reachable=torch.tensor(reachable),
        occupied=torch.tensor(occupied),
        origin=tuple(origin),
        voxel_size=voxel_size,
    )
