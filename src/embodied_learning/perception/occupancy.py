"""Dense batched occupancy mapping from generic rays, without simulator APIs.

Per-frame hit precedence and voxel deduplication follow OctoMap's documented
measurement semantics. This is an independent dense Torch implementation,
not an octree, SLAM system or a port of OctoMap source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from embodied_learning.perception.rays import RayMeasurements, validate_rays


@dataclass(frozen=True)
class OccupancySnapshot:
    """Independent map data: grids [B,X,Y,Z], node origins [B,3] in meters.

    occupied and known_mask provide a conservative three-state classification.
    Observed but uncertain cells have known_mask=False. All tensors are clones
    or newly computed values; caller mutation cannot change the live mapper.
    """

    log_odds: Tensor
    occupied: Tensor
    known_mask: Tensor
    observed: Tensor
    origin: Tensor
    voxel_size: float


class OccupancyMapper:
    """Fuse rays into fixed, separate maps for vectorized environments.

    Grid center [i,j,k] is origin + voxel_size*[i,j,k], matching the field
    builder's query nodes. A cell extends half a voxel around that center.
    World/map coordinates must remain stable between updates. The map does
    not estimate pose, roll with the robot, filter ground or rebuild fields.
    """

    def __init__(
        self,
        num_envs: int,
        shape: tuple[int, int, int],
        *,
        voxel_size: float,
        origin: tuple[float, float, float] | Tensor = (0.0, 0.0, 0.0),
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        hit_log_odds: float = 0.85,
        miss_log_odds: float = -0.4,
        min_log_odds: float = -2.0,
        max_log_odds: float = 3.5,
        free_threshold: float = -0.2,
        occupied_threshold: float = 0.2,
    ) -> None:
        """Configure grid geometry and bounded log-odds evidence increments.

        origin is one center coordinate [3] or separate origins [B,3].
        Shapes need at least three cells per axis for downstream field
        derivatives. Evidence is added once per cell per update (one scan),
        with hits preferred over free-ray traversals in that scan.
        Invalid configuration raises ValueError. Dtype must be float32/64.
        """
        if (
            not isinstance(num_envs, int)
            or isinstance(num_envs, bool)
            or num_envs <= 0
        ):
            raise ValueError("num_envs must be a positive integer.")
        if len(shape) != 3 or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 3 for n in shape
        ):
            raise ValueError("shape must contain three integer dimensions >=3.")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Mapping requires float32 or float64.")
        parameters = (
            voxel_size,
            hit_log_odds,
            miss_log_odds,
            min_log_odds,
            max_log_odds,
            free_threshold,
            occupied_threshold,
        )
        if (
            not all(math.isfinite(v) for v in parameters)
            or voxel_size <= 0
            or hit_log_odds <= 0
            or miss_log_odds >= 0
            or not min_log_odds
            < free_threshold
            < 0
            < occupied_threshold
            < max_log_odds
        ):
            raise ValueError(
                "Invalid voxel size, evidence or classification limits."
            )
        self.num_envs = num_envs
        self.shape = shape
        self.voxel_size = voxel_size
        self.device = torch.device(device)
        self.dtype = dtype
        origin_tensor = torch.as_tensor(origin, device=self.device, dtype=dtype)
        if origin_tensor.shape == (3,):
            origin_tensor = origin_tensor[None].expand(num_envs, -1)
        if (
            origin_tensor.shape != (num_envs, 3)
            or not torch.isfinite(origin_tensor).all()
        ):
            raise ValueError("origin must be finite [3] or [num_envs,3].")
        self._origin = origin_tensor.clone()
        # Canonicalize aliases such as "cuda" to the allocated device index.
        self.device = self._origin.device
        self._log_odds = torch.zeros(
            (num_envs, *shape), device=self.device, dtype=dtype
        )
        self._observed = torch.zeros_like(self._log_odds, dtype=torch.bool)
        self.hit_log_odds = hit_log_odds
        self.miss_log_odds = miss_log_odds
        self.min_log_odds = min_log_odds
        self.max_log_odds = max_log_odds
        self.free_threshold = free_threshold
        self.occupied_threshold = occupied_threshold

    @torch.no_grad()
    def reset(self, env_ids: Tensor | None = None) -> None:
        """Forget all maps, or selected environment indices [M] on map device.

        Origins and grid configuration remain fixed. Invalid indices raise
        ValueError before mutation. Call on episode resets to prevent map
        history from crossing episodes.
        """
        if env_ids is None:
            self._log_odds.zero_()
            self._observed.zero_()
            return
        if (
            env_ids.ndim != 1
            or env_ids.dtype != torch.long
            or env_ids.device != self.device
            or (env_ids < 0).any()
            or (env_ids >= self.num_envs).any()
        ):
            raise ValueError("env_ids must be valid int64 [M] on map device.")
        self._log_odds[env_ids] = 0.0
        self._observed[env_ids] = False

    @torch.no_grad()
    def snapshot(self) -> OccupancySnapshot:
        """Return independent grids and metadata without building a field."""
        occupied = self._observed & (self._log_odds >= self.occupied_threshold)
        free = self._observed & (self._log_odds <= self.free_threshold)
        return OccupancySnapshot(
            self._log_odds.clone(),
            occupied,
            occupied | free,
            self._observed.clone(),
            self._origin.clone(),
            self.voxel_size,
        )

    @torch.no_grad()
    def update(self, rays: RayMeasurements) -> None:
        """Fuse one scan per environment using clipped 3D DDA ray traversal.

        Args:
            rays: World coordinates/masks on the map's dtype/device. Missing,
                nonfinite and zero-length rays are ignored. A valid no-hit
                endpoint is the certified free-space range, not a surface.

        Rays are clipped to the map box; out-of-box hits are not inserted.
        Cells behind a measured hit remain unknown unless previously observed.
        Tied boundary crossings step all tied axes, excluding cells touched
        only at an edge/corner. Segments within 8*dtype-eps in ray parameter
        space are conservatively ignored. CUDA traversal is batched across rays/envs;
        its loop bound is the sum of grid dimensions, without CPU transfers.

        Raises:
            ValueError: Incompatible tensor shapes, frame, precision or device.
        """
        validate_rays(rays)
        if (
            rays.endpoints.shape[0] != self.num_envs
            or rays.endpoints.device != self.device
            or rays.endpoints.dtype != self.dtype
        ):
            raise ValueError("Rays must match mapper batch, dtype and device.")
        size = rays.endpoints.new_tensor(self.shape)
        lower = self._origin[:, None] - self.voxel_size / 2
        valid = (
            rays.valid_mask
            & torch.isfinite(rays.origins).all(-1)
            & torch.isfinite(rays.endpoints).all(-1)
        )
        o = (
            torch.where(valid[..., None], rays.origins, lower) - lower
        ) / self.voxel_size
        e = (
            torch.where(valid[..., None], rays.endpoints, lower) - lower
        ) / self.voxel_size
        d = e - o
        valid &= torch.isfinite(o).all(-1) & torch.isfinite(e).all(-1)
        valid &= torch.isfinite(d).all(-1) & (d != 0).any(-1)
        o = torch.where(valid[..., None], o, 0.0)
        e = torch.where(valid[..., None], e, 0.0)
        d = e - o
        nonparallel = d != 0
        safe_d = torch.where(nonparallel, d, 1.0)
        a, b = -o / safe_d, (size - o) / safe_d
        parallel_inside = (o >= 0) & (o < size)
        lo = torch.where(
            nonparallel,
            torch.minimum(a, b),
            torch.where(parallel_inside, -math.inf, math.inf),
        )
        hi = torch.where(
            nonparallel,
            torch.maximum(a, b),
            torch.where(parallel_inside, math.inf, -math.inf),
        )
        entry = lo.amax(-1).clamp_min(0.0)
        exit_time = hi.amin(-1).clamp_max(1.0)
        tolerance = 8 * torch.finfo(self.dtype).eps
        active = valid & (exit_time - entry > tolerance)
        current_time = entry
        start = o + torch.where(active, entry, 0.0)[..., None] * d
        start = torch.where(active[..., None], start, 0.0)
        cell = start.floor()
        cell -= ((d < 0) & (start == cell)).to(self.dtype)
        cell = torch.minimum(cell.clamp_min(0), size - 1).long()
        step = d.sign().long()
        boundary = cell.to(self.dtype) + (step > 0).to(self.dtype)
        next_crossing = torch.where(nonparallel, (boundary - o) / safe_d, math.inf)
        hit_valid = valid & rays.hit_mask & (e >= 0).all(-1) & (e < size).all(-1)
        hit_cell = torch.minimum(e.clamp_min(0), size - 1).floor().long()
        count = math.prod(self.shape)
        free_count = torch.zeros(
            (self.num_envs, count), device=self.device, dtype=torch.int32
        )
        hit_count = torch.zeros_like(free_count)

        def flat_index(indices: Tensor) -> Tensor:
            bounded = torch.minimum(indices.clamp_min(0), size.long() - 1)
            x, y, z = bounded.unbind(-1)
            return (x * self.shape[1] + y) * self.shape[2] + z

        hit_count.scatter_add_(1, flat_index(hit_cell), hit_valid.int())
        for _ in range(sum(self.shape) + 1):
            if self.device.type == "cpu" and not active.any():
                break
            at_hit = hit_valid & (cell == hit_cell).all(-1)
            next_time = next_crossing.amin(-1)
            positive_span = (
                torch.minimum(next_time, exit_time) - current_time > tolerance
            )
            free_count.scatter_add_(
                1, flat_index(cell), (active & positive_span & ~at_hit).int()
            )
            crossed = (
                nonparallel
                & active[..., None]
                & (next_crossing <= next_time[..., None] + tolerance)
            )
            cell += crossed.long() * step
            # Recompute absolute times to avoid drift from accumulated intervals.
            boundary = cell.to(self.dtype) + (step > 0).to(self.dtype)
            next_crossing = torch.where(
                nonparallel, (boundary - o) / safe_d, math.inf
            )
            current_time = torch.maximum(current_time, next_time)
            active &= (
                (next_time < exit_time - tolerance)
                & (cell >= 0).all(-1)
                & (cell < size).all(-1)
            )
        hits = (hit_count > 0).reshape_as(self._observed)
        frees = (free_count > 0).reshape_as(self._observed) & ~hits
        self._observed |= hits | frees
        self._log_odds += hits.to(self.dtype) * self.hit_log_odds
        self._log_odds += frees.to(self.dtype) * self.miss_log_odds
        self._log_odds.clamp_(self.min_log_odds, self.max_log_odds)
