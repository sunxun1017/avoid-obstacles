"""Sensor-driven CAT fields with explicit freshness and measured coverage.

This module owns no simulator, policy, transport, or localization. Source FMM or
CUDA Eikonal builds changed fields; fresh unchanged scans may reuse a field.
The source occupancy-only interpretation is deliberate: unknown is traversable
for field construction, but never reported as measured free space.
"""

import math
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from embodied_learning.perception.cat_deployment import (
    build_cat_deployment_field,
    prepare_cat_occupancy,
)
from embodied_learning.perception.occupancy import OccupancyMapper, OccupancySnapshot
from embodied_learning.perception.rays import RayMeasurements, validate_rays
from embodied_learning.perception.voxel_field import (
    FieldSample,
    VoxelPotentialField,
    sample_source_field,
)


@dataclass(frozen=True)
class OnlineFieldQuery:
    """Field validity and sensor coverage have different meanings in CAT."""

    sample: FieldSample
    observed: Tensor
    age: Tensor
    versions: Tensor


def default_field_sample(positions: Tensor) -> FieldSample:
    """Source stop/default features, with validity explicitly false."""
    scalar = positions.new_zeros(positions.shape[:2])
    vector = torch.zeros_like(positions)
    invalid = torch.zeros_like(scalar, dtype=torch.bool)
    return FieldSample(
        scalar + 0.5,
        scalar.clone(),
        vector.clone(),
        vector.clone(),
        vector.clone(),
        invalid.clone(),
        invalid.clone(),
        invalid.clone(),
    )


class OnlineFieldMap:
    """Fixed-grid, independent per-environment maps and atomically replaced fields.

    Use one monotonic clock consistently: simulation seconds or monotonic wall time.
    update() accepts a whole batch; reset() clears selected slots. Callers may
    instead publish_snapshot() from another mapper in the same center convention.
    """

    def __init__(
        self,
        mapper: OccupancyMapper,
        *,
        update_period: float = 0.1,
        max_age: float = 0.3,
        preprocess: bool = True,
        field_backend: str = "source",
    ) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (update_period, max_age)):
            raise ValueError(
                "Update period and max age must be finite and positive."
            )
        if mapper.dtype != torch.float32:
            raise ValueError("CAT source fields require float32 mapping.")
        self.mapper = mapper
        self.update_period, self.max_age = update_period, max_age
        self.preprocess = preprocess
        if field_backend not in ("source", "cuda"):
            raise ValueError("field_backend must be source or cuda.")
        self.field_backend = field_backend
        self._builder = None
        self.time = -math.inf
        n, device = mapper.num_envs, mapper.device
        self._fields: list[VoxelPotentialField | None] = [None] * n
        self._known: list[Tensor | None] = [None] * n
        self._goals = torch.full((n, 3), torch.nan, device=device)
        self._stamp = torch.full(
            (n,), -torch.inf, dtype=torch.float64, device=device
        )
        self._attempt = self._stamp.clone()
        self.versions = torch.zeros(n, dtype=torch.long, device=device)
        self.build_counts = torch.zeros_like(self.versions)
        self._cached_occupied: list[Tensor | None] = [None] * n
        self.errors = ["No measurement."] * n

    def _check_time(self, timestamp: float) -> None:
        if not math.isfinite(timestamp) or timestamp < self.time:
            raise ValueError("timestamp must be finite and monotonic.")

    def _check_goals(self, goals: Tensor) -> None:
        if (
            goals.shape != self._goals.shape
            or goals.device != self._goals.device
            or goals.dtype != torch.float32
            or not torch.isfinite(goals).all()
        ):
            raise ValueError("goals must be finite float32 [B,3] on map device.")

    def due(self, timestamp: float, goals: Tensor) -> Tensor:
        """Return slots needing a scan/build; does not advance state."""
        self._check_time(timestamp)
        self._check_goals(goals)
        return (timestamp - self._attempt >= self.update_period - 1e-9) | (
            goals != self._goals
        ).any(-1)

    @torch.no_grad()
    def update(
        self,
        rays: RayMeasurements,
        goals: Tensor,
        *,
        timestamp: float,
        apply_cadence: bool = True,
    ) -> None:
        """Fuse eligible scans; invalid scans do not renew measurement age.

        A worker receiving already selected ray masks disables cadence: only its
        producer owns admission timing, including overwritten/phase-offset scans.
        """
        due = self.due(timestamp, goals)
        if not apply_cadence:
            due = torch.ones_like(due)
        validate_rays(rays)
        if (
            rays.origins.shape[0] != self.mapper.num_envs
            or rays.origins.device != self.mapper.device
            or rays.origins.dtype != self.mapper.dtype
        ):
            raise ValueError("Rays must match mapping batch/device/precision.")
        valid = (
            rays.valid_mask
            & due[:, None]
            & torch.isfinite(rays.origins).all(-1)
            & torch.isfinite(rays.endpoints).all(-1)
            & (rays.endpoints != rays.origins).any(-1)
        )
        changed = (goals != self._goals).any(-1)
        for i in torch.nonzero(changed, as_tuple=True)[0].tolist():
            self._fields[i] = None
            self.errors[i] = "Goal changed; waiting for a fresh field."
        self._goals.copy_(goals)
        self.time = timestamp
        self._attempt[due] = timestamp
        if not valid.any():
            return
        measured = self.mapper.update(
            RayMeasurements(
                rays.origins,
                rays.endpoints,
                valid,
                rays.hit_mask,
            )
        )
        selected = torch.nonzero(measured, as_tuple=True)[0]
        self.publish_snapshot(
            self.mapper.snapshot(), goals, timestamp=timestamp, env_ids=selected
        )

    @torch.no_grad()
    def publish_snapshot(
        self,
        snapshot: OccupancySnapshot,
        goals: Tensor,
        *,
        timestamp: float,
        env_ids: Tensor | None = None,
    ) -> None:
        """Build from an external occupancy snapshot; failure invalidates that slot.

        A snapshot publication certifies a fresh measurement at timestamp.
        Supplying a repeated old snapshot as new data would violate this contract.
        """
        self._check_time(timestamp)
        self._check_goals(goals)
        expected = (self.mapper.num_envs, *self.mapper.shape)
        if (
            snapshot.occupied.shape != expected
            or snapshot.occupied.dtype != torch.bool
            or snapshot.occupied.device != self.mapper.device
            or snapshot.known_mask.shape != expected
            or snapshot.known_mask.dtype != torch.bool
            or snapshot.known_mask.device != self.mapper.device
            or snapshot.origin.shape != goals.shape
            or not torch.isfinite(snapshot.origin).all()
            or snapshot.voxel_size != self.mapper.voxel_size
        ):
            raise ValueError("Snapshot does not match online map layout.")
        ids = self._ids(env_ids)
        self.time = timestamp
        if not len(ids):
            return
        occupied = snapshot.occupied[ids]
        if self.preprocess:
            occupied = prepare_cat_occupancy(occupied)
        for slot, i in enumerate(ids.tolist()):
            changed_goal = not torch.equal(self._goals[i], goals[i])
            self._goals[i] = goals[i]
            self._attempt[i] = timestamp
            origin = tuple(snapshot.origin[i].cpu().tolist())
            cached = self._fields[i]
            reuse = (
                cached is not None
                and not changed_goal
                and cached.origin == origin
                and self._cached_occupied[i] is not None
                and torch.equal(occupied[slot], self._cached_occupied[i])
            )
            try:
                candidate = cached
                if not reuse:
                    builder = build_cat_deployment_field
                    if self.field_backend == "cuda":
                        from embodied_learning.perception.cat_gpu import (
                            CudaCatFieldBuilder,
                        )

                        if self._builder is None:
                            self._builder = CudaCatFieldBuilder(
                                self.mapper.shape, self.mapper.device
                            )
                        builder = self._builder.build
                    candidate = builder(
                        occupied[slot],
                        origin=origin,
                        voxel_size=snapshot.voxel_size,
                        goal=tuple(goals[i].cpu().tolist()),
                    )
                    self.build_counts[i] += 1
                    self._cached_occupied[i] = occupied[slot].clone()
            except ValueError as error:
                self._fields[i] = None
                self.errors[i] = str(error)
                continue
            self._fields[i] = candidate
            self._known[i] = snapshot.known_mask[i].clone()
            self._stamp[i] = timestamp
            self.versions[i] += 1
            self.errors[i] = ""

    def _ids(self, env_ids: Tensor | None) -> Tensor:
        if env_ids is None:
            return torch.arange(self.mapper.num_envs, device=self.mapper.device)
        if (
            env_ids.ndim != 1
            or env_ids.dtype != torch.long
            or env_ids.device != self.mapper.device
            or (env_ids < 0).any()
            or (env_ids >= self.mapper.num_envs).any()
            or env_ids.unique().numel() != env_ids.numel()
        ):
            raise ValueError("env_ids must be unique valid int64 map-device slots.")
        return env_ids

    @torch.no_grad()
    def reset(self, env_ids: Tensor | None = None) -> None:
        """Forget selected episode maps without resetting the shared clock."""
        ids = self._ids(env_ids)
        self.mapper.reset(ids)
        self._goals[ids] = torch.nan
        self._stamp[ids] = self._attempt[ids] = -torch.inf
        self.versions[ids] = 0
        self.build_counts[ids] = 0
        for i in ids.tolist():
            self._fields[i] = self._known[i] = None
            self._cached_occupied[i] = None
            self.errors[i] = "No measurement."

    def map_counts(self) -> tuple[list[int], list[int]]:
        snapshot = self.mapper.snapshot()
        return (
            snapshot.observed.flatten(1).sum(1).cpu().tolist(),
            snapshot.occupied.flatten(1).sum(1).cpu().tolist(),
        )

    @torch.no_grad()
    def sample(self, positions: Tensor, *, timestamp: float) -> OnlineFieldQuery:
        """Query source interpolation with freshness/bounds and coverage masks."""
        self._check_time(timestamp)
        if (
            positions.ndim != 3
            or positions.shape[0] != self.mapper.num_envs
            or positions.shape[-1] != 3
            or positions.dtype != torch.float32
            or positions.device != self.mapper.device
        ):
            raise ValueError("positions must be float32 [B,K,3] on map device.")
        result = default_field_sample(positions)
        observed = torch.zeros(
            positions.shape[:2], dtype=torch.bool, device=positions.device
        )
        age = timestamp - self._stamp
        for i, field in enumerate(self._fields):
            if field is None or age[i] > self.max_age:
                continue
            p = positions[i : i + 1]
            finite = torch.isfinite(p).all(-1)
            safe = torch.where(finite[..., None], p, 0.0)
            index = (safe - p.new_tensor(field.origin)) / field.voxel_size
            maximum = p.new_tensor(field.sdf.shape) - 1
            valid = finite & (index >= 0).all(-1) & (index <= maximum).all(-1)
            source = sample_source_field(field, safe)
            for item in fields(FieldSample):
                dst, src = getattr(result, item.name), getattr(source, item.name)
                mask = valid[..., None] if src.ndim == 3 else valid
                dst[i : i + 1] = torch.where(mask, src, dst[i : i + 1])
            base = torch.minimum(index.clamp_min(0), maximum - 1).long()[0]
            covered = valid[0].clone()
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        covered &= self._known[i][
                            base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz
                        ]
            observed[i] = covered
        return OnlineFieldQuery(result, observed, age.clone(), self.versions.clone())
