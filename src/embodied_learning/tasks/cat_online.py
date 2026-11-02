"""CAT task adapter over independent measured maps; contains no simulator code."""

import time
from dataclasses import fields
from typing import Any

import torch

from embodied_learning.data.field_scenes import SceneManifest
from embodied_learning.perception.occupancy import OccupancyMapper
from embodied_learning.perception.online_field import (
    OnlineFieldMap,
    default_field_sample,
)
from embodied_learning.perception.rays import RayMeasurements
from embodied_learning.perception.voxel_field import FieldSample


class CatOnlineFields:
    """FieldBank-compatible inference adapter, with no offline cache identity."""

    deployment = True
    cache_identity = None
    source_profile = {"version": "cat_source_v1"}

    def __init__(
        self,
        manifest: SceneManifest,
        *,
        num_envs: int,
        device: str | torch.device,
        update_period: float = 0.1,
        max_age: float = 0.3,
        execution: str = "sync",
        field_backend: str = "source",
        mapping_backend: str = "torch",
        max_rays: int = 4096,
    ) -> None:
        if execution not in ("sync", "async"):
            raise ValueError("execution must be sync or async.")
        if field_backend == "cuda" and torch.device(device).type != "cuda":
            raise ValueError("CUDA fields require a CUDA device.")
        self.execution = execution
        first = manifest.scenes[0]
        if any(
            s.shape != first.shape
            or s.voxel_size != first.voxel_size
            or s.origin != first.origin
            for s in manifest.scenes
        ):
            raise ValueError(
                "Online scene batches require one grid origin/shape/resolution."
            )
        if first.voxel_size != 0.04:
            raise ValueError("CAT deployment morphology requires 0.04 m voxels.")
        self.manifest = manifest
        self.identity = manifest.identity
        online_type = OnlineFieldMap
        extra = {}
        if execution == "async":
            from embodied_learning.perception.async_field import AsyncFieldMap

            online_type = AsyncFieldMap
            extra["max_rays"] = max_rays
        self.online = online_type(
            OccupancyMapper(
                num_envs=num_envs,
                shape=first.shape,
                voxel_size=first.voxel_size,
                origin=first.origin,
                device=device,
                backend=mapping_backend,
            ),
            update_period=update_period,
            max_age=max_age,
            field_backend=field_backend,
            **extra,
        )
        self.timestamp = 0.0
        self.last_query = None
        self.last_update_seconds = 0.0
        self.ray_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.query_available = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def start(self) -> None:
        if self.execution == "async":
            self.online.start()

    def close(self) -> None:
        if self.execution == "async":
            self.online.close()

    def reset(self, env_ids: torch.Tensor, scene_ids: torch.Tensor) -> None:
        self.online.reset(env_ids)
        self.ray_counts[env_ids] = 0
        self.query_available[env_ids] = False

    def update(
        self,
        rays: RayMeasurements,
        goals: torch.Tensor,
        *,
        timestamp: float,
        env_origins: torch.Tensor,
        capture_mask: torch.Tensor | None = None,
    ) -> None:
        """Convert world rays; an explicit capture mask owns scan admission.

        The simulator already schedules acquisition. Reapplying map cadence can
        discard fresh scans after a rejected mailbox submission or partial reset.
        Callers without a capture mask retain the map's original admission policy.
        """
        started = time.perf_counter()
        valid = rays.valid_mask
        if capture_mask is None:
            due = self.online.due(timestamp, goals)
        else:
            if (
                capture_mask.shape != valid.shape[:1]
                or capture_mask.dtype != torch.bool
                or capture_mask.device != valid.device
            ):
                raise ValueError("capture_mask must be boolean [B] on ray device.")
            due = capture_mask
            valid = valid & capture_mask[:, None]
        local = RayMeasurements(
            rays.origins - env_origins[:, None],
            rays.endpoints - env_origins[:, None],
            valid,
            rays.hit_mask,
        )
        accepted = self.online.update(
            local, goals, timestamp=timestamp, apply_cadence=capture_mask is None
        )
        if accepted is not False:
            self.ray_counts[due] = rays.valid_mask.sum(-1)[due]
        self.timestamp = timestamp
        self.last_update_seconds = time.perf_counter() - started

    def sample_source(
        self,
        positions: torch.Tensor,
        scene_ids: torch.Tensor,
        goal_ids: torch.Tensor,
    ) -> FieldSample:
        self.last_query = self.online.sample(positions, timestamp=self.timestamp)
        sample = self.last_query.sample
        # A body partly outside the field must not keep a previous moving command.
        usable = sample.guidance_valid.all(-1)
        default = default_field_sample(positions)
        return FieldSample(
            **{
                f.name: torch.where(
                    usable.reshape(-1, *([1] * (getattr(sample, f.name).ndim - 1))),
                    getattr(sample, f.name),
                    getattr(default, f.name),
                )
                for f in fields(FieldSample)
            }
        )

    def diagnostics(self) -> dict[str, Any]:
        measured, occupied = self.online.map_counts()
        return dict(
            execution=self.execution,
            field_backend=self.online.field_backend,
            mapping_backend=self.online.mapper.backend,
            versions=self.online.versions.cpu().tolist(),
            build_counts=self.online.build_counts.cpu().tolist(),
            valid_rays=self.ray_counts.cpu().tolist(),
            measured_voxels=measured,
            occupied_voxels=occupied,
            last_update_seconds=self.last_update_seconds,
            field_available=self.query_available.cpu().tolist(),
            field_age_seconds=(
                self.last_query.age.cpu().tolist()
                if self.last_query is not None
                else None
            ),
            errors=list(self.online.errors),
            worker=self.online.diagnostics() if self.execution == "async" else None,
        )
