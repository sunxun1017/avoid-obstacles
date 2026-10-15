"""CAT task adapter over independent measured maps; contains no simulator code."""

import time
from dataclasses import fields

import torch

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
        manifest,
        *,
        num_envs,
        device,
        update_period=0.1,
        max_age=0.3,
        execution="sync",
        field_backend="source",
        mapping_backend="torch",
        max_rays=4096,
    ):
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

    def start(self):
        if self.execution == "async":
            self.online.start()

    def close(self):
        if self.execution == "async":
            self.online.close()

    def reset(self, env_ids, scene_ids):
        self.online.reset(env_ids)
        self.ray_counts[env_ids] = 0
        self.query_available[env_ids] = False

    def update(self, rays, goals, *, timestamp, env_origins):
        """Convert simulator world rays to the task's stable per-scene frame."""
        started = time.perf_counter()
        due = self.online.due(timestamp, goals)
        local = RayMeasurements(
            rays.origins - env_origins[:, None],
            rays.endpoints - env_origins[:, None],
            rays.valid_mask,
            rays.hit_mask,
        )
        accepted = self.online.update(local, goals, timestamp=timestamp)
        if accepted is not False:
            self.ray_counts[due] = rays.valid_mask.sum(-1)[due]
        self.timestamp = timestamp
        self.last_update_seconds = time.perf_counter() - started

    def sample_source(self, positions, scene_ids, goal_ids):
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

    def diagnostics(self):
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
