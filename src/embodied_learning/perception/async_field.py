"""Nonblocking field consumer with one spawned mapping/building process."""

import multiprocessing as mp
import time

import numpy as np
import torch

from embodied_learning.perception.field_worker import (
    FIELD_ARRAYS,
    FieldTransport,
    layout,
    read_text,
    run_field_worker,
)
from embodied_learning.perception.online_field import OnlineFieldMap
from embodied_learning.perception.rays import validate_rays
from embodied_learning.perception.voxel_field import VoxelPotentialField


class AsyncFieldMap(OnlineFieldMap):
    """Main process only submits/polls/queries; measurement time is wall clock.

    A busy mailbox returns immediately. The producer overwrites pending scans;
    obsolete episode/goal results are rejected per slot before field adoption.
    start() may wait for process initialization, never update()/poll()/sample().
    """

    def __init__(self, mapper, *, max_rays, worker_device=None, **kwargs):
        super().__init__(mapper, **kwargs)
        if type(max_rays) is not int or max_rays < 1:
            raise ValueError("max_rays must be a positive integer.")
        self.max_rays = max_rays
        self._context = mp.get_context("spawn")
        self._transport = None
        self._process = None
        self._closed = False
        self._epochs = np.zeros(mapper.num_envs, dtype=np.int64)
        self._goal_versions = self._epochs.copy()
        self._submitted = self._seen = self._published = self._dropped = 0
        self._counts = np.zeros((mapper.num_envs, 2), dtype=np.int64)
        self.last_update_seconds = 0.0
        self._config = dict(
            mapper=dict(
                num_envs=mapper.num_envs,
                shape=mapper.shape,
                voxel_size=mapper.voxel_size,
                origin=mapper._origin.cpu().tolist(),
                device=str(worker_device or mapper.device),
                dtype=mapper.dtype,
                backend=mapper.backend,
                **{
                    k: getattr(mapper, k)
                    for k in (
                        "hit_log_odds",
                        "miss_log_odds",
                        "min_log_odds",
                        "max_log_odds",
                        "free_threshold",
                        "occupied_threshold",
                    )
                },
            ),
            online=dict(
                update_period=self.update_period,
                max_age=self.max_age,
                preprocess=self.preprocess,
                field_backend=self.field_backend,
            ),
        )

    @property
    def worker_alive(self):
        return self._process is not None and self._process.is_alive()

    def start(self, timeout=30.0):
        if self._closed:
            raise RuntimeError("Field worker is closed.")
        if self._process is not None:
            self._check_worker()
            return
        try:
            self._transport = FieldTransport(
                layout(self.mapper.num_envs, self.mapper.shape, self.max_rays),
                context=self._context,
            )
            t = self._transport
            self._process = self._context.Process(
                target=run_field_worker,
                args=(t.memory.name, t.specs, t.sync, self._config),
                daemon=True,
            )
            self._process.start()
            deadline = time.monotonic() + timeout
            while not t.sync["ready"].wait(0.05):
                self._check_worker()
                if time.monotonic() >= deadline:
                    raise RuntimeError("Field worker initialization timed out.")
            self._check_worker()
        except BaseException:
            self.close()
            raise

    def _check_worker(self):
        if self._closed or self._process is None:
            raise RuntimeError("Field worker is not running; call start first.")
        if not self.worker_alive or self._transport.sync["failed"].value:
            self._fields[:] = [None] * self.mapper.num_envs
            detail = read_text(self._transport.arrays["fatal"])
            self.errors[:] = [
                detail or "Field worker exited."
            ] * self.mapper.num_envs
            raise RuntimeError(
                f"Field worker failed: {detail or self._process.exitcode}"
            )

    @torch.no_grad()
    def update(self, rays, goals, *, timestamp):
        self._check_worker()
        if timestamp > time.monotonic():
            raise ValueError("Measurement timestamp cannot be in the future.")
        due = self.due(timestamp, goals)
        validate_rays(rays)
        b, n = rays.valid_mask.shape
        if (
            b != self.mapper.num_envs
            or n > self.max_rays
            or rays.origins.device != self.mapper.device
            or rays.origins.dtype != torch.float32
        ):
            raise ValueError("Rays must match batch/device/float32 and capacity.")
        changed = (goals != self._goals).any(-1)
        for i in torch.nonzero(changed, as_tuple=True)[0].cpu().tolist():
            self._goal_versions[i] += 1
            self._fields[i] = None
            self.errors[i] = "Goal changed; waiting for worker."
        self._goals.copy_(goals)
        self.time = timestamp
        if not due.any():
            return False
        t = self._transport
        if not t.input_lock.acquire(False):
            return False
        try:
            a = t.arrays
            a["origins"][:, :n] = rays.origins.cpu().numpy()
            a["endpoints"][:, :n] = rays.endpoints.cpu().numpy()
            a["valid"].fill(False)
            a["valid"][:, :n] = (rays.valid_mask & due[:, None]).cpu().numpy()
            a["hit"][:, :n] = rays.hit_mask.cpu().numpy()
            a["in_goals"][:] = goals.cpu().numpy()
            a["in_epoch"][:] = self._epochs
            a["in_goal_version"][:] = self._goal_versions
            a["in_stamp"][0] = timestamp
            self._submitted += 1
            a["in_sequence"][0] = self._submitted
            self._attempt[due] = timestamp
            t.sync["wake"].release()
        finally:
            t.input_lock.release()
        return True

    @torch.no_grad()
    def poll(self):
        self._check_worker()
        t = self._transport
        if not t.output_lock.acquire(False):
            return False
        copies = {}
        try:
            a = t.arrays
            seq = int(a["out_sequence"][0])
            if seq <= self._seen:
                return False
            self._seen = seq
            match = (
                (a["out_epoch"] == self._epochs)
                & (a["out_goal_version"] == self._goal_versions)
                & (a["out_stamp"] <= time.monotonic())
            )
            for i in np.flatnonzero(match):
                meta = {
                    k: a[k][i].copy()
                    for k in (
                        "out_stamp",
                        "out_versions",
                        "out_builds",
                        "out_counts",
                        "out_errors",
                        "out_has_field",
                    )
                }
                if meta["out_has_field"]:
                    meta.update(
                        {k: a[k][i].copy() for k in (*FIELD_ARRAYS, "out_coverage")}
                    )
                copies[int(i)] = meta
            self.last_update_seconds = float(a["out_seconds"][0])
            self._dropped = int(a["out_dropped"][0])
        finally:
            t.output_lock.release()
        for i, data in copies.items():
            field = None
            if data["out_has_field"]:
                arrays = {
                    k: torch.from_numpy(data[k]).to(self.mapper.device)
                    for k in FIELD_ARRAYS
                }
                field = VoxelPotentialField(
                    **arrays,
                    known=torch.ones_like(arrays["occupied"]),
                    origin=tuple(self.mapper._origin[i].cpu().tolist()),
                    voxel_size=self.mapper.voxel_size,
                )
                self._known[i] = torch.from_numpy(data["out_coverage"]).to(
                    self.mapper.device
                )
            self._fields[i] = field
            self._stamp[i] = float(data["out_stamp"])
            self.versions[i] = int(data["out_versions"])
            self.build_counts[i] = int(data["out_builds"])
            self._counts[i] = data["out_counts"]
            self.errors[i] = read_text(data["out_errors"])
        if copies:
            self._published = seq
        return bool(copies)

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        super().reset(ids)
        selected = ids.cpu().numpy()
        self._epochs[selected] += 1
        self._counts[selected] = 0

    def map_counts(self):
        return self._counts[:, 0].tolist(), self._counts[:, 1].tolist()

    def diagnostics(self):
        return dict(
            submitted_sequence=self._submitted,
            published_sequence=self._published,
            dropped_scans=self._dropped,
            worker_alive=self.worker_alive,
            build_counts=self.build_counts.cpu().tolist(),
            errors=list(self.errors),
            last_update_seconds=self.last_update_seconds,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        t = self._transport
        # Lock-free stop flag and semaphore release cannot strand the parent on
        # an Event condition lock if the worker crashes during notification.
        if t is not None:
            t.sync["stop"].value = True
            t.sync["wake"].release()
        if self._process is not None and self._process.pid is not None:
            self._process.join(2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(2.0)
        if t is not None:
            t.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()
