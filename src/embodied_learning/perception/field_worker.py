"""Bounded shared-memory transport and process-owned online field computation."""

import math
import shutil
import time
from multiprocessing import shared_memory

import numpy as np

FIELD_ARRAYS = (
    "sdf",
    "geodesic",
    "sdf_gradient",
    "geodesic_gradient",
    "guidance",
    "occupied",
    "reachable",
)


def layout(batch, shape, max_rays):
    grids = (batch, *shape)
    specs = {
        "origins": ((batch, max_rays, 3), "float32"),
        "endpoints": ((batch, max_rays, 3), "float32"),
        "valid": ((batch, max_rays), "bool"),
        "hit": ((batch, max_rays), "bool"),
        "in_goals": ((batch, 3), "float32"),
        "in_epoch": ((batch,), "int64"),
        "in_goal_version": ((batch,), "int64"),
        "in_stamp": ((1,), "float64"),
        "in_sequence": ((1,), "int64"),
        "out_epoch": ((batch,), "int64"),
        "out_goal_version": ((batch,), "int64"),
        "out_stamp": ((batch,), "float64"),
        "out_sequence": ((1,), "int64"),
        "out_versions": ((batch,), "int64"),
        "out_builds": ((batch,), "int64"),
        "out_has_field": ((batch,), "bool"),
        "out_coverage": (grids, "bool"),
        "out_counts": ((batch, 2), "int64"),
        "out_errors": ((batch, 512), "uint8"),
        "out_seconds": ((1,), "float64"),
        "out_dropped": ((1,), "int64"),
        "fatal": ((2048,), "uint8"),
    }
    for name in FIELD_ARRAYS:
        vector = name in ("sdf_gradient", "geodesic_gradient", "guidance")
        specs[name] = (
            (*grids, 3) if vector else grids,
            "bool" if name in ("occupied", "reachable") else "float32",
        )
    return specs


def write_text(target, value):
    target.fill(0)
    encoded = str(value).encode("utf-8")[: target.size - 1]
    target[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)


def read_text(value):
    return value.tobytes().split(b"\0", 1)[0].decode("utf-8", errors="replace")


class FieldTransport:
    """One replaceable input and one published output, each under its own lock."""

    def __init__(self, specs, *, context=None, name=None, sync=None):
        self.specs = specs
        self.owner = name is None
        offsets, size = {}, 0
        for key, (shape, dtype) in specs.items():
            size = (size + 63) // 64 * 64
            offsets[key] = size
            size += math.prod(shape) * np.dtype(dtype).itemsize
        if self.owner:
            # Linux shared memory may be much smaller than system RAM.
            if shutil.disk_usage("/dev/shm").free < size:
                raise ValueError(f"Insufficient shared memory for {size} bytes.")
            self.memory = shared_memory.SharedMemory(create=True, size=size)
        else:
            self.memory = shared_memory.SharedMemory(name=name)
        self.closed = False
        self.arrays = {}
        try:
            self.sync = (
                dict(
                    input_lock=context.Lock(),
                    output_lock=context.Lock(),
                    ready=context.Event(),
                    wake=context.Semaphore(0),
                    stop=context.RawValue("b", False),
                    failed=context.RawValue("b", False),
                )
                if self.owner
                else sync
            )
            for key, (shape, dtype) in specs.items():
                self.arrays[key] = np.ndarray(
                    shape, dtype=dtype, buffer=self.memory.buf, offset=offsets[key]
                )
            if self.owner:
                for value in self.arrays.values():
                    value.fill(0)
            self.input_lock = self.sync["input_lock"]
            self.output_lock = self.sync["output_lock"]
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.arrays.clear()
        self.memory.close()
        if self.owner:
            self.memory.unlink()


def run_field_worker(name, specs, sync, config):
    """Spawn target: owns all mutable maps and computation on its chosen device."""
    import torch

    from embodied_learning.perception.occupancy import OccupancyMapper
    from embodied_learning.perception.online_field import OnlineFieldMap
    from embodied_learning.perception.rays import RayMeasurements

    channel = FieldTransport(specs, name=name, sync=sync)
    a = channel.arrays
    try:
        torch.set_num_threads(1)
        mapper = OccupancyMapper(**config["mapper"])
        online = OnlineFieldMap(mapper, **config["online"])
        epochs = np.full(mapper.num_envs, -1, dtype=np.int64)
        sequence, dropped = 0, 0
        sync["ready"].set()
        while not sync["stop"].value:
            if not sync["wake"].acquire(timeout=0.1):
                continue
            with channel.input_lock:
                latest = int(a["in_sequence"][0])
                if latest <= sequence:
                    continue
                job = {
                    k: a[k].copy()
                    for k in (
                        "origins",
                        "endpoints",
                        "valid",
                        "hit",
                        "in_goals",
                        "in_epoch",
                        "in_goal_version",
                        "in_stamp",
                    )
                }
            dropped += latest - sequence - 1
            sequence = latest
            started = time.perf_counter()
            reset = np.flatnonzero(epochs != job["in_epoch"])
            if len(reset):
                online.reset(torch.as_tensor(reset, device=mapper.device))
            epochs = job["in_epoch"]
            rays = RayMeasurements(
                *(
                    torch.from_numpy(job[k]).to(mapper.device)
                    for k in ("origins", "endpoints", "valid", "hit")
                )
            )
            online.update(
                rays,
                torch.from_numpy(job["in_goals"]).to(mapper.device),
                timestamp=float(job["in_stamp"][0]),
                apply_cadence=False,
            )
            measured, occupied = online.map_counts()
            # All device work/copies finish before publication becomes visible.
            packed = []
            for i, field in enumerate(online._fields):
                packed.append(
                    None
                    if field is None
                    else {
                        **{
                            key: getattr(field, key).cpu().numpy()
                            for key in FIELD_ARRAYS
                        },
                        "out_coverage": online._known[i].cpu().numpy(),
                    }
                )
            with channel.output_lock:
                a["out_has_field"].fill(False)
                for i, data in enumerate(packed):
                    if data is not None:
                        for key, value in data.items():
                            a[key][i] = value
                        a["out_has_field"][i] = True
                    write_text(a["out_errors"][i], online.errors[i])
                a["out_epoch"][:] = epochs
                a["out_goal_version"][:] = job["in_goal_version"]
                a["out_stamp"][:] = online._stamp.cpu().numpy()
                a["out_versions"][:] = online.versions.cpu().numpy()
                a["out_builds"][:] = online.build_counts.cpu().numpy()
                a["out_counts"][:, 0] = measured
                a["out_counts"][:, 1] = occupied
                a["out_seconds"][0] = time.perf_counter() - started
                a["out_dropped"][0] = dropped
                a["out_sequence"][0] = sequence
    except BaseException as error:
        write_text(a["fatal"], f"{type(error).__name__}: {error}")
        sync["failed"].value = True
        sync["ready"].set()
    finally:
        channel.close()
