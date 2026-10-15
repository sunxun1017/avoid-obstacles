"""Reproduce fresh-build, cache-hit and async transport CAT timings on real rays.

Run from the repository root with the field-training environment. This is a
single stationary sensor replay with changing goals, not a navigation benchmark.
The full configured grid is retained. Cached publications are reported separately.
"""

import argparse
import json
import platform
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from embodied_learning.data.field_scenes import load_scene_manifest
from embodied_learning.perception import online_field
from embodied_learning.perception.async_field import AsyncFieldMap
from embodied_learning.perception.cat_gpu import CudaCatFieldBuilder
from embodied_learning.perception.occupancy import OccupancyMapper
from embodied_learning.perception.online_field import OnlineFieldMap
from embodied_learning.perception.rays import RayMeasurements
from embodied_learning.simulators.mjlab.cat_online import make_cat_online_env
from embodied_learning.simulators.mjlab.cat_sensors import OnlineSensorConfig


def summarize(values):
    return (
        dict(
            count=len(values),
            p50_ms=float(np.quantile(values, 0.5)) * 1000,
            p95_ms=float(np.quantile(values, 0.95)) * 1000,
            max_ms=max(values) * 1000,
        )
        if values
        else None
    )


def compare(source, gpu):
    mask = source.reachable & gpu.reachable
    moving = (
        mask
        & (source.guidance.norm(dim=-1) > 0.1)
        & (gpu.guidance.norm(dim=-1) > 0.1)
    )
    cosine = torch.nn.functional.cosine_similarity(
        source.guidance[moving], gpu.guidance[moving]
    )
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))

    def error(a, b):
        e = (a - b).abs()[mask]
        return dict(
            p50_m=e.quantile(0.5).item(),
            p95_m=e.quantile(0.95).item(),
            p99_m=e.quantile(0.99).item(),
            max_m=e.max().item(),
        )

    return dict(
        sdf=error(source.sdf, gpu.sdf),
        geodesic=error(source.geodesic, gpu.geodesic),
        direction_deg=dict(
            p50=angle.quantile(0.5).item(),
            p95=angle.quantile(0.95).item(),
            max=angle.max().item(),
        ),
        reachable_disagreement=int((source.reachable != gpu.reachable).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=("lidar", "depth"), default="lidar")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup < 2:
        parser.error("At least 3 repeats and 2 warmup cycles are required.")
    torch.set_num_threads(1)
    manifest = load_scene_manifest(Path("configs/experiments/g1_cat_source.yaml"))
    start = time.perf_counter()
    env = make_cat_online_env(
        manifest,
        scene_ids=("wall_detour",),
        num_envs=1,
        device="cuda:0",
        seed=42,
        options=dict(noise_level=0.0, random_pd=False, random_rfi=False, push=False),
        sensor=OnlineSensorConfig(source=args.sensor),
        field_backend="cuda",
        mapping_backend="warp",
    )
    startup = time.perf_counter() - start
    try:
        native = env.unwrapped
        sensor_times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            native.sim.sense()
            rays = native.cat_sensor_reader.read()
            torch.cuda.synchronize()
            sensor_times.append(time.perf_counter() - start)
        local = RayMeasurements(
            rays.origins - native.scene.env_origins[:, None],
            rays.endpoints - native.scene.env_origins[:, None],
            rays.valid_mask,
            rays.hit_mask,
        )
        goals = env.cat_state.goals.clone()
        first = manifest.scenes[0]

        def mapper(backend):
            return OccupancyMapper(
                1,
                first.shape,
                voxel_size=first.voxel_size,
                origin=first.origin,
                device="cuda:0",
                backend=backend,
            )

        def target(i):
            g = goals.clone()
            g[:, 1] += 0.02 if i % 2 else -0.02
            return g

        report = dict(
            device=torch.cuda.get_device_name(),
            python=platform.python_version(),
            torch=torch.__version__,
            sensor=args.sensor,
            grid=list(first.shape),
            voxel_size=first.voxel_size,
            valid_rays=int(local.valid_mask.sum()),
            repeats=args.repeats,
            warmup=args.warmup,
            startup_seconds=startup,
            sensor_latency=summarize(sensor_times),
            workload="stationary real sensor rays; alternating goal y +/- 0.02 m",
        )
        fields = {}
        for backend in ("source", "cuda"):
            p = OnlineFieldMap(
                mapper("torch" if backend == "source" else "warp"),
                field_backend=backend,
            )
            if backend == "cuda":
                p._builder = CudaCatFieldBuilder(first.shape, "cuda:0")
            state = dict(timings={}, enabled=False)

            def instrument(label, function, state=state):
                def wrapped(*a, **kw):
                    if not state["enabled"]:
                        return function(*a, **kw)
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    result = function(*a, **kw)
                    torch.cuda.synchronize()
                    state["timings"].setdefault(label, []).append(
                        time.perf_counter() - t
                    )
                    return result

                return wrapped

            p.mapper.update = instrument("fusion", p.mapper.update)
            builder = (
                online_field.build_cat_deployment_field
                if backend == "source"
                else p._builder.build
            )
            owner = online_field if backend == "source" else p._builder
            attribute = (
                "build_cat_deployment_field" if backend == "source" else "build"
            )
            with (
                patch.object(
                    online_field,
                    "prepare_cat_occupancy",
                    instrument("morphology", online_field.prepare_cat_occupancy),
                ),
                patch.object(owner, attribute, instrument("field", builder)),
            ):
                totals = []
                for i in range(args.warmup + args.repeats):
                    state["enabled"] = i >= args.warmup
                    before = int(p.build_counts[0])
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    p.update(local, target(i), timestamp=i * 0.2)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t
                    assert int(p.build_counts[0]) == before + 1, p.errors
                    if state["enabled"]:
                        totals.append(elapsed)
                fields[backend] = p._fields[0]
            fresh = {k: summarize(v) for k, v in state["timings"].items()}
            fresh["pipeline"] = summarize(totals)
            before = int(p.build_counts[0])
            cached = []
            for j in range(args.repeats):
                torch.cuda.synchronize()
                t = time.perf_counter()
                p.update(local, target(i), timestamp=(i + 1 + j) * 0.2)
                torch.cuda.synchronize()
                cached.append(time.perf_counter() - t)
            assert int(p.build_counts[0]) == before, (
                "Cache workload unexpectedly rebuilt."
            )
            report[backend] = dict(
                fresh=fresh,
                unchanged_scan=summarize(cached),
                fresh_builds=args.repeats,
                cached_new_builds=0,
            )
            # Obstacle insertion/removal stress: translate the measured processed
            # map by one voxel on alternating builds. Synthetic perturbation is
            # deliberately labeled, and these timings exclude fusion/morphology.
            base_map = online_field.prepare_cat_occupancy(
                p.mapper.snapshot().occupied
            )[0]
            shifted = torch.roll(base_map, 1, dims=0)
            if torch.equal(base_map, shifted):
                raise RuntimeError("Changed-map workload did not change occupancy.")
            map_times = []
            for k in range(args.warmup + args.repeats):
                torch.cuda.synchronize()
                t = time.perf_counter()
                builder(
                    base_map if k % 2 else shifted,
                    origin=first.origin,
                    voxel_size=first.voxel_size,
                    goal=tuple(target(i)[0].cpu().tolist()),
                )
                torch.cuda.synchronize()
                if k >= args.warmup:
                    map_times.append(time.perf_counter() - t)
            report[backend]["changed_map_field_only"] = summarize(map_times)
        report["source_comparison"] = compare(fields["source"], fields["cuda"])
        # This includes H2D/D2H/shared-memory transfer, a spawned CUDA context and
        # original wall-clock measurement age, with 10 Hz changing-goal submissions.
        p = AsyncFieldMap(
            mapper("warp"), max_rays=local.valid_mask.shape[1], field_backend="cuda"
        )
        foreground = []
        ages = []
        worker = []
        versions = []
        published = []
        async_start = time.perf_counter()
        with p:
            report["worker_start_seconds"] = time.perf_counter() - async_start
            first_started = time.monotonic()
            p.update(local, target(1), timestamp=first_started)
            while not p.poll():
                if time.monotonic() - first_started > 30.0:
                    raise RuntimeError(f"Worker warmup timed out: {p.diagnostics()}")
                time.sleep(0.005)
            report["worker_first_publication_seconds"] = (
                time.monotonic() - first_started
            )
            # This initial result may already be stale; never stamp it as fresh.
            deadline = time.monotonic()
            measurement_start = None
            for i in range(args.warmup + args.repeats):
                enabled = i >= args.warmup
                if i == args.warmup:
                    measurement_start = time.monotonic()
                t = time.perf_counter()
                p.update(local, target(i), timestamp=time.monotonic())
                if enabled:
                    foreground.append(time.perf_counter() - t)
                for _ in range(5):
                    t = time.perf_counter()
                    changed = p.poll()
                    query = p.sample(goals[:, None], timestamp=time.monotonic())
                    torch.cuda.synchronize()
                    if enabled:
                        foreground.append(time.perf_counter() - t)
                        if changed:
                            ages.append(float(query.age[0]))
                            worker.append(p.last_update_seconds)
                            versions.append(int(p.versions[0]))
                            published.append(time.monotonic())
                    deadline += 0.02
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
            report["async"] = dict(
                foreground=summarize(foreground),
                measurement_age=summarize(ages),
                worker_pipeline=summarize(worker),
                accepted_publications=len(published),
                measured_window_seconds=time.monotonic() - measurement_start,
                full_window_hz=len(published)
                / (time.monotonic() - measurement_start),
                publication_hz=(
                    (len(published) - 1) / (published[-1] - published[0])
                    if len(published) > 1
                    else 0
                ),
                versions=versions,
                diagnostics=p.diagnostics(),
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
