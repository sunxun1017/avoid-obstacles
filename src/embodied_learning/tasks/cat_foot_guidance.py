"""Isolated experiment: change only the foot GF projection radius.

Original point SDF/BF, geodesics, interpolation, body sites and task equations
are preserved. Reblend on the voxel grid BEFORE the original interpolation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild_guidance(field, goal, *, influence_radius):
    if not math.isfinite(influence_radius) or influence_radius <= 0:
        raise ValueError("Guidance radius must be positive and finite.")
    if field.sdf.device.type != "cpu":
        raise ValueError("Build guidance caches on CPU before simulation.")
    sdf = field.sdf.numpy()
    bf = field.sdf_gradient.numpy()
    bnorm = np.linalg.norm(bf, axis=-1, keepdims=True)
    unit = np.zeros_like(bf)
    valid = bnorm[..., 0] > 1e-9
    unit[valid] = bf[valid] / bnorm[valid]
    g = -field.geodesic_gradient.numpy()
    tangent = g - (g * unit).sum(-1, keepdims=True) * unit
    fraction = np.clip(np.maximum(sdf, 0) / (influence_radius + 1e-9), 0, 1)
    weight = (1 - fraction * fraction * (3 - 2 * fraction))[..., None]
    mixed = (1 - weight) * g + weight * tangent
    occupied = field.occupied.numpy()
    mixed[occupied] = unit[occupied]
    magnitude = np.linalg.norm(mixed, axis=-1, keepdims=True)
    direction = np.zeros_like(mixed)
    nonzero = magnitude[..., 0] > 1e-9
    direction[nonzero] = mixed[nonzero] / magnitude[nonzero]
    x = field.origin[0] + np.arange(sdf.shape[0]) * field.voxel_size - goal[0]
    y = field.origin[1] + np.arange(sdf.shape[1]) * field.voxel_size - goal[1]
    xy = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2)
    speed = np.where(xy > 0.3, 0.6, 0.6 * (xy / 0.3) ** 3).astype(np.float32)
    return torch.from_numpy(direction * speed[..., None])


class FootGuidanceBank:
    """Route only published body-site columns 3/4 to the alternative GF."""

    def __init__(self, base, grids, *, radius, manifest_identity):
        if len(grids) != len(base.fields):
            raise ValueError("Foot field scene count mismatch.")
        self.base = base
        fields = []
        for original, overrides in zip(base.fields, grids, strict=True):
            if len(original) != len(overrides):
                raise ValueError("Foot field goal count mismatch.")
            fields.append(
                tuple(
                    replace(f, guidance=g)
                    for f, g in zip(original, overrides, strict=True)
                )
            )
        self.foot = replace(base, fields=tuple(fields))
        for name in ("fields", "identity", "scene_names", "source_profile"):
            setattr(self, name, getattr(base, name))
        self.radius = radius
        self.cache_identity = hashlib.sha256(
            json.dumps(
                {
                    "base_cache_identity": base.cache_identity,
                    "foot_manifest_identity": manifest_identity,
                    "radius_m": radius,
                    "body_site_indices": [3, 4],
                    "implementation": "cat_foot_guidance_v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.experiment_schema = dict(
            version="cat_foot_guidance_v1",
            foot_projection_radius_m=radius,
            other_sites_projection_radius="5 * voxel_size",
            foot_site_indices=[3, 4],
            sdf="original_point_sdf",
            source_interpolation="unchanged",
            base_cache_identity=base.cache_identity,
            foot_manifest_identity=manifest_identity,
        )

    def sample_source(self, positions, scene_ids, goal_ids):
        return self.sample(positions, scene_ids, goal_ids, source=True)

    def sample(self, positions, scene_ids, goal_ids, *, source=False):
        if positions.ndim != 3 or positions.shape[1] != 11:
            raise ValueError("Foot routing requires the ordered 11 body sites.")
        sample = self.base.sample(positions, scene_ids, goal_ids, source=source)
        foot = self.foot.sample(
            positions[:, 3:5], scene_ids, goal_ids, source=source
        )
        guidance = sample.guidance.clone()
        guidance[:, 3:5] = foot.guidance
        return replace(sample, guidance=guidance)


def prepare_foot_cache(base, manifest, directory, *, radius):
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        return load_foot_bank(base, directory, radius=radius)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for scene_id, scene in enumerate(manifest.scenes):
        for goal_id, goal in enumerate(scene.goals):
            field = base.fields[scene_id][goal_id]
            reference = rebuild_guidance(
                field, goal, influence_radius=5 * scene.voxel_size
            )
            reference_error = float((reference - field.guidance).abs().max())
            if reference_error > 2e-6:
                raise ValueError(f"Cannot reproduce original GF: {reference_error}")
            modified = rebuild_guidance(field, goal, influence_radius=radius)
            delta = (modified - field.guidance).abs().amax(-1)
            far = field.sdf >= max(radius, 5 * scene.voxel_size)
            far_error = float(delta[far].max()) if far.any() else 0.0
            if far_error > 2e-6:
                raise ValueError("Guidance changed outside the influence region.")
            name = f"{scene.id}-goal-{goal_id}.npy"
            target = directory / name
            np.save(target, modified.numpy(), allow_pickle=False)
            entries.append(
                dict(
                    scene=scene.id,
                    goal=goal_id,
                    file=name,
                    sha256=sha256(target),
                    reference_max_abs_error=reference_error,
                    far_max_abs_error=far_error,
                    changed_voxels=int((delta > 2e-6).sum()),
                    max_abs_change=float(delta.max()),
                )
            )
    metadata = dict(
        version="cat_foot_guidance_v1",
        radius_m=radius,
        base_cache_identity=base.cache_identity,
        scene_identity=manifest.identity,
        fields=entries,
    )
    temp = directory / "manifest.json.tmp"
    temp.write_text(json.dumps(metadata, indent=2) + "\n")
    temp.replace(directory / "manifest.json")
    return load_foot_bank(base, directory, radius=radius)


def load_foot_bank(base, directory, *, radius):
    directory = Path(directory)
    path = directory / "manifest.json"
    metadata = json.loads(path.read_text())
    if (
        metadata["version"] != "cat_foot_guidance_v1"
        or metadata["radius_m"] != radius
        or metadata["base_cache_identity"] != base.cache_identity
        or metadata["scene_identity"] != base.identity
    ):
        raise ValueError("Foot guidance cache/experiment mismatch.")
    records = {(r["scene"], r["goal"]): r for r in metadata["fields"]}
    expected = {
        (name, g)
        for name, goals in zip(base.scene_names, base.fields, strict=True)
        for g in range(len(goals))
    }
    if set(records) != expected or len(records) != len(metadata["fields"]):
        raise ValueError("Foot guidance fields are incomplete or duplicated.")
    grids = []
    for name, goals in zip(base.scene_names, base.fields, strict=True):
        group = []
        for g, field in enumerate(goals):
            record = records[name, g]
            target = directory / record["file"]
            if (
                target.resolve().parent != directory.resolve()
                or sha256(target) != record["sha256"]
            ):
                raise ValueError("Foot guidance file identity mismatch.")
            array = np.load(target, allow_pickle=False)
            if (
                array.shape != field.guidance.shape
                or array.dtype != np.float32
                or not np.isfinite(array).all()
            ):
                raise ValueError("Invalid foot guidance array.")
            group.append(torch.from_numpy(array).to(field.guidance.device))
        grids.append(tuple(group))
    return FootGuidanceBank(
        base, tuple(grids), radius=radius, manifest_identity=sha256(path)
    )
