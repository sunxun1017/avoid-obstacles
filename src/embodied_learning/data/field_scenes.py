"""Portable known box scenes, shared by physics and offline field generation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from embodied_learning.data.config import load_experiment_config

Point = tuple[float, float, float]


def _point(values: object, name: str) -> Point:
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise ValueError(f"{name} must contain three coordinates.")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
        raise ValueError(f"{name} must contain numbers.")
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"{name} must contain finite numbers.")
    return tuple(float(x) for x in values)


@dataclass(frozen=True)
class SceneBox:
    """Axis-aligned box center and positive half sizes, in scene-local meters."""

    center: Point
    half_size: Point


@dataclass(frozen=True)
class FieldScene:
    """Known obstacle geometry and candidate G1 root starts and target points."""

    id: str
    origin: Point
    shape: tuple[int, int, int]
    voxel_size: float
    boxes: tuple[SceneBox, ...]
    starts: tuple[Point, ...]
    goals: tuple[Point, ...]


@dataclass(frozen=True)
class SceneManifest:
    """Ordered scenes with canonical geometry identity for checkpoint checks."""

    scenes: tuple[FieldScene, ...]

    @property
    def identity(self) -> str:
        """Hash scene order, geometry, grid and candidate start/goal points."""
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


def load_scene_manifest(path: Path) -> SceneManifest:
    """Load and validate known scenes from a YAML or JSON experiment.

    Training configuration may share the file; scene identity includes only
    the scenes. Targets and starts must have room for field interpolation.
    Physics-specific full-body initial clearance is checked by the G1 task.
    """
    payload = load_experiment_config(path)
    records = payload.get("scenes") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("Configuration must contain a nonempty scenes list.")
    scenes = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each scene must be an object.")
        name = record.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Scene ID must contain only letters, numbers, _ or -.")
        origin = _point(record.get("origin"), "origin")
        shape = record.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(n) is not int or n < 3 for n in shape)
        ):
            raise ValueError("Grid shape must contain three integers >=3.")
        spacing = record.get("voxel_size")
        if (
            isinstance(spacing, bool)
            or not isinstance(spacing, (int, float))
            or not math.isfinite(spacing)
            or spacing <= 0
        ):
            raise ValueError("voxel_size must be finite and positive.")
        boxes = []
        box_records = record.get("boxes", [])
        if not isinstance(box_records, list):
            raise ValueError("boxes must be a list.")
        for box in box_records:
            if not isinstance(box, dict) or set(box) != {"center", "half_size"}:
                raise ValueError("Boxes require only center and half_size.")
            center = _point(box["center"], "box center")
            half_size = _point(box["half_size"], "box half_size")
            if any(x <= 0 for x in half_size):
                raise ValueError("Box half sizes must be positive.")
            boxes.append(SceneBox(center, half_size))
        points = {}
        for key in ("starts", "goals"):
            values = record.get(key)
            if not isinstance(values, list) or not values:
                raise ValueError(f"{key} must be a nonempty list.")
            points[key] = tuple(_point(p, key) for p in values)
            for p in points[key]:
                index = (np.asarray(p) - origin) / spacing
                if np.any(index <= 0) or np.any(index >= np.asarray(shape) - 1):
                    raise ValueError(f"{key} must lie strictly inside grid bounds.")
                if any(
                    np.all(np.abs(np.asarray(p) - b.center) <= b.half_size)
                    for b in boxes
                ):
                    raise ValueError(f"{key} must lie outside obstacle geometry.")
        scenes.append(
            FieldScene(
                name,
                origin,
                tuple(shape),
                float(spacing),
                tuple(boxes),
                points["starts"],
                points["goals"],
            )
        )
    if len({s.id for s in scenes}) != len(scenes):
        raise ValueError("Scene IDs must be unique.")
    return SceneManifest(tuple(scenes))


def voxelize_scene(scene: FieldScene) -> NDArray[np.bool_]: # 将障碍物转换成三维布尔网格occupied
    """Sample declared forbidden box volumes at field grid vertices.

    The support ground is not an obstacle. Discretization error remains
    bounded by grid resolution; this is not exact mesh-distance computation.
    """
    axes = np.meshgrid( # 为X,Y,Z三个方向生成坐标轴
        *(# scene.voxel_size 是相邻点的距离，origin
            np.arange(n) * scene.voxel_size + scene.origin[i]
            for i, n in enumerate(scene.shape)
        ),
        indexing="ij",
        sparse=True,
    )
    occupied = np.zeros(scene.shape, dtype=bool)
    for box in scene.boxes: # 只有同时满足X,Y,Z范围的点才会保留为True
        inside = np.ones(scene.shape, dtype=bool)
        for axis, center, half in zip(axes, box.center, box.half_size, strict=True):
            inside &= np.abs(axis - center) <= half + 1e-9
        occupied |= inside
    return occupied
