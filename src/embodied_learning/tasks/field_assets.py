"""Compose known scene geometry with offline fields, without a simulator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from embodied_learning.data.field_scenes import SceneManifest, voxelize_scene
from embodied_learning.perception.voxel_field import (
    FieldSample,
    VoxelPotentialField,
    build_source_voxel_field,
    build_voxel_field,
    sample_source_field,
)

CACHE_VERSION = 1
GRID_NAMES = (
    "sdf",
    "geodesic",
    "sdf_gradient",
    "geodesic_gradient",
    "guidance",
    "known",
    "reachable",
    "occupied",
)


@dataclass(frozen=True)
class FieldCacheManifest:
    """Portable cache manifest path and scene identity."""

    path: Path
    identity: str


@dataclass(frozen=True)
class FieldBank:
    """Scene-ordered, goal-ordered immutable fields on a common device."""

    fields: tuple[tuple[VoxelPotentialField, ...], ...]
    identity: str
    scene_names: tuple[str, ...]
    source_profile: dict | None = None
    cache_identity: str | None = None

    @property
    def num_scenes(self) -> int:
        """Number of scenes available for per-slot routing."""
        return len(self.fields)

    @property
    def num_goals(self) -> tuple[int, ...]:
        """Number of candidate goals for each scene."""
        return tuple(len(goals) for goals in self.fields)

    def sample_source(
        self, positions: Tensor, scene_ids: Tensor, goal_ids: Tensor
    ) -> FieldSample:
        return self.sample(positions, scene_ids, goal_ids, source=True)

    def sample(
        self,
        positions: Tensor,
        scene_ids: Tensor,
        goal_ids: Tensor,
        *,
        source: bool = False,
    ) -> FieldSample:
        """Query scene-local [B,K,3] using independent per-slot scene/goal IDs."""
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [B,K,3].")
        for ids in (scene_ids, goal_ids):
            if (
                ids.shape != (positions.shape[0],)
                or ids.dtype != torch.int64
                or ids.device != positions.device
            ):
                raise ValueError("Scene and goal IDs must be matching int64 [B].")
        if ((scene_ids < 0) | (scene_ids >= self.num_scenes)).any():
            raise ValueError("Scene ID is outside the field bank.")
        counts = torch.tensor(self.num_goals, device=positions.device)
        if ((goal_ids < 0) | (goal_ids >= counts[scene_ids])).any():
            raise ValueError("Goal ID is outside its scene's goal bank.")
        values = {}
        for scene_id, goals in enumerate(self.fields):
            for goal_id, field in enumerate(goals):
                selected = torch.nonzero(
                    (scene_ids == scene_id) & (goal_ids == goal_id),
                    as_tuple=True,
                )[0]
                if selected.numel() == 0:
                    continue
                sampled = (
                    sample_source_field(field, positions[selected])
                    if source
                    else field.sample(positions[selected])
                )
                for item in fields(FieldSample):
                    value = getattr(sampled, item.name)
                    if item.name not in values:
                        values[item.name] = torch.zeros(
                            (positions.shape[0], *value.shape[1:]),
                            dtype=value.dtype,
                            device=value.device,
                        )
                    values[item.name][selected] = value
        if not values:
            raise ValueError("An empty query batch is not supported.")
        return FieldSample(**values)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_fields( # 根据场景中的障碍物和目标，离线计算每个场景-目标组合对应的体素场，训练时可以直接加载，不必每次重新计算
    manifest: SceneManifest, output_dir: Path, *, source_profile: dict | None = None
) -> FieldCacheManifest:
    """Build each scene/goal field once and save pickle-free arrays plus hashes.

    Existing compatible, intact caches are reused. Any incompatible cache
    raises instead of silently replacing training assets in place.
    """
    if source_profile is not None and source_profile != {"version": "cat_source_v1"}:
        raise ValueError("Unknown source profile.")
    output_dir = Path(output_dir)
    path = output_dir / "manifest.json"
    if path.exists():   # 如果输出缓存已经存在，那么就校验并复用
        load_field_cache(path, manifest, device="cpu", source_profile=source_profile)
        return FieldCacheManifest(path, manifest.identity)
    output_dir.mkdir(parents=True, exist_ok=True)   # 创建输出目录
    entries = []
    for scene in manifest.scenes:   # 在yaml文件中定义的场景，逐个场景处理，把障碍物转换成体素
        occupied = voxelize_scene(scene)    # 注意，这里的occupied是固定在世界坐标中的完整场景地图，不是相机看到的地图
        for goal_id, goal in enumerate(scene.goals):    # 这个就是每个场景的多个目标点的遍历
            builder = ( # build_source_voxel_field 构建cat_source_v1的体素场，build_voxel_field 构建普通的体素场
                build_source_voxel_field if source_profile else build_voxel_field
            )
            field = builder(    # 转成某个目标的导航场
                occupied,
                goal=goal,
                voxel_size=scene.voxel_size,
                origin=scene.origin,
            )
            starts = torch.tensor([scene.starts], dtype=torch.float32)
            if not field.sample(starts).geodesic_valid.all():
                raise ValueError(f"Scene {scene.id}: start cannot reach goal.")
            filename = f"{scene.id}-goal-{goal_id}.npz"
            target = output_dir / filename
            np.savez_compressed(
                target, **{name: getattr(field, name).numpy() for name in GRID_NAMES}
            )
            entries.append(
                {
                    "scene": scene.id,
                    "goal": goal_id,
                    "file": filename,
                    "sha256": _digest(target),
                }
            )
    metadata = {
        "version": CACHE_VERSION,
        "source_profile": source_profile,
        "identity": manifest.identity,
        "fields": entries,
    }
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(path)
    return FieldCacheManifest(path, manifest.identity)


def load_field_cache(
    path: Path,
    scene_manifest: SceneManifest,
    *,
    device: str,
    source_profile: dict | None = None,
) -> FieldBank:
    """Load all fields, rejecting identity, array, path or checksum mismatch."""
    path = Path(path)
    metadata = json.loads(path.read_text())
    if (
        metadata.get("version") != CACHE_VERSION
        or metadata.get("identity") != scene_manifest.identity
    ):
        raise ValueError("Field cache version/scene identity mismatch.")
    if metadata.get("source_profile") != source_profile:
        raise ValueError("Field cache source profile mismatch.")
    records = metadata.get("fields", [])
    lookup = {(r["scene"], r["goal"]): r for r in records}
    expected = {
        (s.id, i) for s in scene_manifest.scenes for i in range(len(s.goals))
    }
    if set(lookup) != expected or len(records) != len(lookup):
        raise ValueError("Field cache does not contain the exact scene/goal set.")
    bank = []
    for scene in scene_manifest.scenes:
        goals = []
        for goal_id in range(len(scene.goals)):
            record = lookup[scene.id, goal_id]
            target = path.parent / record["file"]
            if target.resolve().parent != path.parent.resolve():
                raise ValueError("Field files must be in the manifest directory.")
            if not target.is_file() or _digest(target) != record["sha256"]:
                raise ValueError("Field cache array checksum mismatch.")
            try:
                with np.load(target, allow_pickle=False) as data:
                    tensors = {
                        name: torch.from_numpy(data[name].copy())
                        for name in GRID_NAMES
                    }
                if tuple(tensors["sdf"].shape) != scene.shape:
                    raise ValueError("Field cache shape mismatch.")
                if not torch.equal(
                    tensors["occupied"], torch.from_numpy(voxelize_scene(scene))
                ):
                    raise ValueError("Field cache obstacle geometry mismatch.")
                field = VoxelPotentialField(
                    **tensors,
                    origin=scene.origin,
                    voxel_size=scene.voxel_size,
                ).to(device)
            except (KeyError, TypeError, RuntimeError) as error:
                raise ValueError("Field cache arrays are invalid.") from error
            goals.append(field)
        bank.append(tuple(goals))
    return FieldBank(
        tuple(bank),
        scene_manifest.identity,
        tuple(s.id for s in scene_manifest.scenes),
        source_profile,
        hashlib.sha256(
            json.dumps(
                {
                    "identity": metadata["identity"],
                    "source_profile": source_profile,
                    "fields": sorted(records, key=lambda r: (r["scene"], r["goal"])),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    )
