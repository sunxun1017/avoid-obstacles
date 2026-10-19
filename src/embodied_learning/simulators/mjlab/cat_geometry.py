"""Offline known-box clearance for the full native G1 collision envelope.

This is privileged scene geometry, not a sensor-derived field. Cover spheres
provide conservative warning/reward distances; negative clearance does not
certify a physical collision. Native contacts remain the collision authority.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mujoco
import torch
from torch import Tensor

from embodied_learning.data.field_scenes import SceneManifest
from embodied_learning.robots.g1 import G1_BODY_SCHEMA


def _body_group(name: str) -> int:
    local = name.rsplit("/", 1)[-1]
    fixed = {"head_collision": 0, "pelvis_collision": 1, "torso_collision": 2}
    if local in fixed:
        return fixed[local]
    for side, foot, hand, knee, shoulder in (
        ("left", 3, 5, 7, 9),
        ("right", 4, 6, 8, 10),
    ):
        if re.fullmatch(rf"{side}_foot\d+_collision", local):
            return foot
        if local in {
            f"{side}_{part}_collision"
            for part in ("hip", "thigh", "shin", "linkage_brace")
        }:
            return knee
        if local == f"{side}_shoulder_yaw_collision":
            return shoulder
        if local in {
            f"{side}_{part}_collision" for part in ("elbow_yaw", "wrist", "hand")
        }:
            return hand
    raise ValueError(f"Unmapped native robot collision geom: {name}")


@dataclass(frozen=True)
class CollisionEnvelope:
    """Cover spheres in geom-local coordinates, with per-sphere native IDs."""

    geom_ids: Tensor
    local_centers: Tensor
    radii: Tensor
    groups: Tensor
    max_segment: float
    geom_records: tuple[dict, ...]

    @classmethod
    def from_model(cls, model, device, max_segment: float = 0.04):
        if not math.isfinite(max_segment) or max_segment <= 0:
            raise ValueError("max_segment must be finite and positive.")
        prefixed = any(
            model.geom(i).name.startswith("robot/") for i in range(model.ngeom)
        )
        geom_ids, centers, radii, groups, records = [], [], [], [], []
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name
            selected = (
                name.startswith("robot/")
                if prefixed
                else name.endswith("_collision")
            )
            if not selected or not (
                model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
            ):
                continue
            kind = int(model.geom_type[geom_id])
            size = [float(x) for x in model.geom_size[geom_id]]
            if kind not in (
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
            ):
                raise ValueError(f"Unsupported collision geom type {kind}: {name}")
            if (
                not all(math.isfinite(x) for x in size)
                or size[0] <= 0
                or size[1] < 0
            ):
                raise ValueError(f"Invalid native collision geom dimensions: {name}")
            group = _body_group(name)
            half = size[1] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
            segments = max(1, math.ceil(2 * half / max_segment)) if half else 0
            spacing = 2 * half / segments if segments else 0.0
            # Every capsule point lies within r + spacing/2 of an axis sample.
            radius = size[0] + spacing / 2
            for index in range(segments + 1):
                geom_ids.append(geom_id)
                centers.append((0.0, 0.0, -half + index * spacing))
                radii.append(radius)
                groups.append(group)
            records.append(
                dict(
                    geom_id=geom_id,
                    name=name,
                    type=mujoco.mjtGeom(kind).name,
                    size=size,
                    group=group,
                    axis_segments=segments,
                    cover_radius=radius,
                )
            )
        if not geom_ids:
            raise ValueError("No enabled native robot collision geoms found.")
        return cls(
            torch.tensor(geom_ids, dtype=torch.long, device=device),
            torch.tensor(centers, dtype=torch.float32, device=device),
            torch.tensor(radii, dtype=torch.float32, device=device),
            torch.tensor(groups, dtype=torch.long, device=device),
            float(max_segment),
            tuple(records),
        )

    def world_centers(self, geom_xpos, geom_xmat, env_origins) -> Tensor:
        """Transform using FULL native geom arrays, returning scene-local points.

        Nonfinite kinematics propagate to sample_box_envelope's invalid mask;
        malformed shapes/device contracts raise immediately.
        """
        if (
            geom_xpos.ndim != 3
            or geom_xpos.shape[-1] != 3
            or geom_xmat.shape != (*geom_xpos.shape[:2], 3, 3)
            or env_origins.shape != (geom_xpos.shape[0], 3)
            or not geom_xpos.is_floating_point()
            or geom_xmat.dtype != geom_xpos.dtype
            or env_origins.dtype != geom_xpos.dtype
            or geom_xpos.device != self.geom_ids.device
            or geom_xmat.device != geom_xpos.device
            or env_origins.device != geom_xpos.device
            or geom_xpos.shape[1] <= self.geom_records[-1]["geom_id"]
        ):
            raise ValueError(
                "Native geom poses/origins have incompatible shapes or device."
            )
        local = self.local_centers.to(dtype=geom_xpos.dtype)
        rotated = (geom_xmat[:, self.geom_ids] @ local[None, :, :, None]).squeeze(-1)
        return geom_xpos[:, self.geom_ids] + rotated - env_origins[:, None]

    def schema(self) -> dict:
        """JSON-safe cover identity for the independent research contract."""
        return dict(
            version="offline_box_envelope_v1",
            provenance="known_scene_boxes_not_online_perception",
            method="capsule_axis_spheres_radius_plus_half_segment",
            max_segment=self.max_segment,
            group_names=list(G1_BODY_SCHEMA),
            geoms=[
                {**record, "size": list(record["size"])}
                for record in self.geom_records
            ],
            geom_ids=self.geom_ids.cpu().tolist(),
            local_centers=self.local_centers.cpu().tolist(),
            radii=self.radii.cpu().tolist(),
            groups=self.groups.cpu().tolist(),
        )


@dataclass(frozen=True)
class EnvelopeSample:
    clearance: Tensor
    gradient: Tensor
    minimum: Tensor
    valid: Tensor


class BoxEnvelopeQuery:
    """Prevalidated known boxes and cover groups for the native GPU hot path.

    The constructor may transfer static data and synchronize. ``sample`` uses
    fixed-size batches only: no tensor-to-host reads, dynamic ``nonzero``, or
    reconstruction of boxes/groups. Invalid scene IDs fail closed on-device.
    """

    def __init__(
        self, manifest: SceneManifest, envelope: CollisionEnvelope, *, dtype=None
    ):
        if not manifest.scenes:
            raise ValueError("A cached envelope query requires at least one scene.")
        radii, groups = envelope.radii, envelope.groups
        if (
            radii.ndim != 1
            or not len(radii)
            or not radii.is_floating_point()
            or groups.shape != radii.shape
            or groups.dtype != torch.long
            or groups.device != radii.device
        ):
            raise ValueError(
                "Envelope radii and groups have incompatible contracts."
            )
        group_values = groups.cpu().tolist()
        if (
            not torch.isfinite(radii).all()
            or (radii < 0).any()
            or any(group < 0 or group >= 11 for group in group_values)
        ):
            raise ValueError("Envelope radii or group indices are invalid.")
        self.radii = radii.to(dtype=dtype or radii.dtype).detach().clone()
        if (
            not self.radii.is_floating_point()
            or not torch.isfinite(self.radii).all()
        ):
            raise ValueError("Cached envelope precision must retain finite radii.")
        self.group_indices = tuple(
            torch.tensor(
                [
                    index
                    for index, value in enumerate(group_values)
                    if value == group
                ],
                dtype=torch.long,
                device=radii.device,
            )
            for group in range(11)
        )
        self.num_scenes = len(manifest.scenes)
        count = max(1, max(len(scene.boxes) for scene in manifest.scenes))
        centers, sizes, masks = [], [], []
        for scene in manifest.scenes:
            centers.append(
                [box.center for box in scene.boxes]
                + [(0, 0, 0)] * (count - len(scene.boxes))
            )
            sizes.append(
                [box.half_size for box in scene.boxes]
                + [(1, 1, 1)] * (count - len(scene.boxes))
            )
            masks.append(
                [True] * len(scene.boxes) + [False] * (count - len(scene.boxes))
            )
        self.box_centers = self.radii.new_tensor(centers)
        self.box_half_sizes = self.radii.new_tensor(sizes)
        self.box_mask = torch.tensor(masks, device=radii.device, dtype=torch.bool)
        if (
            not torch.isfinite(self.box_centers).all()
            or not torch.isfinite(self.box_half_sizes).all()
            or (self.box_half_sizes <= 0).any()
        ):
            raise ValueError(
                "Known scene boxes must have finite centers and positive sizes."
            )

    def sample(self, centers: Tensor, scene_ids: Tensor) -> EnvelopeSample:
        """Query scene-local cover centers without host synchronization."""
        if (
            centers.ndim != 3
            or centers.shape[0] < 1
            or centers.shape[1:] != (len(self.radii), 3)
            or centers.dtype != self.radii.dtype
            or centers.device != self.radii.device
            or scene_ids.shape != (centers.shape[0],)
            or scene_ids.dtype != torch.long
            or scene_ids.device != centers.device
        ):
            raise ValueError(
                "Cached envelope centers/scene IDs have incompatible contracts."
            )
        scene_valid = (scene_ids >= 0) & (scene_ids < self.num_scenes)
        safe_ids = scene_ids.clamp(0, self.num_scenes - 1)
        valid = scene_valid & torch.isfinite(centers).all((1, 2))
        safe_centers = torch.where(valid[:, None, None], centers, 0.0)
        box_centers = self.box_centers.index_select(0, safe_ids)
        half_sizes = self.box_half_sizes.index_select(0, safe_ids)
        box_mask = self.box_mask.index_select(0, safe_ids)
        delta = safe_centers[:, :, None] - box_centers[:, None]
        q = delta.abs() - half_sizes[:, None]
        outside = q.clamp_min(0)
        length = outside.norm(dim=-1)
        inside, axis = q.max(dim=-1)
        distance = length + inside.clamp_max(0) - self.radii[None, :, None]
        distance = distance.masked_fill(~box_mask[:, None], math.inf)
        sign = torch.where(delta >= 0, 1.0, -1.0)
        outer_normal = sign * outside / length[..., None].clamp_min(1e-30)
        inner_normal = (
            torch.zeros_like(delta).scatter_(-1, axis[..., None], 1.0) * sign
        )
        normals = torch.where((length > 0)[..., None], outer_normal, inner_normal)
        point_distance, nearest_box = distance.min(dim=-1)
        point_normal = normals.gather(
            2, nearest_box[..., None, None].expand(-1, -1, 1, 3)
        ).squeeze(2)
        point_normal = torch.where(
            box_mask.any(-1)[:, None, None], point_normal, 0.0
        )
        clearance = centers.new_full((centers.shape[0], 11), math.inf)
        gradient = centers.new_zeros(centers.shape[0], 11, 3)
        for group, members in enumerate(self.group_indices):
            if members.numel() == 0:  # Static tensor metadata, no device read.
                continue
            value, nearest_point = point_distance.index_select(1, members).min(
                dim=-1
            )
            clearance[:, group] = value
            gradient[:, group] = (
                point_normal.index_select(1, members)
                .gather(1, nearest_point[:, None, None].expand(-1, 1, 3))
                .squeeze(1)
            )
        clearance = torch.where(valid[:, None], clearance, -math.inf)
        gradient = torch.where(valid[:, None, None], gradient, 0.0)
        return EnvelopeSample(clearance, gradient, clearance.min(-1).values, valid)


def sample_box_envelope(
    centers: Tensor,
    radii: Tensor,
    groups: Tensor,
    scene_ids: Tensor,
    manifest: SceneManifest,
) -> EnvelopeSample:
    """Exact point/AABB distance minus cover radius, reduced to 11 body groups.

    Empty obstacle sets and absent groups have +inf clearance and zero gradient.
    Nonfinite point sets invalidate the whole environment, returning -inf and
    zero gradient. Callers must gate observations/termination with ``valid``.
    This function never queries an offline voxel cache or a measured map.
    """
    if (
        centers.ndim != 3
        or centers.shape[-1] != 3
        or not centers.is_floating_point()
    ):
        raise ValueError("centers must be floating [B,P,3].")
    batch, points, _ = centers.shape
    if batch < 1 or points < 1:
        raise ValueError("Envelope queries need nonempty batches and covers.")
    if (
        radii.shape != (points,)
        or radii.dtype != centers.dtype
        or radii.device != centers.device
        or groups.shape != (points,)
        or groups.dtype != torch.long
        or groups.device != centers.device
        or scene_ids.shape != (batch,)
        or scene_ids.dtype != torch.long
        or scene_ids.device != centers.device
    ):
        raise ValueError(
            "Envelope radii/groups/scene IDs have incompatible contracts."
        )
    if (
        not torch.isfinite(radii).all()
        or (radii < 0).any()
        or ((groups < 0) | (groups >= 11)).any()
        or ((scene_ids < 0) | (scene_ids >= len(manifest.scenes))).any()
    ):
        raise ValueError(
            "Envelope radii, group indices, or scene indices are invalid."
        )
    valid = torch.isfinite(centers).all((1, 2))
    safe_centers = torch.where(valid[:, None, None], centers, 0.0)
    clearance = centers.new_full((batch, 11), math.inf)
    gradient = centers.new_zeros(batch, 11, 3)
    for scene_id, scene in enumerate(manifest.scenes):
        slots = torch.nonzero(scene_ids == scene_id, as_tuple=True)[0]
        if not len(slots) or not scene.boxes:
            continue
        box_centers = centers.new_tensor([box.center for box in scene.boxes])
        half = centers.new_tensor([box.half_size for box in scene.boxes])
        if (
            not torch.isfinite(box_centers).all()
            or not torch.isfinite(half).all()
            or (half <= 0).any()
        ):
            raise ValueError(
                "Known scene boxes must have finite centers and positive sizes."
            )
        delta = safe_centers[slots, :, None] - box_centers[None, None]
        q = delta.abs() - half[None, None]
        outside = q.clamp_min(0)
        length = outside.norm(dim=-1)
        inside, axis = q.max(dim=-1)
        distance = length + inside.clamp_max(0) - radii[None, :, None]
        sign = torch.where(delta >= 0, 1.0, -1.0)
        outer_normal = sign * outside / length[..., None].clamp_min(1e-30)
        inner_normal = (
            torch.zeros_like(delta).scatter_(-1, axis[..., None], 1.0) * sign
        )
        normals = torch.where((length > 0)[..., None], outer_normal, inner_normal)
        point_distance, nearest_box = distance.min(dim=-1)
        point_normal = normals.gather(
            2, nearest_box[..., None, None].expand(-1, -1, 1, 3)
        ).squeeze(2)
        for group in range(11):
            members = torch.nonzero(groups == group, as_tuple=True)[0]
            if not len(members):
                continue
            value, nearest_point = point_distance[:, members].min(dim=-1)
            clearance[slots, group] = value
            gradient[slots, group] = (
                point_normal[:, members]
                .gather(1, nearest_point[:, None, None].expand(-1, 1, 3))
                .squeeze(1)
            )
    clearance[~valid] = -math.inf
    gradient[~valid] = 0
    return EnvelopeSample(clearance, gradient, clearance.min(-1).values, valid)
