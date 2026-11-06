"""Paper multi-body field observations, independent of robot and simulator.

Paper equations 1--5 are evaluated from sampled metric fields, rather than
substituting the released progressive tangential velocity field. Point order
and point locations are the caller's explicit robot schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from embodied_learning.perception.potential_field import (
    _positive,
    paper_potential,
    prioritize_guidance,
)
from embodied_learning.perception.voxel_field import FieldSample


@dataclass(frozen=True)
class FieldParameters:
    """Paper scales and explicit numerical regularization, in metric units.

    Empty maps have infinite obstacle distance. Their distance is replaced
    only during field evaluation with empty_scene_distance (outside repulsion
    support), retaining goal guidance. This is an engineering convention;
    returned obstacle distances remain infinite, not artificial obstacles.
    """

    eta: float = 1.0
    xi: float = 1.0
    influence_radius: float = 0.5
    min_distance: float = 0.02
    urgency_scale: float = 1.0
    distance_scale: float = 1.0
    min_approach_speed: float = 0.5
    empty_scene_distance: float = 1.0

    def __post_init__(self) -> None:
        for name in ("eta", "xi", "urgency_scale", "min_approach_speed"):
            _positive(name, getattr(self, name), zero_allowed=True)
        for name in (
            "influence_radius",
            "min_distance",
            "distance_scale",
            "empty_scene_distance",
        ):
            _positive(name, getattr(self, name))
        if self.min_distance >= self.influence_radius:
            raise ValueError("min_distance must be below influence_radius.")
        if self.empty_scene_distance < self.influence_radius:
            raise ValueError("Empty-scene distance must not induce repulsion.")


@dataclass(frozen=True)
class BodyFieldObservation:
    """Ordered guidance [B,K,3], metric scalars and valid_mask [B,K]."""

    guidance: Tensor
    sdf: Tensor
    geodesic: Tensor
    valid_mask: Tensor


def evaluate_body_field(
    sample: FieldSample,
    velocity: Tensor,
    base_priority: Tensor,
    *,
    parameters: FieldParameters,
) -> BodyFieldObservation:
    """Apply paper force and body priority to valid sampled body kinematics.

    Invalid scalar/vector values are excluded before formula evaluation.
    Velocities and sample gradients must use the same coordinate frame.
    """
    scalar = sample.sdf
    if scalar.ndim != 2 or scalar.dtype not in (torch.float32, torch.float64):
        raise ValueError("Field scalar samples must be float32/64 [B,K].")
    vectors = (sample.sdf_gradient, sample.geodesic_gradient, velocity)
    for vector in vectors:
        if (
            vector.shape != (*scalar.shape, 3)
            or vector.device != scalar.device
            or vector.dtype != scalar.dtype
        ):
            raise ValueError("Field gradients/velocity must match [B,K,3].")
    if (
        sample.geodesic.shape != scalar.shape
        or sample.geodesic.device != scalar.device
        or sample.geodesic.dtype != scalar.dtype
    ):
        raise ValueError("Geodesic must match field scalar samples.")
    for mask in (sample.distance_valid, sample.geodesic_valid):
        if (
            mask.shape != scalar.shape
            or mask.dtype != torch.bool
            or mask.device != scalar.device
        ):
            raise ValueError("Field validity must be matching boolean [B,K].")
    valid = (
        sample.distance_valid
        & sample.geodesic_valid
        & ~torch.isnan(scalar)
        & ~torch.isneginf(scalar)
        & torch.isfinite(sample.geodesic)
        & (sample.geodesic >= 0)
    )
    for vector in vectors:
        valid &= torch.isfinite(vector).all(-1)
    sdf = torch.where(
        torch.isposinf(scalar), parameters.empty_scene_distance, scalar
    )
    sdf = torch.where(valid, sdf, parameters.empty_scene_distance)
    geodesic = torch.where(valid, sample.geodesic, 0.0)
    safe_vectors = [torch.where(valid[..., None], v, 0.0) for v in vectors]
    sdf_gradient, goal_gradient, velocity = safe_vectors
    potential = paper_potential(
        geodesic,
        goal_gradient,
        sdf,
        sdf_gradient,
        eta=parameters.eta,
        xi=parameters.xi,
        influence_radius=parameters.influence_radius,
        min_distance=parameters.min_distance,
    )
    guidance = prioritize_guidance(
        potential.force,
        sdf,
        sdf_gradient,
        velocity,
        base_priority,
        urgency_scale=parameters.urgency_scale,
        distance_scale=parameters.distance_scale,
        min_approach_speed=parameters.min_approach_speed,
    )
    return BodyFieldObservation(
        torch.where(valid[..., None], guidance, 0.0),
        torch.where(valid, scalar, 0.0),
        geodesic,
        valid,
    )


def yaw_rotate_vectors(vectors: Tensor, quaternion_wxyz: Tensor) -> Tensor:
    """Rotate world vectors [B,...,3] to gravity-aligned robot yaw coordinates.

    Quaternions use wxyz ordering and need not be normalized. Translation
    does not apply to vectors; roll/pitch do not tilt the navigation frame.
    """
    q = quaternion_wxyz
    if (
        vectors.ndim < 2
        or vectors.shape[-1] != 3
        or vectors.dtype not in (torch.float32, torch.float64)
        or q.shape != (vectors.shape[0], 4)
        or q.dtype != vectors.dtype
        or q.device != vectors.device
        or not torch.isfinite(q).all()
        or not torch.isfinite(vectors).all()
    ):
        raise ValueError("Expected finite matching [B,...,3] vectors and [B,4].")
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    if (norm <= torch.finfo(q.dtype).eps).any():
        raise ValueError("Quaternion must have nonzero norm.")
    w, x, y, z = (q / norm).unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    shape = (vectors.shape[0],) + (1,) * (vectors.ndim - 2)
    c, s = yaw.cos().reshape(shape), yaw.sin().reshape(shape)
    vx, vy, vz = vectors.unbind(-1)
    return torch.stack((c * vx + s * vy, -s * vx + c * vy, vz), dim=-1)
