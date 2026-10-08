"""Paper potentials and body priorities from Click-and-Traverse.

Equations 1--5 of arXiv:2601.16035v2. These functions are independent of
the released code's progressive tangential guidance. Coordinates, gradients,
and velocities must use the same frame. Distances are in meters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PotentialResult:
    """Scalar potentials [B, K] and negative spatial gradient [B, K, 3]."""

    attractive: Tensor
    repulsive: Tensor
    force: Tensor


def _positive(name: str, value: float, *, zero_allowed: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
        raise ValueError(f"{name} must be finite and positive.")


def _validate(scalar: Tensor, *vectors: Tensor) -> None:
    if scalar.dtype not in (torch.float32, torch.float64):
        raise ValueError("Field calculations require float32 or float64.")
    if scalar.ndim != 2 or not scalar.is_floating_point():
        raise ValueError("Scalar samples must be floating tensors [B, K].")
    tensors = (scalar, *vectors)
    for tensor in tensors:
        expected = scalar.shape if tensor is scalar else (*scalar.shape, 3)
        if (
            tensor.shape != expected
            or tensor.device != scalar.device
            or tensor.dtype != scalar.dtype
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Samples must have matching shapes, dtype and device.")


def paper_potential(
    geodesic: Tensor,
    geodesic_gradient: Tensor,
    sdf: Tensor,
    sdf_gradient: Tensor,
    *,
    eta: float = 1.0,
    xi: float = 1.0,
    influence_radius: float = 0.5,
    min_distance: float = 0.02,
) -> PotentialResult:
    """Evaluate paper potentials and their spatial descent direction.

    Args:
        geodesic: Reachable goal distance [B, K], in meters.
        geodesic_gradient: Spatial derivative of geodesic [B, K, 3].
        sdf: Signed obstacle distance [B, K], positive outside.
        sdf_gradient: Spatial derivative of sdf [B, K, 3].
        eta: Nonnegative attraction multiplier.
        xi: Nonnegative repulsion multiplier.
        influence_radius: Repulsion support in meters.
        min_distance: Below this distance, extend the potential linearly
            using its tangent. This engineering regularization avoids the
            singularity and preserves outward force inside obstacles.

    Returns:
        Potentials and force. This vector is guidance, not a physical force.

    Raises:
        ValueError: Samples or parameters are invalid, or dtype is not float32
            or float64. Filter invalid map
            samples before calling; infinite empty-scene distances are not
            accepted by this formula interface.
    """
    for name, value in (("eta", eta), ("xi", xi)):
        _positive(name, value, zero_allowed=True)
    _positive("influence_radius", influence_radius)
    _positive("min_distance", min_distance)
    if min_distance >= influence_radius:
        raise ValueError("min_distance must be smaller than influence_radius.")
    _validate(geodesic, geodesic_gradient, sdf_gradient)
    _validate(sdf, geodesic_gradient, sdf_gradient)
    if (geodesic < 0).any():
        raise ValueError("Geodesic distance must be nonnegative.")
    safe_d = sdf.clamp_min(min_distance)
    inverse_delta = (safe_d.reciprocal() - 1.0 / influence_radius).clamp_min(0)
    coefficient = xi * inverse_delta / safe_d.square()
    repulsive = 0.5 * xi * inverse_delta.square()
    repulsive = repulsive + coefficient * (safe_d - sdf)
    force = -eta * geodesic_gradient + coefficient[..., None] * sdf_gradient
    return PotentialResult(eta * geodesic, repulsive, force)


def prioritize_guidance(
    force: Tensor,
    sdf: Tensor,
    sdf_gradient: Tensor,
    velocity: Tensor,
    base_priority: Tensor,
    *,
    urgency_scale: float = 1.0,
    distance_scale: float = 1.0,
    min_approach_speed: float = 0.5,
) -> Tensor:
    """Apply paper body priorities to a guidance direction, returning [B,K,3].

    Supply base_priority [K] or [B,K], e.g. root=1 and other points=0.5.
    velocity is Cartesian velocity [B,K,3] in the field's frame. distance_scale
    makes exp(-sdf / distance_scale) explicit (paper default: 1 meter).
    A zero force stays zero. The exponent is capped to [-50,50] for numerical
    protection; this cap is an engineering addition to the paper formula.
    """
    _validate(sdf, force, sdf_gradient, velocity)
    for name, value in (
        ("urgency_scale", urgency_scale),
        ("min_approach_speed", min_approach_speed),
    ):
        _positive(name, value, zero_allowed=True)
    _positive("distance_scale", distance_scale)
    if (
        base_priority.shape not in ((sdf.shape[1],), sdf.shape)
        or base_priority.device != sdf.device
        or base_priority.dtype != sdf.dtype
        or not torch.isfinite(base_priority).all()
        or (base_priority < 0).any()
    ):
        raise ValueError("base_priority must be nonnegative [K] or [B,K].")
    approach = -(sdf_gradient * velocity).sum(-1)
    urgency = urgency_scale * approach.clamp_min(min_approach_speed)
    urgency = urgency * torch.exp((-sdf / distance_scale).clamp(-50, 50))
    norm = torch.linalg.vector_norm(force, dim=-1, keepdim=True)
    direction = force / norm.clamp_min(torch.finfo(force.dtype).eps)
    return direction * (base_priority * urgency)[..., None]
