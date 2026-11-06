"""Stable 3D vMF field reward from arXiv:2601.16035v2, equations 7--10.

This module does not implement a discriminator or motion-data prior. The
direction and concentration are derived directly from the environment field.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def vmf_direction_reward(
    guidance: Tensor,
    velocity: Tensor,
    *,
    concentration_scale: float,
    valid_mask: Tensor,
) -> Tensor:
    """Sum paper log-likelihood rewards over valid body points, returning [B].

    Args:
        guidance: Priority-weighted paper field vectors [B,K,3].
        velocity: Body point velocities [B,K,3], in the same coordinates.
        concentration_scale: Nonnegative multiplier for field magnitude.
        valid_mask: Boolean [B,K] indicating usable field/kinematic samples.

    Returns:
        Reward [B], with zero contribution from invalid points. Zero velocity
        has zero directional alignment, rather than being rewarded as aligned.
        Zero concentration uses the uniform sphere density 1/(4*pi).

    Raises:
        ValueError: Incompatible tensors, precision, nonfinite valid inputs,
            invalid scale or overflowed concentration.
    """
    if (
        guidance.ndim != 3
        or guidance.shape[-1] != 3
        or guidance.dtype not in (torch.float32, torch.float64)
        or velocity.shape != guidance.shape
        or velocity.dtype != guidance.dtype
        or velocity.device != guidance.device
        or valid_mask.shape != guidance.shape[:-1]
        or valid_mask.dtype != torch.bool
        or valid_mask.device != guidance.device
    ):
        raise ValueError("Expected matching float32/64 [B,K,3] and bool [B,K].")
    if not math.isfinite(concentration_scale) or concentration_scale < 0:
        raise ValueError("concentration_scale must be finite and nonnegative.")
    g = torch.where(valid_mask[..., None], guidance, 0.0)
    v = torch.where(valid_mask[..., None], velocity, 0.0)
    if not torch.isfinite(g).all() or not torch.isfinite(v).all():
        raise ValueError("Valid guidance and velocity must be finite.")
    kappa = torch.linalg.vector_norm(g, dim=-1) * concentration_scale
    if not torch.isfinite(kappa).all():
        raise ValueError("Field concentration overflowed; inspect field scale.")

    # log C3(k) = log(k) - log(4*pi) - log(sinh(k)). Branch arguments
    # stay safe even when torch.where evaluates both branches eagerly.
    small = kappa.clamp(max=1e-3)
    series = -math.log(4 * math.pi) - small.square() / 6 + small.pow(4) / 180
    safe = kappa.clamp_min(1e-3)
    regular = (
        safe.log()
        - math.log(2 * math.pi)
        - safe
        - torch.log(-torch.expm1(-2 * safe))
    )
    log_normalizer = torch.where(kappa < 1e-3, series, regular)
    speed = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    unit_velocity = v / speed.clamp_min(torch.finfo(v.dtype).eps)
    # kappa * dot(mu, v_hat) simplifies to scale * dot(g, v_hat).
    # Avoid dividing by a zero field and preserve gradients at zero magnitude.
    alignment = concentration_scale * (g * unit_velocity).sum(-1)
    return torch.where(valid_mask, log_normalizer + alignment, 0.0).sum(-1)
