"""G1Cat compute_cmd_from_rtf, Apache-2.0, upstream 866ba392.

Batched PyTorch translation. Retains the source's nonbinary move flag.
"""

import math

import torch
from torch import Tensor


def compute_cat_command(
    root_guidance: Tensor,
    guidance: Tensor,
    boundary: Tensor,
    *,
    deadband: float = 0.2,
) -> Tensor:
    if (
        type(deadband) not in (int, float)
        or not math.isfinite(deadband)
        or deadband <= 0
    ):
        raise ValueError("Command deadband must be positive and finite.")
    b = root_guidance.shape[0]
    if (
        root_guidance.shape != (b, 3)
        or guidance.shape != (b, 5, 3)
        or boundary.shape != guidance.shape
    ):
        raise ValueError("Command inputs must be [B,3] and [B,5,3].")
    if any(
        x.dtype != root_guidance.dtype
        or x.device != root_guidance.device
        or not torch.isfinite(x).all()
        for x in (root_guidance, guidance, boundary)
    ):
        raise ValueError("Command inputs must be finite and share precision/device.")
    v = root_guidance[:, :2] * 0.7
    bf = boundary[:, :, :2]
    bhat = bf / (bf.norm(dim=-1, keepdim=True) + 1e-9)
    ls = (bhat * guidance[:, :, :2]).sum(-1)
    bv = (bhat * v[:, None]).sum(-1)
    delta = (
        (ls - bv)[..., None] / (bhat.square().sum(-1, keepdim=True) + 1e-9)
    ) * bhat
    new = v + torch.where((ls > bv)[..., None], delta, 0.0).mean(1)
    command = (
        torch.cat(
            (torch.ones_like(new[:, :1]), new, torch.zeros_like(new[:, :1])), 1
        )
        * 0.75
    )
    return torch.where(
        (command[:, 1:].norm(dim=-1) < deadband)[:, None], 0.0, command
    )


def retime_cat_arrival(
    command: Tensor, root: Tensor, goals: Tensor, point_sdf: Tensor
) -> Tensor:
    """Experimental root-based arrival speed in a clear goal neighborhood.

    Keep the projected navigation direction and leave speeds near obstacles
    unchanged. The 8 cm stop radius lies inside the unchanged 10 cm metric.
    """
    distance = (root[:, :2] - goals[:, :2]).norm(dim=-1)
    speed = command[:, 1:3].norm(dim=-1)
    clear = (point_sdf >= 0.3).all(-1)
    retime = clear & (distance < 0.35) & (speed > 1e-6)
    target = (0.315 * distance / 0.3).clamp(max=0.315)
    result = command.clone()
    result[:, 1:3] *= torch.where(retime, target / speed.clamp_min(1e-6), 1.0)[
        :, None
    ]
    return torch.where((distance < 0.08)[:, None], 0.0, result)


def corrected_navigation_command(
    root_guidance, guidance, boundary, root, goals, point_sdf
):
    """Root-referenced navigation with an analytic, clear-space goal blend.

    Keep projected obstacle routing. Blend toward the horizontal root-to-goal
    vector only within 35 cm of the goal and with at least 30 cm clearance.
    Never normalize a near-zero sampled direction into a finite walking speed.
    The existing 8 cm stop radius is retained for this isolated experiment.
    """
    command = compute_cat_command(root_guidance, guidance, boundary, deadband=1e-6)
    delta = goals[:, :2] - root[:, :2]
    distance = delta.norm(dim=-1)

    def smoothstep(value):
        value = value.clamp(0, 1)
        return value.square() * (3 - 2 * value)

    blend = smoothstep((0.35 - distance) / 0.10) * smoothstep(
        (point_sdf.amin(dim=-1) - 0.30) / 0.20
    )
    # This direction comes from a metric position error, not a weak field
    # gradient. Inside 30 cm it is simply 1.05 * delta.
    speed = (1.05 * distance).clamp(max=0.315)
    analytic = delta * (speed / distance.clamp_min(1e-9))[:, None]
    result = command.clone()
    result[:, 1:3] = (1 - blend[:, None]) * command[:, 1:3] + blend[
        :, None
    ] * analytic
    result[:, 0] = torch.where(result[:, 1:3].norm(dim=-1) > 1e-6, 0.75, 0.0)
    return torch.where((distance < 0.08)[:, None], 0.0, result)
