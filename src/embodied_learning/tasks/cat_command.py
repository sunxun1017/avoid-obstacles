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
