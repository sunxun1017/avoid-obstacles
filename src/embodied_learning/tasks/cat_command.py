"""G1Cat compute_cmd_from_rtf, Apache-2.0, upstream 866ba392.

Batched PyTorch translation. Retains the source's nonbinary move flag.
"""

import torch
from torch import Tensor


def compute_cat_command(
    root_guidance: Tensor, guidance: Tensor, boundary: Tensor
) -> Tensor:
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
    return torch.where((command[:, 1:].norm(dim=-1) < 0.2)[:, None], 0.0, command)
