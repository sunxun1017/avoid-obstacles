"""Opt-in feet-reward gate based on net root displacement, not foot lifting."""

import torch
from torch import Tensor


class RootProgressWindow:
    """Track the last 25 controls (0.5s) independently for each episode slot.

    The gate is achieved net speed along the current command divided by the
    requested speed, bounded to [0, 1]. A 0.1m/s denominator floor avoids
    turning small near-goal position fluctuations into full progress credit.
    """

    def __init__(self, num_envs: int, device: str):
        self.history = torch.zeros(25, num_envs, 2, device=device)
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.gate = torch.zeros(num_envs, device=device)
        self.cursor = 0
        self.last_step: int | None = None

    def update(
        self, root_xy: Tensor, command_xy: Tensor, *, step: int, initial: Tensor
    ) -> Tensor:
        # Reset all history for new episodes, including resets on the same
        # simulator step. No teleport displacement may become reward credit.
        self.history[:, initial] = root_xy[initial]
        self.age[initial] = 0
        self.gate[initial] = 0
        if step != self.last_step:
            active = ~initial
            self.age[active] = (self.age[active] + 1).clamp(max=25)
            elapsed = self.age.clamp_min(1).to(root_xy.dtype) * 0.02
            displacement = root_xy - self.history[self.cursor]
            speed = command_xy.norm(dim=-1)
            direction = command_xy / speed.clamp_min(1e-6)[:, None]
            net_speed = (displacement * direction).sum(-1) / elapsed
            gate = (net_speed / speed.clamp_min(0.1)).clamp(0, 1)
            self.gate[active] = gate[active]
            self.history[self.cursor, active] = root_xy[active]
            self.cursor = (self.cursor + 1) % 25
            self.last_step = step
        return self.gate.clone()
