"""G1Cat phase/stop state, Apache-2.0, upstream 866ba392.

Explicit per-slot state and batched translation of _update_phase.
"""

import math

import torch
from torch import Tensor


class CatGaitState:
    def __init__(self, num_envs: int, device: str, *, dt: float = 0.02):
        if num_envs < 1 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid gait count or timestep.")
        self.dt = dt
        self.phase = torch.zeros(num_envs, 2, device=device)
        self.phase_dt = torch.zeros(num_envs, device=device)
        self.gait_mask = torch.zeros_like(self.phase)
        self.stop_timestep = torch.full(
            (num_envs,), 100, device=device, dtype=torch.long
        )
        self.last_command = torch.zeros(num_envs, 4, device=device)

    def reset(
        self, env_ids: Tensor, frequency: Tensor, initial_phase: Tensor
    ) -> None:
        if (
            frequency.shape != (len(env_ids),)
            or initial_phase.shape != (len(env_ids), 2)
            or not torch.isfinite(frequency).all()
            or (frequency <= 0).any()
        ):
            raise ValueError("Invalid gait reset arrays.")
        self.phase[env_ids] = initial_phase
        self.phase_dt[env_ids] = 2 * math.pi * self.dt * frequency
        self.gait_mask[env_ids] = 0
        self.stop_timestep[env_ids] = 100
        self.last_command[env_ids] = 0

    def advance(self, command: Tensor) -> Tensor:
        if command.shape != self.last_command.shape:
            raise ValueError("Command must be [B,4].")
        before = self.stop_timestep > 50
        during = ~before & (self.stop_timestep > 0)
        after = ~before & ~during
        transition = (
            (self.last_command[:, 0] == 1.0) & (command[:, 0] == 0.0) & before
        )
        self.stop_timestep = torch.where(transition, 50, self.stop_timestep)
        self.stop_timestep = torch.where(
            during, self.stop_timestep - 1, self.stop_timestep
        )
        result = torch.where(before[:, None], command, 0.0).clone()
        result[:, 0] = torch.where(after, 0.0, 1.0)
        phase = (
            torch.fmod(self.phase + self.phase_dt[:, None] + math.pi, 2 * math.pi)
            - math.pi
        )
        self.phase = torch.where(after[:, None], 0.0, phase)
        cosine = torch.cos(self.phase)
        self.gait_mask = torch.where(
            cosine > 0.6, 1.0, torch.where(cosine < -0.6, -1.0, 0.0)
        )
        self.last_command = result.clone()
        return result
