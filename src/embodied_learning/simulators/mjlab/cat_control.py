"""Source G1Cat incremental targets and explicit PD, Apache-2.0.

Changes: batched PyTorch and native mjlab action term, name-based mapping.
"""

import math
from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from torch import Tensor

from embodied_learning.robots.g1 import (
    CAT_DEFAULT_JOINTS,
    CAT_JOINT_NAMES,
    CAT_KD,
    CAT_KP,
    CAT_TORQUE_LIMIT,
)


class CatJointController:
    def __init__(self, joint_names, soft_lower, soft_upper, num_envs, device):
        if len(joint_names) != 29 or set(joint_names) != set(CAT_JOINT_NAMES):
            raise ValueError("Source G1 requires all 29 named joints.")
        self.indices = torch.tensor(
            [joint_names.index(n) for n in CAT_JOINT_NAMES], device=device
        )
        self.lower = soft_lower[self.indices].to(device)
        self.upper = soft_upper[self.indices].to(device)
        self.default = torch.tensor(
            CAT_DEFAULT_JOINTS, device=device, dtype=torch.float32
        )
        self.targets = self.default.repeat(num_envs, 1)
        self.kp = torch.tensor(CAT_KP, device=device, dtype=torch.float32)
        self.kd = torch.tensor(CAT_KD, device=device, dtype=torch.float32)
        self.limits = torch.tensor(
            CAT_TORQUE_LIMIT, device=device, dtype=torch.float32
        )
        self.kp_scale = torch.ones(num_envs, 1, device=device)
        self.kd_scale = torch.ones_like(self.kp_scale)
        self.rfi_scale = torch.zeros(num_envs, 29, device=device)
        self.actions = torch.zeros(num_envs, 12, device=device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)

    def process_action(self, actions: Tensor) -> None:
        if actions.shape != self.actions.shape or not torch.isfinite(actions).all():
            raise ValueError("Source actions must be finite [B,12].")
        self.last_last_actions.copy_(self.last_actions)
        # Shift the executed controls, not the incoming action: rewards need
        # a[t], a[t-1], a[t-2] even though observations expose a[t].
        self.last_actions.copy_(self.actions)
        self.actions.copy_(actions)
        self.targets[:, :12] = (self.targets[:, :12] + 0.5 * actions).clamp(
            self.lower[:12], self.upper[:12]
        )
        self.targets[:, 12:] = self.default[12:]

    def compute_torques(
        self, joint_pos: Tensor, joint_vel: Tensor, *, rfi: Tensor
    ) -> Tensor:
        torque = (
            self.kp_scale * self.kp * (self.targets - joint_pos)
            - self.kd_scale * self.kd * joint_vel
            + rfi
        )
        return torque.clamp(-self.limits, self.limits)

    def reset(self, env_ids: Tensor) -> None:
        self.targets[env_ids] = self.default
        self.actions[env_ids] = 0
        self.last_actions[env_ids] = 0
        self.last_last_actions[env_ids] = 0


@dataclass(kw_only=True)
class CatActionCfg(ActionTermCfg):
    random_pd: bool = True
    random_rfi: bool = True
    rfi_amplitude_multiplier: float = 1.0

    def __post_init__(self):
        value = self.rfi_amplitude_multiplier
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("RFI attenuation must be finite and between0and1.")

    def build(self, env):
        return CatAction(self, env)


class CatAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        limits = self._entity.data.soft_joint_pos_limits[0]
        self.controller = CatJointController(
            self._entity.joint_names,
            limits[:, 0],
            limits[:, 1],
            env.num_envs,
            env.device,
        )
        env.cat_controller = self.controller

    @property
    def action_dim(self):
        return 12

    @property
    def raw_action(self):
        return self.controller.actions

    def process_actions(self, actions):
        self.controller.process_action(actions)

    def apply_actions(self):
        c = self.controller
        r = self._entity
        q = r.data.joint_pos[:, c.indices]
        v = r.data.joint_vel[:, c.indices]
        noise = (2 * torch.rand_like(q) - 1) * c.rfi_scale
        torques = c.compute_torques(q, v, rfi=noise)
        native = torch.empty_like(q)
        native[:, c.indices] = torques
        r.set_joint_effort_target(native)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        c = self.controller
        c.reset(env_ids)
        n = len(env_ids)
        c.kp_scale[env_ids] = (
            0.75 + 0.5 * torch.rand(n, 1, device=self.device)
            if self.cfg.random_pd
            else 1.0
        )
        c.kd_scale[env_ids] = (
            0.75 + 0.5 * torch.rand(n, 1, device=self.device)
            if self.cfg.random_pd
            else 1.0
        )
        c.rfi_scale[env_ids] = (
            0.1
            * (0.5 + torch.rand(n, 29, device=self.device))
            * c.limits
            * self.cfg.rfi_amplitude_multiplier
            if self.cfg.random_rfi
            else 0.0
        )
