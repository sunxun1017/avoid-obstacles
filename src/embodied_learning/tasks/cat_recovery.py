"""Trainable stopping research, deliberately separate from cat_source_v1.

Commands are world-frame planar velocities, not physical speed guarantees. The
navigator observes actual motion; its calibrated deceleration is an assumption
to test, not a certified bound. No PD gains or source action scaling change here.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from embodied_learning.tasks.cat_reward import SOURCE_SCALES, CatRewardInputs


@dataclass(frozen=True)
class RecoveryConfig:
    dt: float = 0.02
    braking_deceleration: float = 1.0
    latency: float = 0.2
    braking_margin: float = 0.1
    capture_offset: float = 0.04
    zero_target_distance: float = 0.06
    max_speed: float = 0.6
    reference_acceleration: float = 0.5
    goal_radius: float = 0.1
    settle_entry_speed: float = 0.15
    settle_linear_speed: float = 0.1
    settle_angular_speed: float = 0.1
    settle_seconds: float = 0.5
    resume_distance: float = 0.25
    resume_seconds: float = 0.4
    near_resume_seconds: float = 1.0
    fallback_distance: float = 0.3
    approach_gain: float = 1.0
    tracking_weight: float = 4.0
    hold_linear_weight: float = 2.0
    hold_angular_weight: float = 1.0
    hold_position_weight: float = 2.0
    hold_bonus_weight: float = 4.0
    clearance_margin: float = 0.03
    clearance_violation_limit: float = 0.2
    clearance_cost_scale: float = 20.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (field.name != "latency" and value == 0)
            ):
                raise ValueError(f"Invalid recovery configuration: {field.name}.")
        if not (
            self.capture_offset
            < self.zero_target_distance
            < self.goal_radius
            < self.resume_distance
            < self.fallback_distance
        ):
            raise ValueError("Recovery distance thresholds must be ordered.")
        if self.settle_entry_speed < self.settle_linear_speed:
            raise ValueError("Settle entry speed cannot be below hold speed.")
        if self.near_resume_seconds < self.resume_seconds:
            raise ValueError(
                "Small goal drift must use at least the far-goal dwell."
            )


class RecoveryNavigator:
    DRIVE, BRAKE, SETTLE, HOLD = range(4)

    def __init__(self, num_envs: int, device, config: RecoveryConfig | None = None):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("Recovery navigator requires a positive slot count.")
        self.config = config or RecoveryConfig()
        self.mode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.settle_steps = torch.zeros_like(self.mode)
        self.resume_steps = torch.zeros_like(self.mode)
        # Previous reference, not previous measured velocity; used for slew limits.
        self.last_velocity = torch.zeros(num_envs, 3, device=device)
        self.reversal_pending = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )

    def reset(self, env_ids: Tensor) -> None:
        self.mode[env_ids] = self.DRIVE
        self.settle_steps[env_ids] = 0
        self.resume_steps[env_ids] = 0
        self.last_velocity[env_ids] = 0
        self.reversal_pending[env_ids] = False

    def advance(
        self,
        *,
        goal_delta: Tensor,
        velocity: Tensor,
        angular_velocity: Tensor,
        guidance_command: Tensor,
        feet_contact: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Advance active slots once; return coherent command and restart pulses.

        DRIVE/BRAKE retain move_flag=1 so that a moving robot is not forcibly
        frozen into stance. SETTLE/HOLD use move_flag=0. The caller owns phase
        continuity and must not apply the source's irreversible stop latch.
        """
        b = len(self.mode)
        if active.shape != (b,) or active.dtype != torch.bool:
            raise ValueError("Recovery active mask must be boolean [B].")
        if active.device != self.mode.device:
            raise ValueError("Recovery inputs must share the navigator device.")
        for name, value, width in (
            ("goal_delta", goal_delta, 2),
            ("velocity", velocity, 3),
            ("angular_velocity", angular_velocity, 3),
            ("guidance_command", guidance_command, 4),
            ("feet_contact", feet_contact, 2),
        ):
            if value.shape != (b, width) or value.device != self.mode.device:
                raise ValueError(f"Invalid recovery input shape/device: {name}.")
            if not torch.isfinite(value[active]).all():
                raise ValueError(f"Nonfinite active recovery input: {name}.")
        c = self.config
        # Inactive reset/invalid payloads must not poison otherwise valid slots.
        goal_delta = torch.where(active[:, None], goal_delta, 0.0)
        velocity = torch.where(active[:, None], velocity, 0.0)
        angular_velocity = torch.where(active[:, None], angular_velocity, 0.0)
        guidance_command = torch.where(active[:, None], guidance_command, 0.0)
        distance = goal_delta.norm(dim=-1)
        xy_speed = velocity[:, :2].norm(dim=-1)
        speed = velocity.norm(dim=-1)
        angular_speed = angular_velocity.norm(dim=-1)
        supported = (feet_contact > 0).all(-1)
        slow = (speed < c.settle_linear_speed) & (
            angular_speed < c.settle_angular_speed
        )
        goal_slow = slow & supported & (distance < c.goal_radius)
        braking_distance = (
            xy_speed.square() / (2 * c.braking_deceleration)
            + xy_speed * c.latency
            + c.braking_margin
        )
        mode = torch.where(
            (self.mode == self.DRIVE) & (distance <= braking_distance),
            self.BRAKE,
            self.mode,
        )
        mode = torch.where((mode == self.HOLD) & ~goal_slow, self.SETTLE, mode)
        # Crossing the goal at speed must finish braking before reversing.
        away = (velocity[:, :2] * goal_delta).sum(-1) < -0.02 * distance
        pending = self.reversal_pending | (
            (mode == self.BRAKE)
            & away
            & (xy_speed > c.settle_entry_speed)
            & (distance > c.goal_radius)
        )
        settle_candidate = (
            (mode == self.BRAKE)
            & (speed < c.settle_entry_speed)
            & supported
            & ((distance < c.goal_radius) | pending)
        )
        # Keep stepping while the reference ramps down, even if actual speed
        # already fell. The final step to zero must fit the same slew budget.
        enter_settle = settle_candidate & (
            self.last_velocity.norm(dim=-1) <= c.braking_deceleration * c.dt
        )
        mode = torch.where(enter_settle, self.SETTLE, mode)
        stationary = mode >= self.SETTLE
        settle_steps = torch.where(stationary & goal_slow, self.settle_steps + 1, 0)
        mode = torch.where(
            settle_steps >= math.ceil(c.settle_seconds / c.dt), self.HOLD, mode
        )
        resume_steps = torch.where(
            stationary & slow & supported & (distance >= c.goal_radius),
            self.resume_steps + 1,
            0,
        )
        resume_required = torch.where(
            distance > c.resume_distance,
            math.ceil(c.resume_seconds / c.dt),
            math.ceil(c.near_resume_seconds / c.dt),
        )
        restart = active & (resume_steps >= resume_required)
        mode = torch.where(restart, self.DRIVE, mode)
        resume_steps = torch.where(restart, 0, resume_steps)
        pending = pending & ~restart

        guidance = guidance_command[:, 1:3]
        magnitude = guidance.norm(dim=-1)
        fallback = (magnitude < 1e-6) & (distance < c.fallback_distance)
        guidance = torch.where(fallback[:, None], goal_delta, guidance)
        magnitude = torch.where(fallback, c.approach_gain * distance, magnitude)
        direction = guidance / guidance.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        remaining = (distance - c.capture_offset - xy_speed * c.latency).clamp_min(0)
        speed_cap = torch.sqrt(2 * c.braking_deceleration * remaining)
        target_speed = torch.minimum(magnitude.clamp(max=c.max_speed), speed_cap)
        target_speed = torch.where(
            (distance < c.zero_target_distance)
            | settle_candidate
            | pending
            | (mode >= self.SETTLE),
            0.0,
            target_speed,
        )
        target = torch.cat(
            (
                direction * target_speed[:, None],
                torch.zeros_like(target_speed[:, None]),
            ),
            1,
        )
        delta = target - self.last_velocity
        decreasing = target.norm(dim=-1) < self.last_velocity.norm(dim=-1)
        rate = torch.where(
            decreasing, c.braking_deceleration, c.reference_acceleration
        )
        factor = (rate * c.dt / delta.norm(dim=-1).clamp_min(1e-9)).clamp(max=1)
        reference = self.last_velocity + delta * factor[:, None]
        reference = torch.where((mode >= self.SETTLE)[:, None], 0.0, reference)
        self.mode[active] = mode[active]
        self.settle_steps[active] = settle_steps[active]
        self.resume_steps[active] = resume_steps[active]
        self.reversal_pending[active] = pending[active]
        self.last_velocity[active] = reference[active].to(self.last_velocity)
        command = torch.cat(
            (
                (self.mode < self.SETTLE).to(self.last_velocity)[:, None],
                self.last_velocity,
            ),
            1,
        )
        return command, restart


def research_reward_terms(
    source_terms: Mapping[str, Tensor],
    *,
    inputs: CatRewardInputs,
    goal_delta: Tensor,
    angular_velocity: Tensor,
    mode: Tensor,
    source_scales: Mapping[str, float] | None = None,
    config: RecoveryConfig | None = None,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    """Replace source stopping shortcuts and return rates plus their scales.

    ``inputs.sdf`` is research body-envelope clearance. Its six group costs
    cover all eleven probes with capped quadratics. The caller applies terminal
    contact costs once per failed episode, not multiplied by control dt here.
    """
    c = config or RecoveryConfig()
    terms = dict(source_terms)
    scales = dict(SOURCE_SCALES if source_scales is None else source_scales)
    if set(terms) != set(scales):
        raise ValueError("Source reward terms and scales must match.")
    stationary = mode >= RecoveryNavigator.SETTLE
    speed_sq = inputs.root_vel.square().sum(-1)
    angular_sq = angular_velocity.square().sum(-1)
    distance_sq = goal_delta.square().sum(-1)
    terms["tracking_root_field"] = (
        -(inputs.cmd_vel[:, :2] - inputs.root_vel[:, :2]).square().sum(-1)
    )
    scales["tracking_root_field"] = c.tracking_weight
    terms["recovery_linear_speed"] = speed_sq * stationary
    terms["recovery_angular_speed"] = angular_sq * stationary
    terms["recovery_goal_error"] = distance_sq * stationary
    terms["recovery_goal_hold"] = (
        (mode == RecoveryNavigator.HOLD)
        & (speed_sq < c.settle_linear_speed**2)
        & (angular_sq < c.settle_angular_speed**2)
        & (distance_sq < c.goal_radius**2)
    ).to(inputs.root_vel)
    scales.update(
        recovery_linear_speed=-c.hold_linear_weight,
        recovery_angular_speed=-c.hold_angular_weight,
        recovery_goal_error=-c.hold_position_weight,
        recovery_goal_hold=c.hold_bonus_weight,
        recovery_core_clearance=1.0,
    )
    core_violation = (c.clearance_margin - inputs.sdf[:, [1, 2]]).clamp(
        min=0, max=c.clearance_violation_limit
    )
    terms["recovery_core_clearance"] = (
        -c.clearance_cost_scale * core_violation.square().mean(-1)
    )
    speed_ref = inputs.cmd_vel[:, :2].norm(dim=-1)
    for name, ids in (
        ("head", [0]),
        ("feet", [3, 4]),
        ("hands", [5, 6]),
        ("knees", [7, 8]),
        ("shlds", [9, 10]),
    ):
        clearance = inputs.sdf[:, ids]
        violation = (c.clearance_margin - clearance).clamp(
            min=0, max=c.clearance_violation_limit
        )
        terms[name + "df"] = -c.clearance_cost_scale * violation.square().mean(-1)
        if name not in ("head", "feet", "hands"):
            continue
        gf = inputs.gf[:, ids]
        direction = gf / gf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        progress = (direction * inputs.body_velocities[:, ids]).sum(-1)
        # Reward is maximal at commanded progress, not at arbitrarily large speed.
        matched = progress - 2 * (progress - speed_ref[:, None]).clamp_min(0)
        matched = matched.clamp(-c.max_speed, c.max_speed) / c.max_speed
        tau = 0.3 if name == "feet" else 0.5
        field_reward = 5 * matched * torch.sigmoid(40 * (tau - clearance))
        moving = (~stationary & (speed_ref > 1e-6))[:, None]
        if name == "feet":
            moving = moving & (inputs.gait_mask != 1)
        terms[name + "gf"] = torch.where(moving, field_reward, 0.0).mean(-1)
    return terms, scales


def aggregate_recovery_reward(
    terms: Mapping[str, Tensor], scales: Mapping[str, float], *, dt: float
) -> Tensor:
    """Integrate signed reward rates once; no zero clipping hides large errors."""
    if not terms or set(terms) != set(scales) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Recovery reward terms/scales/dt are invalid.")
    if not all(math.isfinite(scale) for scale in scales.values()):
        raise ValueError("Recovery reward scales must be finite.")
    return sum(value * scales[name] for name, value in terms.items()) * dt
