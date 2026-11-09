"""G1Cat source rewards ported to batched PyTorch (Apache-2.0).

Source: Click-and-Traverse commit 866ba392, env_cat.py and env_loco.py.
Changes: explicit inputs, batched operations; no simulation or JAX dependency.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

SOURCE_SCALES = dict(
    tracking_orientation=2.0,
    tracking_root_field=1.0,
    body_motion=-0.5,
    body_rotation=1.0,
    foot_contact=-1.0,
    foot_clearance=-15.0,
    foot_slip=-0.5,
    foot_balance=-30.0,
    straight_knee=-30.0,
    foot_far=0.0,
    smoothness_joint=-1e-6,
    smoothness_action=-1e-3,
    joint_limits=-1.0,
    joint_torque=-1e-4,
    headgf=1.0,
    handsgf=1.0,
    feetgf=1.0,
    headdf=1.0,
    handsdf=1.0,
    feetdf=1.0,
    kneesdf=1.0,
    shldsdf=1.0,
)
OBS_INDICES = tuple(range(19)) + tuple(range(22, 26))


@dataclass
class CatRewardInputs:
    pelvis_rpy: Tensor
    torso_rpy: Tensor
    head_pos: Tensor
    cmd_vel: Tensor
    root_vel: Tensor
    torso_ang_vel: Tensor
    leg_rotation: Tensor
    foot_pos: Tensor
    foot_vel: Tensor
    hands_pos: Tensor
    pelvis_subtree_com: Tensor
    feet_contact: Tensor
    gait_mask: Tensor
    move_flag: Tensor
    foot_height: Tensor
    joint_pos: Tensor
    joint_vel: Tensor
    previous_vel: Tensor
    torques: Tensor
    actions: Tensor
    last_actions: Tensor
    last_last_actions: Tensor
    gf: Tensor
    bf: Tensor
    sdf: Tensor
    navigation_rotation: Tensor
    body_velocities: Tensor


@dataclass
class CatRewardParameters:
    soft_lower: Tensor
    soft_upper: Tensor
    dt: float = 0.02
    chest_height: float = 1.0
    yaw_max: float = 0.5
    foot_height_stance: float = 0.0


def standing_stability_bonus(
    goal_distance: Tensor,
    command: Tensor,
    root_velocity: Tensor,
    gyro: Tensor,
    feet_contact: Tensor,
) -> Tensor:
    """Reward low body motion near the goal during commanded double support."""
    standing = (
        (goal_distance <= 0.10)
        & (command[:, :2].norm(dim=-1) < 0.01)
        & (feet_contact > 0).all(dim=-1)
    )
    motion = (
        root_velocity.square().sum(dim=-1) / 0.20**2
        + gyro.square().sum(dim=-1) / 0.35**2
    )
    quality = (1.0 + motion).reciprocal()
    return torch.where(standing & torch.isfinite(quality), quality, 0.0)


def aggregate_cat_reward(
    terms: dict[str, Tensor], scales: dict[str, float], *, dt: float
) -> Tensor:
    if set(terms) != set(scales) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Reward terms/scales must match and dt must be positive.")
    if not all(math.isfinite(x) for x in scales.values()):
        raise ValueError("Reward scales must be finite.")
    total = sum(
        torch.where(torch.isnan(v), 0.0, v) * scales[k] for k, v in terms.items()
    )
    return (total * dt).clamp(0.0, 10000.0)


def cat_reward_terms(
    x: CatRewardInputs,
    *,
    parameters: CatRewardParameters,
    diagnostics: dict[str, Tensor] | None = None,
    feet_progress_gate: Tensor | None = None,
    whole_body_joint_velocity: bool = False,
    landing_speed_rate: Tensor | None = None,
    contact_force_rate: Tensor | None = None,
    risk_alpha: Tensor | None = None,
    standing_contact_slip: bool = False,
    standing_balance_free: bool = False,
    ducking_allowance: Tensor | None = None,
) -> dict[str, Tensor]:
    b = x.root_vel.shape[0]
    if risk_alpha is not None and (
        risk_alpha.shape != (b, 11)
        or risk_alpha.dtype != x.root_vel.dtype
        or risk_alpha.device != x.root_vel.device
        or not torch.isfinite(risk_alpha).all()
        or (risk_alpha < 0).any()
        or (risk_alpha > 1).any()
    ):
        raise ValueError(
            "Risk alpha must match [B,11] reward inputs and lie in [0,1]."
        )
    if feet_progress_gate is not None and (
        feet_progress_gate.shape != (b,)
        or not torch.isfinite(feet_progress_gate).all()
        or (feet_progress_gate < 0).any()
        or (feet_progress_gate > 1).any()
    ):
        raise ValueError("Feet progress gate must be finite [B] values in [0, 1].")
    expected = {
        "pelvis_rpy": (b, 3),
        "torso_rpy": (b, 3),
        "head_pos": (b, 3),
        "cmd_vel": (b, 3),
        "root_vel": (b, 3),
        "torso_ang_vel": (b, 3),
        "foot_pos": (b, 2, 3),
        "foot_vel": (b, 2, 3),
        "hands_pos": (b, 2, 3),
        "pelvis_subtree_com": (b, 3),
        "feet_contact": (b, 2),
        "gait_mask": (b, 2),
        "move_flag": (b,),
        "foot_height": (b,),
        "gf": (b, 11, 3),
        "bf": (b, 11, 3),
        "sdf": (b, 11),
        "navigation_rotation": (b, 3, 3),
    }
    expected.update(
        {k: (b, 29) for k in ("joint_pos", "joint_vel", "previous_vel", "torques")}
    )
    expected.update(
        {k: (b, 12) for k in ("actions", "last_actions", "last_last_actions")}
    )
    for name, shape in expected.items():
        v = getattr(x, name)
        if (
            v.shape != shape
            or v.device != x.root_vel.device
            or v.dtype != x.root_vel.dtype
            or not v.is_floating_point()
        ):
            raise ValueError(f"{name} has incompatible shape/device/precision.")
    if (
        x.leg_rotation.ndim != 4
        or x.leg_rotation.shape[0] != b
        or x.leg_rotation.shape[-2:] != (3, 3)
    ):
        raise ValueError("leg_rotation must be [B,N,3,3].")
    p = parameters
    lower = p.soft_lower.to(x.joint_pos)
    upper = p.soft_upper.to(x.joint_pos)
    pitch = x.torso_rpy[:, 1].clamp(-math.pi, 0).abs()
    # None preserves archived source rewards. New expert recipes supply a
    # geometry-derived permission, independent of the robot's head height.
    if ducking_allowance is None:
        forward_weight = (x.head_pos[:, 2] > p.chest_height + 0.1).to(x.root_vel)
    else:
        if (
            ducking_allowance.shape != (b,)
            or ducking_allowance.dtype != x.root_vel.dtype
            or ducking_allowance.device != x.root_vel.device
            or not torch.isfinite(ducking_allowance).all()
            or (ducking_allowance < 0).any()
            or (ducking_allowance > 1).any()
        ):
            raise ValueError(
                "Ducking allowance must be matching [B] values in [0,1]."
            )
        forward_weight = 1 - ducking_allowance
    ori = (
        x.pelvis_rpy[:, 0].abs()
        + x.torso_rpy[:, 0].abs()
        + pitch
        + forward_weight * x.torso_rpy[:, 1].abs()
    )
    cmd = x.cmd_vel[:, :2]
    norm = cmd.norm(dim=-1, keepdim=True)
    zero = torch.isclose(norm, torch.zeros_like(norm))
    direction = torch.where(zero, 0.0, cmd / norm.clamp_min(1e-30))
    orth = (
        x.root_vel[:, :2]
        - (x.root_vel[:, :2] * direction).sum(-1, keepdim=True) * direction
    )
    motion = 1.2 * torch.where(
        zero[:, 0], 0.0, orth.square().sum(-1)
    ) + 0.4 * x.torso_ang_vel[:, :2].abs().sum(-1)
    legs = x.navigation_rotation.transpose(-1, -2)[:, None] @ x.leg_rotation
    decay = (
        ((p.yaw_max + 1e-6 - x.cmd_vel[:, 2].abs()) / (p.yaw_max + 1e-6))
        .clamp(0, 1)
        .square()
    )
    axis = legs[:, :, 2, 1].abs().mean(-1) + decay * legs[:, :, 0, 1].abs().mean(-1)
    stance = (x.feet_contact != 0).to(x.root_vel.dtype)
    swing = 1 - stance
    contact = (
        (stance - (x.gait_mask == 1).to(stance)).abs()
        + (swing - (x.gait_mask == -1).to(swing)).abs()
    ) * (x.gait_mask != 0)
    target = p.foot_height_stance + x.foot_height[:, None]
    clearance = (
        (target - x.foot_pos[:, :, 2]).clamp_min(0).square() * (x.gait_mask == -1)
    ).sum(-1) * x.move_flag
    diff = x.foot_pos - x.pelvis_subtree_com[:, None]
    diff = (
        x.navigation_rotation.transpose(-1, -2)[:, None] @ diff[..., None]
    ).squeeze(-1)
    spread = (0.35 - (x.foot_pos[:, 0] - x.foot_pos[:, 1]).norm(dim=-1)).clamp_min(0)
    balance = diff[:, :, :2].sum(1).square().sum(-1) * (1 + 10 * spread)
    if standing_balance_free:
        balance = torch.where(x.move_flag <= 1e-6, 0.0, balance)
    v = x.joint_vel[:, OBS_INDICES]
    acc = (x.previous_vel[:, OBS_INDICES] - v) / p.dt
    act = (
        x.actions.square()
        + (x.actions - x.last_actions).square()
        + (x.actions - 2 * x.last_actions + x.last_last_actions).square()
    ).sum(-1)
    slip_mask = x.gait_mask == 1
    if standing_contact_slip:
        slip_mask = torch.where(
            (x.move_flag <= 1e-6)[:, None], x.feet_contact != 0, slip_mask
        )
    terms = dict(
        tracking_orientation=torch.exp(-0.5 * ori) - pitch,
        tracking_root_field=torch.exp(
            -4 * (cmd - x.root_vel[:, :2]).square().sum(-1)
        ),
        body_motion=motion,
        body_rotation=torch.exp(-5 * axis),
        foot_contact=contact.sum(-1) * x.move_flag,
        foot_clearance=clearance,
        foot_slip=(x.foot_vel.square().sum(-1) * slip_mask).sum(-1),
        foot_balance=balance,
        straight_knee=(0.1 - x.joint_pos[:, [3, 9]]).clamp_min(0).sum(-1),
        foot_far=spread,
        smoothness_joint=(0.01 * v.square() + acc.square()).sum(-1),
        smoothness_action=act,
        joint_limits=(
            (lower - x.joint_pos).clamp_min(0) + (x.joint_pos - upper).clamp_min(0)
        ).sum(-1),
        joint_torque=x.torques.square().sum(-1),
    )
    if whole_body_joint_velocity:
        terms["whole_body_joint_velocity"] = x.joint_vel.square().sum(-1)
    if landing_speed_rate is not None:
        if (
            landing_speed_rate.shape != (b,)
            or landing_speed_rate.device != x.root_vel.device
            or landing_speed_rate.dtype != x.root_vel.dtype
        ):
            raise ValueError(
                "Landing speed rate must match reward batch/device/dtype."
            )
        terms["foot_landing_speed"] = landing_speed_rate
    if contact_force_rate is not None:
        if (
            contact_force_rate.shape != (b,)
            or contact_force_rate.device != x.root_vel.device
            or contact_force_rate.dtype != x.root_vel.dtype
        ):
            raise ValueError(
                "Contact force rate must match reward batch/device/dtype."
            )
        terms["foot_contact_force"] = contact_force_rate
    velocities = x.body_velocities
    if velocities.shape != (b, 11, 3):
        raise ValueError("body_velocities must be [B,11,3].")
    for name, ids in [
        ("head", [0]),
        ("feet", [3, 4]),
        ("hands", [5, 6]),
        ("knees", [7, 8]),
        ("shlds", [9, 10]),
    ]:
        sdf = x.sdf[:, ids]
        terms[name + "df"] = -(20 * F.softplus((0.05 - sdf) / 0.02)).mean(-1)
        if name in ("head", "feet", "hands"):
            g = x.gf[:, ids]
            vel = velocities[:, ids]
            cosine = (
                (g / (g.norm(dim=-1, keepdim=True) + 1e-6))
                * (vel / (vel.norm(dim=-1, keepdim=True) + 1e-6))
            ).sum(-1)
            tau = 0.3 if name == "feet" else 0.5
            crossed = x.move_flag[:, None] < 0.5
            pos = (
                x.head_pos[:, None]
                if name == "head"
                else x.foot_pos
                if name == "feet"
                else x.hands_pos
            )
            if risk_alpha is None:
                crossed = crossed | (pos[:, :, 0] > 1.5)
            if name == "feet":
                crossed = crossed | (x.gait_mask == 1)
            point_reward = torch.where(
                crossed, 4.0, 5 * cosine * torch.sigmoid(40 * (tau - sdf))
            )
            if risk_alpha is not None:
                point_reward = 4.0 + risk_alpha[:, ids] * (point_reward - 4.0)
            terms[name + "gf"] = point_reward.mean(-1)
    if feet_progress_gate is not None:
        # Preserve the source stop reward and all negative directional costs.
        # Only positive feet guidance during commanded travel needs progress.
        moving = (x.move_flag > 0) & (x.cmd_vel[:, :2].norm(dim=-1) > 1e-6)
        gate = torch.where(moving, feet_progress_gate, 1.0)
        feet = terms["feetgf"]
        terms["feetgf"] = feet.clamp_max(0) + feet.clamp_min(0) * gate
    if diagnostics is not None:
        diagnostics["nonfinite_terms"] = torch.stack(
            [~torch.isfinite(v) for v in terms.values()]
        ).sum(0)
    return {k: torch.where(torch.isnan(v), 0.0, v) for k, v in terms.items()}
