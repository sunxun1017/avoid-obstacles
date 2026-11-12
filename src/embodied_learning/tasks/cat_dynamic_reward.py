"""Signed joint-task reward; failures and deadlines are genuine terminals."""

import math
import torch


def gait_cost(terms, scales, *, scale):
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("Frozen gait calibration scale must be at least one.")
    raw = 2*(1-terms["tracking_orientation"]).clamp_min(0)
    raw = raw + (1-terms["body_rotation"]).clamp_min(0)
    for key, value in terms.items():
        if key in ("tracking_orientation", "body_rotation", "tracking_root_field",
                   "standing_stability", "corridor_boundary") or key.endswith("gf"):
            continue
        # DF terms are negative with positive weights; regularizers have
        # positive raw magnitudes with negative weights.
        raw = raw + (-value*scales[key]).clamp_min(0)
    return raw, 2*raw/(scale+raw)


def joint_reward(*, delta_progress, gait_cost, clearance, margins, velocity_xy,
                 nominal_xy, risk, stalled, failed, success, timeout):
    dynamic = ((.10-clearance)/.10).clamp(0, 1).square()
    boundary = ((.10-margins.amin(-1))/.10).clamp(0, 1).square()
    speed = (1-risk)*((velocity_xy-nominal_xy).square().sum(-1)/.30**2).clamp(max=1)
    cost = (gait_cost+2*dynamic+boundary+.3*speed+.5*stalled+.1).clamp(0, 5)
    event = torch.where(failed, -25., torch.where(success, 10., torch.where(timeout, -10., 0.)))
    reward = 2*delta_progress-.02*cost+event
    return reward, dict(joint_gait=gait_cost, joint_dynamic=dynamic,
                       joint_boundary=boundary, joint_speed=speed,
                       joint_stall=stalled.float(), joint_cost=cost, joint_event=event)
