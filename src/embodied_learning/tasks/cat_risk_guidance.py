"""Opt-in continuous body-point collision risk, guidance and Actor transfer."""

import copy
import math

import torch
from torch.nn import functional as F

from embodied_learning.tasks.cat_foot_guidance import sha256

KEY = "risk_guidance"


def validate_schema(value):
    if value is None:
        return None
    keys = {"version", "near_m", "far_m", "margin_m", "horizon_s", "free_gain"}
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["version"] != "sdf_predictive_blend_v1"
    ):
        raise ValueError("Invalid risk guidance schema.")
    result = dict(value)
    for k in keys - {"version"}:
        v = value[k]
        if (
            type(v) not in (float, int)
            or not math.isfinite(v)
            or not 0 <= v <= torch.finfo(torch.float32).max
        ):
            raise ValueError(
                "Risk guidance parameters must be finite nonnegative float32 values."
            )
        result[k] = float(v)
    if result["far_m"] <= result["near_m"] or not 0 < result["free_gain"] <= 1:
        raise ValueError("Risk guidance requires far > near and 0 < free_gain <= 1.")
    if (
        float(torch.tensor(result["far_m"] - result["near_m"], dtype=torch.float32))
        <= 0
    ):
        raise ValueError("Risk guidance transition width underflowed.")
    return result


def risk_weights(sdf, gradient, velocity, predicted_sdf, config):
    """First-order conservative point risk, not a certified swept envelope."""
    cfg = validate_schema(config)
    if sdf.ndim != 2 or not sdf.is_floating_point():
        raise ValueError("Risk distance must be floating [B,K].")
    for x, shape in (
        (gradient, (*sdf.shape, 3)),
        (velocity, (*sdf.shape, 3)),
        (predicted_sdf, sdf.shape),
    ):
        if x.shape != shape or x.dtype != sdf.dtype or x.device != sdf.device:
            raise ValueError("Risk query shapes/device/dtype must match.")
    invalid = (
        torch.isnan(sdf)
        | torch.isneginf(sdf)
        | torch.isnan(predicted_sdf)
        | torch.isneginf(predicted_sdf)
        | ~torch.isfinite(gradient).all(-1)
        | ~torch.isfinite(velocity).all(-1)
    )
    normal = F.normalize(torch.nan_to_num(gradient), dim=-1, eps=1e-6)
    closing = -(normal * torch.nan_to_num(velocity)).sum(-1)
    effective = (
        torch.minimum(sdf - cfg["horizon_s"] * closing.clamp_min(0), predicted_sdf)
        - cfg["margin_m"]
    )
    z = ((effective - cfg["near_m"]) / (cfg["far_m"] - cfg["near_m"])).clamp(0, 1)
    alpha = 1 - z.square() * (3 - 2 * z)
    return torch.where(invalid, 1.0, alpha)


def blend_guidance(safe_guidance, command, alpha, config, *, walk_direction=None):
    """Keep a walking direction; preserve risk-related magnitude after mixing."""
    cfg = validate_schema(config)
    if (
        safe_guidance.ndim != 3
        or safe_guidance.shape[-1] != 3
        or alpha.shape != safe_guidance.shape[:2]
        or command.shape != (len(alpha), 4)
    ):
        raise ValueError("Guidance/command/risk shapes must match.")
    for x in (command, alpha):
        if x.dtype != safe_guidance.dtype or x.device != safe_guidance.device:
            raise ValueError("Guidance precision/device mismatch.")
    if not torch.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError("Risk weights must lie in [0,1].")
    walk = torch.zeros_like(safe_guidance)
    walk[..., :2] = F.normalize(command[:, 1:3], dim=-1, eps=1e-6)[:, None]
    if walk_direction is not None:
        if (
            walk_direction.shape != command[:, 1:].shape
            or walk_direction.dtype != safe_guidance.dtype
            or walk_direction.device != safe_guidance.device
            or not torch.isfinite(walk_direction).all()
            or (walk_direction.norm(dim=-1) > 1 + 1e-5).any()
        ):
            raise ValueError("Transition walking direction must be bounded [B,3].")
        walk = walk_direction[:, None].expand_as(safe_guidance)
    a = alpha[..., None]
    gain = cfg["free_gain"] + (1 - cfg["free_gain"]) * a
    # Deliberately no normalization here: amplitude must reach the Actor.
    return gain * ((1 - a) * walk + a * safe_guidance)


def adapt_risk_transfer(checkpoint, old, target, source, curriculum):
    declaration = curriculum.get("risk_guidance_refinement")
    if declaration is None or (
        source["experiment"] == curriculum["experiment"]
        and source["protocol_sha256"] == curriculum["protocol_sha256"]
    ):
        return old, source, {}
    before = validate_schema(old["schema"].get(KEY))
    after = validate_schema(target["schema"].get(KEY))
    expected = dict(
        version="cat_risk_guidance_actor_transfer_v1",
        parent_sha256=sha256(checkpoint),
        parent_experiment=source["experiment"],
        parent_protocol_sha256=source["protocol_sha256"],
        before=before,
        after=after,
    )
    if declaration != expected or source["stage"] != curriculum["stage"]:
        raise ValueError(
            "Risk guidance refinement parent/declaration/stage mismatch."
        )
    old = copy.deepcopy(old)
    if after is None:
        old["schema"].pop(KEY, None)
    else:
        old["schema"][KEY] = after
    source = {
        **source,
        "experiment": curriculum["experiment"],
        "protocol_sha256": curriculum["protocol_sha256"],
    }
    return (
        old,
        source,
        {KEY: dict(before=before, after=after)} if before != after else {},
    )
