"""Declared existing smoothness costs; original defaults remain unchanged."""

import copy
import math
from pathlib import Path
from typing import Any

from embodied_learning.tasks.cat_foot_guidance import sha256

DEFAULTS = {"smoothness_action": -0.001, "smoothness_joint": -1e-6}


def validate_cost(value: float) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not -3.4028234663852886e38 <= value < 0
    ):
        raise ValueError(
            "Smoothness cost must be finite, negative and representable in float32."
        )
    return value


def apply_joint_smoothness(env: Any, protocol: dict[str, Any]) -> None:
    key = "reward_smoothness_joint_weight"
    if key in protocol:
        weight = validate_cost(protocol[key])
        env.cat_state.scales["smoothness_joint"] = weight
        env.schema[key] = weight


def adapt_smoothness_transfer(
    checkpoint: str | Path,
    old: dict[str, Any],
    target: dict[str, Any],
    source: dict[str, Any],
    curriculum: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    declaration = curriculum.get("smoothness_refinement")
    if declaration is None or (
        source["experiment"] == curriculum["experiment"]
        and source["protocol_sha256"] == curriculum["protocol_sha256"]
    ):
        return old, source, {}
    before = {
        k: validate_cost(old["schema"].get("reward_" + k + "_weight", v))
        for k, v in DEFAULTS.items()
    }
    after = {
        k: validate_cost(target["schema"].get("reward_" + k + "_weight", v))
        for k, v in DEFAULTS.items()
    }
    expected = dict(
        version="cat_smoothness_actor_transfer_v1",
        parent_sha256=sha256(checkpoint),
        parent_experiment=source["experiment"],
        parent_protocol_sha256=source["protocol_sha256"],
        before=before,
        after=after,
    )
    if declaration != expected or source["stage"] != curriculum["stage"]:
        raise ValueError("Smoothness refinement parent/declaration/stage mismatch.")
    old = copy.deepcopy(old)
    changes = {}
    for name in DEFAULTS:
        key = "reward_" + name + "_weight"
        if key in target["schema"]:
            old["schema"][key] = after[name]
        else:
            old["schema"].pop(key, None)
        if before[name] != after[name]:
            changes[key] = dict(before=before[name], after=after[name])
    source = {
        **source,
        "experiment": curriculum["experiment"],
        "protocol_sha256": curriculum["protocol_sha256"],
    }
    return old, source, changes
