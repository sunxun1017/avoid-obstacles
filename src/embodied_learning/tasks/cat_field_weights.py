"""Declared independent CAT field weights and same-stage Actor transfer."""

import copy
import math

from embodied_learning.tasks.cat_foot_guidance import sha256

FIELD_TERMS = frozenset(
    (
        "headgf",
        "feetgf",
        "handsgf",
        "headdf",
        "feetdf",
        "handsdf",
        "kneesdf",
        "shldsdf",
    )
)


def validate_field_weights(weights):
    if not isinstance(weights, dict) or set(weights) - FIELD_TERMS:
        raise ValueError("Unknown field reward overrides.")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0
        for v in weights.values()
    ):
        raise ValueError("Field reward weights must be positive finite numbers.")
    return copy.deepcopy(weights)


def apply_field_weights(env, protocol):
    if "field_reward_overrides" not in protocol:
        return
    weights = validate_field_weights(protocol["field_reward_overrides"])
    if weights:
        env.cat_state.scales.update(weights)
        env.schema["field_reward_overrides"] = weights


def adapt_field_reward_transfer(checkpoint, old, target, source, curriculum):
    declaration = curriculum.get("field_reward_refinement")
    changes = {}
    if declaration is None or (
        source["experiment"] == curriculum["experiment"]
        and source["protocol_sha256"] == curriculum["protocol_sha256"]
    ):
        return old, source, changes
    before = validate_field_weights(old["schema"].get("field_reward_overrides", {}))
    after = validate_field_weights(
        target["schema"].get("field_reward_overrides", {})
    )
    expected = dict(
        version="cat_field_reward_actor_transfer_v1",
        parent_sha256=sha256(checkpoint),
        parent_experiment=source["experiment"],
        parent_protocol_sha256=source["protocol_sha256"],
        before=before,
        after=after,
    )
    if declaration != expected or source["stage"] != curriculum["stage"]:
        raise ValueError(
            "Field reward refinement parent/declaration/stage mismatch."
        )
    old = copy.deepcopy(old)
    if after:
        old["schema"]["field_reward_overrides"] = after
    else:
        old["schema"].pop("field_reward_overrides", None)
    source = {
        **source,
        "experiment": curriculum["experiment"],
        "protocol_sha256": curriculum["protocol_sha256"],
    }
    if before != after:
        changes["field_reward_overrides"] = dict(before=before, after=after)
    return old, source, changes
