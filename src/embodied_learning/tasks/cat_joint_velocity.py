"""Opt-in actual velocity regularization over all29 CAT joints."""

import copy
import math

from embodied_learning.tasks.cat_foot_guidance import sha256

KEY = "whole_body_joint_velocity"


def validate_weight(weight):
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("Whole-body velocity weight must be positive and finite.")
    return float(weight)


def validate_schema(value):
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "weight"}
        or value["version"] != "sum_dq2_29_v1"
    ):
        raise ValueError("Invalid whole-body velocity schema.")
    return dict(version="sum_dq2_29_v1", weight=validate_weight(value["weight"]))


def apply_joint_velocity(env, protocol):
    if "whole_body_joint_velocity_weight" not in protocol:
        return
    weight = validate_weight(protocol["whole_body_joint_velocity_weight"])
    if getattr(env.cat_state, "recovery", None) is not None:
        raise ValueError("Whole-body velocity pilot requires source reward mode.")
    env.cat_state.whole_body_joint_velocity = True
    env.cat_state.scales[KEY] = -weight
    env.schema[KEY] = dict(version="sum_dq2_29_v1", weight=weight)


def adapt_velocity_transfer(checkpoint, old, target, source, curriculum):
    declaration = curriculum.get("whole_body_velocity_refinement")
    if declaration is None or (
        source["experiment"] == curriculum["experiment"]
        and source["protocol_sha256"] == curriculum["protocol_sha256"]
    ):
        return old, source, {}
    before = validate_schema(old["schema"].get(KEY))
    after = validate_schema(target["schema"].get(KEY))
    expected = dict(
        version="cat_whole_body_velocity_actor_transfer_v1",
        parent_sha256=sha256(checkpoint),
        parent_experiment=source["experiment"],
        parent_protocol_sha256=source["protocol_sha256"],
        before=before,
        after=after,
    )
    if declaration != expected or source["stage"] != curriculum["stage"]:
        raise ValueError(
            "Whole-body velocity transfer parent/declaration/stage mismatch."
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
        ({KEY: dict(before=before, after=after)} if before != after else {}),
    )
