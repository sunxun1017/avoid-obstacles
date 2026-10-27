"""Explicit, parent-checked Actor transfer when CAT PD gains change."""

import copy
import math

from embodied_learning.robots.g1 import CAT_JOINT_NAMES
from embodied_learning.tasks.cat_foot_guidance import sha256


def validate_pd(pd):
    if (
        not isinstance(pd, dict)
        or set(pd) != {"version", "joint_names", "kp", "kd"}
        or not isinstance(pd["version"], str)
        or not pd["version"]
        or pd["joint_names"] != list(CAT_JOINT_NAMES)
    ):
        raise ValueError("Invalid controller PD schema or joint order.")
    for key in ("kp", "kd"):
        values = pd[key]
        if (
            not isinstance(values, list)
            or len(values) != 29
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in values
            )
        ):
            raise ValueError("Controller PD requires29positive finite gains.")
    return copy.deepcopy(pd)


def adapt_pd_transfer(checkpoint, old, target, source, curriculum):
    declaration = curriculum.get("controller_pd_transfer")
    if declaration is None or (
        source["experiment"] == curriculum["experiment"]
        and source["protocol_sha256"] == curriculum["protocol_sha256"]
    ):
        return old, source, {}
    before = validate_pd(old["schema"].get("controller_pd"))
    after = validate_pd(target["schema"].get("controller_pd"))
    expected = dict(
        version="cat_controller_pd_actor_transfer_v1",
        parent_sha256=sha256(checkpoint),
        parent_experiment=source["experiment"],
        parent_protocol_sha256=source["protocol_sha256"],
        source_stage=source["stage"],
        target_stage=curriculum["stage"],
        before=before,
        after=after,
    )
    if declaration != expected or curriculum["stage"] not in (
        source["stage"],
        source["stage"] + 1,
    ):
        raise ValueError("Controller PD transfer parent/declaration/stage mismatch.")
    old = copy.deepcopy(old)
    old["schema"]["controller_pd"] = after
    source = {
        **source,
        "experiment": curriculum["experiment"],
        "protocol_sha256": curriculum["protocol_sha256"],
    }
    changes = (
        {"controller_pd": dict(before=before, after=after)}
        if before != after
        else {}
    )
    return old, source, changes
