"""Explicit parent-bound transfer for the standing foot-slip ablation."""

import copy

from embodied_learning.tasks.cat_foot_guidance import sha256


def adapt_standing_slip_transfer(checkpoint, old, target, source, curriculum):
    declaration = curriculum.get("standing_slip_refinement")
    if declaration is None or (
        source["experiment"] == curriculum["experiment"]
        and source["protocol_sha256"] == curriculum["protocol_sha256"]
    ):
        return old, source, {}
    before = old["schema"].get("standing_slip_mode", "phase_mask_v1")
    after = target["schema"].get("standing_slip_mode", "phase_mask_v1")
    expected = dict(
        version="cat_standing_slip_actor_transfer_v1",
        parent_sha256=sha256(checkpoint),
        parent_experiment=source["experiment"],
        parent_protocol_sha256=source["protocol_sha256"],
        before=before,
        after=after,
    )
    if (
        declaration != expected
        or source["stage"] != curriculum["stage"]
        or before != "phase_mask_v1"
        or after not in ("phase_mask_v1", "ground_contact_v1")
    ):
        raise ValueError(
            "Standing slip refinement parent/declaration/stage mismatch."
        )
    old = copy.deepcopy(old)
    if after == "ground_contact_v1":
        old["schema"]["standing_slip_mode"] = after
    else:
        old["schema"].pop("standing_slip_mode", None)
    source = {
        **source,
        "experiment": curriculum["experiment"],
        "protocol_sha256": curriculum["protocol_sha256"],
    }
    changes = (
        {"standing_slip_mode": dict(before=before, after=after)}
        if before != after
        else {}
    )
    return old, source, changes
