"""Same-interface curriculum Actor transfer, distinct from recovery migration."""

import copy
import math
from pathlib import Path

import torch

from embodied_learning.tasks.cat_curriculum import validate_transfer
from embodied_learning.tasks.cat_foot_guidance import sha256


def load_actor_warmstart(
    checkpoint, *, contract, curriculum, training_scenes, device
):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if curriculum is None or "curriculum" not in saved:
        raise ValueError("Actor warmstart requires declared curriculum metadata.")
    old, source = saved["field_contract"], saved["curriculum"]
    target = contract.to_dict()
    from embodied_learning.tasks.cat_joint_velocity import adapt_velocity_transfer

    old, source, velocity_changes = adapt_velocity_transfer(
        checkpoint, old, target, source, curriculum
    )
    from embodied_learning.tasks.cat_field_weights import adapt_field_reward_transfer

    old, source, field_changes = adapt_field_reward_transfer(
        checkpoint, old, target, source, curriculum
    )
    refinement = curriculum.get("reward_weight_warmstart")
    refined = False
    progress = curriculum.get("feetgf_progress_refinement")
    progress_changed = False
    if progress is not None and (
        source["experiment"] != curriculum["experiment"]
        or source["protocol_sha256"] != curriculum["protocol_sha256"]
    ):
        expected = {
            "version": "cat_feetgf_progress_actor_transfer_v1",
            "parent_sha256": sha256(checkpoint),
            "parent_experiment": source["experiment"],
            "parent_protocol_sha256": source["protocol_sha256"],
            "before": None,
            "after": "root_displacement_0p5s_v1",
        }
        if (
            progress != expected
            or refinement is not None
            or source["stage"] != 1
            or curriculum["stage"] != 1
            or old["schema"]["task"].get("feetgf_progress_gate") is not None
            or target["schema"]["task"].get("feetgf_progress_gate")
            != expected["after"]
        ):
            raise ValueError("Feet progress refinement parent/contract mismatch.")
        old = copy.deepcopy(old)
        old["schema"]["task"]["feetgf_progress_gate"] = expected["after"]
        source = {
            **source,
            "experiment": curriculum["experiment"],
            "protocol_sha256": curriculum["protocol_sha256"],
        }
        progress_changed = True
    if refinement and (
        source["experiment"] != curriculum["experiment"]
        or source["protocol_sha256"] != curriculum["protocol_sha256"]
    ):
        expected_keys = {
            "version",
            "parent_sha256",
            "parent_experiment",
            "parent_protocol_sha256",
            "before",
            "after",
        }
        if (
            not isinstance(refinement, dict)
            or set(refinement) != expected_keys
            or refinement["version"] != "cat_root_tracking_weight_v1"
            or refinement["parent_sha256"] != sha256(checkpoint)
            or refinement["parent_experiment"] != source["experiment"]
            or refinement["parent_protocol_sha256"] != source["protocol_sha256"]
        ):
            raise ValueError("Reward refinement parent declaration mismatch.")
        before = old["schema"].get("task", {}).get("root_tracking_weight", 1.0)
        after = target["schema"].get("task", {}).get("root_tracking_weight")
        if (
            refinement["before"] != before
            or refinement["after"] != after
            or type(after) not in (int, float)
            or not math.isfinite(after)
            or after <= 0
        ):
            raise ValueError("Reward refinement tracking weight mismatch.")
        # Adapt compatibility checking only. The saved checkpoint, its real
        # training contract, Critic and optimizer are never rewritten/reused.
        old = copy.deepcopy(old)
        old["schema"].setdefault("task", {})["root_tracking_weight"] = after
        source = {
            **source,
            "experiment": curriculum["experiment"],
            "protocol_sha256": curriculum["protocol_sha256"],
        }
        refined = True
    changes = validate_transfer(
        old,
        target,
        source,
        curriculum,
        actor_warmstart=True,
    )
    changes.update(field_changes)
    changes.update(velocity_changes)
    if refined:
        changes["task.root_tracking_weight"] = {
            "before": before,
            "after": after,
        }
    if progress_changed:
        changes["task.feetgf_progress_gate"] = {
            "before": None,
            "after": "root_displacement_0p5s_v1",
        }
    if saved["training_scenes"] != training_scenes:
        raise ValueError("Actor warmstart training scene selection changed.")
    actor = saved.get("actor_state_dict")
    if (
        not isinstance(actor, dict)
        or not actor
        or any(
            not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
            for v in actor.values()
        )
    ):
        raise ValueError("Invalid Actor warmstart tensors.")
    return saved, dict(
        migration="cat_curriculum_actor_only_v1",
        parent_checkpoint=str(Path(checkpoint).resolve()),
        parent_sha256=sha256(checkpoint),
        parent_completed_iterations=saved["completed_iterations"],
        parent_curriculum=saved["curriculum"],
        contract_changes=changes,
        actor_restored=True,
        actor_normalizer_restored=True,
        critic_restored=False,
        optimizer_restored=False,
        rng_restored=False,
        counters_restored=False,
    )
