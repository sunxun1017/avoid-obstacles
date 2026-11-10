"""Explicit CAT stages and narrowly checked initialization/evaluation transfers."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from embodied_learning.evaluation.cat_obstacle_expert import EXPERT_PROFILE

ROBUST_EXPERT_PROFILE = "cat_obstacle_expert_v2"
ANCHOR_SCENES = ("anchor_flat", "anchor_hurdle", "anchor_ceiling")

# Full bar height/depth, beam underside, free horizontal gap, all in meters.
LEVELS = (
    (0.08, 0.16, 1.40, 0.85),
    (0.12, 0.24, 1.20, 0.70),
    (0.18, 0.32, 1.10, 0.55),
    (0.24, 0.40, 1.06, 0.45),
)


def stage_settings(original, *, profile="legacy_combination_v1"):
    """Keep the bar front at x=.4; move the beam away at easier levels."""
    scene = next(s for s in original["scenes"] if s["id"] == "hurdle_crouch")
    expected = [
        {"center": [-0.3, 0.0, 1.18], "half_size": [0.25, 0.9, 0.12]},
        {"center": [0.6, 0.0, 0.12], "half_size": [0.2, 0.5, 0.12]},
    ]
    if scene["boxes"] != expected:
        raise ValueError("Curriculum requires the verified original geometry.")
    if profile == ROBUST_EXPERT_PROFILE:
        return robust_expert_stages(original)
    stages = []
    for i, (height, depth, underside, gap) in enumerate(LEVELS):
        cfg = copy.deepcopy(original)
        target = next(s for s in cfg["scenes"] if s["id"] == "hurdle_crouch")
        if i < len(LEVELS) - 1:
            beam, bar = target["boxes"]
            beam["center"] = [
                round(0.4 - gap - 0.25, 8),
                0.0,
                round(underside + 0.12, 8),
            ]
            bar["center"] = [round(0.4 + depth / 2, 8), 0.0, height / 2]
            bar["half_size"] = [depth / 2, 0.5, height / 2]
        stages.append(cfg)
    if profile == "legacy_combination_v1":
        return stages
    if profile not in ("foundation_v1", "standing_recovery_v1", EXPERT_PROFILE):
        raise ValueError(f"Unknown curriculum profile: {profile}")
    standing = profile == "standing_recovery_v1"
    foundations = []
    dimensions = ((0.0, 0.0), (0.04, 0.08), (0.08, 0.16))
    if standing:
        dimensions = dimensions[:1]
    for height, depth in dimensions:
        cfg = copy.deepcopy(original)
        target = next(s for s in cfg["scenes"] if s["id"] == "hurdle_crouch")
        target["boxes"] = (
            []
            if not height
            else [
                dict(
                    center=[0.0, 0.0, height / 2],
                    half_size=[depth / 2, 0.5, height / 2],
                )
            ]
        )
        length = 0.8 if height else 0.5
        target["starts"] = [[length if standing else -length, 0.0, 0.8]]
        target["goals"] = [[length, 0.0, 0.8]]
        # Source train_batch's empty task disables field reward coefficients.
        # Isolated hurdles activate only the ground terms.
        cfg["task"].update(ground=1.0 if height else 0.0, lateral=0.0, overhead=0.0)
        foundations.append(cfg)
    if profile == EXPERT_PROFILE:
        ceiling = copy.deepcopy(stages[1])
        scene = next(s for s in ceiling["scenes"] if s["id"] == "hurdle_crouch")
        scene["boxes"] = [scene["boxes"][0]]
        ceiling["task"].update(ground=0.0, lateral=0.0, overhead=1.0)
        stages = [*foundations[1:], ceiling, stages[1]]
        for cfg in stages:
            target = next(s for s in cfg["scenes"] if s["id"] == "hurdle_crouch")
            target["starts"] = [[-1.5, -0.12, 0.8], [-1.5, 0.12, 0.8]]
            target["goals"] = [[1.85, 0.0, 0.8]]
        return stages
    return foundations if standing else foundations + stages


def robust_expert_stages(original):
    """Smaller geometry increments plus permanent rehearsal scenes."""
    from embodied_learning.tasks.cat_overhead import POSTURE_VERSION
    from embodied_learning.tasks.cat_traversal_safety import TRAVERSAL_VERSION

    def bar(height, depth):
        return dict(center=[round(.4 + depth / 2, 8), 0., height / 2],
                    half_size=[depth / 2, .5, height / 2])

    roof = dict(center=[-.55, 0., 1.32], half_size=[.25, .9, .12])
    layouts = [[bar(h, d)] for h, d in ((.04,.08), (.08,.16), (.10,.16), (.10,.24), (.12,.24))]
    layouts += [[roof]] + [[roof, bar(h, .24)] for h in (.08,.10,.12)]
    result = []
    for boxes in layouts:
        cfg = copy.deepcopy(original)
        main = next(s for s in cfg["scenes"] if s["id"] == "hurdle_crouch")
        main.update(boxes=copy.deepcopy(boxes), starts=[[-1.5,-.12,.8],[-1.5,.12,.8]], goals=[[1.85,0.,.8]])
        for name, anchor_boxes in zip(ANCHOR_SCENES, ([], [bar(.08,.16)], [roof]), strict=True):
            anchor = copy.deepcopy(main)
            anchor.update(id=name, boxes=copy.deepcopy(anchor_boxes))
            cfg["scenes"].append(anchor)
        cfg["task"].update(ground=1., lateral=1., overhead=1.,
                           posture_reward=POSTURE_VERSION, traversal_robustness=TRAVERSAL_VERSION)
        result.append(cfg)
    return result


def validate_transfer(
    old, new, source, target, *, evaluation=False, actor_warmstart=False
):
    """Keep resume in-stage; allow declared scene changes for Actor transfer."""
    for value in (old, new):
        if not isinstance(value.get("schema"), dict) or not isinstance(
            value["schema"].get("task", {}), dict
        ):
            raise ValueError("Invalid policy contract task/schema.")
    for key in ("experiment", "protocol_sha256", "robot_fingerprint"):
        if source[key] != target[key]:
            raise ValueError(f"Curriculum {key} mismatch.")
    if not evaluation and not actor_warmstart and target["stage"] != source["stage"]:
        raise ValueError("Resume must stay in the same stage; use Actor warmstart.")
    if not evaluation and target["stage"] not in (
        source["stage"],
        source["stage"] + 1,
    ):
        raise ValueError("Curriculum training must stay or advance one stage.")
    before, after = copy.deepcopy(old), copy.deepcopy(new)
    changes = {}
    if evaluation and after["schema"].get("execution_profile") == "author_playback":
        if (
            before["schema"].get("execution_profile", "source_train")
            != "source_train"
        ):
            raise ValueError("Playback adaptation requires a training checkpoint.")
        quiet = dict(noise_level=0.0, random_pd=False, random_rfi=False, push=False)
        if any(after["schema"]["task"].get(k) != v for k, v in quiet.items()):
            raise ValueError("Invalid author playback perturbation profile.")
        changes["execution_profile"] = dict(
            before="source_train", after="author_playback"
        )
        before["schema"]["task"].update(quiet)
        after["schema"].pop("execution_profile")
    if actor_warmstart or evaluation:
        for key in ("ground", "lateral", "overhead"):
            a = before["schema"].get("task", {}).pop(key, None)
            b = after["schema"].get("task", {}).pop(key, None)
            if a != b:
                if a not in (0, 1) or b not in (0, 1):
                    raise ValueError("Invalid stage field reward coefficient.")
                changes["task." + key] = {"before": a, "after": b}
    for key in ("scene_hash", "field_cache_identity", "robot_model_identity"):
        a, b = before["schema"].pop(key, None), after["schema"].pop(key, None)
        if a != b:
            changes[key] = {"before": a, "after": b}
    for key in ("base_cache_identity", "foot_manifest_identity"):
        a = before["schema"].get("foot_guidance_experiment", {}).pop(key, None)
        b = after["schema"].get("foot_guidance_experiment", {}).pop(key, None)
        if a != b:
            changes["foot_guidance_experiment." + key] = {"before": a, "after": b}
    if json.dumps(before, sort_keys=True) != json.dumps(after, sort_keys=True):
        raise ValueError("Non-geometry policy contract changed.")
    geometry_changes = set(changes) - {"execution_profile"}
    if source["stage"] == target["stage"] and geometry_changes:
        raise ValueError("Same curriculum stage changed its geometry contract.")
    return changes


def require_original_task_contract(reference, actual):
    """Validate the original physical task against the earlier 25 cm reference.

    The declared field-cache/foot-GF/interpolation and trainer differences
    are allowed; scene, robot, observations, action dimensions and reward
    settings must still agree exactly.
    """
    old, new = copy.deepcopy(reference), copy.deepcopy(actual)
    if new["schema"].get("optimization_profile") == "brax_0_12_3_v1":
        if (
            old.get("activation") != "elu"
            or new.get("activation") != "swish"
            or new["schema"].get("policy_activation") != "swish"
        ):
            raise ValueError("Invalid original-task trainer declaration.")
        new["schema"].pop("optimization_profile")
        new["schema"].pop("policy_activation")
        old["activation"] = new["activation"]
    for value in (old, new):
        for key in (
            "field_cache_identity",
            "foot_guidance_experiment",
            "source_interpolation",
        ):
            value["schema"].pop(key, None)
    if old != new:
        raise ValueError("Final original scene/task contract changed.")


def robot_fingerprint(model):
    """Hash robot dynamics separately from allowed world-geometry changes."""
    h = hashlib.sha256()
    for kind, count, attrs in (
        (
            "body",
            model.nbody,
            ("pos", "quat", "ipos", "iquat", "mass", "inertia", "gravcomp"),
        ),
        (
            "geom",
            model.ngeom,
            (
                "type",
                "size",
                "pos",
                "quat",
                "friction",
                "solref",
                "solimp",
                "contype",
                "conaffinity",
                "margin",
                "gap",
            ),
        ),
        (
            "jnt",
            model.njnt,
            ("type", "pos", "axis", "range", "limited", "stiffness"),
        ),
    ):
        lookup = model.joint if kind == "jnt" else getattr(model, kind)
        ids = [i for i in range(count) if lookup(i).name.startswith("robot/")]
        h.update(json.dumps([lookup(i).name for i in ids]).encode())
        for attr in attrs:
            h.update(
                np.ascontiguousarray(getattr(model, f"{kind}_{attr}")[ids]).tobytes()
            )
    for attr in (
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "actuator_gear",
        "actuator_trnid",
        "actuator_ctrlrange",
        "actuator_forcerange",
    ):
        h.update(np.ascontiguousarray(getattr(model, attr)).tobytes())
    for name in sorted(dir(model.opt)):
        value = getattr(model.opt, name)
        if isinstance(value, (int, float)):
            h.update(f"opt.{name}={value!r}".encode())
        elif isinstance(value, np.ndarray):
            h.update(name.encode())
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()
