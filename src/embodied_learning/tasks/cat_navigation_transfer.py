"""SHA-bound Actor-only transfer for the declared navigation pilot."""

import copy
import hashlib
import json

from embodied_learning.tasks.cat_foot_guidance import sha256

PATHS = (
    "task.arrival_control",
    "reward_smoothness_action_weight",
    "reward_smoothness_joint_weight",
    "scene_hash",
    "field_cache_identity",
    "robot_model_identity",
    "foot_guidance_experiment.base_cache_identity",
    "foot_guidance_experiment.foot_manifest_identity",
)


def value(schema, path):
    for key in path.split("."):
        schema = schema.get(key, {})
    return schema


def navigation_changes(old, new):
    return {
        p: dict(before=value(old["schema"], p), after=value(new["schema"], p))
        for p in PATHS
        if value(old["schema"], p) != value(new["schema"], p)
    }


def verify_scene_pool_proof(proof, old, target):
    if not isinstance(proof, dict) or set(proof) != {"before", "after"}:
        return False
    stripped = []
    for name, contract in [("before", old), ("after", target)]:
        scene = proof[name]
        digest = hashlib.sha256(
            json.dumps(scene, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != contract["schema"]["scene_hash"]:
            return False
        scene = copy.deepcopy(scene)
        if set(scene) != {"scenes"} or not isinstance(scene["scenes"], list):
            return False
        for item in scene["scenes"]:
            if not all(key in item for key in ("starts", "goals")):
                return False
            item.pop("starts")
            item.pop("goals")
        stripped.append(scene)
    return stripped[0] == stripped[1]


def adapt_navigation_transfer(checkpoint, old, target, source, curriculum):
    declaration = curriculum.get("navigation_refinement")
    if declaration is None or (
        source["experiment"] == curriculum["experiment"]
        and source["protocol_sha256"] == curriculum["protocol_sha256"]
    ):
        return old, source, {}
    changes = navigation_changes(old, target)
    expected = dict(
        version="cat_navigation_actor_transfer_v1",
        parent_sha256=sha256(checkpoint),
        parent_experiment=source["experiment"],
        parent_protocol_sha256=source["protocol_sha256"],
        changes=changes,
        scene_pool_proof=declaration.get("scene_pool_proof"),
    )
    if (
        declaration != expected
        or not verify_scene_pool_proof(
            declaration.get("scene_pool_proof"), old, target
        )
        or source["stage"] != curriculum["stage"]
        or source["robot_fingerprint"] != curriculum["robot_fingerprint"]
        or old["schema"]["task"].get("arrival_control") != "root_linear_v1"
        or target["schema"]["task"].get("arrival_control")
        not in ("root_linear_v1", "root_planar_blend_v2")
        or target["schema"].get("reward_smoothness_action_weight") != -0.001
        or target["schema"].get("reward_smoothness_joint_weight") != -1e-6
    ):
        raise ValueError("Navigation transfer parent/declaration/contract mismatch.")
    adapted = copy.deepcopy(old)
    for path, change in changes.items():
        node = adapted["schema"]
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = copy.deepcopy(change["after"])
    return (
        adapted,
        {
            **source,
            "experiment": curriculum["experiment"],
            "protocol_sha256": curriculum["protocol_sha256"],
        },
        changes,
    )
