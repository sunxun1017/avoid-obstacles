"""Explicit inference-only field substitution; training contracts stay strict."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from embodied_learning.policies.field_policy import load_student, load_teacher
from embodied_learning.protocols.imitation import PolicyContract


def online_inference_contract(
    online: PolicyContract, saved: PolicyContract
) -> PolicyContract:
    """Validate everything except the deliberately replaced field/sensor assets."""
    schema = dict(online.schema)
    perception = schema.pop("online_perception", {})
    if (
        perception.get("version") != "cat_online_v1"
        or schema.get("task_profile") != "cat_source_v1"
        or schema.get("field_cache_identity") is not None
        or not isinstance(saved.schema.get("field_cache_identity"), str)
        or not saved.schema["field_cache_identity"]
        or (online.actor_dim, online.critic_dim, online.action_dim) != (162, 250, 12)
    ):
        raise ValueError(
            "Expected a CAT training checkpoint and online inference contract."
        )
    schema["field_cache_identity"] = saved.schema["field_cache_identity"]
    schema["robot_model_identity"] = perception.get("sensorless_model_identity")
    if not schema["robot_model_identity"]:
        raise ValueError("Missing sensorless model identity.")
    expected = replace(online, schema=schema)
    expected.require(saved)
    return expected


def load_online_policy(
    checkpoint: str | Path,
    *,
    online_contract: PolicyContract,
    device: str | torch.device,
) -> Any:
    """Load a trusted project CAT checkpoint after validating the online boundary."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actual = PolicyContract.from_dict(saved["field_contract"])
    expected = online_inference_contract(online_contract, actual)
    loader = load_student if "student" in saved else load_teacher
    return loader(checkpoint, expected_contract=expected, device=device)
