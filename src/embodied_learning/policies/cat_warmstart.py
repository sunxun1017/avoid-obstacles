"""Explicit, narrow source-to-recovery initialization; never a resume path."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch

from embodied_learning.protocols.imitation import PolicyContract


def load_cat_warmstart(
    checkpoint: Path,
    *,
    expected_contract: PolicyContract,
    training_scenes: list[int],
    device: str,
) -> tuple[dict, dict]:
    """Validate one supported migration and record the exact trusted parent bytes.

    Task parameters may intentionally differ for research. Physical assets,
    ordering, action semantics and network architecture must remain identical.
    Normalized checkpoints need a separate, explicitly designed migration.
    """
    payload = Path(checkpoint).read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location=device, weights_only=False)
    try:
        parent = PolicyContract.from_dict(saved["field_contract"])
        child = expected_contract
        if (
            parent.schema.get("task_profile") != "cat_source_v1"
            or child.schema.get("task_profile") != "cat_recovery_v1"
            or parent.schema.get("observation") != "cat_source_actor162_critic250_v1"
            or child.schema.get("observation")
            != "cat_recovery_actor171_critic256_v1"
            or (parent.actor_dim, parent.critic_dim, parent.action_dim)
            != (162, 250, 12)
            or (child.actor_dim, child.critic_dim, child.action_dim)
            != (171, 256, 12)
            or parent.normalization is not False
            or child.normalization is not False
            or parent.schema.get("normalization") is not False
            or child.schema.get("normalization") is not False
            or parent.schema.get("action_transform") != "latent_tanh_v1"
            or parent.schema.get("source_profile") != {"version": "cat_source_v1"}
            or saved["training_scenes"] != training_scenes
            or "recovery" in parent.schema
            or not isinstance(child.schema.get("recovery"), dict)
        ):
            raise ValueError(
                "Warmstart only supports matching source-to-recovery PPO."
            )
        required = (
            "field_cache_identity",
            "robot_model_identity",
            "scene_hash",
            "scene_names",
            "body_sites",
            "joint_names",
            "action_names",
            "obs_joint_names",
            "physics",
        )
        if any(not parent.schema.get(key) for key in required):
            raise ValueError("Warmstart parent lacks physical identity or ordering.")
        source, target = parent.to_dict(), child.to_dict()
        for contract in (source, target):
            contract.pop("actor_dim")
            contract.pop("critic_dim")
            for key in ("task_profile", "observation", "task", "recovery"):
                contract["schema"].pop(key, None)
        if source != target:
            raise ValueError("Warmstart physical/schema/network contract mismatch.")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(
            "Warmstart checkpoint contract is missing or invalid."
        ) from error
    provenance = dict(
        migration="cat_source162_250_to_recovery171_256_v1",
        parent_checkpoint=str(Path(checkpoint).resolve()),
        parent_sha256=hashlib.sha256(payload).hexdigest(),
        parent_contract=parent.to_dict(),
        parent_ppo_config=saved.get("ppo_config"),
        parent_completed_iterations=saved.get("completed_iterations"),
        optimizer_restored=False,
    )
    return saved, provenance


def initialize_cat_warmstart(actor, critic, saved: dict) -> None:
    """Copy both networks, adding zero columns only to their input layers."""
    migrated = []
    for model, group, old_width, new_width in (
        (actor, "actor", 162, 171),
        (critic, "critic", 250, 256),
    ):
        target = model.state_dict()
        source = saved.get(group + "_state_dict")
        if not isinstance(source, dict) or set(source) != set(target):
            raise ValueError(f"Warmstart {group} state keys mismatch.")
        expanded = {}
        for key, value in source.items():
            expected = target[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != expected.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Warmstart {group}.{key} has invalid tensor data.")
            if key == "mlp.0.weight":
                if (
                    value.ndim != 2
                    or expected.ndim != 2
                    or value.shape != (expected.shape[0], old_width)
                    or expected.shape[1] != new_width
                ):
                    raise ValueError(f"Warmstart {group} input layer mismatch.")
                expanded[key] = torch.zeros_like(expected)
                expanded[key][:, :old_width] = value
            elif value.shape != expected.shape:
                raise ValueError(f"Warmstart {group}.{key} shape mismatch.")
            else:
                expanded[key] = value
        migrated.append((model, expanded))
    # Validate both models before modifying either one.
    for model, state in migrated:
        model.load_state_dict(state, strict=True)
