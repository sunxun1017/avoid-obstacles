"""Smoothness refinement cannot bypass unrelated compatibility checks."""

import copy
from types import SimpleNamespace

import pytest
import torch
from test_cat_curriculum import fixture_contract, fixture_metadata

from embodied_learning.policies.cat_actor_warmstart import load_actor_warmstart
from embodied_learning.tasks.cat_foot_guidance import sha256


def trial(tmp_path, after=-0.005):
    old = fixture_contract(4)
    old["schema"]["reward_smoothness_action_weight"] = -0.001
    saved = dict(
        field_contract=old,
        curriculum=fixture_metadata(4),
        training_scenes=[2],
        actor_state_dict={"w": torch.ones(3)},
        completed_iterations=128,
    )
    p = tmp_path / "parent.pt"
    torch.save(saved, p)
    target = copy.deepcopy(old)
    target["schema"]["reward_smoothness_action_weight"] = after
    target["schema"]["reward_smoothness_joint_weight"] = -5e-6
    meta = {
        **fixture_metadata(4),
        "experiment": "trial",
        "protocol_sha256": "trial-protocol",
        "smoothness_refinement": dict(
            version="cat_smoothness_actor_transfer_v1",
            parent_sha256=sha256(p),
            parent_experiment="test",
            parent_protocol_sha256="protocol",
            before={"smoothness_action": -0.001, "smoothness_joint": -1e-6},
            after={"smoothness_action": after, "smoothness_joint": -5e-6},
        ),
    }
    return p, saved, target, meta


def load(p, saved, target, meta):
    return load_actor_warmstart(
        p,
        contract=SimpleNamespace(to_dict=lambda: target),
        curriculum=meta,
        training_scenes=saved["training_scenes"],
        device="cpu",
    )


@pytest.mark.parametrize("after", [-0.001, -0.005])
def test_explicit_same_stage_transfer_preserves_saved_model(tmp_path, after):
    p, saved, target, meta = trial(tmp_path, after)
    actual, info = load(p, saved, target, meta)
    assert actual["field_contract"] == saved["field_contract"]
    assert not info["critic_restored"] and not info["optimizer_restored"]
    expected = {"reward_smoothness_joint_weight": {"before": -1e-6, "after": -5e-6}}
    if after != -0.001:
        expected["reward_smoothness_action_weight"] = {
            "before": -0.001,
            "after": after,
        }
    assert info["contract_changes"] == expected


@pytest.mark.parametrize(
    "change",
    [
        "sha",
        "before",
        "after",
        "stage",
        "geometry",
        "push",
        "robot",
        "missing",
        "extra",
    ],
)
def test_refinement_rejects_other_changes(tmp_path, change):
    p, saved, target, meta = trial(tmp_path)
    d = meta["smoothness_refinement"]
    if change == "sha":
        d["parent_sha256"] = "wrong"
    if change == "before":
        d["before"] = -0.007
    if change == "after":
        d["after"] = -0.002
    if change == "stage":
        meta["stage"] = 3
    if change == "geometry":
        target["schema"]["scene_hash"] = "changed"
    if change == "push":
        target["schema"]["task"]["push"] = False
    if change == "robot":
        meta["robot_fingerprint"] = "wrong"
    if change == "missing":
        meta.pop("smoothness_refinement")
    if change == "extra":
        d["extra"] = 1
    with pytest.raises(ValueError):
        load(p, saved, target, meta)


@pytest.mark.parametrize("after", [0, 1, True, float("nan"), float("inf"), -1e100])
def test_refinement_rejects_invalid_cost(tmp_path, after):
    p, saved, target, meta = trial(tmp_path, after)
    with pytest.raises(ValueError):
        load(p, saved, target, meta)


def test_full_resume_still_rejects_weight_change(tmp_path):
    from embodied_learning.tasks.cat_curriculum import validate_transfer

    p, saved, target, meta = trial(tmp_path)
    with pytest.raises(ValueError):
        validate_transfer(
            saved["field_contract"], target, saved["curriculum"], saved["curriculum"]
        )


def test_runtime_only_changes_two_costs():
    from embodied_learning.tasks.cat_hurdle import apply_runtime_settings

    env = SimpleNamespace(
        unwrapped=SimpleNamespace(cat_controller=object()),
        cat_state=SimpleNamespace(
            scales={
                "smoothness_action": -0.001,
                "smoothness_joint": -1e-6,
                "body_motion": -0.5,
            }
        ),
        schema={},
        device="cpu",
    )
    apply_runtime_settings(
        env,
        {
            "reward_smoothness_action_weight": -0.005,
            "reward_smoothness_joint_weight": -5e-6,
        },
    )
    assert env.cat_state.scales == {
        "smoothness_action": -0.005,
        "smoothness_joint": -5e-6,
        "body_motion": -0.5,
    }
    assert env.schema["reward_smoothness_joint_weight"] == -5e-6


@pytest.mark.parametrize("value", [0, 1, True, float("nan"), float("inf"), -1e100])
def test_runtime_joint_weight_rejects_invalid_values(value):
    from embodied_learning.tasks.cat_hurdle import apply_runtime_settings

    env = SimpleNamespace(
        unwrapped=SimpleNamespace(cat_controller=object()),
        cat_state=SimpleNamespace(scales={}),
        schema={},
        device="cpu",
    )
    with pytest.raises(ValueError):
        apply_runtime_settings(env, {"reward_smoothness_joint_weight": value})
