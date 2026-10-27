"""Independent field weights must preserve unrelated policy contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch
from test_cat_curriculum import fixture_contract, fixture_metadata

from embodied_learning.policies.cat_actor_warmstart import load_actor_warmstart
from embodied_learning.tasks.cat_foot_guidance import sha256
from embodied_learning.tasks.cat_hurdle import apply_runtime_settings


@pytest.mark.parametrize("term", ["feetgf", "feetdf", "headgf", "headdf"])
def test_runtime_changes_only_the_requested_field_term(term):
    scales = {
        "feetgf": 1.0,
        "feetdf": 1.0,
        "headgf": 1.0,
        "headdf": 1.0,
        "tracking_root_field": 1.0,
        "smoothness_action": -0.001,
    }
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(cat_controller=object()),
        cat_state=SimpleNamespace(scales=scales.copy()),
        schema={},
        device="cpu",
    )
    apply_runtime_settings(env, {"field_reward_overrides": {term: 1.5}})
    assert env.cat_state.scales == {**scales, term: 1.5}
    assert env.schema["field_reward_overrides"] == {term: 1.5}


@pytest.mark.parametrize(
    "weights",
    [
        {"bad": 1.5},
        {"feetgf": 0},
        {"feetdf": -1},
        {"headgf": True},
        {"headgf": float("nan")},
        {"feetdf": float("inf")},
        [1.5],
    ],
)
def test_runtime_rejects_invalid_field_weights(weights):
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(cat_controller=object()),
        cat_state=SimpleNamespace(scales={}),
        schema={},
        device="cpu",
    )
    with pytest.raises(ValueError):
        apply_runtime_settings(env, {"field_reward_overrides": weights})


def trial(tmp_path, weights):
    old = fixture_contract(4)
    old["schema"]["task"].update(
        feetgf_progress_gate="root_displacement_0p5s_v1", push=True
    )
    saved = dict(
        field_contract=old,
        curriculum=fixture_metadata(4),
        training_scenes=[2],
        actor_state_dict={"w": torch.ones(3)},
        completed_iterations=480,
    )
    p = tmp_path / "parent.pt"
    torch.save(saved, p)
    target = copy.deepcopy(old)
    if weights:
        target["schema"]["field_reward_overrides"] = weights
    meta = {
        **fixture_metadata(4),
        "experiment": "trial",
        "protocol_sha256": "trial-protocol",
        "field_reward_refinement": dict(
            version="cat_field_reward_actor_transfer_v1",
            parent_sha256=sha256(p),
            parent_experiment="test",
            parent_protocol_sha256="protocol",
            before={},
            after=weights,
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


@pytest.mark.parametrize("weights", [{}, {"feetgf": 1.5}, {"feetdf": 1.5}])
def test_declared_same_stage_actor_transfer_allows_only_field_weights(
    tmp_path, weights
):
    p, saved, target, meta = trial(tmp_path, weights)
    actual, info = load(p, saved, target, meta)
    assert actual["field_contract"] == saved["field_contract"]
    assert not info["critic_restored"] and not info["optimizer_restored"]
    assert info["contract_changes"] == (
        {"field_reward_overrides": {"before": {}, "after": weights}}
        if weights
        else {}
    )


@pytest.mark.parametrize(
    "change",
    [
        "sha",
        "before",
        "after",
        "stage",
        "geometry",
        "push",
        "missing",
        "robot",
        "declaration_extra",
    ],
)
def test_reward_transfer_does_not_bypass_parent_and_physics_checks(tmp_path, change):
    p, saved, target, meta = trial(tmp_path, {"feetgf": 1.5})
    if change == "sha":
        meta["field_reward_refinement"]["parent_sha256"] = "wrong"
    if change == "before":
        meta["field_reward_refinement"]["before"] = {"feetgf": 2.0}
    if change == "after":
        meta["field_reward_refinement"]["after"] = {"feetdf": 1.5}
    if change == "stage":
        meta["stage"] = 5
    if change == "geometry":
        target["schema"]["scene_hash"] = "changed"
    if change == "push":
        target["schema"]["task"]["push"] = False
    if change == "missing":
        meta.pop("field_reward_refinement")
    if change == "robot":
        meta["robot_fingerprint"] = "wrong"
    if change == "declaration_extra":
        meta["field_reward_refinement"]["extra"] = 1
    with pytest.raises(ValueError):
        load(p, saved, target, meta)


@pytest.mark.parametrize("protocol", [{}, {"field_reward_overrides": {}}])
def test_absent_or_empty_overrides_preserve_runtime_contract(protocol):
    scales = {"feetgf": 1.0, "feetdf": 1.0}
    schema = {"task": {"ground": 1.0}}
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(cat_controller=object()),
        cat_state=SimpleNamespace(scales=scales.copy()),
        schema=copy.deepcopy(schema),
        device="cpu",
    )
    apply_runtime_settings(env, protocol)
    assert env.cat_state.scales == scales
    assert env.schema == schema


def test_field_transfer_retains_historical_gate_declaration(tmp_path):
    p, saved, target, meta = trial(tmp_path, {"feetgf": 1.5})
    meta["feetgf_progress_refinement"] = {
        "version": "historical",
        "parent_sha256": "earlier-ungated-parent",
    }
    _, info = load(p, saved, target, meta)
    assert info["contract_changes"] == {
        "field_reward_overrides": {"before": {}, "after": {"feetgf": 1.5}}
    }


def test_full_resume_cannot_change_field_weight(tmp_path):
    from test_field_ppo import TinyEnv

    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist

    env = TinyEnv()
    cfg = PpoConfig(
        iterations=1, rollout_steps=4, hidden_dims=(16, 8), epochs=1, mini_batches=1
    )
    parent = train_specialist(env, cfg, tmp_path / "parent")
    env.schema = {**env.schema, "field_reward_overrides": {"feetgf": 1.5}}
    with pytest.raises(ValueError):
        train_specialist(env, cfg, tmp_path / "bad-resume", resume=parent)
