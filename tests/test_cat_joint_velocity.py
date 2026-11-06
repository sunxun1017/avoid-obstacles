from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_cat_field_weights import load, trial

from embodied_learning.tasks.cat_foot_guidance import sha256
from embodied_learning.tasks.cat_hurdle import apply_runtime_settings
from embodied_learning.tasks.cat_joint_velocity import validate_weight
from embodied_learning.tasks.cat_reward import (
    SOURCE_SCALES,
    CatRewardInputs,
    CatRewardParameters,
    cat_reward_terms,
)


def inputs():
    f = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    return CatRewardInputs(
        **{
            k[6:]: torch.from_numpy(f[k]).clone()
            for k in f.files
            if k.startswith("input_")
        }
    ), CatRewardParameters(
        torch.from_numpy(f["soft_lower"]), torch.from_numpy(f["soft_upper"])
    )


@pytest.mark.parametrize("joint", range(29))
def test_all_actual_velocities_including_wrists_enter_new_term(joint):
    x, p = inputs()
    x.joint_vel.zero_()
    x.joint_vel[:, joint] = 2.0
    t = cat_reward_terms(x, parameters=p, whole_body_joint_velocity=True)
    torch.testing.assert_close(
        t["whole_body_joint_velocity"],
        torch.full_like(t["whole_body_joint_velocity"], 4.0),
    )
    assert "whole_body_joint_velocity" not in cat_reward_terms(x, parameters=p)


@pytest.mark.parametrize(
    "weight", [0, -1, True, float("inf"), float("nan"), "0.05", {}, []]
)
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError):
        validate_weight(weight)


def test_runtime_optin_default_and_exact_weight():
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(cat_controller=object()),
        cat_state=SimpleNamespace(scales=SOURCE_SCALES.copy(), recovery=None),
        schema={},
        device="cpu",
    )
    apply_runtime_settings(env, {})
    assert env.cat_state.scales == SOURCE_SCALES and env.schema == {}
    apply_runtime_settings(env, {"whole_body_joint_velocity_weight": 0.05})
    assert env.cat_state.whole_body_joint_velocity is True
    assert env.cat_state.scales == {
        **SOURCE_SCALES,
        "whole_body_joint_velocity": -0.05,
    }
    assert env.schema["whole_body_joint_velocity"] == {
        "version": "sum_dq2_29_v1",
        "weight": 0.05,
    }


@pytest.mark.parametrize(
    "change",
    [
        None,
        "sha",
        "stage",
        "push",
        "geometry",
        "robot",
        "weight",
        "missing",
        "extra",
    ],
)
def test_transfer_allows_only_declared_velocity_reward(tmp_path, change):
    p, saved, target, meta = trial(tmp_path, {})
    meta.pop("field_reward_refinement")
    target["schema"]["whole_body_joint_velocity"] = {
        "version": "sum_dq2_29_v1",
        "weight": 0.05,
    }
    meta["whole_body_velocity_refinement"] = dict(
        version="cat_whole_body_velocity_actor_transfer_v1",
        parent_sha256=sha256(p),
        parent_experiment="test",
        parent_protocol_sha256="protocol",
        before=None,
        after=target["schema"]["whole_body_joint_velocity"],
    )
    if change == "sha":
        meta["whole_body_velocity_refinement"]["parent_sha256"] = "bad"
    if change == "stage":
        meta["stage"] = 3
    if change == "push":
        target["schema"]["task"]["push"] = False
    if change == "geometry":
        target["schema"]["scene_hash"] = "bad"
    if change == "robot":
        meta["robot_fingerprint"] = "bad"
    if change == "weight":
        target["schema"]["whole_body_joint_velocity"] = {
            **target["schema"]["whole_body_joint_velocity"],
            "weight": 0.1,
        }
    if change == "missing":
        meta.pop("whole_body_velocity_refinement")
    if change == "extra":
        meta["whole_body_velocity_refinement"]["unexpected"] = 1
    if change:
        with pytest.raises(ValueError):
            load(p, saved, target, meta)
    else:
        actual, info = load(p, saved, target, meta)
        assert actual["field_contract"] == saved["field_contract"]
        assert not info["critic_restored"] and not info["optimizer_restored"]
        assert info["contract_changes"]["whole_body_joint_velocity"] == {
            "before": None,
            "after": {"version": "sum_dq2_29_v1", "weight": 0.05},
        }


def test_recovery_not_silently_given_source_only_reward():
    env = SimpleNamespace(
        cat_state=SimpleNamespace(scales={}, recovery=object()), schema={}
    )
    from embodied_learning.tasks.cat_joint_velocity import apply_joint_velocity

    with pytest.raises(ValueError):
        apply_joint_velocity(env, {"whole_body_joint_velocity_weight": 0.05})


def test_task_refresh_integrates_velocity_reward_without_changing_other_terms():
    import copy

    from test_cat_review_regressions import task_case

    baseline, controller, payload, _ = task_case()
    # Safe positive-return fixture: do not let reward clipping hide the delta.
    from dataclasses import fields

    x = payload["reward_inputs"]
    for item in fields(x):
        getattr(x, item.name).zero_()
    x.joint_pos.copy_(controller.default.expand_as(x.joint_pos))
    x.joint_vel[:, 19] = 1.0  # A wrist excluded from old smoothness.
    x.foot_pos[:, 0, 1] = 0.1
    x.foot_pos[:, 1, 1] = -0.1
    x.head_pos[:, 2] = 1.0
    x.feet_contact.fill_(1)
    x.navigation_rotation.copy_(torch.eye(3).expand_as(x.navigation_rotation))
    x.leg_rotation.copy_(torch.eye(3).expand_as(x.leg_rotation))
    baseline.bank.sample_source = lambda points, *args: SimpleNamespace(
        guidance=torch.zeros_like(points),
        sdf_gradient=torch.zeros_like(points),
        sdf=torch.full(points.shape[:2], 0.6),
    )
    candidate = copy.deepcopy(baseline)
    other_controller = copy.deepcopy(controller)
    other_payload = copy.deepcopy(payload)
    env = SimpleNamespace(cat_state=candidate, schema={})
    from embodied_learning.tasks.cat_joint_velocity import apply_joint_velocity

    apply_joint_velocity(env, {"whole_body_joint_velocity_weight": 0.05})
    rng = torch.get_rng_state()
    baseline.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    torch.set_rng_state(rng)
    candidate.refresh(
        other_payload, other_controller, torch.zeros(16, dtype=torch.long), 0
    )
    for key in baseline.terms:
        torch.testing.assert_close(candidate.terms[key], baseline.terms[key])
    expected_velocity = other_payload["reward_inputs"].joint_vel.square().sum(-1)
    torch.testing.assert_close(
        candidate.terms["whole_body_joint_velocity"], expected_velocity
    )
    rates = sum(baseline.terms[k] * baseline.scales[k] for k in baseline.terms)
    expected_reward = ((rates - 0.05 * expected_velocity) * 0.02).clamp(0, 10000)
    torch.testing.assert_close(candidate.reward(), expected_reward)
    assert bool((baseline.reward() > 0).all())
    assert bool((candidate.reward() > 0).all())
    torch.testing.assert_close(
        candidate.reward() - baseline.reward(), -0.05 * expected_velocity * 0.02
    )
