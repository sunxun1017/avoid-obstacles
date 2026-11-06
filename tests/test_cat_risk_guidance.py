"""Risk priority must survive observations without weakening safety costs."""

import copy
from types import SimpleNamespace

import pytest
import torch
from test_cat_curriculum import fixture_contract, fixture_metadata

from embodied_learning.tasks.cat_foot_guidance import sha256
from embodied_learning.tasks.cat_risk_guidance import (
    adapt_risk_transfer,
    blend_guidance,
    risk_weights,
    validate_schema,
)

SPEC = dict(
    version="sdf_predictive_blend_v1",
    near_m=0.05,
    far_m=0.35,
    margin_m=0.05,
    horizon_s=0.2,
    free_gain=0.25,
)


def weights(d, vx=0.0, pred=None):
    s = torch.tensor(d, dtype=torch.float32).reshape(1, -1)
    normal = torch.zeros(*s.shape, 3)
    normal[..., 0] = 1
    v = normal * vx
    return risk_weights(
        s, normal, v, s if pred is None else torch.tensor(pred).reshape_as(s), SPEC
    )


def test_near_far_monotonic_and_body_independence():
    a = weights([0.0, 0.1, 0.2, 0.3, 0.4, 0.8])
    assert a[0, 0] == 1 and a[0, -1] == 0
    assert (a[:, :-1] >= a[:, 1:]).all() and 0 < a[0, 2] < 1


def test_approaching_or_predicted_hazard_increases_risk():
    base = weights([0.4])
    assert weights([0.4], vx=-1) > base
    assert weights([0.6], pred=[0.08]) == 1
    assert weights([0.4], vx=1) == base


def test_empty_scene_safe_but_invalid_query_conservative():
    a = weights([float("inf"), float("nan"), -float("inf")])
    torch.testing.assert_close(a, torch.tensor([[0.0, 1.0, 1.0]]))


def test_blending_preserves_risk_amplitude_and_normal_walking():
    safe = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    cmd = torch.tensor([[1.0, 0.2, 0.0, 0.0]])
    out = blend_guidance(safe, cmd, torch.tensor([[0.0, 1.0]]), SPEC)
    torch.testing.assert_close(
        out, torch.tensor([[[0.25, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    )
    assert torch.isfinite(
        blend_guidance(safe, torch.zeros_like(cmd), torch.ones(1, 2), SPEC)
    ).all()


@pytest.mark.parametrize(
    "bad",
    [
        {"near_m": 0.35},
        {"far_m": 0.01},
        {"margin_m": -1},
        {"horizon_s": float("nan")},
        {"free_gain": 0},
        {"free_gain": 2},
        {"extra": 1},
        {"version": "wrong"},
        {"near_m": True},
    ],
)
def test_schema_rejects_invalid_settings(bad):
    with pytest.raises(ValueError):
        validate_schema({**SPEC, **bad})


def test_transfer_only_adapts_declared_risk_schema(tmp_path):
    p = tmp_path / "model.pt"
    p.write_bytes(b"parent")
    old = fixture_contract(4)
    new = copy.deepcopy(old)
    new["schema"]["risk_guidance"] = SPEC
    source = fixture_metadata(4)
    target = {
        **source,
        "experiment": "risk",
        "protocol_sha256": "new",
        "risk_guidance_refinement": dict(
            version="cat_risk_guidance_actor_transfer_v1",
            parent_sha256=sha256(p),
            parent_experiment=source["experiment"],
            parent_protocol_sha256=source["protocol_sha256"],
            before=None,
            after=SPEC,
        ),
    }
    adapted, meta, changes = adapt_risk_transfer(p, old, new, source, target)
    assert "risk_guidance" not in old["schema"] and adapted == new
    assert changes == {"risk_guidance": {"before": None, "after": SPEC}}
    for key in ["parent_sha256", "parent_protocol_sha256", "after"]:
        broken = copy.deepcopy(target)
        broken["risk_guidance_refinement"][key] = "bad"
        with pytest.raises(ValueError):
            adapt_risk_transfer(p, old, new, source, broken)
    broken = copy.deepcopy(target)
    broken["stage"] = 3
    with pytest.raises(ValueError):
        adapt_risk_transfer(p, old, new, source, broken)


def test_risk_reward_preserves_non_gf_and_progress_gate():
    from test_cat_review_regressions import task_case

    from embodied_learning.tasks.cat_reward import (
        CatRewardParameters,
        cat_reward_terms,
    )

    st, c, p, _ = task_case()
    x = p["reward_inputs"]
    params = CatRewardParameters(c.lower, c.upper)
    old = cat_reward_terms(x, parameters=params)
    far = cat_reward_terms(x, parameters=params, risk_alpha=torch.zeros(16, 11))
    for key in old:
        if key.endswith("gf"):
            torch.testing.assert_close(far[key], torch.full((16,), 4.0))
        else:
            torch.testing.assert_close(far[key], old[key])
    x.move_flag.fill_(1)
    x.cmd_vel[:, :2] = 1
    gated = cat_reward_terms(
        x,
        parameters=params,
        risk_alpha=torch.zeros(16, 11),
        feet_progress_gate=torch.zeros(16),
    )
    assert (gated["feetgf"] == 0).all()


def test_risk_mode_removes_absolute_x_completion_threshold():
    from test_cat_review_regressions import task_case

    from embodied_learning.tasks.cat_reward import (
        CatRewardParameters,
        cat_reward_terms,
    )

    st, c, p, _ = task_case()
    x = p["reward_inputs"]
    x.move_flag.fill_(1)
    x.gait_mask.fill_(-1)
    x.foot_pos[:, :, 0] = 1.51
    params = CatRewardParameters(c.lower, c.upper)
    a = cat_reward_terms(x, parameters=params, risk_alpha=torch.ones(16, 11))
    x.foot_pos[:, :, 0] = 1.49
    b = cat_reward_terms(x, parameters=params, risk_alpha=torch.ones(16, 11))
    torch.testing.assert_close(a["feetgf"], b["feetgf"])


def test_task_integration_changes_only_field_inputs_and_gf_rewards():
    from test_cat_review_regressions import task_case

    from embodied_learning.tasks.cat_traversal import CatTaskState

    old, c, p, _ = task_case()
    new = CatTaskState(
        old.manifest,
        old.bank,
        old.allowed,
        16,
        "cpu",
        {"noise_level": 0.0},
        risk_guidance=SPEC,
    )
    new.reset(torch.arange(16))
    new.gait = copy.deepcopy(old.gait)
    old.refresh(copy.deepcopy(p), c, torch.zeros(16, dtype=torch.long), 0)
    new.refresh(copy.deepcopy(p), c, torch.zeros(16, dtype=torch.long), 0)
    assert new.risk_alpha.shape == (16, 11) and torch.isfinite(new.risk_alpha).all()
    for key in old.terms:
        if not key.endswith("gf"):
            torch.testing.assert_close(new.terms[key], old.terms[key])
    for group, width in [("actor", 162), ("critic", 250)]:
        assert new.observations()[group].shape == (16, width)
        torch.testing.assert_close(
            new.observations()[group][:, :85], old.observations()[group][:, :85]
        )
    torch.testing.assert_close(new.gait.phase, old.gait.phase)
    torch.testing.assert_close(new.gait.last_command, old.gait.last_command)


def test_full_loader_guard_and_original_checkpoint_preserved(tmp_path):
    from embodied_learning.policies.cat_actor_warmstart import load_actor_warmstart

    old = fixture_contract(4)
    source = fixture_metadata(4)
    p = tmp_path / "parent.pt"
    saved = dict(
        field_contract=old,
        curriculum=source,
        training_scenes=[2],
        actor_state_dict={"w": torch.ones(3)},
        completed_iterations=320,
    )
    torch.save(saved, p)
    target = copy.deepcopy(old)
    target["schema"]["risk_guidance"] = SPEC
    meta = {
        **source,
        "experiment": "risk",
        "protocol_sha256": "new",
        "risk_guidance_refinement": dict(
            version="cat_risk_guidance_actor_transfer_v1",
            parent_sha256=sha256(p),
            parent_experiment=source["experiment"],
            parent_protocol_sha256=source["protocol_sha256"],
            before=None,
            after=SPEC,
        ),
    }

    def load(t, m):
        return load_actor_warmstart(
            p,
            contract=SimpleNamespace(to_dict=lambda: t),
            curriculum=m,
            training_scenes=[2],
            device="cpu",
        )

    actual, info = load(target, meta)
    assert (
        actual["field_contract"] == old
        and not info["critic_restored"]
        and not info["optimizer_restored"]
    )
    for which in ["task", "geometry", "robot", "missing"]:
        t = copy.deepcopy(target)
        m = copy.deepcopy(meta)
        if which == "task":
            t["schema"]["task"]["push"] = False
        if which == "geometry":
            t["schema"]["scene_hash"] = "wrong"
        if which == "robot":
            m["robot_fingerprint"] = "wrong"
        if which == "missing":
            m.pop("risk_guidance_refinement")
        with pytest.raises(ValueError):
            load(t, m)


def test_partial_reset_keeps_continuing_risk_aligned_with_cached_observations():
    from test_cat_review_regressions import task_case

    state, controller, payload, _ = task_case()
    state.risk_guidance = SPEC
    direction = torch.zeros(16, 11, 3)
    direction[..., 0] = 1
    state.bank.sample_source = lambda *args: SimpleNamespace(
        guidance=direction,
        sdf_gradient=-direction,
        sdf=torch.full((16, 11), 0.4),
    )
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    payload["body_positions"][..., 0] += 0.03
    payload["root_position"][..., 0] += 0.03
    state.refresh(payload, controller, torch.ones(16, dtype=torch.long), 1)
    risk = state.risk_alpha.clone()
    delayed = state.risk_alpha_delayed.clone()
    obs = state.observations()["actor"].clone()
    assert (risk > 0.99).all()
    state.reset(torch.tensor([0]))
    steps = torch.ones(16, dtype=torch.long)
    steps[0] = 0
    state.refresh(payload, controller, steps, 1)
    torch.testing.assert_close(state.risk_alpha[1:], risk[1:])
    torch.testing.assert_close(state.risk_alpha_delayed[1:], delayed[1:])
    torch.testing.assert_close(state.observations()["actor"][1:], obs[1:])
    assert (state.risk_alpha[0] < 0.01).all()
