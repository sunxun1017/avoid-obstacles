"""Research wiring must change the observed behavior without changing source."""

from types import SimpleNamespace

import pytest
import torch
from test_cat_review_regressions import task_case

from embodied_learning.tasks.cat_traversal import CatTaskState


def recovery_case():
    base, controller, payload, _ = task_case()
    state = CatTaskState(
        base.manifest,
        base.bank,
        base.allowed,
        16,
        "cpu",
        {"noise_level": 0.0},
        task_profile="cat_recovery_v1",
        recovery_options={},
    )
    state.reset(torch.arange(16))
    state.goals[:, 0] = 1.0
    payload["reward_inputs"].navigation_rotation = torch.eye(3).repeat(16, 1, 1)
    payload["reward_inputs"].feet_contact.fill_(1)
    payload["reward_inputs"].root_vel.zero_()
    payload["body_envelope"] = SimpleNamespace(
        clearance=torch.full((16, 11), 0.4),
        gradient=torch.zeros(16, 11, 3),
        minimum=torch.full((16,), 0.4),
        valid=torch.ones(16, dtype=torch.bool),
    )
    return state, controller, payload


def test_recovery_actor_observes_velocity_envelope_and_consistent_command():
    state, controller, payload = recovery_case()
    payload["root_local_vel"][:, 0] = 2.2
    payload["reward_inputs"].root_vel[:, 0] = 2.2
    payload["body_envelope"].clearance[:, 0] = -0.02
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    state.refresh(payload, controller, torch.ones(16, dtype=torch.long), 1)
    obs = state.observations()
    assert obs["actor"].shape == (16, 171)
    assert obs["critic"].shape == (16, 256)
    torch.testing.assert_close(obs["actor"][:, 162:165], payload["root_local_vel"])
    torch.testing.assert_close(obs["actor"][:, 76:80], obs["critic"][:, 76:80])
    torch.testing.assert_close(obs["actor"][:, 91], torch.full((16,), -0.02))
    assert torch.isfinite(state.reward()).all()


def test_recovery_repeated_refresh_and_partial_reset_preserve_other_slots():
    state, controller, payload = recovery_case()
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    for step in range(1, 5):
        state.refresh(payload, controller, torch.full((16,), step), step)
    obs, phase = state.observations().clone(), state.gait.phase.clone()
    velocity = state.recovery.last_velocity.clone()
    state.refresh(payload, controller, torch.full((16,), 4), 4)
    torch.testing.assert_close(state.observations()["actor"], obs["actor"])
    torch.testing.assert_close(state.recovery.last_velocity, velocity)
    state.reset(torch.tensor([0]))
    state.refresh(payload, controller, torch.full((16,), 4), 4)
    torch.testing.assert_close(state.observations()["actor"][1:], obs["actor"][1:])
    torch.testing.assert_close(state.gait.phase[1:], phase[1:])
    torch.testing.assert_close(state.recovery.last_velocity[1:], velocity[1:])
    assert not state.recovery.last_velocity[0].any()


def test_envelope_invalidity_stops_and_reports_invalid_without_fake_collision():
    state, controller, payload = recovery_case()
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    payload["body_envelope"].valid[0] = False
    payload["body_envelope"].clearance[0] = -torch.inf
    # Conservative lower bound alone is not proof of actual contact.
    payload["body_envelope"].clearance[1] = -0.2
    state.bank.sample_source = lambda p, *args: SimpleNamespace(
        guidance=torch.zeros_like(p),
        sdf_gradient=torch.zeros_like(p),
        sdf=torch.ones(p.shape[:2]),
    )
    state.refresh(payload, controller, torch.full((16,), 60), 60)
    assert state.outcomes()["invalid"][0]
    assert not state.gait.last_command[0].any()
    assert not state.outcomes()["collision"][1]
    assert torch.isfinite(state.observations()["actor"]).all()
    assert torch.isfinite(state.reward()).all()


def test_source_rejects_research_options_without_changing_profile():
    base, _, _, _ = task_case()
    with pytest.raises(ValueError, match="profile"):
        CatTaskState(
            base.manifest,
            base.bank,
            base.allowed,
            16,
            "cpu",
            {},
            recovery_options={"max_speed": 0.4},
        )


@pytest.mark.parametrize("speed", [torch.inf, 1e19, 1e20])
def test_nonfinite_motion_gets_finite_failure_cost_and_zero_command(speed):
    state, controller, payload = recovery_case()
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    payload["reward_inputs"].root_vel[0, 0] = speed
    payload["root_local_vel"][0, 0] = speed
    state.refresh(payload, controller, torch.ones(16, dtype=torch.long), 1)
    assert state.outcomes()["invalid"][0]
    assert not state.gait.last_command[0].any()
    assert torch.isfinite(state.reward()).all()
    assert state.reward()[0] == -10.0
    assert torch.isfinite(state.observations()["actor"]).all()


def test_native_contact_cost_is_once_per_episode_without_changing_termination():
    state, controller, payload = recovery_case()
    state.bank.sample_source = lambda p, *args: SimpleNamespace(
        guidance=torch.zeros_like(p),
        sdf_gradient=torch.zeros_like(p),
        sdf=torch.ones(p.shape[:2]),
    )
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    payload["physical_collision"][0] = True
    payload["body_envelope"].clearance[1] = -0.2
    state.refresh(payload, controller, torch.full((16,), 60), 60)
    assert state.saved_outcomes["physical_collision"][0]
    assert not state.outcomes()["collision"][0]
    assert state.terms["recovery_first_contact"][0] * 0.02 == -5.0
    assert state.terms["recovery_first_contact"][1] == 0.0
    payload["physical_collision"][0] = False
    state.refresh(payload, controller, torch.full((16,), 61), 61)
    payload["physical_collision"][0] = True
    state.refresh(payload, controller, torch.full((16,), 62), 62)
    assert state.terms["recovery_first_contact"][0] == 0.0
    state.reset(torch.tensor([0]))
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 62)
    state.refresh(payload, controller, torch.ones(16, dtype=torch.long), 63)
    assert state.terms["recovery_first_contact"][0] * 0.02 == -5.0


@pytest.mark.parametrize(
    ("task_options", "recovery_options"),
    [({"ground": 1e50}, {}), ({}, {"tracking_weight": 1e50})],
)
def test_recovery_rejects_scales_that_cannot_be_used_in_float32_rewards(
    task_options,
    recovery_options,
):
    base, _, _, _ = task_case()
    with pytest.raises(ValueError, match="float32"):
        CatTaskState(
            base.manifest,
            base.bank,
            base.allowed,
            16,
            "cpu",
            task_options,
            task_profile="cat_recovery_v1",
            recovery_options=recovery_options,
        )
