"""Inference-only support probes must not replace real contact observations."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodied_learning.tasks.cat_recovery import RecoveryNavigator


@pytest.fixture
def diagnostic(monkeypatch):
    tool_dir = Path(__file__).parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tool_dir))
    spec = importlib.util.spec_from_file_location(
        "cat_probe_diagnostic", tool_dir / "diagnose_cat_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def research_state(count):
    return SimpleNamespace(
        task_profile="cat_recovery_v1", recovery=RecoveryNavigator(count, "cpu")
    )


def test_contact_override_selects_only_active_stationary_reference_braking_slots(
    diagnostic,
):
    state = research_state(9)
    nav = state.recovery
    nav.mode[:] = torch.tensor(
        [nav.BRAKE, nav.DRIVE, nav.SETTLE, nav.HOLD] + [nav.BRAKE] * 5
    )
    nav.last_velocity = torch.zeros(9, 3, dtype=torch.float64)
    nav.last_velocity[4, 0] = 1e-6  # Strictly below, not equal to the threshold.
    velocity = torch.zeros(9, 3, dtype=torch.float64)
    velocity[5, 0] = nav.config.settle_entry_speed
    velocity[7, 2] = 0.14
    velocity[8, 2] = 0.16  # The gate is 3D speed, not just planar speed.
    active = torch.ones(9, dtype=torch.bool)
    active[6] = False
    contacts = torch.tensor([[0.0, 1.0]] * 9)
    originals = [x.clone() for x in (contacts, velocity, active, nav.mode)]

    result = diagnostic.probe_settle_contacts(
        state, feet_contact=contacts, velocity=velocity, active=active
    )

    expected = contacts.clone()
    expected[[0, 7]] = 1.0
    torch.testing.assert_close(result, expected)
    assert result.data_ptr() != contacts.data_ptr()
    for actual, original in zip(
        (contacts, velocity, active, nav.mode), originals, strict=True
    ):
        torch.testing.assert_close(actual, original)


def test_source_profile_rejects_probe_and_reports_no_recovery_telemetry(diagnostic):
    state = SimpleNamespace(task_profile="cat_source_v1", recovery=None)
    with pytest.raises(ValueError, match="cat_recovery_v1"):
        diagnostic.probe_settle_contacts(
            state,
            feet_contact=torch.zeros(1, 2),
            velocity=torch.zeros(1, 3),
            active=torch.ones(1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="cat_recovery_v1"):
        with diagnostic.settle_without_support_probe(
            state, torch.ones(1, dtype=torch.bool)
        ):
            pytest.fail("Source probe must be rejected before entering its scope")
    assert diagnostic.recovery_diagnostic_values(state, {}) == {}


def test_probe_changes_only_navigator_input_and_restores_method(diagnostic):
    from test_cat_recovery_task import recovery_case

    state, controller, payload = recovery_case()
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    nav = state.recovery
    nav.mode.fill_(nav.BRAKE)
    nav.last_velocity.zero_()
    nav.reversal_pending.fill_(True)
    payload["reward_inputs"].root_vel.zero_()
    payload["reward_inputs"].feet_contact.zero_()
    real_contacts = payload["reward_inputs"].feet_contact
    selected = torch.zeros(16, dtype=torch.bool)
    selected[0] = True
    original_advance = nav.advance

    with diagnostic.settle_without_support_probe(state, selected):
        state.refresh(payload, controller, torch.ones(16, dtype=torch.long), 1)

    assert nav.advance == original_advance
    assert "advance" not in nav.__dict__
    assert nav.mode[0] == nav.SETTLE
    assert (nav.mode[1:] == nav.BRAKE).all()
    assert payload["reward_inputs"].feet_contact is real_contacts
    assert not real_contacts.any()
    assert state.gait.last_command[0, 0] == 0
    assert (state.gait.last_command[1:, 0] == 1).all()
    assert not state.saved_outcomes["fall"].any()
    assert not state.saved_outcomes["collision"].any()
    with pytest.raises(RuntimeError, match="probe scope failure"):
        with diagnostic.settle_without_support_probe(state, selected):
            raise RuntimeError("probe scope failure")
    assert nav.advance == original_advance
    assert "advance" not in nav.__dict__


def test_recovery_telemetry_preserves_real_state_and_optional_envelope(diagnostic):
    state = research_state(2)
    nav = state.recovery
    nav.mode[:] = torch.tensor([nav.BRAKE, nav.SETTLE])
    nav.reversal_pending[0] = True
    nav.resume_steps[:] = torch.tensor([0, 4])
    nav.settle_steps[:] = torch.tensor([0, 2])
    nav.last_velocity[0, 0] = 0.2
    clearance = torch.tensor([[0.3, -0.1], [0.2, 0.4]])
    payload = {"body_envelope": SimpleNamespace(clearance=clearance)}
    values = diagnostic.recovery_diagnostic_values(state, payload)
    expected = {
        "navigation_mode": nav.mode,
        "reversal_pending": nav.reversal_pending,
        "resume_steps": nav.resume_steps,
        "settle_steps": nav.settle_steps,
        "reference_velocity": nav.last_velocity,
        "body_envelope_clearance": clearance,
    }
    assert values.keys() == expected.keys()
    for key, actual in expected.items():
        torch.testing.assert_close(values[key], actual)
        assert values[key].data_ptr() != actual.data_ptr()
    assert "body_envelope_clearance" not in diagnostic.recovery_diagnostic_values(
        state, {}
    )
    nav.mode.zero_()
    assert values["navigation_mode"].tolist() == [1, 2]


def invoke_diagnostic_main(diagnostic, monkeypatch, tmp_path, variant, env):
    from embodied_learning.algorithms import dagger
    from embodied_learning.cli import field_training

    run = tmp_path / "run"
    (run / "fields").mkdir(parents=True)
    checkpoint = run / "synthetic-checkpoint"
    checkpoint.write_bytes(b"CLI wiring fixture only; no policy weights")
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_cat_checkpoint.py",
            "--run-dir",
            str(run),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(run / "unused-config"),
            "--scene",
            "synthetic",
            "--result-dir",
            str(output),
            "--num-envs",
            str(env.num_envs),
            "--variant",
            variant,
        ],
    )

    def fake_loader(_args):
        return dagger.evaluate_student(
            env, SimpleNamespace(predict=lambda _obs: None), steps=1
        )

    monkeypatch.setattr(field_training, "main", fake_loader)
    return diagnostic.main(), output


def test_cli_source_probe_is_rejected_before_rollout(
    diagnostic, monkeypatch, tmp_path
):
    env = SimpleNamespace(
        num_envs=1,
        cat_state=SimpleNamespace(task_profile="cat_source_v1", recovery=None),
        unwrapped=None,
    )
    with pytest.raises(ValueError, match="cat_recovery_v1"):
        invoke_diagnostic_main(
            diagnostic, monkeypatch, tmp_path, "settle_without_support", env
        )


@pytest.mark.parametrize("variant", ["baseline", "settle_without_support"])
def test_cli_records_true_contacts_and_recovery_telemetry(
    diagnostic, monkeypatch, tmp_path, variant
):
    from test_cat_recovery_task import recovery_case

    state, controller, payload = recovery_case()
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    state.recovery.mode.fill_(state.recovery.BRAKE)
    state.recovery.last_velocity.zero_()
    state.recovery.reversal_pending.fill_(True)
    payload["reward_inputs"].root_vel.zero_()
    payload["reward_inputs"].feet_contact.zero_()
    native = SimpleNamespace(
        cat_obstacle_contacts=[],
        cat_controller=controller,
        sim=SimpleNamespace(
            data=SimpleNamespace(
                nacon=torch.tensor([0]),
                contact=SimpleNamespace(
                    dist=torch.zeros(0),
                    worldid=torch.zeros(0, dtype=torch.long),
                    geom=torch.zeros(0, 2, dtype=torch.long),
                    frame=torch.zeros(0, 3, 3),
                ),
            )
        ),
    )

    class SnapshotEnv:
        num_envs, device = 16, "cpu"
        cat_state, unwrapped = state, native

        def get_observations(self):
            return state.observations()

        def step(self, _actions):
            # Terminal fixture also falls on the diagnostic trace's stride.
            state.refresh(payload, controller, torch.full((16,), 1000), 5)
            return (
                self.get_observations(),
                state.reward(),
                torch.ones(16),
                {
                    "field_outcomes": state.saved_outcomes,
                    "cat_metrics": state.last_metrics,
                    "cat_metric_definitions": state.metric_definitions,
                },
            )

    _, output = invoke_diagnostic_main(
        diagnostic, monkeypatch, tmp_path, variant, SnapshotEnv()
    )
    expected = (
        state.recovery.SETTLE if variant != "baseline" else state.recovery.BRAKE
    )
    metadata = json.loads((output / "metadata.json").read_text())
    assert ("navigator_override" in metadata) == (variant != "baseline")
    events = json.loads((output / "events.json").read_text())
    assert len(events) == 16
    assert all(event["navigation_mode"] == expected for event in events)
    assert all(event["feet_ground_contact"] == [0.0, 0.0] for event in events)
    assert not payload["reward_inputs"].feet_contact.any()
    with np.load(output / "trace.npz") as trace:
        assert (trace["navigation_mode"] == expected).all()
        assert (trace["feet_ground_contact"] == 0).all()
        np.testing.assert_array_equal(
            trace["body_envelope_clearance"][0],
            payload["body_envelope"].clearance[:8].numpy(),
        )
