"""Episode metrics must retain contacts and measure physical settling."""

from types import SimpleNamespace

import pytest
import torch
from test_cat_review_regressions import task_case

from embodied_learning.algorithms.dagger import evaluate_student


def quiet_case():
    state, controller, payload, _ = task_case()
    state.bank.sample_source = lambda p, *args: SimpleNamespace(
        guidance=torch.zeros_like(p),
        sdf_gradient=torch.zeros_like(p),
        sdf=torch.ones(p.shape[:2]),
    )
    payload["reward_inputs"].root_vel.zero_()
    payload["root_local_vel"].zero_()
    payload["gyro"].zero_()
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    return state, controller, payload


def advance(state, controller, payload, step):
    state.refresh(payload, controller, torch.full((16,), step), step)


def test_contact_is_retained_until_reset_and_disqualifies_success():
    state, controller, payload = quiet_case()
    # Physical contact during the source's 50-step grace period still counts.
    payload["physical_collision"][0] = True
    payload["self_contact"][1] = True
    advance(state, controller, payload, 1)
    assert not state.outcomes()["collision"].any()
    payload["physical_collision"].zero_()
    payload["self_contact"].zero_()
    advance(state, controller, payload, 1000)
    outcomes = state.saved_outcomes
    assert outcomes["goal"].all()
    assert outcomes["physical_collision"][:3].tolist() == [True, False, False]
    assert outcomes["success"][:3].tolist() == [False, False, True]

    state.reset(torch.tensor([0]))
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 1000)
    # Post-reset observation refresh cannot overwrite the terminal report.
    assert state.saved_outcomes is outcomes
    assert outcomes["physical_collision"][0]
    advance(state, controller, payload, 1001)
    assert not state.saved_outcomes["physical_collision"].any()
    assert state.saved_outcomes["success"][:3].tolist() == [True, False, True]


def test_stop_command_does_not_mean_moving_or_rotating_robot_has_settled():
    state, controller, payload = quiet_case()
    payload["root_local_vel"][0, 0] = 1.0
    payload["gyro"][1, 2] = 0.5
    payload["root_local_vel"][2, 2] = 0.5
    for step in range(1, 26):
        advance(state, controller, payload, step)
    metrics = state.last_metrics
    assert metrics["stop_command"].all()
    assert metrics["settled_stop"][:4].tolist() == [False, False, False, True]
    assert "stopped" not in metrics


def test_settling_requires_continuous_time_and_resets_only_selected_slots():
    state, controller, payload = quiet_case()
    for step in range(1, 25):
        advance(state, controller, payload, step)
    assert not state.last_metrics["settled_stop"].any()
    # Repeated reads must not add another control period.
    for _ in range(3):
        advance(state, controller, payload, 24)
    assert not state.last_metrics["settled_stop"].any()
    advance(state, controller, payload, 25)
    assert state.last_metrics["settled_stop"].all()

    state.reset(torch.tensor([0]))
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 25)
    advance(state, controller, payload, 26)
    assert not state.last_metrics["settled_stop"][0]
    assert state.last_metrics["settled_stop"][1:].all()
    payload["root_local_vel"][1, 0] = 0.5
    payload["up_z"][2] = -1.0  # A motionless fallen robot has not stopped safely.
    payload["invalid"][3] = True
    advance(state, controller, payload, 27)
    assert not state.last_metrics["settled_stop"][:4].any()
    payload["root_local_vel"][1, 0] = 0.0
    advance(state, controller, payload, 28)
    assert not state.last_metrics["settled_stop"][1]


def test_evaluation_uses_latched_collisions_and_separate_stop_metrics():
    class EpisodeEnv:
        num_envs = 16

        def __init__(self):
            self.state, self.controller, self.payload = quiet_case()
            self.steps = 0

        def get_observations(self):
            return self.state.observations()

        def step(self, _actions):
            self.steps += 1
            self.payload["physical_collision"][0] = self.steps == 1
            advance(self.state, self.controller, self.payload, self.steps)
            return (
                self.get_observations(),
                self.state.reward(),
                torch.full((16,), self.steps == 25),
                {
                    "field_outcomes": self.state.saved_outcomes,
                    "cat_metrics": self.state.last_metrics,
                    "cat_metric_definitions": self.state.metric_definitions,
                },
            )

    policy = SimpleNamespace(predict=lambda _obs: torch.zeros(16, 12))
    result = evaluate_student(EpisodeEnv(), policy, steps=25)
    assert result["rates"]["physical_collision"] == 1 / 16
    assert result["rates"]["goal"] == 1.0
    assert result["rates"]["success"] == 15 / 16
    assert result["stop_command_fraction"] == 1.0
    assert result["settled_stop_fraction"] == pytest.approx(1 / 25)
    assert "stopped_step_fraction" not in result
    assert result["metric_definitions"]["settled_stop"] == {
        "linear_speed_below_m_s": 0.1,
        "angular_speed_below_rad_s": 0.1,
        "continuous_seconds": 0.5,
        "control_dt_seconds": 0.02,
    }
