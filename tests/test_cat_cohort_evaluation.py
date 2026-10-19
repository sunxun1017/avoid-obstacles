"""A complete-episode cohort must not select only fast terminating slots."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "cohort", Path(__file__).parents[1] / "tools/evaluate_cat_checkpoint.py"
)
cohort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort)


class DifferentDurationEnv:
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self.age = torch.zeros(2, dtype=torch.long)
        self.lengths = torch.tensor([1, 3])

    def get_observations(self):
        return {}

    def step(self, actions):
        self.age += 1
        done = self.age == self.lengths
        goal = torch.tensor([False, True])
        flags = dict(
            goal=goal,
            success=goal,
            physical_collision=~goal,
            fall=~goal,
            collision=~goal,
            invalid=torch.zeros(2, dtype=torch.bool),
            timeout=goal & done,
        )
        metrics = dict(
            goal_distance=torch.tensor([1.0, 0.0]),
            min_goal_distance=torch.tensor([1.0, 0.0]),
            stop_command=goal,
            settled_stop=goal & done,
            root_speed=torch.tensor([1.0, 0.0]),
            root_angular_speed=torch.tensor([0.2, 0.0]),
        )
        self.age[done] = 0
        return (
            {},
            torch.ones(2),
            done,
            dict(
                field_outcomes=flags, cat_metrics=metrics, cat_metric_definitions={}
            ),
        )


def test_cohort_waits_for_slow_slot_and_excludes_fast_slots_extra_episodes(tmp_path):
    path = tmp_path / "episodes.jsonl"
    result = cohort.evaluate_cohort(
        DifferentDurationEnv(),
        SimpleNamespace(predict=lambda obs: None),
        episodes=2,
        episode_path=path,
    )
    assert result["episodes"] == 2
    assert result["included_transitions"] == 4
    assert result["simulation_steps"] == 3
    assert result["rates"]["physical_collision"] == 0.5
    assert result["rates"]["success"] == 0.5
    assert result["rates"]["timeout"] == 0.5
    assert result["rates"]["timeout_without_goal"] == 0
    assert result["stop_command_fraction"] == 0.75
    assert result["settled_stop_fraction"] == 0.25
    assert result["first_goal_seconds"]["mean"] == pytest.approx(0.02)
    assert len(path.read_text().splitlines()) == 2


def test_uneven_slot_quotas_include_exactly_designated_episodes():
    result = cohort.evaluate_cohort(
        DifferentDurationEnv(),
        SimpleNamespace(predict=lambda obs: None),
        episodes=5,
    )
    assert result["episodes"] == 5
    assert result["included_transitions"] == 9
    assert result["simulation_steps"] == 6
    assert result["outcomes"]["success"] == 2
    assert result["first_goal_seconds"]["count"] == 2


def test_wilson_interval_does_not_claim_certainty_at_zero_events():
    low, high = cohort.wilson(0, 1000)
    assert low == pytest.approx(0)
    assert 0 < high < 0.004


class HoldThenLeaveEnv:
    """Two designated short episodes finish while a slow slot is still active."""

    num_envs, device = 2, "cpu"

    def __init__(self):
        self.step_number = 0

    def get_observations(self):
        return {}

    def step(self, actions):
        self.step_number += 1
        # Slot 0 reaches and holds, leaves, then finishes a no-goal episode.
        # Its extra fourth-step hold is outside its assigned quota.
        table = (
            (
                [0.05, 0.05],
                [True, False],
                [0.02, 0.9],
                [0.01, 0.2],
                [False, False],
                [True, True],
            ),
            (
                [0.3, 0.05],
                [False, True],
                [0.4, 0.04],
                [0.2, 0.02],
                [True, False],
                [True, True],
            ),
            (
                [1.0, 0.05],
                [False, True],
                [0.8, 0.03],
                [0.3, 0.01],
                [True, False],
                [False, True],
            ),
            (
                [0.05, 0.05],
                [True, True],
                [0.01, 0.02],
                [0.01, 0.01],
                [True, True],
                [True, True],
            ),
        )
        distance, settled, speed, angular, done, goal = table[self.step_number - 1]
        flags = dict(
            goal=torch.tensor(goal),
            success=torch.tensor(goal),
            physical_collision=torch.zeros(2, dtype=torch.bool),
            fall=torch.zeros(2, dtype=torch.bool),
            collision=torch.zeros(2, dtype=torch.bool),
            invalid=torch.zeros(2, dtype=torch.bool),
            timeout=torch.tensor(done),
        )
        metrics = dict(
            goal_distance=torch.tensor(distance),
            min_goal_distance=torch.tensor([0.05 if goal[0] else 1.0, 0.05]),
            stop_command=torch.tensor(settled),
            settled_stop=torch.tensor(settled),
            root_speed=torch.tensor(speed),
            root_angular_speed=torch.tensor(angular),
        )
        return (
            {},
            torch.ones(2),
            torch.tensor(done),
            dict(
                field_outcomes=flags, cat_metrics=metrics, cat_metric_definitions={}
            ),
        )


def test_goal_hold_distinguishes_leaving_final_hold_and_ignored_extra_episodes(
    tmp_path,
):
    path = tmp_path / "episodes.jsonl"
    result = cohort.evaluate_cohort(
        HoldThenLeaveEnv(),
        SimpleNamespace(predict=lambda obs: None),
        episodes=3,
        episode_path=path,
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    first, second, slow = records
    assert result["included_transitions"] == 7
    assert result["goal_hold_fraction"] == pytest.approx(4 / 7)
    assert result["ever_goal_hold_count"] == 2
    assert result["ever_goal_hold_rate"] == pytest.approx(2 / 3)
    assert result["final_goal_hold_count"] == 1
    assert result["final_goal_hold_rate"] == pytest.approx(1 / 3)
    assert first["ever_goal_hold"] and not first["final_goal_hold"]
    assert first["final_root_speed"] == pytest.approx(0.4)
    assert first["final_root_angular_speed"] == pytest.approx(0.2)
    assert not second["ever_goal_hold"] and not second["final_goal_hold"]
    assert second["first_goal_root_speed"] is None
    assert slow["ever_goal_hold"] and slow["final_goal_hold"]
    assert slow["first_goal_root_speed"] == pytest.approx(0.9)
    assert result["first_goal_root_speed"]["mean"] == pytest.approx(0.46)
    assert result["final_root_speed"]["count"] == 3
    assert result["final_root_angular_speed"]["count"] == 3
    assert result["outcomes"]["success"] == 2
    assert "ever_goal_hold" not in result["outcomes"]
    assert "final_goal_hold" not in result["rates"]
    assert result["metric_definitions"]["goal_hold"]["goal_distance_below_m"] == 0.1


@pytest.mark.parametrize("condition", ["fall", "invalid", "distance", "unsettled"])
def test_goal_hold_requires_in_goal_settled_valid_upright_state(condition):
    class UnsafeHoldEnv(DifferentDurationEnv):
        def step(self, actions):
            obs, reward, done, extras = super().step(actions)
            done[:] = True
            metrics, flags = extras["cat_metrics"], extras["field_outcomes"]
            metrics["goal_distance"][:] = 0.05
            metrics["settled_stop"][:] = True
            flags["fall"][:] = False
            if condition in ("fall", "invalid"):
                flags[condition][:] = True
            elif condition == "distance":
                metrics["goal_distance"][:] = 0.1
            else:
                metrics["settled_stop"][:] = False
            return obs, reward, done, extras

    result = cohort.evaluate_cohort(
        UnsafeHoldEnv(), SimpleNamespace(predict=lambda obs: None), episodes=2
    )
    assert result["goal_hold_fraction"] == 0
    assert result["ever_goal_hold_count"] == 0
    assert result["final_goal_hold_count"] == 0
