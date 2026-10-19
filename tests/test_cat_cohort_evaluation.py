"""A complete-episode cohort must not select only fast terminating slots."""

import importlib.util
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
