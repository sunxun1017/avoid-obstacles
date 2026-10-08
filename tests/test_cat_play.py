"""Inference-only playback and cleanup, including partial checkpoints."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodied_learning.algorithms.dagger import evaluate_student
from embodied_learning.algorithms.ppo import PpoConfig, train_specialist
from embodied_learning.cli.field_play import play_policy
from embodied_learning.policies.field_policy import FieldStudent, load_teacher
from embodied_learning.protocols.imitation import PolicyContract


class PlayEnv:
    num_envs = 1
    num_actions = 12
    max_episode_length = 1000
    device = "cpu"
    cfg = {}
    schema = {"task_profile": "cat_source_v1", "normalization": False}
    scene_ids = torch.zeros(1, dtype=torch.long)
    episode_length_buf = torch.zeros(1, dtype=torch.long)

    def __init__(self):
        self.closed = False
        self.steps = 0

    def get_observations(self):
        return TensorDict(
            {"actor": torch.ones(1, 162), "critic": torch.ones(1, 250)},
            batch_size=[1],
        )

    def step(self, actions):
        self.steps += 1
        return (
            self.get_observations(),
            -actions.square().sum(-1),
            torch.tensor([self.steps % 2 == 0]),
            {
                "field_outcomes": {
                    "goal": torch.tensor([True]),
                    "collision": torch.tensor([False]),
                },
                "cat_metrics": {
                    "goal_distance": torch.tensor([0.3]),
                    "min_goal_distance": torch.tensor([0.05]),
                    "stopped": torch.tensor([True]),
                },
            },
        )

    def close(self):
        self.closed = True


def install_viewer(monkeypatch, *, fail=False):
    class Viewer:
        def __init__(self, env, policy, **kwargs):
            self.env = env
            self.policy = policy

        def run(self, **kwargs):
            if fail:
                raise RuntimeError("viewer error")
            for _ in range(2):
                self.env.step(self.policy(self.env.get_observations()))

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "mjlab.viewer",
        SimpleNamespace(NativeMujocoViewer=Viewer, ViserPlayViewer=Viewer),
    )


def test_play_no_parameter_or_optimizer_updates(monkeypatch):
    env = PlayEnv()
    policy = FieldStudent(PolicyContract.from_env(env, (16,)), "cpu")
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    install_viewer(monkeypatch)
    play_policy(env, policy, viewer="native", port=8765, steps=2)
    assert env.closed and env.steps == 2
    assert all(torch.equal(before[k], v) for k, v in policy.state_dict().items())


def test_play_closes_env_on_error(monkeypatch):
    env = PlayEnv()
    policy = FieldStudent(PolicyContract.from_env(env, (16,)), "cpu")
    install_viewer(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="viewer error"):
        play_policy(env, policy, viewer="native", port=8765)
    assert env.closed


def test_play_loads_intermediate_source_checkpoint(tmp_path):
    env = PlayEnv()
    config = PpoConfig(
        iterations=2,
        rollout_steps=4,
        hidden_dims=(16,),
        critic_hidden_dims=(32,),
        normalization=False,
        mini_batches=1,
        epochs=1,
        save_interval=1,
    )
    train_specialist(env, config, tmp_path)
    checkpoint = tmp_path / "model_0.pt"
    expected = PolicyContract.from_env(env, (16,), critic_hidden_dims=(32,))
    teacher = load_teacher(checkpoint, expected_contract=expected, device="cpu")
    assert teacher.predict(env.get_observations()).shape == (1, 12)
    with pytest.raises(ValueError, match="contract"):
        load_teacher(
            checkpoint,
            expected_contract=replace(expected, actor_dim=173),
            device="cpu",
        )


def test_evaluation_reports_ever_goal_and_final_distances():
    env = PlayEnv()
    policy = FieldStudent(PolicyContract.from_env(env, (16,)), "cpu")
    metrics = evaluate_student(env, policy, steps=2)
    assert metrics["rates"]["goal"] == 1.0
    assert metrics["mean_final_goal_distance"] == pytest.approx(0.3)
    assert metrics["mean_min_goal_distance"] == pytest.approx(0.05)
    assert metrics["stopped_step_fraction"] == 1.0
    empty = evaluate_student(PlayEnv(), policy, steps=1)
    assert (
        empty["rates"]["goal"] is None and empty["mean_final_goal_distance"] is None
    )


def test_source_has_explicit_contact_capacity():
    from embodied_learning.simulators.mjlab.cat_traversal import build_cat_cfg

    cfg = build_cat_cfg(num_envs=1, seed=42)
    # Autodetection undersizes one-env playback once arms/body hit the floor.
    assert cfg.sim.nconmax is not None and cfg.sim.nconmax >= 70
