"""DAgger must label immutable student-visited states before stepping."""

from dataclasses import replace

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from embodied_learning.algorithms.dagger import (
    DaggerConfig,
    evaluate_student,
    train_dagger,
)
from embodied_learning.data.aggregation import AggregationBuffer
from embodied_learning.protocols.imitation import PolicyContract


class Env:
    num_envs, num_actions, device = 2, 1, "cpu"
    schema = {"action_scale": [1.0], "scene_hash": "tiny"}
    max_episode_length, cfg = 20, {}
    training_step = 0
    episode_length_buf = torch.zeros(2, dtype=torch.long)

    def __init__(self):
        self.state = torch.tensor([[1.0], [10.0]])
        self.scene_ids = torch.tensor([0, 1])
        self.executed = []

    def get_observations(self):
        return TensorDict(
            {"actor": self.state, "critic": self.state}, batch_size=[2]
        )

    def step(self, action):
        self.training_step += 1
        self.executed.append(action.clone())
        self.state.add_(action)
        return self.get_observations(), torch.ones(2), torch.zeros(2).long(), {}


class Student(nn.Module):
    def __init__(self, contract):
        super().__init__()
        self.contract = contract
        self.bias = nn.Parameter(torch.tensor(0.25))

    def forward(self, obs):
        return self.bias.expand(len(obs), 1)

    @torch.no_grad()
    def predict(self, obs):
        return self(obs)

    def update_normalization(self, obs):
        pass


class Teacher(nn.Module):
    def __init__(self, contract, tag):
        super().__init__()
        self.contract = contract
        self.tag = nn.Parameter(torch.tensor(float(tag)))
        self.visits = []

    def predict(self, obs):
        self.visits.append(obs["actor"].clone())
        return obs["actor"] + self.tag


def setup():
    env = Env()
    contract = PolicyContract.from_env(env, (8,))
    student = Student(contract)
    experts = {0: Teacher(contract, 0), 1: Teacher(contract, 100)}
    config = DaggerConfig(
        iterations=1,
        rollout_steps=2,
        updates=1,
        batch_size=4,
        capacity=100,
        beta_start=0.0,
    )
    return env, student, experts, config


def run(tmp_path):
    env, student, experts, config = setup()
    path = train_dagger(env, student, experts, config, tmp_path)
    return env, student, experts, config, torch.load(path, weights_only=False)


def test_labels_student_visited_states(tmp_path):
    _, _, experts, _, _ = run(tmp_path)
    assert torch.equal(experts[0].visits[1], torch.tensor([[1.25]]))
    assert torch.equal(experts[1].visits[1], torch.tensor([[10.25]]))


def test_teacher_scene_routing(tmp_path):
    _, _, _, _, saved = run(tmp_path)
    assert torch.equal(
        saved["buffer"]["labels"][:4],
        torch.tensor([[1.0], [110.0], [1.25], [110.25]]),
    )


def test_labels_before_env_step(tmp_path):
    _, _, _, _, saved = run(tmp_path)
    assert torch.equal(
        saved["buffer"]["observations"][:2], torch.tensor([[1.0], [10.0]])
    )


def test_old_rounds_retained(tmp_path):
    env, student, experts, config = setup()
    path = train_dagger(
        env, student, experts, replace(config, iterations=2), tmp_path
    )
    data = torch.load(path, weights_only=False)["buffer"]
    assert data["size"] == 8
    assert torch.equal(data["observations"][:2], torch.tensor([[1.0], [10.0]]))


def test_beta_zero_executes_student(tmp_path):
    env, _, _, _, _ = run(tmp_path)
    assert torch.equal(env.executed[0], torch.full((2, 1), 0.25))


def test_teacher_no_grad(tmp_path):
    _, _, experts, _, _ = run(tmp_path)
    assert all(e.tag.grad is None for e in experts.values())
    assert experts[1].tag.item() == 100


def test_capacity_bound():
    buffer = AggregationBuffer(3, 1, 1, "cpu")
    buffer.add(torch.arange(5.0).reshape(-1, 1), torch.arange(5.0).reshape(-1, 1))
    assert len(buffer) == 3
    assert set(buffer.state_dict()["observations"].flatten().tolist()) == {2, 3, 4}
    x, y = buffer.sample(100)
    assert torch.equal(x, y)


def test_resume_buffer_optimizer_schedule(tmp_path):
    env, student, experts, config = setup()
    first = train_dagger(env, student, experts, config, tmp_path / "first")
    saved = torch.load(first, weights_only=False)
    env2, student2, experts2, _ = setup()
    second = train_dagger(
        env2, student2, experts2, config, tmp_path / "second", resume=first
    )
    resumed = torch.load(second, weights_only=False)
    assert resumed["completed_rounds"] == 2
    assert resumed["buffer"]["seen"] == 8
    assert torch.equal(
        resumed["buffer"]["observations"][:4], saved["buffer"]["observations"][:4]
    )
    assert resumed["optimizer"]["state"][0]["step"] == 2
    assert resumed["scheduler"]["last_epoch"] == 2
    assert saved["environment_steps"] == 2
    assert resumed["environment_steps"] == 4


def test_eval_never_uses_teacher():
    env, student, experts, _ = setup()
    result = evaluate_student(env, student, steps=2)
    assert result["steps"] == 2
    assert not any(e.visits for e in experts.values())


def test_action_contract_mismatch(tmp_path):
    env, student, experts, config = setup()
    experts[0].contract = replace(experts[0].contract, action_dim=2)
    with pytest.raises(ValueError, match="contract"):
        train_dagger(env, student, experts, config, tmp_path)
    assert not env.executed


def test_real_dagger_with_ppo_teachers(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Real mjlab DAgger requires CUDA.")
    from pathlib import Path

    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist
    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.policies.field_policy import (
        FieldStudent,
        load_student,
        load_teacher,
    )
    from embodied_learning.simulators.mjlab.field_traversal import make_field_env
    from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields

    manifest = load_scene_manifest(Path("configs/experiments/g1_field.json"))
    cache = prepare_fields(manifest, tmp_path / "fields")
    bank = load_field_cache(cache.path, manifest, device="cuda:0")
    names = ("wall_detour", "passage")
    experts = {}
    ppo = PpoConfig(
        iterations=2, rollout_steps=4, epochs=1, mini_batches=1, hidden_dims=(32, 16)
    )
    for i, name in enumerate(names):
        teacher_env = make_field_env(
            manifest, bank, scene_ids=(name,), num_envs=4, device="cuda:0", seed=42
        )
        try:
            path = train_specialist(teacher_env, ppo, tmp_path / name)
            experts[i] = load_teacher(
                path,
                expected_contract=PolicyContract.from_env(
                    teacher_env, ppo.hidden_dims
                ),
                device="cuda:0",
            )
        finally:
            teacher_env.close()
    env = make_field_env(
        manifest, bank, scene_ids=names, num_envs=2, device="cuda:0", seed=42
    )
    try:
        contract = PolicyContract.from_env(env, (32, 16))
        student = FieldStudent(contract, "cuda:0")
        before = [p.detach().clone() for p in student.parameters()]
        teacher_before = {
            i: {k: v.clone() for k, v in e.model.state_dict().items()}
            for i, e in experts.items()
        }
        config = DaggerConfig(
            iterations=2,
            rollout_steps=4,
            updates=2,
            capacity=100,
            batch_size=8,
            beta_start=0.0,
        )
        path = train_dagger(env, student, experts, config, tmp_path / "student")
        saved = torch.load(path, weights_only=False)
        assert saved["buffer"]["seen"] == 16
        assert any(
            not torch.equal(a, b)
            for a, b in zip(before, student.parameters(), strict=True)
        )
        for i, expert in experts.items():
            for name, value in expert.model.state_dict().items():
                assert torch.equal(value, teacher_before[i][name])
        path = train_dagger(
            env,
            student,
            experts,
            replace(config, iterations=1),
            tmp_path / "resumed",
            resume=path,
        )
        assert torch.load(path, weights_only=False)["buffer"]["seen"] == 24
        loaded = load_student(path, expected_contract=contract, device="cuda:0")
        assert evaluate_student(env, loaded, steps=4)["steps"] == 4
    finally:
        env.close()
