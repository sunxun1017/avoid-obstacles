"""Strict actor contracts and real official PPO optimization."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from embodied_learning.algorithms.ppo import PpoConfig, train_specialist
from embodied_learning.policies.field_policy import load_teacher
from embodied_learning.protocols.imitation import PolicyContract


class TinyEnv:
    num_envs, num_actions, max_episode_length = 4, 2, 10
    device, cfg = "cpu", {}
    schema = {
        "body_sites": ["root"],
        "joint_names": ["a", "b"],
        "action_scale": [1.0, 1.0],
        "scene_hash": "test",
    }
    scene_ids = torch.zeros(4, dtype=torch.long)
    episode_length_buf = torch.zeros(4, dtype=torch.long)

    def get_observations(self):
        return TensorDict(
            {
                "actor": torch.arange(12, dtype=torch.float32).reshape(4, 3),
                "critic": torch.arange(12, dtype=torch.float32).reshape(4, 3),
            },
            batch_size=[4],
        )

    def step(self, actions):
        return (
            self.get_observations(),
            -actions.square().sum(1),
            torch.zeros(4, dtype=torch.long),
            {},
        )


@pytest.fixture
def checkpoint(tmp_path):
    env = TinyEnv()
    config = PpoConfig(
        iterations=2, rollout_steps=4, hidden_dims=(16, 8), epochs=1, mini_batches=1
    )
    path = train_specialist(env, config, tmp_path)
    return env, config, path


@pytest.mark.parametrize("change", ["schema", "actor_dim", "action_dim"])
def test_reject_schema_action_scene_mismatch(checkpoint, change):
    env, config, path = checkpoint
    contract = PolicyContract.from_env(env, config.hidden_dims)
    replacement = {
        "schema": {**contract.schema, "scene_hash": "different"},
        "actor_dim": 4,
        "action_dim": 3,
    }
    with pytest.raises(ValueError, match="contract"):
        load_teacher(
            path,
            expected_contract=replace(contract, **{change: replacement[change]}),
            device="cpu",
        )


def test_teacher_loads_normalizer(checkpoint):
    env, config, path = checkpoint
    teacher = load_teacher(
        path,
        expected_contract=PolicyContract.from_env(env, config.hidden_dims),
        device="cpu",
    )
    saved = torch.load(path, weights_only=False)
    assert torch.equal(
        teacher.model.state_dict()["obs_normalizer._mean"],
        saved["actor_state_dict"]["obs_normalizer._mean"],
    )
    assert teacher.model.obs_normalizer.count > 0


def test_teacher_deterministic_frozen(checkpoint):
    env, config, path = checkpoint
    teacher = load_teacher(
        path,
        expected_contract=PolicyContract.from_env(env, config.hidden_dims),
        device="cpu",
    )
    obs = env.get_observations()
    a, b = teacher.predict(obs), teacher.predict(obs)
    assert torch.equal(a, b)
    assert not a.requires_grad
    assert not teacher.model.training
    assert all(not p.requires_grad for p in teacher.model.parameters())


def test_checkpoint_retains_contract(checkpoint):
    env, config, path = checkpoint
    saved = torch.load(path, weights_only=False)
    assert saved["field_contract"]["schema"] == env.schema
    assert saved["completed_iterations"] == 2
    assert saved["optimizer_state_dict"]["state"]
    assert saved["parameter_delta"] > 0


def test_ppo_update_and_resume(checkpoint, tmp_path):
    env, config, path = checkpoint
    before = torch.load(path, weights_only=False)
    resumed = train_specialist(
        env, replace(config, iterations=1), tmp_path / "resume", resume=path
    )
    after = torch.load(resumed, weights_only=False)
    assert after["completed_iterations"] == 3
    assert after["parameter_delta"] > 0
    assert not torch.equal(
        before["actor_state_dict"]["mlp.0.weight"],
        after["actor_state_dict"]["mlp.0.weight"],
    )


def test_real_two_scene_ppo(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Real mjlab PPO requires CUDA.")
    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.simulators.mjlab.field_traversal import make_field_env
    from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields

    manifest = load_scene_manifest(Path("configs/experiments/g1_field.json"))
    cache = prepare_fields(manifest, tmp_path / "fields")
    bank = load_field_cache(cache.path, manifest, device="cuda:0")
    config = PpoConfig(
        iterations=2, rollout_steps=4, hidden_dims=(32, 16), epochs=1, mini_batches=1
    )
    for name in ("wall_detour", "passage"):
        env = make_field_env(
            manifest, bank, scene_ids=(name,), num_envs=4, device="cuda:0", seed=42
        )
        try:
            assert len(env.schema['action_scale']) == 29
            path = train_specialist(env, config, tmp_path / name)
            teacher = load_teacher(
                path,
                expected_contract=PolicyContract.from_env(env, config.hidden_dims),
                device="cuda:0",
            )
            assert torch.isfinite(teacher.predict(env.get_observations())).all()
            resumed = train_specialist(
                env,
                replace(config, iterations=1),
                tmp_path / (name + "-resume"),
                resume=path,
            )
            saved = torch.load(resumed, weights_only=False)
            assert saved["completed_iterations"] == 3
            assert saved["parameter_delta"] > 0
        finally:
            env.close()
