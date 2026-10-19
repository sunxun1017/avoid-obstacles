"""Explicit source-to-recovery migration never weakens strict PPO resume."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from embodied_learning.algorithms.ppo import PpoConfig, train_specialist
from embodied_learning.cli.field_training import _parser, main, preflight_teachers
from embodied_learning.data.config import load_experiment_config
from embodied_learning.protocols.imitation import PolicyContract
from embodied_learning.robots.g1 import G1_BODY_SCHEMA


class CatContractEnv:
    num_envs, num_actions, max_episode_length = 4, 12, 1000
    device, cfg = "cpu", {}

    def __init__(self, *, recovery=False):
        self.actor_dim, self.critic_dim = (171, 256) if recovery else (162, 250)
        self.schema = dict(
            task_profile="cat_recovery_v1" if recovery else "cat_source_v1",
            observation="cat_recovery_actor171_critic256_v1"
            if recovery
            else "cat_source_actor162_critic250_v1",
            field_cache_identity="same-fields",
            robot_model_identity="same-physics",
            source_commit="866ba392f1c1e84b92ad75fa66550f26e8af8e48",
            scene_hash="same-scenes",
            scene_names=["wall_detour"],
            body_sites=list(G1_BODY_SCHEMA),
            joint_names=[f"joint_{i}" for i in range(29)],
            action_names=[f"joint_{i}" for i in range(12)],
            obs_joint_names=[f"joint_{i}" for i in range(23)],
            action_scale=0.5,
            action_transform="latent_tanh_v1",
            task={"noise_level": 0.0 if recovery else 1.0},
            source_profile={"version": "cat_source_v1"},
            physics={"timestep": 0.002, "decimation": 10},
            normalization=False,
        )
        if recovery:
            self.schema["recovery"] = {"braking_radius": 0.4}
        self.scene_ids = torch.zeros(self.num_envs, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.training_step = 0

    def get_observations(self):
        return TensorDict(
            {
                "actor": torch.linspace(-1, 1, self.actor_dim).repeat(
                    self.num_envs, 1
                ),
                "critic": torch.linspace(-1, 1, self.critic_dim).repeat(
                    self.num_envs, 1
                ),
            },
            batch_size=[self.num_envs],
        )

    def step(self, actions):
        self.training_step += 1
        return (
            self.get_observations(),
            -actions.tanh().square().sum(-1),
            torch.zeros(self.num_envs, dtype=torch.long),
            {},
        )


@pytest.fixture
def source_checkpoint(tmp_path):
    config = PpoConfig(
        iterations=2,
        rollout_steps=2,
        hidden_dims=(8,),
        critic_hidden_dims=(12,),
        normalization=False,
        epochs=1,
        mini_batches=1,
        save_interval=1,
    )
    path = train_specialist(CatContractEnv(), config, tmp_path / "source")
    return config, path


def recovery_contract(config):
    return PolicyContract.from_env(
        CatContractEnv(recovery=True),
        config.hidden_dims,
        critic_hidden_dims=config.critic_hidden_dims,
        normalization=False,
    )


def recovery_models(config):
    observations = CatContractEnv(recovery=True).get_observations()
    groups = {"actor": ["actor"], "critic": ["critic"]}
    models = []
    for group, output in (("actor", 12), ("critic", 1)):
        model_config = config.runner_config(action_transform="latent_tanh_v1")[group]
        model_config.pop("class_name")
        models.append(MLPModel(observations, groups, group, output, **model_config))
    return models


def test_cli_warmstart_is_explicit_and_mutually_exclusive_with_resume():
    args = _parser().parse_args(["ppo", "--warmstart", "parent.pt"])
    assert args.warmstart == Path("parent.pt")
    with pytest.raises(SystemExit):
        _parser().parse_args(
            ["ppo", "--warmstart", "parent.pt", "--resume", "existing.pt"]
        )


def test_warmstart_preserves_old_features_and_ignores_added_features(
    source_checkpoint,
):
    from embodied_learning.policies.cat_warmstart import (
        initialize_cat_warmstart,
        load_cat_warmstart,
    )

    config, path = source_checkpoint
    saved, provenance = load_cat_warmstart(
        path,
        expected_contract=recovery_contract(config),
        training_scenes=[0],
        device="cpu",
    )
    actor, critic = recovery_models(config)
    initialize_cat_warmstart(actor, critic, saved)
    for model, group, prefix in ((actor, "actor", 162), (critic, "critic", 250)):
        state = model.state_dict()
        original = saved[group + "_state_dict"]
        for key, value in original.items():
            if key == "mlp.0.weight":
                torch.testing.assert_close(state[key][:, :prefix], value)
                assert torch.count_nonzero(state[key][:, prefix:]) == 0
            else:
                torch.testing.assert_close(state[key], value)
        obs = CatContractEnv(recovery=True).get_observations()
        before = model(obs).detach()
        obs[group][:, prefix:] = 1e6
        torch.testing.assert_close(model(obs), before)
    assert (
        provenance["parent_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert provenance["parent_contract"] == saved["field_contract"]


@pytest.mark.parametrize(
    "drift",
    ["profile", "normalization", "scene", "model", "cache", "actions", "network"],
)
def test_warmstart_rejects_unsupported_migration(source_checkpoint, drift):
    from embodied_learning.policies.cat_warmstart import load_cat_warmstart

    config, path = source_checkpoint
    contract = copy.deepcopy(recovery_contract(config))
    if drift == "normalization":
        contract = replace(contract, normalization=True)
    elif drift == "network":
        contract = replace(contract, hidden_dims=(16,))
    elif drift != "scene":
        key, value = {
            "profile": ("task_profile", "cat_other_v1"),
            "model": ("robot_model_identity", "other-physics"),
            "cache": ("field_cache_identity", "other-fields"),
            "actions": (
                "action_names",
                list(reversed(contract.schema["action_names"])),
            ),
        }[drift]
        contract.schema[key] = value
    with pytest.raises(ValueError, match="[Ww]armstart"):
        load_cat_warmstart(
            path,
            expected_contract=contract,
            training_scenes=[1] if drift == "scene" else [0],
            device="cpu",
        )


def test_warmstart_rejects_unrecognized_state_keys(source_checkpoint):
    from embodied_learning.policies.cat_warmstart import initialize_cat_warmstart

    config, path = source_checkpoint
    saved = torch.load(path, weights_only=False)
    saved["actor_state_dict"]["unexpected"] = torch.ones(1)
    with pytest.raises(ValueError, match="[Ww]armstart"):
        initialize_cat_warmstart(*recovery_models(config), saved)


def test_warmstart_restarts_training_and_resume_retains_provenance(
    source_checkpoint, tmp_path
):
    config, source = source_checkpoint
    parent = torch.load(source, weights_only=False)
    env = CatContractEnv(recovery=True)
    config = replace(config, iterations=1)
    path = train_specialist(env, config, tmp_path / "recovery", warmstart=source)
    warmed = torch.load(path, weights_only=False)
    assert warmed["completed_iterations"] == 1
    assert warmed["environment_steps"] == 2
    assert warmed["warmstart"]["parent_contract"] == parent["field_contract"]
    assert warmed["parameter_delta"] > 0
    assert all(
        state["step"] == 1
        for state in warmed["optimizer_state_dict"]["state"].values()
    )
    periodic = torch.load(path.parent / "model_0.pt", weights_only=False)
    assert periodic["warmstart"] == warmed["warmstart"]
    resumed = train_specialist(env, config, tmp_path / "resumed", resume=path)
    after = torch.load(resumed, weights_only=False)
    assert after["completed_iterations"] == 2
    assert after["warmstart"] == warmed["warmstart"]
    with pytest.raises(ValueError, match="contract"):
        train_specialist(env, config, tmp_path / "invalid-resume", resume=source)
    with pytest.raises(ValueError, match="configuration"):
        train_specialist(
            env,
            replace(config, mini_batches=2),
            tmp_path / "changed-resume",
            resume=path,
        )
    records = preflight_teachers(
        {"wall_detour": path},
        scene_hash="same-scenes",
        scene_names=("wall_detour",),
        task_profile="cat_recovery_v1",
    )
    assert records["wall_detour"]["completed_iterations"] == 1


def test_warmstart_cannot_overwrite_an_existing_training_run(
    source_checkpoint, tmp_path
):
    config, source = source_checkpoint
    original = source.read_bytes()
    with pytest.raises(ValueError, match="new output directory"):
        train_specialist(
            CatContractEnv(recovery=True),
            config,
            source.parent,
            warmstart=source,
        )
    assert source.read_bytes() == original


def test_cli_research_options_and_yaml_save_interval_reach_checkpoints(
    tmp_path, monkeypatch
):
    from embodied_learning.simulators.mjlab import cat_traversal
    from embodied_learning.tasks import field_assets

    # Replace unavailable GPU physics, retaining real CLI/config/PPO/checkpoints.
    def factory(
        manifest,
        bank,
        *,
        scene_ids,
        num_envs,
        device,
        seed,
        options,
        task_profile="cat_source_v1",
        recovery_options=None,
    ):
        env = CatContractEnv(recovery=task_profile == "cat_recovery_v1")
        env.schema.update(
            scene_hash=manifest.identity,
            scene_names=[scene.id for scene in manifest.scenes],
            task=options,
        )
        if recovery_options is not None:
            env.schema["recovery"] = recovery_options
        env.close = lambda: None
        return env

    monkeypatch.setattr(cat_traversal, "make_cat_env", factory)
    monkeypatch.setattr(field_assets, "load_field_cache", lambda *args, **kw: None)
    settings = load_experiment_config(Path("configs/experiments/g1_cat_source.yaml"))
    settings["ppo"].update(
        hidden_dims=[8],
        critic_hidden_dims=[12],
        save_interval=1,
        rollout_steps=2,
        epochs=1,
        mini_batches=1,
    )
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps(settings))
    common = [
        "ppo",
        "--config",
        str(config_path),
        "--scene",
        "wall_detour",
        "--num-envs",
        "4",
        "--device",
        "cpu",
        "--iterations",
        "2",
    ]
    source_run, research_run = tmp_path / "source", tmp_path / "research"
    assert main([*common, "--output-dir", str(source_run)]) == 0
    parent = source_run / "specialists/wall_detour/latest.pt"
    settings.update(task_profile="cat_recovery_v1", recovery={"braking_radius": 0.6})
    config_path.write_text(json.dumps(settings))
    assert (
        main(
            [
                *common,
                "--output-dir",
                str(research_run),
                "--warmstart",
                str(parent),
            ]
        )
        == 0
    )
    checkpoint = research_run / "specialists/wall_detour/latest.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["field_contract"]["schema"]["recovery"] == {"braking_radius": 0.6}
    assert saved["field_contract"]["actor_dim"] == 171
    assert saved["ppo_config"]["save_interval"] == 1
    assert (checkpoint.parent / "model_1.pt").is_file()
