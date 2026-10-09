"""Source task contracts and actual training/resume gates."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from embodied_learning.algorithms.ppo import PpoConfig, train_specialist
from embodied_learning.data.config import load_experiment_config
from embodied_learning.policies.field_policy import FieldStudent, load_teacher
from embodied_learning.protocols.imitation import PolicyContract


def test_asymmetric_networks_and_no_normalization():
    config = PpoConfig(
        hidden_dims=(256, 128, 64),
        critic_hidden_dims=(512, 256, 128),
        normalization=False,
        gamma=0.97,
        learning_rate=3e-4,
        rollout_steps=20,
        epochs=4,
    )
    runner = config.runner_config()
    assert runner["actor"]["hidden_dims"] == [256, 128, 64]
    assert runner["critic"]["hidden_dims"] == [512, 256, 128]
    assert runner["actor"]["obs_normalization"] is False
    assert runner["algorithm"]["gamma"] == 0.97
    c = PolicyContract(
        {"task_profile": "cat_source_v1", "normalization": False},
        162,
        250,
        12,
        (256, 128, 64),
        normalization=False,
        critic_hidden_dims=(512, 256, 128),
    )
    assert PolicyContract.from_dict(c.to_dict()) == c
    student = FieldStudent(c, "cpu")
    assert isinstance(student.model.obs_normalizer, torch.nn.Identity)


def test_source_contract_rejects_old_teacher_and_cache():
    c = PolicyContract(
        {"task_profile": "cat_source_v1"}, 162, 250, 12, (32,), normalization=False
    )
    with pytest.raises(ValueError, match="contract"):
        c.require(PolicyContract({}, 173, 173, 29, (32,)), network=False)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native source PPO requires CUDA"
)
def test_source_ppo_update_resume_and_dagger_frozen_teachers(tmp_path):
    from embodied_learning.algorithms.dagger import DaggerConfig, train_dagger
    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.simulators.mjlab.cat_traversal import make_cat_env
    from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields

    manifest = load_scene_manifest(Path("configs/experiments/g1_field.yaml"))
    profile = {"version": "cat_source_v1"}
    cache = prepare_fields(manifest, tmp_path / "fields", source_profile=profile)
    bank = load_field_cache(
        cache.path, manifest, device="cuda:0", source_profile=profile
    )
    config = PpoConfig(
        iterations=2,
        rollout_steps=4,
        hidden_dims=(32, 16),
        critic_hidden_dims=(64, 32),
        normalization=False,
        gamma=0.97,
        learning_rate=3e-4,
        epochs=1,
        mini_batches=1,
    )
    options = {
        "noise_level": 0.0,
        "random_pd": False,
        "random_rfi": False,
        "push": False,
    }
    experts = {}
    for i, scene in enumerate(manifest.scenes):
        env = make_cat_env(
            manifest,
            bank,
            scene_ids=(scene.id,),
            num_envs=4,
            device="cuda:0",
            seed=42,
            options=options,
        )
        try:
            path = train_specialist(env, config, tmp_path / scene.id)
            expected = PolicyContract.from_env(
                env,
                config.hidden_dims,
                critic_hidden_dims=config.critic_hidden_dims,
                normalization=False,
            )
            experts[i] = load_teacher(
                path, expected_contract=expected, device="cuda:0"
            )
            resumed = train_specialist(
                env,
                replace(config, iterations=1),
                tmp_path / (scene.id + "-resume"),
                resume=path,
            )
            saved = torch.load(resumed, weights_only=False)
            assert (
                saved["completed_iterations"] == 3 and saved["parameter_delta"] > 0
            )
        finally:
            env.close()
    env = make_cat_env(
        manifest,
        bank,
        scene_ids=tuple(s.id for s in manifest.scenes),
        num_envs=4,
        device="cuda:0",
        seed=42,
        options=options,
    )
    try:
        before = {
            i: {k: v.clone() for k, v in t.model.state_dict().items()}
            for i, t in experts.items()
        }
        student = FieldStudent(PolicyContract.from_env(env, (32, 16)), "cuda:0")
        config = DaggerConfig(
            iterations=2, rollout_steps=4, updates=2, batch_size=8, capacity=100
        )
        path = train_dagger(env, student, experts, config, tmp_path / "dagger")
        train_dagger(
            env,
            student,
            experts,
            replace(config, iterations=1),
            tmp_path / "dagger-resume",
            resume=path,
        )
        assert (
            torch.load(tmp_path / "dagger-resume/latest.pt", weights_only=False)[
                "completed_rounds"
            ]
            == 3
        )
        for i, t in experts.items():
            assert all(
                torch.equal(v, before[i][k]) for k, v in t.model.state_dict().items()
            )
    finally:
        env.close()


def test_source_cli_configuration_defaults():
    from embodied_learning.cli.field_training import _parser

    settings = load_experiment_config(Path("configs/experiments/g1_cat_source.yaml"))
    args = _parser().parse_args(
        [
            "ppo",
            "--config",
            "configs/experiments/g1_cat_source.yaml",
            "--scene",
            "wall_detour",
        ]
    )
    assert args.hidden_dims is None and args.rollout_steps is None
    config = PpoConfig(**settings["ppo"])
    assert config.runner_config()["critic"]["hidden_dims"] == [512, 256, 128]
    assert config.rollout_steps == 20 and config.epochs == 4
    assert all(s["voxel_size"] == 0.04 for s in settings["scenes"])


def test_wrong_rsl_editable_path_has_actionable_diagnostic(monkeypatch):
    import sys
    from types import SimpleNamespace

    from embodied_learning.cli.field_training import check_rsl_rl

    monkeypatch.setitem(
        sys.modules,
        "rsl_rl",
        SimpleNamespace(__file__="/example/amp/rsl_rl/__init__.py"),
    )
    monkeypatch.setitem(sys.modules, "rsl_rl.models", None)
    with pytest.raises(RuntimeError, match="/example/amp/rsl_rl") as error:
        check_rsl_rl()
    assert ".venv-field" in str(error.value)
