"""CLI checkpoint consumers preserve the trained observation normalizer."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodied_learning.cli.field_training import main
from embodied_learning.data.config import load_experiment_config
from embodied_learning.data.field_scenes import load_scene_manifest
from embodied_learning.policies.field_policy import load_teacher
from embodied_learning.protocols.imitation import PolicyContract
from embodied_learning.robots.g1 import G1_BODY_SCHEMA


class CpuSourceEnv:
    """Replace GPU physics only; CLI, models, optimization and loading stay real."""

    num_envs, num_actions, max_episode_length = 2, 12, 1000
    device, cfg = "cpu", {}

    def __init__(self, manifest):
        self.schema = dict(
            task_profile="cat_source_v1",
            normalization=False,
            body_sites=list(G1_BODY_SCHEMA),
            scene_hash=manifest.identity,
            scene_names=[scene.id for scene in manifest.scenes],
            field_cache_identity="verified-cache",
            robot_model_identity="robot",
            action_transform="latent_tanh_v1",
        )
        self.scene_ids = torch.zeros(self.num_envs, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.field_state = SimpleNamespace(allowed=[0])
        self.actor_dim = 162
        self.steps = 0
        self.actions = []
        self.closed = False

    def get_observations(self):
        return TensorDict(
            {
                "actor": torch.arange(self.num_envs * self.actor_dim)
                .reshape(self.num_envs, self.actor_dim)
                .float()
                / 100
                + self.steps * 0.01,
                "critic": torch.arange(self.num_envs * 250)
                .reshape(self.num_envs, 250)
                .float()
                / 100
                + self.steps * 0.01,
            },
            batch_size=[self.num_envs],
        )

    def step(self, actions):
        self.actions.append(actions.clone())
        self.steps += 1
        return (
            self.get_observations(),
            -actions.tanh().square().sum(-1),
            torch.zeros(self.num_envs, dtype=torch.long),
            {},
        )

    def close(self):
        self.closed = True


@pytest.fixture(params=[False, True], ids=["unnormalized", "normalized"])
def trained_cli(tmp_path, monkeypatch, request):
    from embodied_learning.simulators.mjlab import cat_traversal
    from embodied_learning.tasks import field_assets

    settings = load_experiment_config(Path("configs/experiments/g1_cat_source.yaml"))
    settings["ppo"].update(
        hidden_dims=[8], critic_hidden_dims=[8], normalization=request.param
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings))
    manifest = load_scene_manifest(config)
    environments = []
    drift = {}

    def factory(*args, **kwargs):
        env = CpuSourceEnv(manifest)
        env.schema.update(drift.get("schema", {}))
        env.actor_dim = drift.get("actor_dim", 162)
        environments.append(env)
        return env

    monkeypatch.setattr(cat_traversal, "make_cat_env", factory)
    monkeypatch.setattr(field_assets, "load_field_cache", lambda *a, **k: None)
    common = [
        "--config",
        str(config),
        "--output-dir",
        str(tmp_path),
        "--scene",
        "wall_detour",
        "--device",
        "cpu",
        "--num-envs",
        "2",
    ]
    assert (
        main(
            [
                "ppo",
                *common,
                "--iterations",
                "1",
                "--rollout-steps",
                "2",
                "--epochs",
                "1",
                "--mini-batches",
                "1",
            ]
        )
        == 0
    )
    checkpoint = tmp_path / "specialists/wall_detour/latest.pt"
    saved = torch.load(checkpoint, weights_only=False)
    contract = PolicyContract.from_dict(saved["field_contract"])
    teacher = load_teacher(checkpoint, expected_contract=contract, device="cpu")
    return SimpleNamespace(
        common=common,
        checkpoint=checkpoint,
        teacher=teacher,
        environments=environments,
        drift=drift,
        normalization=request.param,
        output=tmp_path,
        factory=factory,
        manifest=manifest,
        config=config,
    )


@pytest.mark.parametrize("stage", ["evaluate", "play"])
def test_cli_inference_loads_the_trained_normalizer(trained_cli, stage, monkeypatch):
    if stage == "play":
        import mjlab.viewer

        class HeadlessViewer:
            def __init__(self, env, policy):
                self.env, self.policy = env, policy

            def run(self, *, num_steps):
                for _ in range(num_steps):
                    self.env.step(self.policy(self.env.get_observations()))

            def close(self):
                pass

        monkeypatch.setattr(mjlab.viewer, "NativeMujocoViewer", HeadlessViewer)
    case = trained_cli
    expected = case.teacher.predict(CpuSourceEnv(case.manifest).get_observations())
    assert (
        main(
            [
                stage,
                *case.common,
                "--checkpoint",
                str(case.checkpoint),
                "--steps",
                "2",
            ]
        )
        == 0
    )
    env = case.environments[-1]
    assert env.closed and len(env.actions) == 2
    torch.testing.assert_close(env.actions[0], expected)


@pytest.mark.parametrize("drift", ["dimension", "schema"])
def test_cli_normalizer_loading_still_rejects_incompatible_env(trained_cli, drift):
    case = trained_cli
    if drift == "dimension":
        case.drift["actor_dim"] = 161
    else:
        case.drift["schema"] = {"scene_hash": "different-scene"}
    with pytest.raises(ValueError, match="contract"):
        main(
            [
                "evaluate",
                *case.common,
                "--checkpoint",
                str(case.checkpoint),
                "--steps",
                "2",
            ]
        )
    assert not case.environments[-1].actions


def test_cli_dagger_trains_and_resumes_with_teacher_normalization(trained_cli):
    case = trained_cli
    arguments = [
        "dagger",
        *case.common,
        "--iterations",
        "1",
        "--rollout-steps",
        "2",
        "--updates",
        "1",
        "--batch-size",
        "2",
        "--capacity",
        "8",
    ]
    assert main(arguments) == 0
    checkpoint = case.output / "generalist/latest.pt"
    first = torch.load(checkpoint, weights_only=False)
    assert first["field_contract"]["normalization"] is case.normalization
    assert first["completed_rounds"] == 1
    assert main([*arguments, "--resume", str(checkpoint)]) == 0
    resumed = torch.load(checkpoint, weights_only=False)
    assert resumed["completed_rounds"] == 2
    assert resumed["buffer"]["seen"] == 8
    if case.normalization:
        assert (
            resumed["student"]["model.obs_normalizer.count"]
            > first["student"]["model.obs_normalizer.count"]
        )


def test_online_cli_loads_the_trained_normalizer(trained_cli, monkeypatch):
    from embodied_learning.cli.cat_online import main as online_main
    from embodied_learning.simulators.mjlab import cat_online

    case = trained_cli

    def online_factory(*args, **kwargs):
        env = case.factory(*args, **kwargs)
        env.schema.update(
            field_cache_identity=None,
            robot_model_identity="with-camera",
            online_perception=dict(
                version="cat_online_v1", sensorless_model_identity="robot"
            ),
        )
        env.cat_state = SimpleNamespace(
            options={}, bank=SimpleNamespace(diagnostics=lambda: {})
        )
        env.unwrapped = SimpleNamespace(
            action_manager=SimpleNamespace(
                get_term=lambda _: SimpleNamespace(cfg=SimpleNamespace())
            ),
            cat_controller=SimpleNamespace(
                kp_scale=torch.ones(2, 1),
                kd_scale=torch.ones(2, 1),
                rfi_scale=torch.zeros(2, 29),
            ),
            reset=lambda: None,
        )
        return env

    monkeypatch.setattr(cat_online, "make_cat_online_env", online_factory)
    expected = case.teacher.predict(CpuSourceEnv(case.manifest).get_observations())
    assert (
        online_main(
            [
                "--config",
                str(case.config),
                "--checkpoint",
                str(case.checkpoint),
                "--device",
                "cpu",
                "--num-envs",
                "2",
                "--steps",
                "2",
                "--execution",
                "sync",
            ]
        )
        == 0
    )
    env = case.environments[-1]
    assert env.closed and len(env.actions) == 2
    torch.testing.assert_close(env.actions[0], expected)
