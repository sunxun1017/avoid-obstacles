"""Observable regressions identified by the final independent review."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodied_learning.robots.g1 import CAT_JOINT_NAMES
from embodied_learning.simulators.mjlab.cat_control import CatJointController
from embodied_learning.simulators.mjlab.cat_traversal import CatEnv
from embodied_learning.tasks.cat_reward import CatRewardInputs
from embodied_learning.tasks.cat_traversal import CatTaskState


def task_case():
    fixture = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    b = 16

    class Bank:
        source_profile = {"version": "cat_source_v1"}

        def sample_source(self, p, scene, goal):
            return SimpleNamespace(
                guidance=torch.from_numpy(fixture["input_gf"]),
                sdf_gradient=torch.from_numpy(fixture["input_bf"]),
                sdf=torch.from_numpy(fixture["input_sdf"]),
            )

    state = CatTaskState(
        SimpleNamespace(), Bank(), (0,), b, "cpu", {"noise_level": 0.0}
    )
    c = CatJointController(
        CAT_JOINT_NAMES, torch.full((29,), -2.0), torch.full((29,), 2.0), b, "cpu"
    )
    inputs = CatRewardInputs(
        **{
            k[6:]: torch.from_numpy(fixture[k]).clone()
            for k in fixture.files
            if k.startswith("input_")
        }
    )
    p = torch.zeros(b, 11, 3)
    p[:, 0, 2] = 1.0
    payload = dict(
        reward_inputs=inputs,
        body_positions=p,
        root_position=torch.zeros(b, 3),
        root_rotation=torch.eye(3).repeat(b, 1, 1),
        gyro=torch.zeros(b, 3),
        gravity=torch.zeros(b, 3),
        up_z=torch.ones(b),
        self_contact=torch.zeros(b, dtype=torch.bool),
        physical_collision=torch.zeros(b, dtype=torch.bool),
        root_local_vel=torch.zeros(b, 3),
        invalid=torch.zeros(b, dtype=torch.bool),
    )
    state.reset(torch.arange(b))
    return state, c, payload, fixture


def test_reset_normalization_and_command_match_executed_author():
    state, c, p, f = task_case()
    state.refresh(p, c, torch.zeros(16, dtype=torch.long), 0)
    obs = state.observations()
    torch.testing.assert_close(
        obs["critic"][:, 88:91], torch.from_numpy(f["reset_gf"][:, 0])
    )
    torch.testing.assert_close(
        obs["critic"][:, 91:94], torch.from_numpy(f["reset_bf"][:, 0])
    )
    torch.testing.assert_close(
        obs["critic"][:, 76:80], torch.from_numpy(f["reset_command"])
    )


def test_resume_counter_does_not_advance_physical_state():
    state, c, p, _ = task_case()
    state.refresh(p, c, torch.zeros(16, dtype=torch.long), 0)
    before = state.observations().clone()
    phase = state.gait.phase.clone()
    env = CatEnv.__new__(CatEnv)
    native = SimpleNamespace(common_step_counter=0)
    native.unwrapped = native
    env.env = native
    env.cat_state = state
    env.training_step = 123
    state.refresh(p, c, torch.zeros(16, dtype=torch.long), env.training_step)
    torch.testing.assert_close(state.gait.phase, phase)
    torch.testing.assert_close(state.observations()["actor"], before["actor"])


def test_legacy_checkpoint_and_config_resume_in_original_directory(tmp_path):
    from test_cat_play import PlayEnv

    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist

    class LegacyEnv(PlayEnv):
        schema = {"legacy": True}

    env = LegacyEnv()
    config = PpoConfig(
        iterations=2, rollout_steps=4, hidden_dims=(16,), mini_batches=1, epochs=1
    )
    path = train_specialist(env, config, tmp_path)
    saved = torch.load(path, weights_only=False)
    old_config = json.loads((tmp_path / "config.json").read_text())
    for key in ("critic_hidden_dims", "normalization", "gamma"):
        saved["ppo_config"].pop(key)
        old_config.pop(key)
    saved["field_contract"].pop("critic_hidden_dims")
    torch.save(saved, path)
    (tmp_path / "config.json").write_text(json.dumps(old_config))
    train_specialist(env, replace(config, iterations=1), tmp_path, resume=path)
    assert torch.load(path, weights_only=False)["completed_iterations"] == 3


@__import__("pytest").mark.skipif(
    not torch.cuda.is_available(), reason="Native adapter requires CUDA"
)
def test_source_site_velocity_and_asset_contracts(tmp_path):
    import pytest

    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.protocols.imitation import PolicyContract
    from embodied_learning.robots.g1 import G1_BODY_SCHEMA
    from embodied_learning.simulators.mjlab.cat_traversal import (
        make_cat_env,
        read_cat_reward_inputs,
    )
    from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields

    m = load_scene_manifest(Path("configs/experiments/g1_field.json"))
    profile = {"version": "cat_source_v1"}
    cache = prepare_fields(m, tmp_path, source_profile=profile)
    bank = load_field_cache(cache.path, m, device="cuda:0", source_profile=profile)
    env = make_cat_env(
        m,
        bank,
        scene_ids=("wall_detour",),
        num_envs=1,
        device="cuda:0",
        seed=42,
        options={
            "noise_level": 0.0,
            "random_pd": False,
            "random_rfi": False,
            "push": False,
        },
    )
    try:
        robot = env.unwrapped.scene["robot"]
        state = torch.cat(
            (robot.data.root_link_pose_w, robot.data.root_link_vel_w), 1
        ).clone()
        state[:, 7:] = 0.0
        state[:, 11] = 1.0  # World pitch angular velocity.
        robot.write_root_state_to_sim(state)
        env.unwrapped.sim.forward()
        env.unwrapped.sim.sense()
        payload = read_cat_reward_inputs(env.unwrapped, env.cat_state)
        site = robot.site_names.index(G1_BODY_SCHEMA[1])
        expected = robot.data.site_vel_w[:, site, :3]
        assert (expected - robot.data.root_link_vel_w[:, :3]).norm() > 0.05
        torch.testing.assert_close(payload["reward_inputs"].root_vel, expected)
        assert env.schema["field_cache_identity"] == bank.cache_identity
        identity = env.schema["robot_model_identity"]
        assert isinstance(identity, str) and len(identity) == 64
        contract = PolicyContract.from_env(env, (16,))
        drift = replace(
            contract, schema={**contract.schema, "robot_model_identity": "changed"}
        )
        with pytest.raises(ValueError, match="contract"):
            contract.require(drift)
    finally:
        env.close()


def test_model_identity_changes_with_native_numeric_asset():
    from embodied_learning.simulators.mjlab.cat_traversal import (
        native_model_identity,
    )

    model = SimpleNamespace(
        body_mass=np.array([1.0, 2.0]), body_inertia=np.ones((2, 3)), names=b"robot"
    )
    before = native_model_identity(model)
    model.body_mass[1] = 3.0
    assert native_model_identity(model) != before


def test_verified_cache_identity_changes_for_a_different_valid_cache(tmp_path):
    import hashlib

    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields

    m = load_scene_manifest(Path("configs/experiments/g1_field.json"))
    profile = {"version": "cat_source_v1"}
    cache = prepare_fields(m, tmp_path, source_profile=profile)
    before = load_field_cache(cache.path, m, device="cpu", source_profile=profile)
    metadata = json.loads(cache.path.read_text())
    record = metadata["fields"][0]
    target = tmp_path / record["file"]
    with np.load(target) as arrays:
        values = {k: arrays[k].copy() for k in arrays.files}
    values["guidance"] += 0.001
    np.savez_compressed(target, **values)
    record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    cache.path.write_text(json.dumps(metadata))
    after = load_field_cache(cache.path, m, device="cpu", source_profile=profile)
    assert before.cache_identity != after.cache_identity


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native CLI evaluation requires CUDA"
)
def test_no_perturbations_cleans_the_first_cached_observation(tmp_path, monkeypatch):
    from embodied_learning.algorithms import dagger
    from embodied_learning.cli.field_training import main
    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.policies.field_policy import FieldStudent
    from embodied_learning.protocols.imitation import PolicyContract
    from embodied_learning.simulators.mjlab import cat_traversal
    from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields

    config = json.loads(Path("configs/experiments/g1_field.json").read_text())
    options = dict(noise_level=1.0, random_pd=False, random_rfi=False, push=False)
    config.update(
        task_profile="cat_source_v1",
        source_profile={"version": "cat_source_v1"},
        task=options,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    m = load_scene_manifest(path)
    profile = config["source_profile"]
    cache = prepare_fields(m, tmp_path / "fields", source_profile=profile)
    bank = load_field_cache(cache.path, m, device="cuda:0", source_profile=profile)
    env = cat_traversal.make_cat_env(
        m,
        bank,
        scene_ids=("wall_detour",),
        num_envs=1,
        device="cuda:0",
        seed=42,
        options=options,
    )
    contract = PolicyContract.from_env(env, (16,))
    student = FieldStudent(contract, "cuda:0")
    checkpoint = tmp_path / "student.pt"
    torch.save(
        {"field_contract": contract.to_dict(), "student": student.state_dict()},
        checkpoint,
    )
    monkeypatch.setattr(cat_traversal, "make_cat_env", lambda *_a, **_k: env)

    def evaluate(e, p, *, steps):
        obs = e.get_observations()
        torch.testing.assert_close(obs["actor"][:, :52], obs["critic"][:, :52])
        assert e.cat_state.options["noise_level"] == 0.0
        return {}

    monkeypatch.setattr(dagger, "evaluate_student", evaluate)
    main(
        [
            "evaluate",
            "--config",
            str(path),
            "--output-dir",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint),
            "--scene",
            "wall_detour",
            "--num-envs",
            "1",
            "--steps",
            "1",
            "--no-perturbations",
        ]
    )
