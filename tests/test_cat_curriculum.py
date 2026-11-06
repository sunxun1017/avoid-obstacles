"""Curriculum geometry, promotion gates, and actual optimizer continuity."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from embodied_learning.tasks import cat_curriculum as curriculum


def test_last_stage_is_exact_original_and_easy_bar_stays_on_floor():
    from embodied_learning.data.config import load_experiment_config

    original = load_experiment_config(Path("configs/experiments/g1_cat_source.yaml"))
    before = copy.deepcopy(original)
    stages = curriculum.stage_settings(original)
    assert original == before
    assert stages[-1] == original
    easy = stages[0]["scenes"][2]["boxes"]
    assert easy[1]["center"][2] - easy[1]["half_size"][2] == 0
    assert 2 * easy[1]["half_size"][2] == pytest.approx(0.08)
    assert 2 * easy[1]["half_size"][0] == pytest.approx(0.16)
    assert easy[1]["center"][0] - easy[1]["half_size"][0] == pytest.approx(0.4)
    assert easy[0]["center"][2] - easy[0]["half_size"][2] == pytest.approx(1.40)
    gap = 0.4 - (easy[0]["center"][0] + easy[0]["half_size"][0])
    assert gap == pytest.approx(0.85)
    for stage in stages:
        assert stage["task"] == original["task"]
        assert stage["scenes"][:2] == original["scenes"][:2]
        assert stage["scenes"][2]["starts"] == original["scenes"][2]["starts"]


def test_foundation_starts_empty_then_isolated_bars_and_keeps_final_original():
    from embodied_learning.data.config import load_experiment_config

    original = load_experiment_config(Path("configs/experiments/g1_cat_source.yaml"))
    stages = curriculum.stage_settings(original, profile="foundation_v1")
    assert len(stages) == 7 and stages[-1] == original
    empty = stages[0]["scenes"][2]
    assert empty["boxes"] == []
    assert empty["starts"] == [[-0.5, 0.0, 0.8]]
    assert empty["goals"] == [[0.5, 0.0, 0.8]]
    assert all(stages[0]["task"][k] == 0 for k in ("ground", "lateral", "overhead"))
    for stage, height, depth in zip(
        stages[1:3], (0.04, 0.08), (0.08, 0.16), strict=True
    ):
        scene = stage["scenes"][2]
        assert len(scene["boxes"]) == 1
        bar = scene["boxes"][0]
        assert bar["center"][2] == height / 2
        assert bar["half_size"] == [depth / 2, 0.5, height / 2]
        assert scene["starts"] == [[-0.8, 0.0, 0.8]]
        assert scene["goals"] == [[0.8, 0.0, 0.8]]
        assert stage["task"]["ground"] == 1
        assert stage["task"]["lateral"] == stage["task"]["overhead"] == 0
    assert stages[3:] == curriculum.stage_settings(original)
    with pytest.raises(ValueError):
        curriculum.stage_settings(original, profile="typo")


def test_actor_stage_warmstart_keeps_actor_and_normalizer_but_fresh_critic_optimizer(
    tmp_path,
):
    from test_field_ppo import TinyEnv

    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist

    env = TinyEnv()
    env.curriculum_metadata = fixture_metadata(0)
    cfg = PpoConfig(
        iterations=2, rollout_steps=4, hidden_dims=(16, 8), epochs=1, mini_batches=1
    )
    parent = train_specialist(env, cfg, tmp_path / "parent")
    before = torch.load(parent, weights_only=False)
    env.schema = {**env.schema, "scene_hash": "bar"}
    env.curriculum_metadata = fixture_metadata(1)
    child = train_specialist(
        env, replace(cfg, iterations=1), tmp_path / "child", warmstart_actor=parent
    )
    after = torch.load(child, weights_only=False)
    init = torch.load(tmp_path / "child/actor-initialization.pt", weights_only=False)
    for key, value in before["actor_state_dict"].items():
        torch.testing.assert_close(
            init["actor_state_dict"][key], value, rtol=0, atol=0
        )
    assert any(
        not torch.equal(value, init["critic_state_dict"][key])
        for key, value in before["critic_state_dict"].items()
    )
    assert init["optimizer_state_dict"]["state"] == {}
    assert after["completed_iterations"] == 1
    assert {
        float(s["step"]) for s in after["optimizer_state_dict"]["state"].values()
    } == {1}
    assert after["warmstart"]["migration"] == "cat_curriculum_actor_only_v1"
    assert after["warmstart"]["critic_restored"] is False
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_specialist(
            env, cfg, tmp_path / "bad", resume=parent, warmstart_actor=parent
        )


def fixture_contract(stage):
    return dict(
        schema=dict(
            scene_hash=f"scene{stage}",
            field_cache_identity=f"fields{stage}",
            robot_model_identity=f"model{stage}",
            task={"ground": 1},
            observation="actor162",
            body_sites=["head", "foot"],
            foot_guidance_experiment=dict(
                foot_projection_radius_m=0.30,
                base_cache_identity=f"base{stage}",
                foot_manifest_identity=f"foot{stage}",
            ),
        ),
        actor_dim=162,
        critic_dim=250,
    )


def fixture_metadata(stage):
    return dict(
        stage=stage,
        experiment="test",
        protocol_sha256="protocol",
        robot_fingerprint="robot",
    )


def test_declared_reward_weight_actor_transfer_checks_parent_and_only_weight(
    tmp_path,
):
    from test_field_ppo import TinyEnv

    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist
    from embodied_learning.policies.cat_actor_warmstart import load_actor_warmstart
    from embodied_learning.protocols.imitation import PolicyContract
    from embodied_learning.tasks.cat_foot_guidance import sha256

    env = TinyEnv()
    env.schema = {**env.schema, "task": {}}
    env.curriculum_metadata = fixture_metadata(0)
    cfg = PpoConfig(
        iterations=1, rollout_steps=4, hidden_dims=(16, 8), epochs=1, mini_batches=1
    )
    parent = train_specialist(env, cfg, tmp_path / "parent")
    before = torch.load(parent, weights_only=False)
    env.schema = {
        **env.schema,
        "scene_hash": "bar",
        "task": {"root_tracking_weight": 1.5},
    }
    env.curriculum_metadata = {
        **fixture_metadata(1),
        "experiment": "tracking15",
        "protocol_sha256": "tracking15-protocol",
        "reward_weight_warmstart": dict(
            version="cat_root_tracking_weight_v1",
            parent_sha256=sha256(parent),
            parent_experiment="test",
            parent_protocol_sha256="protocol",
            before=1.0,
            after=1.5,
        ),
    }
    child = train_specialist(env, cfg, tmp_path / "child", warmstart_actor=parent)
    after = torch.load(child, weights_only=False)
    init = torch.load(tmp_path / "child/actor-initialization.pt", weights_only=False)
    for key, value in before["actor_state_dict"].items():
        torch.testing.assert_close(
            init["actor_state_dict"][key], value, rtol=0, atol=0
        )
    assert init["optimizer_state_dict"]["state"] == {}
    assert after["warmstart"]["contract_changes"]["task.root_tracking_weight"] == {
        "before": 1.0,
        "after": 1.5,
    }
    target = PolicyContract.from_dict(after["field_contract"])
    metadata = copy.deepcopy(env.curriculum_metadata)
    metadata["reward_weight_warmstart"]["parent_sha256"] = "wrong"
    with pytest.raises(ValueError, match="parent"):
        load_actor_warmstart(
            parent,
            contract=target,
            curriculum=metadata,
            training_scenes=before["training_scenes"],
            device="cpu",
        )
    altered = copy.deepcopy(after["field_contract"])
    altered["schema"]["task"]["noise_level"] = 0.0
    with pytest.raises(ValueError, match="contract"):
        load_actor_warmstart(
            parent,
            contract=PolicyContract.from_dict(altered),
            curriculum=env.curriculum_metadata,
            training_scenes=before["training_scenes"],
            device="cpu",
        )


def test_transfer_allows_only_declared_scene_changes_and_forward_adjacent_stage():
    old, new = fixture_contract(0), fixture_contract(1)
    source, target = fixture_metadata(0), fixture_metadata(1)
    changes = curriculum.validate_transfer(
        old, new, source, target, actor_warmstart=True
    )
    assert "scene_hash" in changes
    for bad in ("observation", "body_sites", "task"):
        altered = copy.deepcopy(new)
        altered["schema"][bad] = "bad"
        with pytest.raises(ValueError, match="contract"):
            curriculum.validate_transfer(
                old, altered, source, target, actor_warmstart=True
            )
    altered = copy.deepcopy(new)
    altered["schema"]["foot_guidance_experiment"]["foot_projection_radius_m"] = 0.35
    with pytest.raises(ValueError, match="contract"):
        curriculum.validate_transfer(
            old, altered, source, target, actor_warmstart=True
        )
    for name, value in (
        ("stage", 2),
        ("protocol_sha256", "other"),
        ("robot_fingerprint", "changed"),
    ):
        with pytest.raises(ValueError):
            curriculum.validate_transfer(
                old, new, source, {**target, name: value}, actor_warmstart=True
            )
    with pytest.raises(ValueError):
        curriculum.validate_transfer(old, new, source, source)


def test_scene_transition_keeps_optimizer_steps_and_global_iterations(tmp_path):
    from test_field_ppo import TinyEnv

    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist

    env = TinyEnv()
    env.schema = {**env.schema, "scene_hash": "easy"}
    env.curriculum_metadata = fixture_metadata(0)
    config = PpoConfig(
        iterations=2, rollout_steps=4, hidden_dims=(16, 8), epochs=1, mini_batches=1
    )
    first = train_specialist(env, config, tmp_path / "easy")
    before = torch.load(first, weights_only=False)
    assert before["curriculum"]["stage"] == 0
    assert {
        float(s["step"]) for s in before["optimizer_state_dict"]["state"].values()
    } == {2}
    env.schema = {**env.schema, "scene_hash": "harder"}
    env.curriculum_metadata = fixture_metadata(1)
    with pytest.raises(ValueError, match="Resume.*stage"):
        train_specialist(
            env, replace(config, iterations=1), tmp_path / "harder", resume=first
        )
    env.schema = {**env.schema, "scene_hash": "easy"}
    env.curriculum_metadata = fixture_metadata(0)
    second = train_specialist(
        env, replace(config, iterations=1), tmp_path / "same-stage", resume=first
    )
    after = torch.load(second, weights_only=False)
    assert after["completed_iterations"] == 3
    assert after["curriculum"]["stage"] == 0
    assert {
        float(s["step"]) for s in after["optimizer_state_dict"]["state"].values()
    } == {3}
    assert after["curriculum_transition"]["parent_completed_iterations"] == 2
    assert after["parameter_delta"] > 0
