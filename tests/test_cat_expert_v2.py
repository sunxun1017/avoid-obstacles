import copy
from pathlib import Path

import pytest
import torch

from embodied_learning.data.config import load_experiment_config
from embodied_learning.tasks.cat_curriculum import stage_settings


def test_v2_curriculum_changes_one_bar_dimension_at_a_time_and_keeps_anchors():
    original = load_experiment_config(Path("configs/experiments/g1_cat_source.yaml"))
    before = copy.deepcopy(original)
    stages = stage_settings(original, profile="cat_obstacle_expert_v2")
    assert original == before
    assert len(stages) == 9
    expected = [
        (0.04, 0.08),
        (0.08, 0.16),
        (0.10, 0.16),
        (0.10, 0.24),
        (0.12, 0.24),
        None,
        (0.08, 0.24),
        (0.10, 0.24),
        (0.12, 0.24),
    ]
    for i, (stage, dims) in enumerate(zip(stages, expected, strict=True)):
        main = stage["scenes"][2]
        bars = [b for b in main["boxes"] if b["center"][2] < 0.3]
        if dims is None:
            assert not bars
        else:
            assert len(bars) == 1
            assert (
                2 * bars[0]["half_size"][2],
                2 * bars[0]["half_size"][0],
            ) == pytest.approx(dims)
        assert [s["id"] for s in stage["scenes"][3:]] == [
            "anchor_flat",
            "anchor_hurdle",
            "anchor_ceiling",
        ]
        assert stage["scenes"][3:] == stages[0]["scenes"][3:]
        assert main["starts"] == [[-1.5, -0.12, 0.8], [-1.5, 0.12, 0.8]]
        if i >= 6:
            roof, bar = main["boxes"]
            assert bar["center"][0] - bar["half_size"][0] - (
                roof["center"][0] + roof["half_size"][0]
            ) == pytest.approx(0.70)
    assert (
        stages[-1]["scenes"][2]["boxes"]
        == stage_settings(original, profile="cat_obstacle_expert_v1")[-1]["scenes"][
            2
        ]["boxes"]
    )


def test_phase_push_is_once_per_episode_resets_per_slot_and_falls_back_to_timer():
    from embodied_learning.tasks.cat_phase_push_curriculum import TraversalPhasePush

    p = TraversalPhasePush(4, "cpu")
    front, back = torch.full((4,), 0.4), torch.full((4,), 0.64)
    feet = torch.tensor([[0.19, 0.0], [0.5, 0.2], [0.8, 0.9], [-1.0, -1.0]])
    contact = torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    deadline = torch.full((4,), 275)
    ids = p.select(torch.full((4,), 100), deadline, feet, contact, front, back)
    assert ids.tolist() == [0, 1, 2]
    assert (
        p.select(torch.full((4,), 101), deadline, feet, contact, front, back).numel()
        == 0
    )
    assert p.select(
        torch.full((4,), 275), deadline, feet, contact, front, back
    ).tolist() == [0, 1, 2, 3]
    # Caller reschedules the normal deadline after every returned push.
    deadline[:] = 550
    p.reset(torch.tensor([1]))
    feet[:] = -1
    assert (
        p.select(torch.full((4,), 549), deadline, feet, contact, front, back).numel()
        == 0
    )
    assert p.select(
        torch.full((4,), 550), deadline, feet, contact, front, back
    ).tolist() == [0, 1, 2, 3]
    assert p.counts.tolist() == [1, 1, 1]


def test_new_recipe_declares_training_mix_but_target_only_evaluation(tmp_path):
    import json

    from train_cat_curriculum import initialize, scene_selection

    original = load_experiment_config(Path("configs/experiments/g1_cat_source.yaml"))
    parent = tmp_path / "parent.pt"
    torch.save(
        {
            "field_contract": {"schema": {"task": original["task"]}},
            "curriculum": {"experiment": "p", "protocol_sha256": "p", "stage": 3},
            "completed_iterations": 600,
            "ppo_config": {"seed": 4},
        },
        parent,
    )
    root = tmp_path / "v2"
    initialize(
        root,
        recipe="cat_obstacle_expert_v2",
        parent_checkpoint=parent,
        training_seed=7,
        terminal_gate_seeds=(8, 9),
        training_disturbance_profile="unitree_g1_bridge_v1",
    )
    protocol = json.loads((root / "protocol.json").read_text())
    assert scene_selection(protocol, training=True) == ["hurdle_crouch"] * 5 + [
        "anchor_flat",
        "anchor_hurdle",
        "anchor_ceiling",
    ]
    assert scene_selection(protocol, training=False) == ["hurdle_crouch"]
    assert protocol["progression"] == "cat_obstacle_expert_v1"
    for stage in stage_settings(
        json.loads((root / "original.yaml").read_text()),
        profile=protocol["curriculum_profile"],
    ):
        assert stage["task"]["traversal_robustness"] == "obstacle_clearance_v1"
        assert stage["task"]["posture_reward"] == "overhead_clearance_v1"


def test_pinned_actor_bootstrap_migrates_v2_mix_but_strict_resume_rejects_old_rules(
    tmp_path,
):
    from types import SimpleNamespace

    from test_cat_curriculum import fixture_metadata
    from test_field_ppo import TinyEnv

    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist
    from embodied_learning.tasks.cat_foot_guidance import sha256

    env = TinyEnv()
    names = ["wall_detour", "passage", "hurdle_crouch"]
    env.schema = {
        **env.schema,
        "scene_names": names,
        "task": {"posture_reward": "overhead_clearance_v1"},
    }
    env.field_state = SimpleNamespace(allowed=(2,))
    env.curriculum_metadata = fixture_metadata(3)
    cfg = PpoConfig(
        iterations=1, rollout_steps=4, hidden_dims=(16, 8), epochs=1, mini_batches=1
    )
    parent = train_specialist(env, cfg, tmp_path / "parent")
    saved = torch.load(parent, weights_only=False)
    env.field_state.allowed = (2, 2, 2, 2, 2, 3, 4, 5)
    env.schema = {
        **env.schema,
        "scene_hash": "new",
        "scene_names": names + ["anchor_flat", "anchor_hurdle", "anchor_ceiling"],
        "task": {
            **env.schema["task"],
            "traversal_robustness": "obstacle_clearance_v1",
        },
    }
    env.curriculum_metadata = {
        **fixture_metadata(0),
        "experiment": "v2",
        "protocol_sha256": "v2",
        "expert_bootstrap": {
            "sha256": sha256(parent),
            "experiment": "test",
            "protocol_sha256": "protocol",
            "stage": 3,
            "completed_updates": 1,
            "training_seed": saved["ppo_config"]["seed"],
            "parent_rfi_multiplier": 1.0,
        },
    }
    child = train_specialist(env, cfg, tmp_path / "child", warmstart_actor=parent)
    got = torch.load(child, weights_only=False)
    changes = got["warmstart"]["contract_changes"]
    assert changes["task.traversal_robustness"] == {
        "before": None,
        "after": "obstacle_clearance_v1",
    }
    assert changes["training_scenes"] == {
        "before": [2],
        "after": [2, 2, 2, 2, 2, 3, 4, 5],
    }
    env.schema["task"] = {"posture_reward": "overhead_clearance_v1"}
    with pytest.raises(ValueError):
        train_specialist(env, cfg, tmp_path / "child", resume=child)
