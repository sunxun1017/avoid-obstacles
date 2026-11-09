"""No low-head reward exemption without local overhead geometry."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from embodied_learning.data.field_scenes import FieldScene, SceneBox, SceneManifest
from embodied_learning.tasks.cat_reward import (
    CatRewardInputs,
    CatRewardParameters,
    cat_reward_terms,
)


def scene(boxes):
    return FieldScene(
        "case",
        (-3.0, -3.0, -0.2),
        (151, 151, 76),
        0.04,
        tuple(boxes),
        ((-1.0, 0.0, 0.8),),
        ((1.0, 0.0, 0.8),),
    )


def allowance(boxes, *, x=0.0, head_z=1.05, vx=0.0):
    from embodied_learning.tasks.cat_overhead import OverheadClearance

    query = OverheadClearance(SceneManifest((scene(boxes),)), "cpu")
    return query.allowance(
        torch.tensor([[x, 0.0, 0.8]]),
        torch.tensor([[x, 0.0, head_z]]),
        torch.tensor([[vx, 0.0]]),
        torch.tensor([0]),
    )


@pytest.mark.parametrize(
    "boxes",
    [
        [],
        [SceneBox((0.0, 0.0, 0.02), (0.04, 0.5, 0.02))],
        [SceneBox((0.0, 0.0, 0.75), (0.1, 0.5, 0.75))],
        [SceneBox((0.0, 0.0, 2.0), (0.3, 0.5, 0.1))],
    ],
)
def test_empty_ground_hurdle_wall_and_high_ceiling_do_not_allow_ducking(boxes):
    assert allowance(boxes).item() == 0.0


def test_low_ceiling_permission_does_not_depend_on_current_head_height():
    boxes = [SceneBox((0.0, 0.0, 1.3), (0.3, 0.5, 0.1))]
    for height in (0.75, 1.05, 1.35):
        assert allowance(boxes, head_z=height).item() == 1.0
    assert allowance(boxes, x=1.0).item() == 0.0


def test_ceiling_permission_fades_out_and_anticipates_approach():
    boxes = [SceneBox((0.0, 0.0, 1.3), (0.3, 0.5, 0.1))]
    values = [allowance(boxes, x=x).item() for x in (0.45, 0.50, 0.55, 0.61)]
    assert values[0] == pytest.approx(1.0)
    assert 1.0 > values[1] > values[2] > 0.0
    assert values[3] == 0.0
    assert allowance(boxes, x=-0.65, vx=1.0).item() == pytest.approx(1.0)
    assert allowance(boxes, x=0.65, vx=1.0).item() == 0.0


def reward_case():
    f = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    x = CatRewardInputs(
        **{
            k[6:]: torch.from_numpy(f[k]).clone()
            for k in f.files
            if k.startswith("input_")
        }
    )
    x.pelvis_rpy.zero_()
    x.torso_rpy.zero_()
    x.torso_rpy[:, 1] = math.radians(37)
    x.head_pos[:, 2] = 1.05
    p = CatRewardParameters(
        torch.from_numpy(f["soft_lower"]),
        torch.from_numpy(f["soft_upper"]),
    )
    return x, p


def test_low_head_in_free_space_is_penalized_and_raising_head_has_no_cliff():
    x, p = reward_case()
    zero = torch.zeros(len(x.head_pos))
    low = cat_reward_terms(x, parameters=p, ducking_allowance=zero)
    assert low["tracking_orientation"][0].item() == pytest.approx(0.7240565)
    x.head_pos[:, 2] = 1.2
    high = cat_reward_terms(x, parameters=p, ducking_allowance=zero)
    torch.testing.assert_close(
        low["tracking_orientation"], high["tracking_orientation"]
    )
    x.torso_rpy[:, 1] = 0.0
    upright = cat_reward_terms(x, parameters=p, ducking_allowance=zero)
    assert (upright["tracking_orientation"] > low["tracking_orientation"]).all()


def test_real_overhead_allows_forward_pitch_but_keeps_backwards_penalty():
    x, p = reward_case()
    one = torch.ones(len(x.head_pos))
    forward = cat_reward_terms(x, parameters=p, ducking_allowance=one)
    torch.testing.assert_close(forward["tracking_orientation"], one)
    x.torso_rpy[:, 1] *= -1
    backward = cat_reward_terms(x, parameters=p, ducking_allowance=one)
    assert (backward["tracking_orientation"] < forward["tracking_orientation"]).all()


def test_task_refresh_applies_geometry_gate_even_when_head_field_weight_is_zero():
    from test_cat_review_regressions import task_case

    from embodied_learning.tasks.cat_traversal import CatTaskState

    old, controller, payload, _ = task_case()
    manifest = SceneManifest((scene([]),))
    state = CatTaskState(
        manifest,
        old.bank,
        (0,),
        16,
        "cpu",
        {
            "noise_level": 0.0,
            "overhead": 0.0,
            "posture_reward": "overhead_clearance_v1",
        },
    )
    state.reset(torch.arange(16))
    x = payload["reward_inputs"]
    x.pelvis_rpy.zero_()
    x.torso_rpy.zero_()
    x.torso_rpy[:, 1] = math.radians(37)
    x.head_pos[:, 2] = 1.05
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    assert (state.terms["tracking_orientation"] < 0.8).all()


def test_new_expert_recipe_enables_geometry_posture_in_every_stage(tmp_path):
    from train_cat_curriculum import initialize, stages_for

    from embodied_learning.data.config import load_experiment_config

    original = load_experiment_config(Path("configs/experiments/g1_cat_source.yaml"))
    parent = tmp_path / "parent.pt"
    torch.save(
        {
            "field_contract": {"schema": {"task": original["task"]}},
            "curriculum": {
                "experiment": "parent",
                "protocol_sha256": "p",
                "stage": 4,
            },
            "completed_iterations": 50,
            "ppo_config": {"seed": 42},
        },
        parent,
    )
    root = tmp_path / "expert"
    initialize(
        root,
        recipe="cat_obstacle_expert_v1",
        parent_checkpoint=parent,
        training_seed=7,
        terminal_gate_seeds=(8, 9),
        training_disturbance_profile="unitree_g1_bridge_v1",
    )
    protocol = json.loads((root / "protocol.json").read_text())
    for stage in stages_for(root, protocol):
        assert stage["task"]["posture_reward"] == "overhead_clearance_v1"


def test_invalid_physics_terminates_only_bad_slot_without_aborting_reward_batch():
    from test_cat_review_regressions import task_case

    from embodied_learning.tasks.cat_traversal import CatTaskState

    old, controller, payload, _ = task_case()
    manifest = SceneManifest(
        (
            scene(
                [
                    SceneBox((0.0, 0.0, 1.3), (0.3, 0.5, 0.1)),
                ]
            ),
        )
    )
    state = CatTaskState(
        manifest,
        old.bank,
        (0,),
        16,
        "cpu",
        {"noise_level": 0.0, "posture_reward": "overhead_clearance_v1"},
    )
    state.reset(torch.arange(16))
    payload["body_positions"][0, 0, 0] = torch.nan
    x = payload["reward_inputs"]
    x.pelvis_rpy.zero_()
    x.torso_rpy.zero_()
    x.torso_rpy[:, 1] = math.radians(37)
    x.head_pos[:, 2] = 1.05
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    assert state._outcomes["invalid"].tolist() == [True] + [False] * 15
    assert torch.isfinite(state._reward).all()
    assert state.terms["tracking_orientation"][0] < 0.8
    torch.testing.assert_close(
        state.terms["tracking_orientation"][1:], torch.ones(15)
    )


def test_expert_actor_bootstrap_records_posture_change_but_resume_rejects_it(
    tmp_path,
):
    from test_cat_curriculum import fixture_metadata
    from test_field_ppo import TinyEnv

    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist
    from embodied_learning.tasks.cat_foot_guidance import sha256

    env = TinyEnv()
    env.schema = {**env.schema, "task": {}}
    env.curriculum_metadata = fixture_metadata(4)
    cfg = PpoConfig(
        iterations=1,
        rollout_steps=4,
        hidden_dims=(16, 8),
        epochs=1,
        mini_batches=1,
    )
    parent = train_specialist(env, cfg, tmp_path / "parent")
    saved = torch.load(parent, weights_only=False)
    env.schema = {
        **env.schema,
        "scene_hash": "expert-scene",
        "task": {"posture_reward": "overhead_clearance_v1"},
    }
    env.curriculum_metadata = {
        **fixture_metadata(0),
        "experiment": "expert",
        "protocol_sha256": "expert",
        "expert_bootstrap": {
            "sha256": sha256(parent),
            "experiment": "test",
            "protocol_sha256": "protocol",
            "stage": 4,
            "completed_updates": 1,
            "training_seed": saved["ppo_config"]["seed"],
            "parent_rfi_multiplier": 1.0,
        },
    }
    child = train_specialist(env, cfg, tmp_path / "child", warmstart_actor=parent)
    result = torch.load(child, weights_only=False)
    assert result["warmstart"]["contract_changes"]["task.posture_reward"] == {
        "before": None,
        "after": "overhead_clearance_v1",
    }
    env.schema = {**env.schema, "task": {}}
    with pytest.raises(ValueError):
        train_specialist(env, cfg, tmp_path / "child", resume=child)


def test_geometry_permission_does_not_leak_between_scenes_or_from_distant_roof():
    from embodied_learning.tasks.cat_overhead import OverheadClearance

    roof = scene([SceneBox((0.0, 0.0, 1.3), (0.3, 0.5, 0.1))])
    query = OverheadClearance(SceneManifest((roof, replace(roof, boxes=()))), "cpu")
    root = torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.8], [2.0, 0.0, 0.8]])
    head = root + torch.tensor([0.0, 0.0, 0.25])
    got = query.allowance(root, head, torch.zeros(3, 2), torch.tensor([0, 1, 0]))
    torch.testing.assert_close(got, torch.tensor([1.0, 0.0, 0.0]))
