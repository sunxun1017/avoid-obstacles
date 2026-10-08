"""Field task state and real G1 integration contracts."""

from pathlib import Path

import pytest
import torch

from embodied_learning.data.field_scenes import load_scene_manifest
from embodied_learning.robots.g1 import G1_BODY_SCHEMA
from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields
from embodied_learning.tasks.field_traversal import FieldTaskState


def test_body_schema_order():
    assert G1_BODY_SCHEMA == (
        "head",
        "imu_in_pelvis",
        "imu_in_torso",
        "left_foot",
        "right_foot",
        "left_palm",
        "right_palm",
        "left_knee",
        "right_knee",
        "left_shoulder",
        "right_shoulder",
    )


def test_multi_env_field_coordinates():
    world = torch.tensor([[[11.0, 2.0, 3.0]], [[-3.0, 5.0, 1.0]]])
    origin = torch.tensor([[10.0, 0.0, 0.0], [-4.0, 4.0, 0.0]])
    assert torch.equal(
        FieldTaskState.local_points(world, origin),
        torch.tensor([[[1.0, 2.0, 3.0]], [[1.0, 1.0, 1.0]]]),
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("Real mjlab integration requires CUDA.")
    from embodied_learning.simulators.mjlab.field_traversal import make_field_env

    manifest = load_scene_manifest(Path("configs/experiments/g1_field.json"))
    cache = prepare_fields(manifest, tmp_path_factory.mktemp("fields"))
    bank = load_field_cache(cache.path, manifest, device="cuda:0")
    instance = make_field_env(
        manifest,
        bank,
        scene_ids=("wall_detour", "passage"),
        num_envs=2,
        device="cuda:0",
        seed=42,
    )
    yield instance
    instance.close()


def test_reset_goal_isolation(env):
    env.reset()
    before = env.field_state.goal_ids.clone()
    env.reset_slots(torch.tensor([0], device=env.device))
    assert env.field_state.goal_ids[1] == before[1]
    assert torch.isfinite(env.get_observations()["actor"]).all()


def test_actor_contains_all_body_fields(env):
    obs = env.get_observations()["actor"]
    # 3 linear + 3 angular + 3 gravity + 3*29 joints/actions + 11*(3+3+1).
    assert obs.shape == (2, 173)
    assert torch.isfinite(obs).all()
    assert env.field_state.current(env.unwrapped).valid_mask.all()


def test_reward_precedes_reset(env):
    env.reset()
    env.unwrapped.episode_length_buf[:] = env.max_episode_length - 1
    _, rewards, dones, extras = env.step(torch.zeros(2, 29, device=env.device))
    assert dones.bool().all()
    assert torch.isfinite(rewards).all()
    assert extras["field_outcomes"]["timeout"].all()
    assert (env.episode_length_buf == 0).all()


def test_goal_collision_fall_timeout(env):
    flags = FieldTaskState.classify(
        goal_distance=torch.tensor([0.0, 2.0, 2.0, 2.0]),
        contact=torch.tensor([False, True, False, False]),
        upright=torch.tensor([1.0, 1.0, 0.0, 1.0]),
        root_height=torch.tensor([0.8, 0.8, 0.2, 0.8]),
        valid=torch.ones(4, dtype=torch.bool),
        elapsed=torch.tensor([0, 0, 0, 10]),
        limit=10,
    )
    assert flags["goal"].tolist() == [True, False, False, False]
    assert flags["collision"].tolist() == [False, True, False, False]
    assert flags["fall"].tolist() == [False, False, True, False]
    assert flags["timeout"].tolist() == [False, False, False, True]


def test_invalid_field_termination():
    flags = FieldTaskState.classify(
        goal_distance=torch.tensor([2.0]),
        contact=torch.tensor([False]),
        upright=torch.tensor([1.0]),
        root_height=torch.tensor([0.8]),
        valid=torch.tensor([False]),
        elapsed=torch.tensor([0]),
        limit=10,
    )
    assert flags["invalid"].item()
    assert not flags["goal"].item()


def test_support_ground_not_collision(env):
    env.reset()
    assert not env.field_state.outcomes(env.unwrapped)["collision"].any()
    # Move slot zero into the declared physical wall, then run physics.
    robot = env.unwrapped.scene["robot"]
    scene = env.field_state.manifest.scenes[env.scene_ids[0].item()]
    center = scene.boxes[0].center
    pose = torch.tensor(
        [[center[0], center[1], 0.8, 1.0, 0.0, 0.0, 0.0]], device=env.device
    )
    pose[:, :3] += env.unwrapped.scene.env_origins[:1]
    robot.write_root_link_pose_to_sim(
        pose, env_ids=torch.tensor([0], device=env.device)
    )
    _, _, _, extras = env.step(torch.zeros(2, 29, device=env.device))
    assert extras["field_outcomes"]["collision"][0]


def test_configurable_reward_parameters():
    from embodied_learning.tasks.field_traversal import RewardParameters

    assert RewardParameters(vmf_weight=2.0).vmf_weight == 2.0
    with pytest.raises(ValueError):
        RewardParameters(collision_weight=-1.0)


def test_g1_clearance_rejects_body_intersection():
    from embodied_learning.data.field_scenes import (
        FieldScene,
        SceneBox,
        SceneManifest,
    )
    from embodied_learning.simulators.mjlab.field_traversal import (
        validate_g1_clearance,
    )

    scene = FieldScene(
        "bad",
        (-2.0, -2.0, -0.2),
        (41, 41, 23),
        0.1,
        (SceneBox((0.25, 0.0, 1.0), (0.1, 0.1, 0.5)),),
        ((0.0, 0.0, 0.8),),
        ((1.5, 0.0, 0.8),),
    )
    # Root is outside the box, but the arm/body geometry overlaps it.
    with pytest.raises(ValueError, match="clearance"):
        validate_g1_clearance(SceneManifest((scene,)))


def test_training_noise_disturbance_curriculum_options():
    from embodied_learning.tasks.field_traversal import parse_task_options

    _, _, _, training = parse_task_options(
        {
            "training": {
                "observation_noise": True,
                "push_velocity": 0.1,
                "push_interval_s": 0.5,
                "field_warmup_steps": 100,
            }
        }
    )
    assert training.observation_noise
    assert training.push_velocity == 0.1
    assert training.field_factor(0) == 0
    assert training.field_factor(50) == 0.5
    assert training.field_factor(100) == 1
    with pytest.raises(ValueError):
        parse_task_options({"training": {"push_interval_s": 0.0}})


def test_real_training_options_and_resumed_curriculum(env, tmp_path):
    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist
    from embodied_learning.simulators.mjlab.field_traversal import make_field_env

    instance = make_field_env(
        env.field_state.manifest,
        env.field_state.bank,
        scene_ids=("wall_detour",),
        num_envs=2,
        device="cuda:0",
        seed=42,
        options={
            "training": {
                "observation_noise": True,
                "push_velocity": 0.1,
                "push_interval_s": 0.1,
                "field_warmup_steps": 100,
            }
        },
    )
    try:
        assert instance.unwrapped.cfg.observations["actor"].enable_corruption
        assert not instance.unwrapped.cfg.observations["critic"].enable_corruption
        assert "push_robot" in instance.unwrapped.cfg.events
        instance.training_step = 0
        early_reward = instance.field_state.reward(instance.unwrapped)
        instance.training_step = 100
        full_reward = instance.field_state.reward(instance.unwrapped)
        assert not torch.allclose(early_reward, full_reward)
        instance.training_step = 0
        config = PpoConfig(
            iterations=1,
            rollout_steps=4,
            epochs=1,
            mini_batches=1,
            hidden_dims=(16, 8),
        )
        path = train_specialist(instance, config, tmp_path / "initial")
        saved = torch.load(path, weights_only=False)
        assert saved["environment_steps"] == 4
        instance.training_step = 0
        path = train_specialist(instance, config, tmp_path / "resumed", resume=path)
        saved = torch.load(path, weights_only=False)
        assert saved["environment_steps"] == 8
        assert (
            instance.field_state.training.field_factor(instance.training_step)
            == 0.08
        )
    finally:
        instance.close()
