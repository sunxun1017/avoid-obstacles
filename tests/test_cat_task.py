from pathlib import Path

import numpy as np
import pytest
import torch

from embodied_learning.robots.g1 import CAT_JOINT_NAMES
from embodied_learning.simulators.mjlab.cat_control import CatJointController
from embodied_learning.tasks.cat_observation import build_cat_observations
from embodied_learning.tasks.cat_traversal import CatTaskState


def test_source_observation_fixture_162_250():
    x = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    payload = {
        k[4:]: torch.from_numpy(x[k])
        for k in x.files
        if k.startswith("obs_") and k not in ("obs_actor", "obs_critic")
    }
    obs = build_cat_observations(**payload, noise_level=0.0)
    torch.testing.assert_close(
        obs["actor"], torch.from_numpy(x["obs_actor"]), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        obs["critic"], torch.from_numpy(x["obs_critic"]), rtol=1e-5, atol=1e-6
    )
    assert obs["actor"].shape == (16, 162) and obs["critic"].shape == (16, 250)


def test_reward_history_keeps_the_two_preceding_executed_actions():
    c = CatJointController(
        CAT_JOINT_NAMES, torch.full((29,), -2.0), torch.full((29,), 2.0), 1, "cpu"
    )
    c.process_action(torch.ones(1, 12))
    c.process_action(torch.ones(1, 12) * 2)
    c.process_action(torch.ones(1, 12) * 4)
    torch.testing.assert_close(c.actions, torch.full((1, 12), 4.0))
    torch.testing.assert_close(c.last_actions, torch.full((1, 12), 2.0))
    torch.testing.assert_close(c.last_last_actions, torch.ones(1, 12))


def test_action_smoothness_uses_real_history_and_observation_keeps_latest_action():
    from test_cat_review_regressions import task_case

    state, controller, payload, _ = task_case()
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    for step, value in enumerate((0.2, -0.1, 0.3), start=1):
        action = torch.zeros(16, 12)
        action[:, 0] = value
        controller.process_action(action)
        state.refresh(payload, controller, torch.full((16,), step), step)
    # .3^2 + (.3-(-.1))^2 + (.3-2*(-.1)+.2)^2 = .74.
    torch.testing.assert_close(
        state.terms["smoothness_action"], torch.full((16,), 0.74)
    )
    # At the next policy decision, the previous executed action is still .3.
    for group in ("actor", "critic"):
        torch.testing.assert_close(state.observations()[group][:, 52:64], action)


def test_action_history_partial_reset_does_not_leak_between_episodes():
    c = CatJointController(
        CAT_JOINT_NAMES, torch.full((29,), -2.0), torch.full((29,), 2.0), 2, "cpu"
    )
    c.process_action(torch.full((2, 12), 0.1))
    c.process_action(torch.full((2, 12), 0.2))
    c.reset(torch.tensor([0]))
    c.process_action(torch.full((2, 12), 0.3))
    torch.testing.assert_close(c.last_actions[0], torch.zeros(12))
    torch.testing.assert_close(c.last_last_actions[0], torch.zeros(12))
    torch.testing.assert_close(c.last_actions[1], torch.full((12,), 0.2))
    torch.testing.assert_close(c.last_last_actions[1], torch.full((12,), 0.1))


def test_source_collision_grace_49_50():
    args = dict(
        up_z=torch.ones(2),
        head_z=torch.ones(2),
        self_contact=torch.tensor([True, True]),
        sdf=torch.ones(2, 11),
        invalid=torch.zeros(2, dtype=torch.bool),
        steps=torch.tensor([49, 50]),
        threshold=0.04,
    )
    out = CatTaskState.classify(**args)
    assert out["collision"].tolist() == [False, True]


def test_ever_reached_goal_latched_until_episode_end():
    state = CatTaskState.__new__(CatTaskState)
    state.ever_goal = torch.zeros(2, dtype=torch.bool)
    state.record_goal(torch.tensor([0.05, 1.0]), torch.tensor([True, True]))
    state.record_goal(torch.tensor([1.0, 1.0]), torch.tensor([True, True]))
    assert state.ever_goal.tolist() == [True, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native G1 requires CUDA")
def test_reads_do_not_advance_state(tmp_path):
    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.simulators.mjlab.cat_traversal import make_cat_env
    from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields

    manifest = load_scene_manifest(Path("configs/experiments/g1_field.yaml"))
    profile = {"version": "cat_source_v1"}
    cache = prepare_fields(manifest, tmp_path, source_profile=profile)
    bank = load_field_cache(
        cache.path, manifest, device="cuda:0", source_profile=profile
    )
    env = make_cat_env(
        manifest,
        bank,
        scene_ids=("wall_detour",),
        num_envs=4,
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
        before = env.cat_state.gait.phase.clone()
        first = env.get_observations()
        second = env.get_observations()
        torch.testing.assert_close(first["actor"], second["actor"])
        torch.testing.assert_close(env.cat_state.gait.phase, before)
        for _ in range(20):
            obs, reward, done, extras = env.step(torch.zeros(4, 12, device="cuda:0"))
        assert obs["critic"].shape == (4, 250)
        assert torch.isfinite(reward).all()
        env.reset_slots(torch.tensor([0], device="cuda:0"))
        assert env.cat_state.gait.stop_timestep[0] == 100
        # Start every slot from a fresh physical state so the test does not
        # depend on whether a zero-action rollout happened to make G1 fall.
        env.reset_slots(torch.arange(4, device="cuda:0"))
        # Force simultaneous resets: continuing slots and reset slots differ.
        env.episode_length_buf[:2] = 999
        env.cat_state.gait.phase[1] = 0.3
        phase = env.cat_state.gait.phase.clone()
        phase_dt = env.cat_state.gait.phase_dt.clone()
        obs, reward, done, extras = env.step(torch.zeros(4, 12, device="cuda:0"))
        assert done[0] and extras["field_outcomes"]["timeout"][0]
        assert env.episode_length_buf[0] == 0
        from embodied_learning.tasks.cat_reward import aggregate_cat_reward

        expected = aggregate_cat_reward(
            env.cat_state.saved_terms, env.cat_state.scales, dt=0.02
        )
        torch.testing.assert_close(reward, expected)
        # Other slots may fall and reset in this same step. Only continuing
        # episodes advance the old phase; reset slots get a new phase/frequency.
        active = ~done
        assert active.any()
        expected_phase = (
            torch.fmod(
                phase[active] + phase_dt[active, None] + torch.pi,
                2 * torch.pi,
            )
            - torch.pi
        )
        torch.testing.assert_close(env.cat_state.gait.phase[active], expected_phase)
        assert done[1] and extras["field_outcomes"]["timeout"][1]
        assert (env.episode_length_buf[done] == 0).all()
        torch.testing.assert_close(
            env.cat_state.gait.phase[done].abs().sum(-1),
            torch.full_like(phase_dt[done], torch.pi),
        )
        torch.testing.assert_close(
            obs["actor"][1:], env.get_observations()["actor"][1:]
        )
    finally:
        env.close()


def test_observation_noise_is_bounded_and_critic_is_clean():
    x = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    payload = {
        k[4:]: torch.from_numpy(x[k])
        for k in x.files
        if k.startswith("obs_") and k not in ("obs_actor", "obs_critic")
    }
    clean = build_cat_observations(**payload, noise_level=0.0)
    noisy = build_cat_observations(**payload, noise_level=1.0)
    delta = (noisy["actor"] - clean["actor"]).abs()
    for begin, end, bound in (
        (0, 3, 0.2),
        (3, 6, 0.05),
        (6, 29, 0.03),
        (29, 52, 1.5),
    ):
        assert (delta[:, begin:end] <= bound + 1e-6).all()
    torch.testing.assert_close(noisy["actor"][:, 52:], clean["actor"][:, 52:])
    torch.testing.assert_close(noisy["critic"], clean["critic"])


def test_invalid_pose_terminates_without_querying_nan():
    from types import SimpleNamespace

    from embodied_learning.tasks.cat_reward import CatRewardInputs

    class Bank:
        source_profile = {"version": "cat_source_v1"}

        def sample_source(self, p, scene, goal):
            assert torch.isfinite(p).all(), "mapping query received invalid pose"
            return SimpleNamespace(
                guidance=torch.zeros_like(p),
                sdf_gradient=torch.zeros_like(p),
                sdf=torch.ones(p.shape[:2]),
            )

    state = CatTaskState(
        SimpleNamespace(), Bank(), (0,), 1, "cpu", {"noise_level": 0.0}
    )
    c = CatJointController(
        CAT_JOINT_NAMES, torch.full((29,), -2.0), torch.full((29,), 2.0), 1, "cpu"
    )
    x = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    inputs = CatRewardInputs(
        **{
            k[6:]: torch.from_numpy(x[k][:1]).clone()
            for k in x.files
            if k.startswith("input_")
        }
    )
    payload = dict(
        reward_inputs=inputs,
        body_positions=torch.full((1, 11, 3), float("nan")),
        root_position=torch.full((1, 3), float("nan")),
        root_rotation=torch.eye(3)[None],
        gyro=torch.zeros(1, 3),
        gravity=torch.zeros(1, 3),
        up_z=torch.ones(1),
        self_contact=torch.zeros(1, dtype=torch.bool),
        physical_collision=torch.zeros(1, dtype=torch.bool),
        root_local_vel=torch.zeros(1, 3),
        invalid=torch.ones(1, dtype=torch.bool),
    )
    state.refresh(payload, c, torch.tensor([1]), 1)
    assert state.outcomes()["invalid"][0]
    assert torch.isfinite(state.observations()["actor"]).all()


def test_diagnostic_clean_observation_preserves_noise_draws():
    with np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz") as fixture:
        payload = {
            key[4:]: torch.from_numpy(fixture[key])
            for key in fixture.files
            if key.startswith("obs_") and key not in ("obs_actor", "obs_critic")
        }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        noisy = build_cat_observations(**payload, noise_level=1.0)
        noisy_rng = torch.random.get_rng_state()
        torch.manual_seed(123)
        clean = build_cat_observations(**payload, noise_level=0.0, consume_noise_draws=True)
        assert torch.equal(noisy_rng, torch.random.get_rng_state())
        reference = build_cat_observations(**payload, noise_level=0.0)
    assert not torch.equal(noisy["actor"], clean["actor"])
    assert torch.equal(clean["actor"], reference["actor"])
    assert torch.equal(clean["critic"], reference["critic"])
