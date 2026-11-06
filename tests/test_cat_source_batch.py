"""Guard the time/environment boundaries when matching source PPO batches."""

import pytest
import torch
from tensordict import TensorDict


def make_algorithm():
    from rsl_rl.storage import RolloutStorage

    from embodied_learning.algorithms.brax_ppo import SourceMLPModel, SourcePPO

    obs = TensorDict({"actor": torch.zeros(4, 2), "critic": torch.zeros(4, 3)}, [4])
    groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = SourceMLPModel(
        obs,
        groups,
        "actor",
        1,
        hidden_dims=[4],
        activation="swish",
        distribution_cfg={
            "class_name": "embodied_learning.algorithms.brax_ppo:SourceNormalTanh"
        },
    )
    critic = SourceMLPModel(
        obs, groups, "critic", 1, hidden_dims=[4], activation="swish"
    )
    storage = RolloutStorage("rl", 4, 8, obs, [1])
    algorithm = SourcePPO(
        actor,
        critic,
        storage,
        source_unroll_steps=4,
        num_learning_epochs=1,
        num_mini_batches=2,
        schedule="fixed",
        use_clipped_value_loss=False,
        value_loss_coef=0.25,
    )
    return algorithm


def test_segmented_batch_keeps_environment_time_and_bootstrap_boundaries():
    # Swapping time/environment axes or bootstrapping all segments from the final
    # observation must fail: every time and slot has an independent marker.
    algorithm = make_algorithm()
    storage = algorithm.storage
    marker = torch.arange(8)[:, None] * 10 + torch.arange(4)[None]
    for key in ("actor", "critic"):
        storage.observations[key].copy_(marker[..., None])
        algorithm.last_observations[key].copy_(
            torch.tensor([80.0, 81.0, 82.0, 83.0])[:, None]
        )
    storage.rewards.copy_((marker + 0.5)[..., None])
    storage.actions.copy_((marker + 0.25)[..., None])
    storage.actions_log_prob.copy_((-marker.float())[..., None])
    storage.dones.zero_()
    storage.dones[2, 1] = 1
    algorithm.truncations.zero_()
    algorithm.truncations[6, 2] = 1
    length, count, data, bootstrap = algorithm._trajectory_batches()
    assert (length, count) == (4, 8)
    expected = torch.tensor(
        [
            [0, 10, 20, 30],
            [1, 11, 21, 31],
            [2, 12, 22, 32],
            [3, 13, 23, 33],
            [40, 50, 60, 70],
            [41, 51, 61, 71],
            [42, 52, 62, 72],
            [43, 53, 63, 73],
        ]
    )
    torch.testing.assert_close(
        data["observations"]["critic"][..., 0], expected.float()
    )
    torch.testing.assert_close(data["rewards"][..., 0], expected + 0.5)
    torch.testing.assert_close(data["actions"][..., 0], expected + 0.25)
    torch.testing.assert_close(data["log_prob"][..., 0], -expected.float())
    torch.testing.assert_close(
        bootstrap["critic"][:, 0],
        torch.tensor([40.0, 41.0, 42.0, 43.0, 80.0, 81.0, 82.0, 83.0]),
    )
    assert data["dones"].nonzero().tolist() == [[1, 2, 0]]
    assert data["truncations"].nonzero().tolist() == [[6, 2, 0]]


def test_source_sized_batch_reaches_runner_without_lengthening_gae():
    from embodied_learning.algorithms.ppo import PpoConfig

    config = PpoConfig(
        optimization_profile="brax_0_12_3_v1",
        normalization=False,
        rollout_steps=160,
        source_unroll_steps=20,
        mini_batches=32,
    )
    runner = config.runner_config(action_transform="latent_tanh_v1")
    assert runner["num_steps_per_env"] == 160
    assert runner["algorithm"]["source_unroll_steps"] == 20
    assert runner["algorithm"]["num_mini_batches"] == 32


def test_segmented_update_uses_short_unrolls_and_all_trajectories():
    algorithm = make_algorithm()
    obs = algorithm.storage.observations[0].clone()
    for step in range(8):
        obs["actor"].fill_(step * 0.01)
        obs["critic"].fill_(step * 0.02)
        algorithm.act(obs)
        algorithm.process_env_step(
            obs, torch.ones(4), torch.zeros(4, dtype=torch.bool), {}
        )
    algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert algorithm.last_batch_shapes == [(4, 4), (4, 4)]
    assert algorithm.storage.step == 0


@pytest.mark.parametrize("steps", [0, 3, True])
def test_rejects_invalid_source_unroll_boundaries(steps):
    from embodied_learning.algorithms.ppo import PpoConfig

    with pytest.raises(ValueError, match="unroll"):
        PpoConfig(
            optimization_profile="brax_0_12_3_v1",
            normalization=False,
            rollout_steps=8,
            source_unroll_steps=steps,
        )
