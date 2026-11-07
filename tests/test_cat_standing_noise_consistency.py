import pytest
import torch

from embodied_learning.algorithms.ppo import PpoConfig
from embodied_learning.tasks.cat_observation import standing_noise_pairs
from test_cat_review_regressions import task_case


def test_pairs_replace_only_noisy_proprioception_and_use_true_state_for_gate():
    state, controller, payload, _ = task_case()
    payload["reward_inputs"].feet_contact.fill_(1)
    payload["reward_inputs"].feet_contact[3, 0] = 0
    payload["root_local_vel"][1, 0] = .3
    payload["gyro"][2, 0] = 1.1
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    observations = state.observations().clone()
    observations["critic"][:, 77:79] = 0
    observations["critic"][4, 77] = .15
    observations["actor"][:, :52] += 0.1
    observations["actor"][2, 0] = 0.
    original = observations["actor"].clone()

    mask, clean = standing_noise_pairs(observations)
    assert mask.tolist() == [True, False, False, False, False] + [True] * 11
    torch.testing.assert_close(clean["actor"][:, :52], observations["critic"][mask, :52])
    torch.testing.assert_close(clean["actor"][:, 52:], original[mask, 52:])
    torch.testing.assert_close(observations["actor"], original)


def test_standing_consistency_is_opt_in_to_source_profile():
    assert "source_standing_consistency_weight" not in PpoConfig().runner_config()["algorithm"]
    with pytest.raises(ValueError):
        PpoConfig(source_standing_consistency_weight=.1)
    with pytest.raises(ValueError):
        PpoConfig(optimization_profile="brax_0_12_3_v1", source_standing_consistency_weight=float("nan"))
    cfg = PpoConfig(optimization_profile="brax_0_12_3_v1", normalization=False,
                    source_standing_consistency_weight=.1)
    assert cfg.runner_config(action_transform="latent_tanh_v1")["algorithm"]["source_standing_consistency_weight"] == .1


def test_source_optimizer_reports_auxiliary_loss_only_for_eligible_samples():
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict

    from embodied_learning.algorithms.brax_ppo import SourceMLPModel, SourcePPO

    actor_input = torch.zeros(8, 162)
    critic_input = torch.zeros(8, 250)
    actor_input[:, :52] = 0.1
    critic_input[:4, 217:219] = 1
    obs = TensorDict(dict(actor=actor_input, critic=critic_input), batch_size=[8])
    groups = dict(actor=["actor"], critic=["critic"])
    actor = SourceMLPModel(
        obs, groups, "actor", 12, hidden_dims=[8], activation="swish",
        distribution_cfg=dict(class_name="embodied_learning.algorithms.brax_ppo:SourceNormalTanh"),
    )
    critic = SourceMLPModel(obs, groups, "critic", 1, hidden_dims=[8], activation="swish")
    algorithm = SourcePPO(
        actor, critic, RolloutStorage("rl", 8, 4, obs, [12]),
        num_learning_epochs=1, num_mini_batches=2,
        schedule="fixed", use_clipped_value_loss=False, value_loss_coef=.25,
        source_standing_consistency_weight=.1,
    )
    for _ in range(4):
        algorithm.act(obs)
        algorithm.process_env_step(obs, torch.ones(8), torch.zeros(8, dtype=torch.bool), {})
    algorithm.compute_returns(obs)
    result = algorithm.update()
    assert result["standing_noise_consistency"] > 0
    assert result["standing_noise_fraction"] == .5
