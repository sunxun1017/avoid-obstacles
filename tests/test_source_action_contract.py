import math
from types import SimpleNamespace

import mujoco
import pytest
import torch


def test_source_action_transform_reaches_environment_once(monkeypatch):
    from mjlab.rl import RslRlVecEnvWrapper

    from embodied_learning.simulators.mjlab.cat_traversal import CatEnv

    received = []
    monkeypatch.setattr(
        RslRlVecEnvWrapper,
        "step",
        lambda self, actions: (
            received.append(actions.clone()) or (None, torch.zeros(2), None, {})
        ),
    )
    env = CatEnv.__new__(CatEnv)
    env.cat_state = SimpleNamespace(
        saved_outcomes={},
        last_metrics={},
        metric_definitions={},
        saved_terms={},
        saved_reward_diagnostics={},
        scales={},
    )
    raw = torch.tensor([[-100.0, 0.0, 0.5], [100.0, -0.5, 1.0]])
    env.step(raw)
    torch.testing.assert_close(received[0], raw.tanh())
    with pytest.raises(ValueError, match="finite"):
        env.step(torch.full_like(raw, float("inf")))
    assert len(received) == 1


def test_model_identity_includes_solver_options():
    from embodied_learning.simulators.mjlab.cat_traversal import (
        native_model_identity,
    )

    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    before = native_model_identity(model)
    model.opt.iterations += 1
    assert native_model_identity(model) != before


def test_cli_uses_yaml_defaults_and_explicit_overrides():
    from embodied_learning.cli.field_training import (
        _parser,
        resolve_training_defaults,
    )

    settings = {"seed": 91, "ppo": {"mini_batches": 7}}
    args = _parser().parse_args(["ppo"])
    resolve_training_defaults(args, settings)
    assert (args.seed, args.mini_batches) == (91, 7)
    args = _parser().parse_args(["ppo", "--seed", "42", "--mini-batches", "4"])
    resolve_training_defaults(args, settings)
    assert (args.seed, args.mini_batches) == (42, 4)
    args = _parser().parse_args(["ppo"])
    with pytest.raises(ValueError, match="seed"):
        resolve_training_defaults(args, {"seed": True})


def test_tanh_distribution_density_entropy_and_extreme_gradients():
    from embodied_learning.policies.source_distribution import (
        LatentTanhGaussianDistribution,
    )

    distribution = LatentTanhGaussianDistribution(2)
    mean = torch.zeros(2, 2, requires_grad=True)
    distribution.update(mean)
    raw = torch.tensor([[0.2, -0.3], [100.0, -100.0]])
    stable_jacobian = 2 * (
        math.log(2) - raw - torch.nn.functional.softplus(-2 * raw)
    )
    expected = (
        torch.distributions.Normal(mean, 1).log_prob(raw) - stable_jacobian
    ).sum(-1)
    torch.testing.assert_close(distribution.log_prob(raw), expected)
    torch.manual_seed(11)
    sample = distribution._distribution.rsample()
    correction = 2 * (
        math.log(2) - sample - torch.nn.functional.softplus(-2 * sample)
    )
    expected_entropy = distribution._distribution.entropy().sum(-1) + correction.sum(
        -1
    )
    torch.manual_seed(11)
    torch.testing.assert_close(distribution.entropy, expected_entropy)
    (distribution.log_prob(raw).sum() + distribution.entropy.sum()).backward()
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(distribution.std_param.grad).all()


def test_source_runner_uses_matching_distribution():
    from embodied_learning.algorithms.ppo import PpoConfig

    cfg = PpoConfig().runner_config(action_transform="latent_tanh_v1")
    assert cfg["actor"]["distribution_cfg"]["class_name"].endswith(
        "LatentTanhGaussianDistribution"
    )
    assert (
        PpoConfig().runner_config()["actor"]["distribution_cfg"]["class_name"]
        == "GaussianDistribution"
    )
