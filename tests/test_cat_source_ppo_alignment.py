"""Numerical contracts for the isolated source-alignment experiment."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict


def source_gae():
    """Execute the unchanged pinned Brax function with a NumPy scan shim."""
    path = Path(__file__).parent / "fixtures/brax_losses_v0_12_3.py"
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_gae"
    )

    def scan(fn, carry, xs, *, length, reverse):
        out = [None] * length
        for i in reversed(range(length)) if reverse else range(length):
            carry, out[i] = fn(carry, tuple(x[i] for x in xs))
        return carry, np.stack(out)

    namespace = dict(
        jnp=np,
        jax=SimpleNamespace(
            lax=SimpleNamespace(scan=scan, stop_gradient=lambda x: x)
        ),
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["compute_gae"]


def test_gae_matches_executed_brax_with_midrollout_timeout_and_failure():
    from embodied_learning.algorithms.brax_ppo import source_gae as actual

    rng = np.random.default_rng(20261021)
    shape = (20, 7)
    truncation, termination = (
        np.zeros(shape, np.float32),
        np.zeros(shape, np.float32),
    )
    truncation[[0, 9, 19], [0, 1, 2]] = 1
    termination[[2, 10, 19], [3, 4, 5]] = 1
    rewards, values = (rng.normal(size=shape).astype(np.float32) for _ in range(2))
    bootstrap = rng.normal(size=(7,)).astype(np.float32)
    expected = source_gae()(
        truncation, termination, rewards, values, bootstrap, 0.95, 0.97
    )
    args = [
        torch.from_numpy(a)
        for a in (truncation, termination, rewards, values, bootstrap)
    ]
    for observed, reference in zip(
        actual(*args, lam=0.95, gamma=0.97), expected, strict=True
    ):
        torch.testing.assert_close(
            observed, torch.from_numpy(reference), rtol=1e-5, atol=1e-6
        )


def test_state_dependent_scale_and_tanh_density_gradients():
    from embodied_learning.algorithms.brax_ppo import SourceNormalTanh

    distribution = SourceNormalTanh(2)
    logits = torch.tensor(
        [[0.2, -0.3, -2.0, 1.0], [0.1, 0.4, 2.0, -3.0]], requires_grad=True
    )
    distribution.update(logits)
    torch.testing.assert_close(
        distribution.std, torch.nn.functional.softplus(logits[:, 2:]) + 0.001
    )
    assert not list(distribution.parameters())
    assert distribution.input_dim == 4
    torch.testing.assert_close(
        distribution.deterministic_output(logits), logits[:, :2]
    )
    raw = torch.tensor([[0.7, -0.2], [100.0, -100.0]])
    base = torch.distributions.Normal(logits[:, :2], distribution.std)
    jac = 2 * (np.log(2) - raw - torch.nn.functional.softplus(-2 * raw))
    torch.testing.assert_close(
        distribution.log_prob(raw), (base.log_prob(raw) - jac).sum(-1)
    )
    loss = distribution.log_prob(raw).sum() + distribution.entropy.sum()
    loss.backward()
    assert torch.isfinite(logits.grad).all() and bool(
        (logits.grad[:, 2:] != 0).all()
    )


def test_profile_contract_and_model_roundtrip():
    from rsl_rl.utils import resolve_callable

    from embodied_learning.algorithms.ppo import PpoConfig
    from embodied_learning.policies.field_policy import _make_model
    from embodied_learning.protocols.imitation import PolicyContract

    config = PpoConfig(optimization_profile="brax_0_12_3_v1", normalization=False)
    cfg = config.runner_config(action_transform="latent_tanh_v1")
    assert cfg["actor"]["activation"] == cfg["critic"]["activation"] == "swish"
    assert cfg["algorithm"]["schedule"] == "fixed"
    assert cfg["algorithm"]["value_loss_coef"] == 0.25
    assert cfg["algorithm"]["use_clipped_value_loss"] is False
    contract = PolicyContract(
        dict(optimization_profile="brax_0_12_3_v1"),
        4,
        5,
        2,
        (8, 4),
        activation="swish",
        normalization=False,
    )
    obs = TensorDict(
        dict(actor=torch.randn(7, 4), critic=torch.randn(7, 5)), batch_size=[7]
    )
    model = resolve_callable(cfg["actor"].pop("class_name"))(
        obs,
        dict(actor=["actor"], critic=["critic"]),
        "actor",
        2,
        **{**cfg["actor"], "hidden_dims": [8, 4]},
    )
    assert model.mlp[-1].out_features == 4
    for layer in model.mlp:
        if isinstance(layer, torch.nn.Linear):
            assert bool((layer.bias == 0).all())
            assert float(layer.weight.abs().max()) <= (3 / layer.in_features) ** 0.5
    loaded = _make_model(contract, "cpu", stochastic=True)
    loaded.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model(obs), loaded(obs))


def test_source_update_preserves_time_sequences_raw_rewards_and_fixed_lr():
    from rsl_rl.storage import RolloutStorage

    from embodied_learning.algorithms.brax_ppo import SourceMLPModel, SourcePPO

    torch.manual_seed(42)
    obs = TensorDict(
        dict(actor=torch.randn(8, 4), critic=torch.randn(8, 5)), batch_size=[8]
    )
    groups = dict(actor=["actor"], critic=["critic"])
    actor = SourceMLPModel(
        obs,
        groups,
        "actor",
        2,
        hidden_dims=[8],
        activation="swish",
        distribution_cfg=dict(
            class_name="embodied_learning.algorithms.brax_ppo:SourceNormalTanh"
        ),
    )
    critic = SourceMLPModel(
        obs, groups, "critic", 1, hidden_dims=[8], activation="swish"
    )
    storage = RolloutStorage("rl", 8, 4, obs, [2])
    algorithm = SourcePPO(
        actor,
        critic,
        storage,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=0.0003,
        gamma=0.97,
        lam=0.95,
        schedule="fixed",
        use_clipped_value_loss=False,
        value_loss_coef=0.25,
    )
    before = torch.nn.utils.parameters_to_vector(actor.parameters()).detach().clone()
    for step in range(4):
        algorithm.act(obs)
        dones = torch.zeros(8, dtype=torch.bool)
        timeouts = torch.zeros_like(dones)
        if step == 1:
            dones[0] = timeouts[0] = True
        algorithm.process_env_step(
            obs, torch.ones(8), dones, dict(time_outs=timeouts)
        )
    assert bool((storage.rewards == 1).all()), (
        "Source masks truncation instead of adding RSL timeout reward"
    )
    algorithm.compute_returns(obs)
    result = algorithm.update()
    assert all(np.isfinite(v) for v in result.values())
    assert algorithm.optimizer.param_groups[0]["lr"] == 0.0003
    assert {float(s["step"]) for s in algorithm.optimizer.state.values()} == {4.0}
    assert storage.step == 0
    assert not torch.equal(
        before, torch.nn.utils.parameters_to_vector(actor.parameters())
    )
    assert algorithm.last_batch_shapes == [(4, 4)] * 4


def test_mismatched_optimization_schema_is_rejected_before_training(tmp_path):
    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist

    config = PpoConfig(optimization_profile="brax_0_12_3_v1", normalization=False)
    with pytest.raises(ValueError, match="optimization profile"):
        train_specialist(SimpleNamespace(schema={}), config, tmp_path)


def test_original_scene_check_allows_declared_trainer_change_only():
    import copy

    from embodied_learning.tasks.cat_curriculum import require_original_task_contract

    old = dict(
        actor_dim=162,
        critic_dim=250,
        action_dim=12,
        activation="elu",
        hidden_dims=[256, 128, 64],
        schema=dict(
            scene_hash="original",
            task={"ground": 1},
            robot_model_identity="robot",
            field_cache_identity="old",
            foot_guidance_experiment={"foot_projection_radius_m": 0.25},
        ),
    )
    new = copy.deepcopy(old)
    new["activation"] = "swish"
    new["schema"].update(
        optimization_profile="brax_0_12_3_v1",
        policy_activation="swish",
        field_cache_identity="new",
        source_interpolation="xyz_weights_v1",
        foot_guidance_experiment={"foot_projection_radius_m": 0.30},
    )
    require_original_task_contract(old, new)
    for key, bad in (
        ("task", {"ground": 2}),
        ("scene_hash", "changed"),
        ("robot_model_identity", "different"),
        ("optimization_profile", "unknown"),
    ):
        changed = copy.deepcopy(new)
        changed["schema"][key] = bad
        with pytest.raises(ValueError, match="original"):
            require_original_task_contract(old, changed)
    changed = copy.deepcopy(new)
    changed["actor_dim"] = 171
    with pytest.raises(ValueError, match="original"):
        require_original_task_contract(old, changed)


@pytest.mark.parametrize("activation", [None, "elu"])
def test_source_profile_rejects_missing_or_conflicting_activation(
    tmp_path, activation
):
    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist

    schema = dict(
        optimization_profile="brax_0_12_3_v1", action_transform="latent_tanh_v1"
    )
    if activation is not None:
        schema["policy_activation"] = activation
    config = PpoConfig(optimization_profile="brax_0_12_3_v1", normalization=False)
    with pytest.raises(ValueError, match="activation"):
        train_specialist(SimpleNamespace(schema=schema), config, tmp_path)


def test_source_loader_rejects_conflicting_network_activation():
    from embodied_learning.policies.field_policy import _make_model
    from embodied_learning.protocols.imitation import PolicyContract

    contract = PolicyContract(
        dict(optimization_profile="brax_0_12_3_v1"),
        4,
        5,
        2,
        (8,),
        activation="elu",
        normalization=False,
    )
    with pytest.raises(ValueError, match="activation"):
        _make_model(contract, "cpu", stochastic=True)
