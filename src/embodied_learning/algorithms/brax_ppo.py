"""Pinned Brax 0.12.3 MLP/PPO semantics over the existing native environment.

Reference: google/brax v0.12.3, training/agents/ppo/{losses,networks}.py
and training/{networks,distribution}.py (Apache-2.0). This is a Torch
translation, not the original JAX runtime. Environment count and physics
remain explicitly different from the author's public training setup.
"""

import math
from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import Distribution
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from embodied_learning.policies.source_distribution import (
    LatentTanhGaussianDistribution,
)


class _MeanSlice(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, logits):
        return logits[..., : self.output_dim]


class SourceNormalTanh(LatentTanhGaussianDistribution):
    """State-dependent softplus scale; actions stay latent until CatEnv.step."""

    def __init__(self, output_dim):
        Distribution.__init__(self, output_dim)
        self._distribution = None
        Normal.set_default_validate_args(False)

    @property
    def input_dim(self):
        return 2 * self.output_dim

    def update(self, mlp_output):
        mean, scale = mlp_output.split(self.output_dim, dim=-1)
        self._distribution = Normal(mean, F.softplus(scale) + 0.001)

    def deterministic_output(self, mlp_output):
        return mlp_output[..., : self.output_dim]

    def as_deterministic_output_module(self):
        return _MeanSlice(self.output_dim)


class SourceMLPModel(MLPModel):
    """LeCun uniform kernels and zero biases, as in Brax's Flax MLP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(
                    layer.weight,
                    -math.sqrt(3 / layer.in_features),
                    math.sqrt(3 / layer.in_features),
                )
                nn.init.zeros_(layer.bias)


@torch.no_grad()
def source_gae(
    truncation, termination, rewards, values, bootstrap_value, *, lam, gamma
):
    """Brax returns and its one-step advantage over the lambda return.

    In particular, truncation masks the timeout transition. It does not
    add an RSL timeout bootstrap reward. Values/targets are recomputed for
    each trajectory minibatch using the current critic.
    """
    mask = 1 - truncation
    next_values = torch.cat((values[1:], bootstrap_value[None]), dim=0)
    deltas = (rewards + gamma * (1 - termination) * next_values - values) * mask
    acc = torch.zeros_like(bootstrap_value)
    differences = torch.empty_like(values)
    for step in reversed(range(len(rewards))):
        acc = deltas[step] + gamma * (1 - termination[step]) * mask[step] * lam * acc
        differences[step] = acc
    vs = differences + values
    next_vs = torch.cat((vs[1:], bootstrap_value[None]), dim=0)
    advantages = (rewards + gamma * (1 - termination) * next_vs - values) * mask
    return vs, advantages


class SourcePPO(PPO):
    """Trajectory minibatches and the pinned Brax loss/gradient semantics."""

    def __init__(self, *args, source_unroll_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self.actor.is_recurrent
            or self.critic.is_recurrent
            or self.rnd
            or self.symmetry
            or self.is_multi_gpu
        ):
            raise ValueError(
                "This experiment supports feedforward single-GPU PPO only."
            )
        if (
            self.schedule != "fixed"
            or self.use_clipped_value_loss
            or self.value_loss_coef != 0.25
        ):
            raise ValueError(
                "Source profile requires fixed LR and unclipped "
                "quarter-MSE value loss."
            )
        self.source_unroll_steps = (
            self.storage.num_transitions_per_env
            if source_unroll_steps is None
            else source_unroll_steps
        )
        if (
            type(self.source_unroll_steps) is not int
            or self.source_unroll_steps < 1
            or self.storage.num_transitions_per_env % self.source_unroll_steps
        ):
            raise ValueError(
                "Source unroll steps must divide the collected rollout."
            )
        trajectories = self.storage.num_envs * (
            self.storage.num_transitions_per_env // self.source_unroll_steps
        )
        if trajectories % self.num_mini_batches:
            raise ValueError(
                "Minibatches must divide the number of short trajectories."
            )
        self.truncations = torch.zeros_like(self.storage.rewards)
        self.last_observations = self.storage.observations[0].clone()
        self.last_batch_shapes = []

    def process_env_step(self, obs, rewards, dones, extras):
        timeout = extras.get("time_outs", torch.zeros_like(dones))
        self.truncations[self.storage.step].copy_(timeout.reshape(-1, 1))
        # Reuse storage/normalization handling without RSL's timeout reward.
        super().process_env_step(
            obs,
            rewards,
            dones,
            {k: v for k, v in extras.items() if k != "time_outs"},
        )

    def compute_returns(self, obs):
        # Keep a regular, preallocated buffer: collection runs in inference_mode.
        self.last_observations.copy_(obs)

    def _trajectory_batches(self):
        """Split collection into separate short unrolls, as in Brax scan.

        Fewer simultaneous worlds may collect several unrolls before an update.
        Each has its own next observation for bootstrapping; GAE never crosses
        a short-unroll boundary merely because both came from the same world.
        """
        st = self.storage
        length, worlds = self.source_unroll_steps, st.num_envs
        rounds = st.num_transitions_per_env // length

        def pack(value):
            suffix = value.shape[2:]
            return (
                value.reshape(rounds, length, worlds, *suffix)
                .transpose(1, 2)
                .reshape(rounds * worlds, length, *suffix)
            )

        data = dict(
            observations=pack(st.observations),
            rewards=pack(st.rewards),
            dones=pack(st.dones),
            actions=pack(st.actions),
            log_prob=pack(st.actions_log_prob),
            truncations=pack(self.truncations),
        )
        next_observations = [
            st.observations[t]
            for t in range(length, st.num_transitions_per_env, length)
        ]
        next_observations.append(self.last_observations)
        bootstrap = torch.stack(next_observations, dim=0).reshape(rounds * worlds)
        return length, rounds * worlds, data, bootstrap

    def update(self):
        st = self.storage
        if st.step != st.num_transitions_per_env:
            raise ValueError("Source update requires a complete rollout.")
        length, count, data, bootstrap_observations = self._trajectory_batches()
        width = count // self.num_mini_batches
        sums = dict(value=0.0, surrogate=0.0, entropy=0.0, global_grad_norm=0.0)
        self.last_batch_shapes = []
        parameters = list(chain(self.actor.parameters(), self.critic.parameters()))
        for _ in range(self.num_learning_epochs):
            # Shuffle complete 20-step unrolls, not individual transitions.
            order = torch.randperm(count, device=self.device)
            for offset in range(0, count, width):
                slots = order[offset : offset + width]
                observations = (
                    data["observations"][slots].transpose(0, 1).reshape(-1)
                )
                values = self.critic(observations).reshape(length, width)
                with torch.no_grad():
                    bootstrap = self.critic(bootstrap_observations[slots]).reshape(
                        width
                    )
                    truncation = data["truncations"][slots, :, 0].transpose(0, 1)
                    termination = data["dones"][slots, :, 0].transpose(
                        0, 1
                    ).float() * (1 - truncation)
                    targets, advantages = source_gae(
                        truncation,
                        termination,
                        data["rewards"][slots, :, 0].transpose(0, 1),
                        values,
                        bootstrap,
                        lam=self.lam,
                        gamma=self.gamma,
                    )
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std(correction=0) + 1e-8
                    )
                logits = self.actor.mlp(self.actor.get_latent(observations))
                self.actor.distribution.update(logits)
                raw = (
                    data["actions"][slots]
                    .transpose(0, 1)
                    .reshape(-1, *st.actions_shape)
                )
                log_prob = self.actor.get_output_log_prob(raw).reshape(length, width)
                ratio = (
                    log_prob - data["log_prob"][slots, :, 0].transpose(0, 1)
                ).exp()
                policy_loss = -torch.minimum(
                    ratio * advantages,
                    ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
                    * advantages,
                ).mean()
                value_loss = 0.25 * (targets - values).square().mean()
                entropy = self.actor.output_entropy.mean()
                loss = policy_loss + value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                # Brax clips the combined actor + critic gradient tree.
                norm = nn.utils.clip_grad_norm_(
                    parameters, self.max_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()
                for key, value in zip(
                    sums, (value_loss, policy_loss, entropy, norm), strict=True
                ):
                    sums[key] += float(value.detach())
                self.last_batch_shapes.append((length, width))
        st.clear()
        updates = self.num_learning_epochs * self.num_mini_batches
        return {key: value / updates for key, value in sums.items()}
