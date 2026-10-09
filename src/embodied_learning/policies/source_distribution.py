"""Brax-style tanh actions with RSL-RL rollouts stored in latent coordinates.

Sampling and deterministic outputs deliberately return the Gaussian latent.
CatEnv applies tanh once at the simulator boundary, for PPO, play and DAgger.
Log densities and entropy include the tanh Jacobian. KL is invariant under
this common bijection, so the inherited Gaussian KL remains valid.
"""

import math

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.nn import functional as F


class LatentTanhGaussianDistribution(GaussianDistribution):
    @staticmethod
    def _log_jacobian(raw: torch.Tensor) -> torch.Tensor:
        # Stable even where floating-point tanh has saturated to +/-1.
        return 2 * (math.log(2) - raw - F.softplus(-2 * raw))

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return super().log_prob(outputs) - self._log_jacobian(outputs).sum(-1)

    @property
    def entropy(self) -> torch.Tensor:
        raw = self._distribution.rsample()
        return super().entropy + self._log_jacobian(raw).sum(-1)


def distribution_class(action_transform: str | None) -> str:
    if action_transform is None:
        return "GaussianDistribution"
    if action_transform != "latent_tanh_v1":
        raise ValueError(f"Unknown action transform: {action_transform}")
    return (
        "embodied_learning.policies.source_distribution:"
        "LatentTanhGaussianDistribution"
    )
