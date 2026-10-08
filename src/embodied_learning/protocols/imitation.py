"""Simulator-independent training and imitation contracts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from tensordict import TensorDict
from torch import Tensor, nn


@dataclass(frozen=True)
class PolicyContract:
    """Ordered physical schema, input dimensions and actor architecture."""

    schema: dict[str, Any]
    actor_dim: int
    critic_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...]
    activation: str = "elu"
    normalization: bool = True
    critic_hidden_dims: tuple[int, ...] | None = None

    @classmethod
    def from_env(
        cls,
        env: FieldTrainingEnvironment,
        hidden_dims: tuple[int, ...],
        *,
        critic_hidden_dims: tuple[int, ...] | None = None,
        normalization: bool | None = None,
    ) -> PolicyContract:
        obs = env.get_observations()
        return cls(
            json.loads(json.dumps(env.schema)),
            obs["actor"].shape[1],
            obs["critic"].shape[1],
            env.num_actions,
            hidden_dims,
            normalization=env.schema.get("normalization", True)
            if normalization is None
            else normalization,
            critic_hidden_dims=critic_hidden_dims,
        )

    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON-compatible form for strict checkpoint validation."""
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyContract:
        value = dict(value)
        value["hidden_dims"] = tuple(value["hidden_dims"])
        if value.get("critic_hidden_dims") is not None:
            value["critic_hidden_dims"] = tuple(value["critic_hidden_dims"])
        return cls(**value)

    def require(self, actual: PolicyContract, *, network: bool = True) -> None:
        """Reject any schema/dimension drift; optionally allow a new network."""
        expected, observed = self.to_dict(), actual.to_dict()
        if not network:
            for item in ("hidden_dims", "activation", "critic_hidden_dims"):
                expected.pop(item)
                observed.pop(item)
        if expected != observed:
            raise ValueError(
                "Policy contract mismatch (schema/actions/observations)."
            )


class FieldTrainingEnvironment(Protocol):
    """Native vector environment interface accepted by official RSL-RL."""

    num_envs: int
    num_actions: int
    device: str
    cfg: Any
    schema: dict[str, Any]
    scene_ids: Tensor
    episode_length_buf: Tensor
    max_episode_length: int

    def get_observations(self) -> TensorDict: ...
    def step(self, actions: Tensor) -> tuple[TensorDict, Tensor, Tensor, dict]: ...
    def close(self) -> None: ...


class ExpertPolicy(Protocol):
    """Frozen deterministic policy preserving its training normalizer."""

    contract: PolicyContract

    def predict(self, observations: TensorDict) -> Tensor: ...


class StudentPolicy(ExpertPolicy, Protocol):
    """Trainable module with deterministic inference and explicit statistics."""

    def __call__(self, observations: TensorDict) -> Tensor: ...
    def parameters(self) -> Iterator[nn.Parameter]: ...
    def train(self, mode: bool = True) -> StudentPolicy: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
    def update_normalization(self, observations: TensorDict) -> None: ...


ImitationEnvironment = FieldTrainingEnvironment
