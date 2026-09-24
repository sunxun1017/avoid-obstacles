"""Common vector-environment contract used by training algorithms.

Observations and actions intentionally remain backend-native objects (usually
PyTorch tensors or tensor dictionaries). Adapters normalize the call contract,
not device placement, observation structure, or simulator behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EnvReset:
    """Result of resetting a vectorized environment."""

    observations: Any
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnvStep:
    """One vectorized transition with separate terminal and timeout flags."""

    observations: Any
    rewards: Any
    terminated: Any
    truncated: Any
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorEnvironment(Protocol):
    """Minimal vector environment interface for rollout collectors."""

    @property
    def num_envs(self) -> int: ...

    @property
    def device(self) -> str: ...

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: Any | None = None,
        options: dict[str, Any] | None = None,
    ) -> EnvReset: ...

    def step(self, actions: Any) -> EnvStep: ...

    def close(self) -> None: ...
