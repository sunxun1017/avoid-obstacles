"""Shared helpers for simulator environment adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embodied_learning.protocols.environment import EnvReset, EnvStep


def unpack_reset(result: Any) -> EnvReset:
    """Normalize Gymnasium-style reset output while preserving observations."""
    if isinstance(result, tuple) and len(result) == 2:
        observations, info = result
        normalized_info = dict(info) if isinstance(info, Mapping) else {}
        return EnvReset(observations, normalized_info)
    return EnvReset(result)


def unpack_step(result: Any) -> EnvStep:
    """Normalize five-value and legacy four-value vector step results."""
    if not isinstance(result, tuple):
        raise TypeError("Environment step() must return a tuple.")

    if len(result) == 5:
        observations, rewards, terminated, truncated, info = result
        return EnvStep(
            observations,
            rewards,
            terminated,
            truncated,
            dict(info) if isinstance(info, Mapping) else {},
        )

    if len(result) == 4:
        observations, rewards, done, info = result
        info_dict = dict(info) if isinstance(info, Mapping) else {}
        truncated = info_dict.get("TimeLimit.truncated", False)
        terminated = _logical_and_not(done, truncated)
        return EnvStep(observations, rewards, terminated, truncated, info_dict)

    raise ValueError(
        f"Expected 4- or 5-value step result; received {len(result)} values."
    )


def _logical_and_not(left: Any, right: Any) -> Any:
    """Compute ``left and not right`` for Python, NumPy, or Torch values."""
    if isinstance(right, bool):
        return left if not right else _zeros_like(left)
    try:
        return left & ~right
    except (TypeError, ValueError):
        return bool(left) and not bool(right)


def _zeros_like(value: Any) -> Any:
    """Create a false mask matching common array/tensor values without imports."""
    zeros_like = getattr(value, "new_zeros", None)
    if zeros_like is not None:
        return zeros_like(value.shape, dtype=getattr(value, "dtype", None))
    zeros_like = getattr(value, "zeros_like", None)
    if zeros_like is not None:
        return zeros_like(value)
    try:
        return value & False
    except TypeError:
        return False


class VectorEnvAdapter:
    """Base wrapper that exposes a backend environment through our protocol."""

    backend_name = "unknown"

    def __init__(self, env: Any) -> None:
        self.env = env

    @property
    def num_envs(self) -> int:
        value = getattr(self.env, "num_envs", None)
        if value is None:
            message = (
                f"{self.backend_name} environment has no num_envs attribute."
            )
            raise AttributeError(message)
        return int(value)

    @property
    def device(self) -> str:
        return str(getattr(self.env, "device", "cpu"))

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: Any | None = None,
        options: dict[str, Any] | None = None,
    ) -> EnvReset:
        raise NotImplementedError

    def step(self, actions: Any) -> EnvStep:
        return unpack_step(self.env.step(actions))

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()
