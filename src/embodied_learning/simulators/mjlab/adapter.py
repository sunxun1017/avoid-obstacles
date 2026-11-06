"""Adapter for mjlab manager-based vector RL environments."""

from __future__ import annotations

from typing import Any

from embodied_learning.protocols.environment import EnvReset
from embodied_learning.simulators._adapter import VectorEnvAdapter, unpack_reset


class MjlabEnvAdapter(VectorEnvAdapter):
    """Expose an already-created mjlab environment through project APIs.

    The native environment can be created by an mjlab task/config. No mjlab
    import is required to import this adapter. Configure ``auto_reset`` in the
    native task according to the collector: when enabled, completed slots are
    reset inside ``step``; when disabled, the rollout owner must call
    ``reset(env_ids=...)`` before stepping those slots again.
    """

    backend_name = "mjlab"

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: Any | None = None,
        options: dict[str, Any] | None = None,
    ) -> EnvReset:
        if env_ids is not None:
            # Current mjlab accepts env_ids for partial reset. Keep options
            # native because task reset options are project-specific.
            result = self.env.reset(env_ids=env_ids, seed=seed, options=options)
        else:
            result = self.env.reset(seed=seed, options=options)
        return unpack_reset(result)
