"""Adapter for Isaac Lab manager-based and direct vector RL environments."""

from __future__ import annotations

from typing import Any

from embodied_learning.protocols.environment import EnvReset
from embodied_learning.simulators._adapter import VectorEnvAdapter, unpack_reset


class IsaacLabEnvAdapter(VectorEnvAdapter):
    """Expose an already-created Isaac Lab environment through project APIs.

    Construct Isaac Lab environments through their official task/config workflow
    (typically after starting ``AppLauncher``), then pass the native environment
    here. This adapter deliberately does not import Isaac Lab at module import
    time, so the rest of the package remains usable without Isaac Sim installed.

    Isaac Lab vector environments auto-reset completed sub-environments during
    ``step``. The returned observation for those slots is therefore post-reset;
    terminal observations and sensor reset timing remain backend-defined extras.
    """

    backend_name = "Isaac Lab"

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: Any | None = None,
        options: dict[str, Any] | None = None,
    ) -> EnvReset:
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if env_ids is not None:
            # ManagerBasedRLEnv supports env_ids, while DirectRLEnv versions may
            # expose full reset only. Let the underlying version report support.
            kwargs["env_ids"] = env_ids
        if options is not None:
            kwargs["options"] = options
        return unpack_reset(self.env.reset(**kwargs))
