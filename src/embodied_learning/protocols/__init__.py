"""Framework-independent interfaces shared by project modules."""

from embodied_learning.protocols.environment import (
    EnvReset,
    EnvStep,
    VectorEnvironment,
)

__all__ = ["EnvReset", "EnvStep", "VectorEnvironment"]
