"""Field actors with saved native RSL-RL normalization and strict contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import Tensor, nn

from embodied_learning.protocols.imitation import PolicyContract


def _make_model(
    contract: PolicyContract, device: str, *, stochastic: bool
) -> MLPModel:
    obs = TensorDict(
        {
            "actor": torch.zeros(1, contract.actor_dim, device=device),
            "critic": torch.zeros(1, contract.critic_dim, device=device),
        },
        batch_size=[1],
    )
    distribution = (
        {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
        if stochastic
        else None
    )
    return MLPModel(
        obs,
        {"actor": ["actor"], "critic": ["critic"]},
        "actor",
        contract.action_dim,
        hidden_dims=contract.hidden_dims,
        activation=contract.activation,
        obs_normalization=contract.normalization,
        distribution_cfg=distribution,
    ).to(device)


class FieldStudent(nn.Module):
    """Trainable deterministic generalist; normalization is saved with weights."""

    def __init__(self, contract: PolicyContract, device: str) -> None:
        super().__init__()
        self.contract = contract
        self.model = _make_model(contract, device, stochastic=False)

    def forward(self, observations: TensorDict) -> Tensor:
        return self.model(observations)

    @torch.no_grad()
    def predict(self, observations: TensorDict) -> Tensor:
        return self(observations)

    def update_normalization(self, observations: TensorDict) -> None:
        self.model.update_normalization(observations)


class FrozenTeacher:
    """Deterministic eval-only specialist with immutable normalization."""

    def __init__(
        self,
        model: MLPModel,
        contract: PolicyContract,
        training_scenes: tuple[int, ...],
    ) -> None:
        self.model, self.contract = model, contract
        self.training_scenes = training_scenes
        model.eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(self, observations: TensorDict) -> Tensor:
        return self.model(observations)


def load_teacher(
    checkpoint: Path, *, expected_contract: PolicyContract, device: str
) -> FrozenTeacher:
    """Load a trusted PPO actor, including all normalizer buffers, strictly."""
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    actual = PolicyContract.from_dict(saved["field_contract"])
    expected_contract.require(actual)
    model = _make_model(actual, device, stochastic=True)
    model.load_state_dict(saved["actor_state_dict"], strict=True)
    teacher = FrozenTeacher(model, actual, tuple(saved["training_scenes"]))
    teacher.identity = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    return teacher


def load_student(
    checkpoint: Path, *, expected_contract: PolicyContract, device: str
) -> FieldStudent:
    """Load a trusted DAgger student strictly, for pure-student evaluation."""
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    actual = PolicyContract.from_dict(saved["field_contract"])
    expected_contract.require(actual)
    student = FieldStudent(actual, device)
    student.load_state_dict(saved["student"], strict=True)
    student.eval().requires_grad_(False)
    return student
