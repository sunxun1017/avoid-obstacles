"""Bounded FIFO aggregation with uniform sampling over retained DAgger data."""

from __future__ import annotations

import torch
from torch import Tensor


class AggregationBuffer:
    """Keep the most recent capacity examples, including earlier rounds.

    Uniform sampling uses all retained rounds, not just the current rollout.
    At capacity, oldest examples retire explicitly; this is not a reservoir.
    CPU storage is recommended to keep replay data out of simulation VRAM.
    """

    def __init__(
        self, capacity: int, observation_dim: int, action_dim: int, device: str
    ) -> None:
        if any(
            type(v) is not int or v < 1
            for v in (capacity, observation_dim, action_dim)
        ):
            raise ValueError("Buffer dimensions/capacity must be positive integers.")
        self.capacity = capacity
        self.observations = torch.zeros(capacity, observation_dim, device=device)
        self.labels = torch.zeros(capacity, action_dim, device=device)
        self.size = self.seen = 0

    def __len__(self) -> int:
        return self.size

    def add(self, observations: Tensor, labels: Tensor) -> None:
        """Detach and copy aligned immutable rows; overwrite only oldest rows."""
        if (
            observations.ndim != 2
            or labels.ndim != 2
            or observations.shape[0] != labels.shape[0]
            or observations.shape[1] != self.observations.shape[1]
            or labels.shape[1] != self.labels.shape[1]
            or not torch.isfinite(observations).all()
            or not torch.isfinite(labels).all()
        ):
            raise ValueError(
                "Aggregation requires finite aligned observations/labels."
            )
        count = len(observations)
        retained = min(count, self.capacity)
        indices = (
            torch.arange(
                self.seen + count - retained,
                self.seen + count,
                device=self.observations.device,
            )
            % self.capacity
        )
        if retained:
            self.observations[indices] = (
                observations[-retained:].detach().to(self.observations)
            )
            self.labels[indices] = labels[-retained:].detach().to(self.labels)
        self.seen += count
        self.size = min(self.seen, self.capacity)

    def sample(self, count: int) -> tuple[Tensor, Tensor]:
        """Sample uniformly with replacement from every retained example."""
        if self.size == 0 or count < 1:
            raise ValueError("Cannot sample empty data or a nonpositive count.")
        indices = torch.randint(self.size, (count,), device=self.observations.device)
        return self.observations[indices], self.labels[indices]

    def state_dict(self) -> dict:
        """Portable storage including cursor and history size."""
        return dict(
            capacity=self.capacity,
            size=self.size,
            seen=self.seen,
            observations=self.observations[: self.size].cpu().clone(),
            labels=self.labels[: self.size].cpu().clone(),
        )

    def load_state_dict(self, state: dict) -> None:
        """Reject a different buffer layout or corrupt cursor before loading."""
        size, seen = state["size"], state["seen"]
        if (
            state["capacity"] != self.capacity
            or type(size) is not int
            or type(seen) is not int
            or not 0 <= size <= self.capacity
            or seen < size
            or size != min(seen, self.capacity)
            or state["observations"].shape != (size, self.observations.shape[1])
            or state["labels"].shape != (size, self.labels.shape[1])
            or not torch.isfinite(state["observations"]).all()
            or not torch.isfinite(state["labels"]).all()
        ):
            raise ValueError("Aggregation buffer checkpoint mismatch/corruption.")
        self.observations[:size] = state["observations"].to(self.observations)
        self.labels[:size] = state["labels"].to(self.labels)
        self.size, self.seen = size, seen
