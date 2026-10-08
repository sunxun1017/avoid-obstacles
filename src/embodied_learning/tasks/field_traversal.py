"""Compose scene-local body fields with simulator-supplied kinematics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from embodied_learning.data.field_scenes import SceneManifest
from embodied_learning.perception.field_observation import (
    BodyFieldObservation,
    FieldParameters,
    evaluate_body_field,
    yaw_rotate_vectors,
)
from embodied_learning.perception.field_reward import vmf_direction_reward
from embodied_learning.robots.g1 import G1_BODY_SCHEMA
from embodied_learning.tasks.field_assets import FieldBank


@dataclass(frozen=True)
class RewardParameters:
    """Explicit engineering shaping alongside the raw paper log-vMF reward."""

    vmf_weight: float = 1.0
    concentration_scale: float = 1.0
    speed_weight: float = 1.0
    alive_weight: float = 30.0
    goal_weight: float = 50.0
    collision_weight: float = 50.0
    fall_weight: float = 50.0
    invalid_weight: float = 50.0
    tilt_weight: float = 0.05
    energy_weight: float = 0.00001
    joint_limit_weight: float = 1.0
    action_rate_weight: float = 0.1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")


def parse_task_options(
    options: dict | None,
) -> tuple[FieldParameters, RewardParameters, float]:
    """Reject unknown options and parse field/reward scales before physics."""
    options = options or {}
    if set(options) - {"field", "rewards", "episode_length_s"}:
        raise ValueError("Unknown field task option.")
    field = FieldParameters(**options.get("field", {}))
    rewards = RewardParameters(**options.get("rewards", {}))
    duration = options.get("episode_length_s", 20.0)
    if (
        not isinstance(duration, (float, int))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("episode_length_s must be finite and positive.")
    return field, rewards, float(duration)


class FieldTaskState:
    """Per-slot scene/goal state; all field arrays remain on the task device."""

    def __init__(
        self,
        manifest: SceneManifest,
        bank: FieldBank,
        allowed: tuple[int, ...],
        num_envs: int,
        device: str,
        options: dict | None = None,
    ) -> None:
        if bank.identity != manifest.identity:
            raise ValueError("Scene and field identities differ.")
        self.manifest, self.bank, self.allowed = manifest, bank, allowed
        self.scene_ids = torch.full(
            (num_envs,), allowed[0], dtype=torch.long, device=device
        )
        self.goal_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.goals = torch.zeros(num_envs, 3, device=device)
        self.parameters, self.rewards, duration = parse_task_options(options)
        self.settings = dict(
            field=asdict(self.parameters),
            rewards=asdict(self.rewards),
            episode_length_s=duration,
        )
        self.priority = torch.full((11,), 0.5, device=device)
        self.priority[1] = 1.0
        self.generation = 0
        self._key = None
        self._current = None
        self.step_outcomes: dict[str, Tensor] = {}

    @staticmethod
    def local_points(world: Tensor, origins: Tensor) -> Tensor:
        """Subtract environment translation, retaining metric axes."""
        return world - origins[:, None, :]

    @staticmethod
    def classify(
        *,
        goal_distance: Tensor,
        contact: Tensor,
        upright: Tensor,
        root_height: Tensor,
        valid: Tensor,
        elapsed: Tensor,
        limit: int,
    ) -> dict[str, Tensor]:
        """Independent outcome flags; success requires a safe valid state."""
        collision = contact
        fall = (upright < math.cos(math.radians(70))) | (root_height < 0.35)
        invalid = ~valid
        goal = (goal_distance < 0.1) & ~collision & ~fall & ~invalid
        return dict(
            goal=goal,
            collision=collision,
            fall=fall,
            invalid=invalid,
            timeout=elapsed >= limit,
        )

    def current(self, env: Any) -> BodyFieldObservation:
        """Sample every ordered body site at the current physics state."""
        key = (env.common_step_counter, self.generation)
        if key != self._key:
            robot = env.scene["robot"]
            indices = [robot.site_names.index(n) for n in G1_BODY_SCHEMA]
            self.positions = self.local_points(
                robot.data.site_pos_w[:, indices], env.scene.env_origins
            )
            self.velocity = robot.data.site_lin_vel_w[:, indices]
            sampled = self.bank.sample(self.positions, self.scene_ids, self.goal_ids)
            self._current = evaluate_body_field(
                sampled,
                self.velocity,
                self.priority,
                parameters=self.parameters,
            )
            self._key = key
        return self._current

    def observation(self, env: Any) -> Tensor:
        """Yaw-local guidance/velocity and bounded metric obstacle distances."""
        field = self.current(env)
        quat = env.scene["robot"].data.root_link_quat_w
        guidance = yaw_rotate_vectors(field.guidance, quat)
        velocity = yaw_rotate_vectors(self.velocity, quat)
        distance = field.sdf.nan_to_num(posinf=10.0).clamp(-1.0, 10.0)
        return torch.cat((guidance.flatten(1), velocity.flatten(1), distance), 1)

    def outcomes(self, env: Any) -> dict[str, Tensor]:
        """Obstacle-only contact flags; the support plane is excluded."""
        field = self.current(env)
        contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in env.field_contact_names:
            contact |= env.scene[name].data.found.flatten(1).any(1)
        robot = env.scene["robot"]
        root = robot.data.root_link_pos_w - env.scene.env_origins
        q = robot.data.root_link_quat_w
        upright = 1 - 2 * (q[:, 1].square() + q[:, 2].square())
        self.goal_distance = (root[:, :2] - self.goals[:, :2]).norm(dim=-1)
        flags = self.classify(
            goal_distance=self.goal_distance,
            contact=contact,
            upright=upright,
            root_height=root[:, 2],
            valid=field.valid_mask.all(1),
            elapsed=env.episode_length_buf,
            limit=env.max_episode_length,
        )
        self.step_outcomes = {k: v.clone() for k, v in flags.items()}
        return flags

    def reward(self, env: Any) -> Tensor:
        """Paper log vMF direction objective plus explicit locomotion shaping."""
        field = self.current(env)
        direction = vmf_direction_reward(
            field.guidance,
            self.velocity,
            concentration_scale=self.rewards.concentration_scale,
            valid_mask=field.valid_mask,
        )
        flags = self.outcomes(env)
        params = self.rewards
        speed = (field.guidance[:, 1] * self.velocity[:, 1]).sum(-1)
        robot = env.scene["robot"]
        quat = robot.data.root_link_quat_w
        upright = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
        energy = robot.data.qfrc_actuator.square().sum(-1)
        return (
            params.vmf_weight * direction
            + params.speed_weight * speed
            + params.alive_weight * field.valid_mask.all(1)
            + params.goal_weight * flags["goal"]
            - params.collision_weight * flags["collision"]
            - params.fall_weight * flags["fall"]
            - params.invalid_weight * flags["invalid"]
            - params.tilt_weight * (1 - upright).square()
            - params.energy_weight * energy
        )
