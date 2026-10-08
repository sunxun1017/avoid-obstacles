"""Compose scene-local body fields with simulator-supplied kinematics."""

from __future__ import annotations

import math

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


class FieldTaskState:
    """Per-slot scene/goal state; all field arrays remain on the task device."""

    def __init__(
        self,
        manifest: SceneManifest,
        bank: FieldBank,
        allowed: tuple[int, ...],
        num_envs: int,
        device: str,
    ) -> None:
        if bank.identity != manifest.identity:
            raise ValueError("Scene and field identities differ.")
        self.manifest, self.bank, self.allowed = manifest, bank, allowed
        self.scene_ids = torch.full(
            (num_envs,), allowed[0], dtype=torch.long, device=device
        )
        self.goal_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.goals = torch.zeros(num_envs, 3, device=device)
        self.parameters = FieldParameters()
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

    def current(self, env) -> BodyFieldObservation:
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

    def observation(self, env) -> Tensor:
        """Yaw-local guidance/velocity and bounded metric obstacle distances."""
        field = self.current(env)
        quat = env.scene["robot"].data.root_link_quat_w
        guidance = yaw_rotate_vectors(field.guidance, quat)
        velocity = yaw_rotate_vectors(self.velocity, quat)
        distance = field.sdf.nan_to_num(posinf=10.0).clamp(-1.0, 10.0)
        return torch.cat((guidance.flatten(1), velocity.flatten(1), distance), 1)

    def outcomes(self, env) -> dict[str, Tensor]:
        """Obstacle-only contact flags; the support plane is excluded."""
        field = self.current(env)
        contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in env.field_contact_names:
            contact |= env.scene[name].data.found.flatten(1).any(1)
        robot = env.scene["robot"]
        root = robot.data.root_link_pos_w - env.scene.env_origins
        q = robot.data.root_link_quat_w
        upright = 1 - 2 * (q[:, 1].square() + q[:, 2].square())
        flags = self.classify(
            goal_distance=(root[:, :2] - self.goals[:, :2]).norm(dim=-1),
            contact=contact,
            upright=upright,
            root_height=root[:, 2],
            valid=field.valid_mask.all(1),
            elapsed=env.episode_length_buf,
            limit=env.max_episode_length,
        )
        self.step_outcomes = {k: v.clone() for k, v in flags.items()}
        return flags

    def reward(self, env) -> Tensor:
        """Paper log vMF direction objective plus explicit locomotion shaping."""
        field = self.current(env)
        direction = vmf_direction_reward(
            field.guidance,
            self.velocity,
            concentration_scale=1.0,
            valid_mask=field.valid_mask,
        )
        flags = self.outcomes(env)
        # Remove the state-independent uniform-density baseline only.
        direction += field.valid_mask.sum(1) * math.log(4 * math.pi)
        speed = (field.guidance[:, 1] * self.velocity[:, 1]).sum(-1)
        safety = flags["collision"] | flags["fall"] | flags["invalid"]
        return direction + speed + 50 * flags["goal"] - 50 * safety
