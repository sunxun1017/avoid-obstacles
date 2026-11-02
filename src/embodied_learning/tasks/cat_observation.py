"""Source G1Cat observation layout (Apache-2.0, 866ba392).

Pure Tensor composition with explicit noisy/delayed inputs.
"""

import torch
from tensordict import TensorDict

from embodied_learning.tasks.cat_reward import OBS_INDICES

GROUPS = ((0,), (1,), (2,), (3, 4), (5, 6), (7, 8), (9, 10))


def field_features(
    gf: torch.Tensor, bf: torch.Tensor, sdf: torch.Tensor
) -> torch.Tensor:
    return torch.cat(
        [
            x
            for ids in GROUPS
            for x in (
                gf[:, ids].flatten(1),
                bf[:, ids].flatten(1),
                sdf[:, ids].flatten(1),
            )
        ],
        1,
    )


def build_cat_observations(
    *,
    gyro: torch.Tensor,
    gravity: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    default_joints: torch.Tensor,
    last_actions: torch.Tensor,
    targets: torch.Tensor,
    command: torch.Tensor,
    foot_height: torch.Tensor,
    phase: torch.Tensor,
    root_local_vel: torch.Tensor,
    gf: torch.Tensor,
    bf: torch.Tensor,
    sdf: torch.Tensor,
    body_positions: torch.Tensor,
    body_velocities: torch.Tensor,
    torso_rpy: torch.Tensor,
    gait_mask: torch.Tensor,
    feet_contact: torch.Tensor,
    kp_scale: torch.Tensor,
    kd_scale: torch.Tensor,
    rfi_scale: torch.Tensor,
    navigation_rotation: torch.Tensor,
    gf_delayed: torch.Tensor,
    bf_delayed: torch.Tensor,
    sdf_delayed: torch.Tensor,
    command_delayed: torch.Tensor,
    noise_level: float = 1.0,
) -> TensorDict:
    # The empty-map extension represents absent obstacles by +inf. Encode this
    # as free space at the actor's existing distance horizon, rather than
    # letting nan_to_num turn the Critic input into float32's maximum value.
    # Every finite SDF (including the unbounded source Critic distances) is kept.
    sdf = torch.where(torch.isposinf(sdf), 0.5, sdf)

    def noisy(x: torch.Tensor, scale: float) -> torch.Tensor:
        return (
            x + (2 * torch.rand_like(x) - 1) * noise_level * scale
            if noise_level
            else x
        )

    gait_phase = torch.cat((torch.cos(phase), torch.sin(phase)), 1)
    common = [
        gyro,
        gravity,
        (joint_pos - default_joints)[:, OBS_INDICES],
        joint_vel[:, OBS_INDICES],
        last_actions,
        targets[:, :12],
        command,
        foot_height[:, None],
        gait_phase,
    ]
    critic = torch.cat(
        common
        + [
            root_local_vel,
            field_features(gf, bf, sdf),
            body_positions[:, 0],
            body_velocities[:, 0],
            body_positions[:, 1],
            body_positions[:, 2],
            body_positions[:, 3:5].flatten(1),
            body_velocities[:, 3:5].flatten(1),
            body_positions[:, 5:7].flatten(1),
            body_velocities[:, 5:7].flatten(1),
            body_positions[:, 7:9].flatten(1),
            body_positions[:, 9:11].flatten(1),
            torso_rpy[:, :2],
            gait_mask,
            feet_contact,
            kp_scale,
            kd_scale,
            rfi_scale,
        ],
        1,
    )
    rotation = navigation_rotation.transpose(-1, -2)

    def rotate(v: torch.Tensor) -> torch.Tensor:
        return (rotation[:, None] @ v[..., None]).squeeze(-1)

    actor_command = command.clone()
    actor_command[:, 1:] = (rotation @ command_delayed[:, 1:, None]).squeeze(-1)
    actor_command[:, -1] = 0.0
    actor = torch.cat(
        [
            noisy(gyro, 0.2),
            noisy(gravity, 0.05),
            (noisy(joint_pos, 0.03) - default_joints)[:, OBS_INDICES],
            noisy(joint_vel, 1.5)[:, OBS_INDICES],
            last_actions,
            targets[:, :12],
            actor_command,
            foot_height[:, None],
            gait_phase,
            field_features(
                rotate(gf_delayed),
                rotate(bf_delayed) * (sdf_delayed < 0.5)[..., None],
                sdf_delayed.clamp(-1, 0.5),
            ),
        ],
        1,
    )
    if actor.shape[1] != 162 or critic.shape[1] != 250:
        raise ValueError("Source observation dimensions must be 162/250.")
    return TensorDict(
        {"actor": torch.nan_to_num(actor), "critic": torch.nan_to_num(critic)},
        batch_size=[len(actor)],
    )
