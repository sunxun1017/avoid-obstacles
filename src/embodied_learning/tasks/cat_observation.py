"""Source G1Cat observation layout (Apache-2.0, 866ba392).

Pure Tensor composition with explicit noisy/delayed inputs.
"""

import torch
from tensordict import TensorDict

from embodied_learning.tasks.cat_reward import OBS_INDICES

GROUPS = ((0,), (1,), (2,), (3, 4), (5, 6), (7, 8), (9, 10))


def field_features(gf, bf, sdf):
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
    gyro,
    gravity,
    joint_pos,
    joint_vel,
    default_joints,
    last_actions,
    targets,
    command,
    foot_height,
    phase,
    root_local_vel,
    gf,
    bf,
    sdf,
    body_positions,
    body_velocities,
    torso_rpy,
    gait_mask,
    feet_contact,
    kp_scale,
    kd_scale,
    rfi_scale,
    navigation_rotation,
    gf_delayed,
    bf_delayed,
    sdf_delayed,
    command_delayed,
    noise_level=1.0,
):
    # The empty-map extension represents absent obstacles by +inf. Encode this
    # as free space at the actor's existing distance horizon, rather than
    # letting nan_to_num turn the Critic input into float32's maximum value.
    # Every finite SDF (including the unbounded source Critic distances) is kept.
    sdf = torch.where(torch.isposinf(sdf), 0.5, sdf)

    def noisy(x, scale):
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

    def rotate(v):
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
