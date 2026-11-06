"""Evaluation-only probe for noisy Actor joint velocity feedback."""

from types import SimpleNamespace

# Actor layout: gyro 3, gravity 3, positions 23, velocities 23.
ACTOR_JOINT_VELOCITY = slice(29, 52)
ALPHA = 0.45
DWELL_TICKS = 10


def velocity_ema_step(raw, previous, count, eligible, *, where):
    count = where(eligible, count + 1, 0)
    enabled = count >= DWELL_TICKS
    blended = ALPHA * raw + (1 - ALPHA) * previous
    return count, where(enabled[..., None], blended, raw), enabled


def stationary_velocity_probe(policy, env, ground, active):
    import torch

    state, native = env.cat_state, env.unwrapped
    count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    previous = torch.zeros((env.num_envs, 23), device=env.device)
    applied = torch.zeros_like(count)

    def predict(observations):
        nonlocal count, previous
        metrics = state.last_metrics
        if "goal_distance" not in metrics:
            return policy.predict(observations)
        support = (
            ground[0].data.found.flatten(1).any(1)
            & ground[1].data.found.flatten(1).any(1)
        )
        eligible = (
            active & (native.episode_length_buf > 0) & support
            & (state.gait.last_command[:, 0] < 0.5)
            & (metrics["goal_distance"] <= 0.1)
            & (metrics["root_speed"] < 0.2)
            & (metrics["root_angular_speed"] < 1.0)
        )
        actor = observations["actor"]
        if actor.shape != (env.num_envs, 162):
            raise ValueError("Velocity filter requires the CAT 162-channel Actor contract")
        count, previous, enabled = velocity_ema_step(
            actor[:, ACTOR_JOINT_VELOCITY], previous, count, eligible, where=torch.where,
        )
        applied.add_(enabled)
        # Never replace a tensor in the environment's cached observation.
        filtered = observations.clone(recurse=False)
        actor = actor.clone()
        actor[:, ACTOR_JOINT_VELOCITY] = previous
        filtered["actor"] = actor
        return policy.predict(filtered)

    return SimpleNamespace(predict=predict), applied
