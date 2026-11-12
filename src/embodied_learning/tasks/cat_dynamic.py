"""Versioned tensor geometry and task memory for the joint CAT expert."""

import torch

JOINT_VERSION = "cat_static_dynamic_expert_v1"
ACTOR_DIM, CRITIC_DIM = 191, 279


def predict_sphere_clearance(centers, radii, velocities, obstacle_position,
                             obstacle_velocity, obstacle_radius, *,
                             times=(0., .2, .4, .8)):
    """Minimum surface separation over every cover and prediction instant."""
    if (centers.ndim != 3 or centers.shape[-1] != 3
            or velocities.shape != centers.shape
            or radii.shape != (centers.shape[1],)
            or obstacle_position.shape != (centers.shape[0], 3)
            or obstacle_velocity.shape != obstacle_position.shape
            or obstacle_radius.shape != (centers.shape[0],)
            or not times or min(times) < 0):
        raise ValueError("Incompatible dynamic geometry shapes/times.")
    arrays = (radii, velocities, obstacle_position, obstacle_velocity, obstacle_radius)
    if any(x.device != centers.device or x.dtype != centers.dtype for x in arrays):
        raise ValueError("Dynamic geometry must share dtype/device.")
    if (radii <= 0).any() or (obstacle_radius <= 0).any():
        raise ValueError("Dynamic geometry radii must be positive.")
    t = centers.new_tensor(times)[None, None, :, None]
    relative = centers[:, :, None] - obstacle_position[:, None, None]
    velocity = velocities[:, :, None] - obstacle_velocity[:, None, None]
    d = (relative + t * velocity).norm(dim=-1)
    d -= radii[None, :, None] + obstacle_radius[:, None, None]
    return d.amin((1, 2))


def head_allowance(centers, radii, velocities, obstacle_position,
                   obstacle_velocity, obstacle_radius):
    """Caller supplies upright head geometry, never the actual crouched head."""
    distance = predict_sphere_clearance(
        centers, radii, velocities, obstacle_position, obstacle_velocity,
        obstacle_radius, times=(0., .1, .2),
    )
    return ((.05 - distance) / .05).clamp(0, 1)


def nominal_risk(reference_clearance, valid):
    return torch.where(valid, ((.25-reference_clearance)/.20).clamp(0, 1), 0.)


class JointMemory:
    """Progress has a high-water mark; retreat cannot collect it twice."""

    def __init__(self, count, device):
        self.best = torch.zeros(count, device=device)
        self.accumulated = torch.zeros_like(self.best)
        self.stall_seconds = torch.zeros_like(self.best)
        self.clear_seconds = torch.zeros_like(self.best)

    def reset(self, ids):
        for value in (self.best, self.accumulated, self.stall_seconds, self.clear_seconds):
            value[ids] = 0

    def update(self, progress, risk, active, dt=.02):
        delta = (progress-self.best).clamp_min(0) * active
        self.best += delta
        accumulated = self.accumulated + delta
        advanced = accumulated >= .02-1e-7
        self.accumulated = torch.where(active, torch.where(advanced, 0., accumulated), self.accumulated)
        self.stall_seconds = torch.where(active,
            torch.where(advanced, 0., self.stall_seconds+dt), self.stall_seconds)
        self.clear_seconds = torch.where(active,
            torch.where(risk < .2, self.clear_seconds+dt, 0.), self.clear_seconds)
        return delta

    @property
    def stalled(self):
        return (self.stall_seconds >= 2.-1e-6) & (self.clear_seconds >= 1.-1e-6)

    def features(self, route_length):
        return torch.stack((self.best/route_length.clamp_min(.01),
                            self.accumulated/.02,
                            (self.stall_seconds/2).clamp(0, 1),
                            self.clear_seconds.clamp(0, 1)), -1)

    def snapshot(self, ids):
        return {name: getattr(self, name)[ids].clone() for name in
                ("best", "accumulated", "stall_seconds", "clear_seconds")}

    def restore(self, ids, snapshot):
        for name in ("best", "accumulated", "stall_seconds", "clear_seconds"):
            getattr(self, name)[ids] = snapshot[name]
