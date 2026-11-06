"""Training-only S0 impulses in the failure modes measured by phase probes."""

import torch


class S0PhasePushCurriculum:
    version = "s0_phase_push_v1"
    near_goal_version = "s0_phase_push_near_goal_v1"
    phase_names = ("walk", "brake", "stop")

    def __init__(self, num_envs, device, *, near_goal_starts=False):
        self.near_goal_starts = near_goal_starts
        slots = torch.arange(num_envs, device=device)
        self.phase = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 2], device=device)[slots % 8]
        # Each phase covers both signs. Walking and braking are pushed laterally.
        sign = torch.where(((slots // 8 + (slots % 8 == 7)) % 2) == 0, 1.0, -1.0)
        sagittal = self.phase == 2
        self.direction = torch.stack((sign * sagittal, sign * ~sagittal), dim=-1)
        self.used = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.still = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.counts = torch.zeros(3, dtype=torch.long, device=device)
        self.near_goal_reset_count = torch.zeros((), dtype=torch.long, device=device)

    def near_goal_reset_slots(self, ids):
        """Use both signs of half of the stationary impulse slots for goal starts."""
        if not self.near_goal_starts:
            return ids[:0]
        group = ids % 16
        selected = ids[(group == 6) | (group == 7)]
        self.near_goal_reset_count += len(selected)
        return selected

    def select(self, tick, interval, distance, command_speed, planar_speed):
        """Return slots eligible for one extra impulse in the current episode."""
        reset = tick == 1
        self.used &= ~reset
        stopped = (command_speed < .01) & (planar_speed < .15) & (distance <= .32)
        self.still = torch.where(stopped & ~reset, self.still + 1, 0)

        walk = (self.phase == 0) & (tick >= 25) & (distance > .45)
        walk &= (command_speed >= .20) & (planar_speed > .10)
        brake = (self.phase == 1) & (tick >= 25) & (.10 < distance)
        brake &= (distance <= .32) & (planar_speed >= .15)
        stop = (self.phase == 2) & (tick >= 200) & (self.still >= 15)
        elapsed = tick % interval
        separated = ((tick < interval) | (elapsed > 50)) & (interval - elapsed > 50)
        selected = ~self.used & (tick <= 600) & separated & (walk | brake | stop)
        self.used |= selected
        self.counts += torch.stack([(selected & (self.phase == p)).sum() for p in range(3)])
        return torch.nonzero(selected, as_tuple=True)[0]

    def impulse(self, slots, training_step):
        """Ramp the additional impulse cap from 0.20 to 0.45 m/s over 300 updates."""
        fraction = min(max(training_step / 48000, 0.), 1.)
        cap = .20 + .25 * fraction
        magnitude = .10 + (cap - .10) * torch.rand(len(slots), device=slots.device)
        return magnitude[:, None] * self.direction[slots]
