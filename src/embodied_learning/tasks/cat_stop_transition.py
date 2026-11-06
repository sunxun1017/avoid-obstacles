"""Opt-in stop8/resume10 hysteresis and coordinated gait/GF transitions."""

import torch
import torch.nn.functional as F

from embodied_learning.tasks.cat_gait import CatPlaybackGaitState

VERSION = "stop8_resume10_v1"


class CatStopTransitionGait(CatPlaybackGaitState):
    """Normal transitions fade; strong dynamics retain immediate recovery.

    No random draws. The latent walking phase is preserved through normal
    stops. Phase feature amplitude interpolates to the standing representation.
    Emergency restarts keep the source phase reset instead of delaying balance.
    """

    EXTRA = ("parked", "urgent", "activation", "walk_vector", "phase_features")

    def __init__(self, num_envs, device, *, dt=0.02, command_deadband=0.01):
        super().__init__(num_envs, device, dt=dt, command_deadband=command_deadband)
        self.parked = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.urgent = torch.zeros_like(self.parked)
        self.activation = torch.zeros(num_envs, device=device)
        self.walk_vector = torch.zeros(num_envs, 3, device=device)
        self.phase_features = torch.zeros(num_envs, 4, device=device)
        self.phase_features[:, :2] = 1

    def reset(self, env_ids, frequency, initial_phase):
        super().reset(env_ids, frequency, initial_phase)
        self.parked[env_ids] = False
        self.urgent[env_ids] = False
        self.activation[env_ids] = 0
        self.walk_vector[env_ids] = 0
        self.phase_features[env_ids] = self.phase_features.new_tensor([1, 1, 0, 0])

    def snapshot(self):
        """Extra state; task refresh separately snapshots inherited gait state."""
        return {k: getattr(self, k).clone() for k in self.EXTRA}

    def restore(self, snapshot, ids):
        for k, v in snapshot.items():
            getattr(self, k)[ids] = v[ids]

    def seed(self, command, distance, ids):
        moving = command[ids, 1:3].norm(dim=-1) > self.command_deadband
        self.parked[ids] = distance[ids] < 0.08
        self.activation[ids] = moving.to(command.dtype)
        self.walk_vector[ids] = 0
        self.walk_vector[ids, :2] = F.normalize(command[ids, 1:3], dim=-1, eps=1e-6)
        phase = torch.cat((self.phase[ids].cos(), self.phase[ids].sin()), -1)
        stand = phase.new_tensor([1, 1, 0, 0])
        a = self.activation[ids, None]
        self.phase_features[ids] = a * phase + (1 - a) * stand

    def filter_command(self, command, root, goals, velocity, gyro):
        d = (root[:, :2] - goals[:, :2]).norm(dim=-1)
        dynamic = (velocity[:, :2].norm(dim=-1) > 0.35) | (gyro.norm(dim=-1) > 0.8)
        self.urgent.copy_(
            dynamic & (command[:, 1:3].norm(dim=-1) > self.command_deadband)
        )
        self.parked.copy_(
            torch.where(d < 0.08, True, torch.where(d >= 0.10, False, self.parked))
        )
        # A strong disturbance must not be held inside the position latch while
        # the root navigator is already asking for a valid recovery step.
        self.parked[self.urgent] = False
        return torch.where(self.parked[:, None], 0.0, command)

    def advance(self, command):
        has_velocity = command[:, 1:3].norm(dim=-1) > self.command_deadband
        restart = ~self.has_velocity & has_velocity
        old_a = self.activation.clone()
        active = has_velocity | (old_a > 0)
        phase = (
            torch.remainder(
                self.phase + self.phase_dt[:, None] + torch.pi, 2 * torch.pi
            )
            - torch.pi
        )
        self.phase = torch.where(active[:, None], phase, self.phase)
        self.phase[restart & ((self.step == 0) | self.urgent)] = (
            self.phase.new_tensor([0.0, torch.pi])
        )
        rate = self.dt / 0.2
        target_a = has_velocity.to(command.dtype)
        self.activation += (target_a - self.activation).clamp(-rate, rate)
        self.activation[self.urgent & has_velocity] = 1
        target = torch.zeros_like(self.walk_vector)
        target[:, :2] = F.normalize(command[:, 1:3], dim=-1, eps=1e-6)
        delta = target - self.walk_vector
        length = delta.norm(dim=-1, keepdim=True)
        self.walk_vector += delta * (rate / length.clamp_min(rate))
        self.walk_vector[self.urgent & has_velocity] = target[
            self.urgent & has_velocity
        ]
        phase_features = torch.cat((self.phase.cos(), self.phase.sin()), -1)
        stand = phase_features.new_tensor([1, 1, 0, 0])
        a = self.activation[:, None]
        self.phase_features.copy_(a * phase_features + (1 - a) * stand)
        cosine = self.phase.cos()
        mask = torch.where(cosine > 0.6, 1.0, torch.where(cosine < -0.6, -1.0, 0.0))
        self.gait_mask = torch.where(a > 0, mask, 1.0)
        result = command.clone()
        result[:, 0] = self.activation
        self.last_command = result.clone()
        self.has_velocity.copy_(has_velocity)
        self.stop_timestep = torch.where(
            has_velocity, self.step + 1, self.stop_timestep
        )
        self.step += 1
        return result


def adapt_stop_transfer(checkpoint, old, target, source, curriculum):
    """Explicit Actor-only transfer; strict full resume remains unchanged."""
    import copy

    from embodied_learning.tasks.cat_foot_guidance import sha256

    declaration = curriculum.get("stop_transition_refinement")
    if declaration is None or (
        source["experiment"] == curriculum["experiment"]
        and source["protocol_sha256"] == curriculum["protocol_sha256"]
    ):
        return old, source, {}
    before = old["schema"]["task"].get("stop_transition")
    after = target["schema"]["task"].get("stop_transition")
    expected = dict(
        version="cat_stop_transition_actor_transfer_v1",
        parent_sha256=sha256(checkpoint),
        parent_experiment=source["experiment"],
        parent_protocol_sha256=source["protocol_sha256"],
        before=before,
        after=after,
    )
    if (
        declaration != expected
        or source["stage"] != curriculum["stage"]
        or before not in (None, VERSION)
        or after not in (None, VERSION)
        or target["schema"]["task"].get("arrival_control") != "root_planar_blend_v2"
    ):
        raise ValueError("Stop transition parent/declaration mismatch.")
    adapted = copy.deepcopy(old)
    if after is None:
        adapted["schema"]["task"].pop("stop_transition", None)
    else:
        adapted["schema"]["task"]["stop_transition"] = after
    return (
        adapted,
        {
            **source,
            "experiment": curriculum["experiment"],
            "protocol_sha256": curriculum["protocol_sha256"],
        },
        {"task.stop_transition": dict(before=before, after=after)}
        if before != after
        else {},
    )
