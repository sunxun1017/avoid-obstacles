"""Adapt Unitree's 480-input Isaac Lab G1 policy to the CAT controller."""

import torch

from embodied_learning.robots.g1 import CAT_JOINT_NAMES
from embodied_learning.tasks.cat_reward import OBS_INDICES


class IsaacLabVelocityHistory:
    """Five oldest-first frames flattened by observation term, as in deployment."""

    widths = (3, 3, 3, 29, 29, 29)

    def __init__(self, slots, device):
        self.frames = torch.zeros(slots, 5, sum(self.widths), device=device)

    def push(self, sample, reset):
        frames = self.frames.roll(-1, dims=1)
        frames[:, -1] = sample
        self.frames = torch.where(reset[:, None, None], sample[:, None], frames)
        return torch.cat([term.flatten(1) for term in self.frames.split(self.widths, dim=-1)], dim=-1)


def native_joint_values(values, indexes):
    native = torch.empty_like(values)
    native[..., indexes] = values
    return native


class IsaacLabG1Stand:
    """Expose official Isaac Lab position targets and gains to the handoff actor."""

    def __init__(self, env, parameters, session, seed, joint_angle_noise_ratio=1/3):
        self.native = env.unwrapped
        self.robot = self.native.scene["robot"]
        control = self.control = self.native.cat_controller
        names = [self.robot.joint_names[int(i)] for i in control.indices.tolist()]
        if names != list(CAT_JOINT_NAMES) or parameters["step_dt"] != .02:
            raise ValueError("Expected the published G1 29-joint layout at 50 Hz.")
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.rng = torch.Generator(device=env.device).manual_seed(seed + 100_000_000)
        self.joint_angle_noise_ratio = joint_angle_noise_ratio
        as_tensor = control.targets.new_tensor
        ids = parameters["joint_ids_map"]
        if sorted(ids) != list(range(29)):
            raise ValueError("Expected a complete Unitree motor permutation.")
        self.indexes = torch.tensor(ids, device=env.device)
        observations = parameters["observations"]
        terms = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos_rel", "joint_vel_rel", "last_action")
        if tuple(observations) != terms or any(
            term["clip"] is not None or term["history_length"] != 5 for term in observations.values()
        ):
            raise ValueError("Unexpected Unitree observation history contract.")
        self.observation_scale = as_tensor([v for term in observations.values() for v in term["scale"]])
        action = parameters["actions"]["JointPositionAction"]
        if action["clip"] is not None or action["joint_ids"] is not None:
            raise ValueError("Expected the published all-joint position action.")

        def to_native(values):
            return native_joint_values(as_tensor(values), self.indexes)

        self.default = to_native(parameters["default_joint_pos"])
        self.scale = to_native(action["scale"])
        self.offset = to_native(action["offset"])
        control.kp.copy_(as_tensor(parameters["stiffness"]))
        control.kd.copy_(as_tensor(parameters["damping"]))
        self.zero_features = control.targets.new_zeros(env.num_envs, 5)
        self.last_action = torch.zeros_like(control.targets)
        self.pending_target = control.targets.clone()
        self.history = IsaacLabVelocityHistory(env.num_envs, env.device)
        self.observed = list(OBS_INDICES)
        self.unobserved = [j for j in range(29) if j not in self.observed]

    def predict(self, observations):
        reset = self.native.episode_length_buf == 0
        self.last_action[reset] = 0
        data = self.robot.data
        ids = self.control.indices
        q = data.joint_pos[:, ids] - self.default
        v = data.joint_vel[:, ids]
        noise = observations["actor"][:, :52] - observations["critic"][:, :52]
        gyro = data.root_link_ang_vel_b + noise[:, :3]
        gravity = data.projected_gravity_b + noise[:, 3:6]
        q[:, self.observed] += self.joint_angle_noise_ratio * noise[:, 6:29]
        v[:, self.observed] += noise[:, 29:52]
        extra = torch.rand((len(q), 2, len(self.unobserved)), device=q.device, generator=self.rng) * 2 - 1
        q[:, self.unobserved] += .03 * self.joint_angle_noise_ratio * extra[:, 0]
        v[:, self.unobserved] += 1.5 * extra[:, 1]
        x = self.indexes
        frame = torch.cat((gyro, gravity, self.zero_features[:, :3], q[:, x], v[:, x], self.last_action[:, x]), dim=1)
        inputs = self.history.push(frame * self.observation_scale, reset).cpu().numpy()
        action = torch.from_numpy(self.session.run(None, {self.input_name: inputs})[0]).to(q.device)
        if not torch.isfinite(action).all():
            raise ValueError("Non-finite Unitree action.")
        action = native_joint_values(action, x)
        self.pending_target.copy_(self.offset + self.scale * action)
        return action
