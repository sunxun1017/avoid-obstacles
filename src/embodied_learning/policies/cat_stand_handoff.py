"""Hysteretic handoff between CAT navigation and Unitree's zero-command Actor."""

import torch


class StandHandoff:
    """Require supported low-speed approach and let small pushes settle first."""

    version = "cat_unitree_stand_handoff_v1"
    enter_distance_m = 0.08
    return_distance_m = 0.10
    immediate_return_distance_m = 0.30
    enter_linear_m_s = 0.16
    recovery_linear_m_s = 0.20
    angular_rad_s = 0.60
    dwell_steps = 5
    blend_steps = 10
    target_delta_rad = 0.06

    def __init__(self, slots, device):
        self.standing = torch.zeros(slots, dtype=torch.bool, device=device)
        self.transitioning = torch.zeros_like(self.standing)
        self._dwell = torch.zeros(slots, dtype=torch.int32, device=device)
        self._blend = torch.zeros_like(self._dwell)
        self.weight = torch.zeros(slots, device=device)

    def update(self, reset, distance, linear, angular, supported):
        self.standing[reset] = False
        self._dwell[reset] = 0
        self._blend[reset] = 0
        stable = supported & (angular < self.angular_rad_s)
        approach = stable & (linear < self.enter_linear_m_s) & (distance <= self.enter_distance_m)
        displaced = stable & (linear < self.recovery_linear_m_s) & (distance > self.return_distance_m)
        ready = torch.where(self.standing, displaced, approach) & ~reset
        self._dwell.copy_(torch.where(ready, self._dwell + 1, 0))
        switch = (self._dwell >= self.dwell_steps) | (
            self.standing & (distance > self.immediate_return_distance_m)
        )
        self.standing.logical_xor_(switch)
        self._dwell[switch] = 0
        previous = self._blend.clone()
        self._blend.add_(torch.where(self.standing, 1, -1)).clamp_(0, self.blend_steps)
        self.transitioning.copy_(self._blend != previous)
        self.weight.copy_(self._blend / self.blend_steps)

    def mix_targets(self, previous, walk, stand):
        target = torch.lerp(walk, stand, self.weight[:, None])
        limited = previous + (target - previous).clamp(-self.target_delta_rad, self.target_delta_rad)
        return torch.where(self.transitioning[:, None], limited, target)


class CatUnitreeHandoffActor:
    """Keep both actors live and blend targets and gains in the common joint order.

    The standing adapter supplies Unitree's absolute targets. CAT continues to
    observe the actual executed target and the effective bounded leg increment.
    """

    def __init__(self, env, navigation, make_standing_adapter):
        if any(env.cat_state.manifest.scenes[i].boxes for i in env.cat_state.allowed):
            raise ValueError("The initial handoff experiment requires an empty scene.")
        self.native = env.unwrapped
        self.controller = self.native.cat_controller
        self.robot = self.native.scene["robot"]
        self.navigation = navigation
        self.handoff = StandHandoff(env.num_envs, env.device)
        self.walk_kp, self.walk_kd = self.controller.kp.clone(), self.controller.kd.clone()
        process_navigation = self.controller.process_action
        self.standing = make_standing_adapter()
        self.stand_kp, self.stand_kd = self.controller.kp.clone(), self.controller.kd.clone()
        self.controller.kp = self.walk_kp.repeat(env.num_envs, 1)
        self.controller.kd = self.walk_kd.repeat(env.num_envs, 1)

        def process(actions):
            previous = self.controller.targets.clone()
            process_navigation(actions)
            targets = self.handoff.mix_targets(previous, self.controller.targets, self.standing.pending_target)
            effective = ((targets[:, :12] - previous[:, :12]) / 0.5).clamp(-1, 1)
            blended = (self.handoff.weight > 0) | self.handoff.transitioning
            self.controller.actions.copy_(torch.where(blended[:, None], effective, actions))
            self.controller.targets.copy_(targets)
            weight = self.handoff.weight[:, None]
            self.controller.kp.copy_(torch.lerp(self.walk_kp, self.stand_kp, weight))
            self.controller.kd.copy_(torch.lerp(self.walk_kd, self.stand_kd, weight))

        self.controller.process_action = process

    def predict(self, observations):
        reset = self.native.episode_length_buf == 0
        double_support = self.native.scene["cat_left_ground"].data.found.flatten(1).any(1)
        double_support &= self.native.scene["cat_right_ground"].data.found.flatten(1).any(1)
        root = self.robot.data.root_link_pos_w - self.native.scene.env_origins
        distance = (root[:, :2] - self.native.cat_state.goals[:, :2]).norm(dim=-1)
        linear = self.robot.data.root_link_lin_vel_w.norm(dim=-1)
        angular = self.robot.data.root_link_ang_vel_b.norm(dim=-1)
        self.handoff.update(reset, distance, linear, angular, double_support)
        self.standing.last_action.copy_(
            (self.controller.targets - self.standing.offset) / self.standing.scale
        )
        self.standing.predict(observations)
        return self.navigation.predict(observations)


class UnitreeGoalCommands:
    """Recenter in an empty scene with the same official velocity Actor and gains."""

    start_distance_m = 0.10
    stop_distance_m = 0.04
    min_speed_m_s = 0.12
    max_speed_m_s = 0.30
    gain_s_inv = 2.0
    gait_period_steps = 30  # 0.60 s at the official 50 Hz control rate.

    def __init__(self, slots, device):
        self.recentering = torch.zeros(slots, dtype=torch.bool, device=device)

    def update(self, reset, active, body_error, linear, angular, supported, step):
        distance = body_error.norm(dim=-1)
        start = (
            (distance > self.start_distance_m)
            & supported
            & (linear < StandHandoff.recovery_linear_m_s)
            & (angular < StandHandoff.angular_rad_s)
        )
        keep = self.recentering & (distance > self.stop_distance_m)
        self.recentering.copy_(active & ~reset & (start | keep))

        speed = (self.gain_s_inv * distance).clamp(self.min_speed_m_s, self.max_speed_m_s)
        velocity = body_error * (speed / distance.clamp_min(1e-6))[:, None]
        phase = ((step + 1) % self.gait_period_steps) * (2 * torch.pi / self.gait_period_steps)
        zero_yaw = torch.zeros_like(velocity[:, :1])
        features = torch.cat((velocity, zero_yaw, phase.sin()[:, None], phase.cos()[:, None]), dim=1)
        return torch.where(self.recentering[:, None], features, 0.0)
