"""Joint CAT state layered on the unchanged source observation/control path."""

import torch

from embodied_learning.tasks.cat_dynamic import (
    JointMemory, head_allowance, nominal_risk, predict_sphere_clearance,
)
from embodied_learning.tasks.cat_dynamic_reward import gait_cost, joint_reward
from embodied_learning.tasks.cat_reward import CatRewardParameters, cat_reward_terms
from embodied_learning.tasks.cat_traversal import CatTaskState
from embodied_learning.tasks.cat_traversal_safety import TraversalSafety


class JointTraversalSafety(TraversalSafety):
    def corridor(self, command, root, scene_ids):
        return command, torch.zeros_like(root[:, 0])


class JointTaskState(CatTaskState):
    def __init__(self, *args, joint_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_config = dict(joint_config)
        self.gait_scale = float(joint_config['gait_scale'])
        if self.gait_scale < 1 or not 1 <= joint_config['stage'] <= 4:
            raise ValueError('Invalid joint stage/calibration.')
        device, count = self.scene_ids.device, len(self.scene_ids)
        self.traversal_safety = JointTraversalSafety(self.manifest, device)
        self.memory = JointMemory(count, device)
        self.start_x = torch.zeros(count, device=device)
        self.hold = torch.zeros_like(self.start_x)
        self.roof_entered = torch.zeros(count, dtype=torch.bool, device=device)
        self.roof_passed = self.roof_entered.clone()
        self.bar_entered = self.roof_entered.clone()
        self.bar_passed = self.roof_entered.clone()
        self.joint_phase = torch.zeros(count, 4, dtype=torch.bool, device=device)
        self.terrain = torch.zeros(len(self.manifest.scenes), 6, device=device)
        self.width = torch.full((len(self.manifest.scenes),), .6, device=device)
        for i, scene in enumerate(self.manifest.scenes):
            if scene.id.startswith('legacy_'):
                self.width[i] = .32
            for box in scene.boxes:
                c, h = box.center, box.half_size
                if c[2]-h[2] <= .01 and c[2]+h[2] <= .35:
                    self.terrain[i, :3] = self.terrain.new_tensor((c[0]-h[0], c[0]+h[0], c[2]+h[2]))
                elif c[2]-h[2] > .7:
                    self.terrain[i, 3:] = self.terrain.new_tensor((c[0]-h[0], c[0]+h[0], c[2]-h[2]))
        self.previous_cover = None

    def reset(self, env_ids):
        super().reset(env_ids)
        self.memory.reset(env_ids)
        self.hold[env_ids] = 0
        for value in (self.roof_entered, self.roof_passed, self.bar_entered, self.bar_passed, self.joint_phase):
            value[env_ids] = False

    def refresh(self, payload, controller, steps, common_step):
        if self._key == (common_step, self.generation):
            return
        initial = self.initial.clone()
        same = self._last_common == common_step
        active = ~initial & (not same)
        previous = self._obs
        # Source refresh preserves continuing slots on partial reset. Its
        # cached observations must retain the source dimensions for that copy.
        if previous is not None:
            self._obs = previous.clone()
            self._obs['actor'] = previous['actor'][:, :162]
            self._obs['critic'] = previous['critic'][:, :250]
        super().refresh(payload, controller, steps, common_step)
        x = payload['reward_inputs']
        raw_root = payload['root_position']
        root = torch.nan_to_num(raw_root)
        self.start_x[initial] = root[initial, 0]
        geometry, moving = payload['joint_geometry'], payload['dynamic']
        raw_cover = geometry['centers']
        cover = torch.nan_to_num(raw_cover)
        if self.previous_cover is None:
            self.previous_cover = cover.clone()
        self.previous_cover[initial] = cover[initial]
        body_velocity = (cover-self.previous_cover)/.02
        radii = geometry['radii']
        position = moving['observed_position']
        velocity = moving['observed_velocity']
        valid = moving['valid'] & ((position-root).norm(dim=-1) <= 3.)
        nominal = torch.nan_to_num(x.cmd_vel)
        reference_velocity = nominal[:, None].expand_as(cover)
        d_reference = predict_sphere_clearance(cover, radii, reference_velocity,
            position, velocity, moving['radius'])
        risk = nominal_risk(d_reference, valid)
        clearance = predict_sphere_clearance(cover, radii, body_velocity,
            moving['position'], moving['velocity'], moving['radius'])
        clearance = torch.where(moving['episode_dynamic'], clearance, torch.inf)
        upright = geometry['upright_head']
        allowance = head_allowance(upright, geometry['head_radii'],
            nominal[:, None].expand_as(upright), moving['position'],
            moving['velocity'], moving['radius']) * moving['episode_dynamic']
        if self.overhead_clearance is not None:
            allowance = torch.maximum(allowance, self.overhead_clearance.allowance(
                root, payload['body_positions'][:, 0], nominal[:, :2], self.scene_ids))
        terms = dict(self.terms)
        terms.update(cat_reward_terms(x, parameters=CatRewardParameters(controller.lower, controller.upper),
            ducking_allowance=allowance, risk_alpha=self.risk_alpha,
            standing_contact_slip=True,
            standing_balance_free=getattr(self, 'standing_balance_free', False)))
        for name in ('foot_contact', 'foot_clearance'):
            terms[name] *= 1-risk
        direction = nominal[:, :2]/nominal[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        orthogonal = x.root_vel[:, :2]-(x.root_vel[:, :2]*direction).sum(-1, keepdim=True)*direction
        lateral = 1.2*orthogonal.square().sum(-1)*(nominal[:, :2].norm(dim=-1)>1e-6)
        terms['body_motion'] = (terms['body_motion']-risk*lateral).clamp_min(0)
        # Actual supporting feet, not the clock, select contact/slip/clearance.
        terms['foot_slip'] = (x.foot_vel.square().sum(-1)*(x.feet_contact != 0)).sum(-1)
        foot_cover, foot_radii, foot_groups = payload['foot_cover']
        actual_gait = torch.where(x.feet_contact != 0, 1, -1)
        terms['obstacle_foot_clearance'] = self.traversal_safety.foot_clearance_cost(
            foot_cover, foot_radii, foot_groups, actual_gait, self.scene_ids)
        raw_gait, scaled_gait = gait_cost(terms, self.scales, scale=self.gait_scale)
        terrain = self.terrain[self.scene_ids]
        has_bar, has_roof = terrain[:, 2]>0, terrain[:, 5]>0
        feet_lo = torch.stack([(foot_cover[:, foot_groups==i, 0]-foot_radii[foot_groups==i]).amin(1)
                              for i in (0, 1)], 1)
        feet_hi = torch.stack([(foot_cover[:, foot_groups==i, 0]+foot_radii[foot_groups==i]).amax(1)
                              for i in (0, 1)], 1)
        self.bar_entered |= active & has_bar & (feet_lo < terrain[:, :1]).any(1)
        self.bar_passed |= active & self.bar_entered & (feet_lo > terrain[:, 1:2]).all(1)
        inside_roof = has_roof & (root[:, 0] >= terrain[:, 3]) & (root[:, 0] <= terrain[:, 4])
        self.roof_entered |= active & inside_roof
        self.roof_passed |= active & self.roof_entered & (cover[:, :, 0]-radii).amin(1).gt(terrain[:, 4])
        swing_over = (((feet_hi >= terrain[:, :1]) & (feet_lo <= terrain[:, 1:2])) & (x.feet_contact==0)).any(1)
        trailing = (feet_lo > terrain[:, 1:2]).any(1) & (feet_lo <= terrain[:, 1:2]).any(1)
        self.joint_phase[:, 0] = has_bar & swing_over
        self.joint_phase[:, 1] = has_bar & trailing
        self.joint_phase[:, 2] = inside_roof & ((terrain[:, 4]-root[:, 0]) >= .35)
        self.joint_phase[:, 3] = has_roof & has_bar & (root[:, 0]>terrain[:, 4]) & (feet_hi<terrain[:, :1]).all(1)
        length = (self.goals[:, 0]-self.start_x).clamp_min(.01)
        progress = (root[:, 0]-self.start_x).clamp_min(0).minimum(length)
        for blocked, boundary in ((has_bar & ~self.bar_passed, terrain[:, 1]),
                                  (has_roof & ~self.roof_passed, terrain[:, 4])):
            progress = torch.where(blocked, progress.minimum((boundary-self.start_x).clamp_min(0)), progress)
        delta = self.memory.update(progress, risk, active)
        margins = torch.stack((self.width[self.scene_ids]+root[:, 1], self.width[self.scene_ids]-root[:, 1]), -1)
        outside = (margins < 0).any(1)
        invalid = self._outcomes['invalid'] | ~torch.isfinite(raw_cover).all((1,2))
        invalid |= ~torch.isfinite(raw_gait) | ~torch.isfinite(clearance) & moving['episode_dynamic']
        failed = self._outcomes['collision'] | self._outcomes['fall'] | invalid | outside
        skills_done = (~has_bar | self.bar_passed) & (~has_roof | self.roof_passed)
        upright_ok = (x.torso_rpy[:, 1].abs()<.20) & (x.torso_rpy[:, 0].abs()<.15)
        upright_ok &= (allowance>0) | (x.head_pos[:, 2]>=1.25)
        holding = (self.goal_distance<.1) & (x.root_vel.norm(dim=-1)<.2) & upright_ok & skills_done & ~failed
        self.hold = torch.where(active, torch.where(holding, self.hold+.02, 0.), self.hold)
        success = self.hold >= .5-1e-6
        reward, diagnostics = joint_reward(delta_progress=delta,
            gait_cost=torch.nan_to_num(scaled_gait, nan=2.), clearance=torch.nan_to_num(clearance, nan=-1., posinf=10.),
            margins=margins, velocity_xy=torch.nan_to_num(x.root_vel[:, :2]), nominal_xy=nominal[:, :2],
            risk=risk, stalled=self.memory.stalled & (self.goal_distance>=.1),
            failed=failed, success=success, timeout=self._outcomes['timeout'])
        self._outcomes.update(invalid=invalid, outside_corridor=outside, complete=success & ~failed)
        self._reward = torch.where(active, reward, torch.zeros_like(reward))
        rotation = x.navigation_rotation.transpose(-1,-2)
        rotate = lambda v: (rotation @ v[..., None]).squeeze(-1)
        goal3 = self.goals-root
        token = torch.cat(((rotate(position-root)/2).clamp(-2,2),
                           rotate(velocity-x.root_vel).clamp(-2,2), moving['radius'][:,None]/.2,
                           valid[:,None], (moving['age'][:,None]/.3).clamp(0,1)), -1)
        token = torch.where(valid[:,None], token, 0.)
        local_terrain = terrain.clone()
        local_terrain[:, [0,1,3,4]] = ((terrain[:, [0,1,3,4]]-root[:, :1])/2).clamp(-2,2)
        local_terrain[:, [2,5]] /= 1.5
        local_terrain[:, :3] *= has_bar[:, None]
        local_terrain[:, 3:] *= has_roof[:, None]
        extra = torch.cat((rotate(x.root_vel).clamp(-2,2), (x.feet_contact!=0).float(),
                           (rotate(goal3)[:, :2]/2).clamp(-2,2), (margins/2).clamp(-2,2),
                           risk[:,None], token, local_terrain, self.memory.features(length)), -1)
        source_actor = self._obs['actor']
        self._obs['anchor'] = torch.cat((source_actor, (~moving['episode_dynamic'])[:,None]), -1)
        self._obs['actor'] = torch.nan_to_num(torch.cat((source_actor, extra), -1))
        self._obs['critic'] = torch.nan_to_num(torch.cat((self._obs['critic'], extra), -1))
        if same and previous is not None:
            for key in ('actor','critic','anchor'):
                self._obs[key][~initial] = previous[key][~initial]
        if not same and active.any():
            self.saved_outcomes.update({k:v.clone() for k,v in self._outcomes.items()})
            self.saved_outcomes['success'] = (success & ~failed).clone()
            self.saved_reward_diagnostics.update({k:v.clone() for k,v in diagnostics.items()})
            self.saved_reward_diagnostics.update(joint_gait_raw=raw_gait.clone(), joint_risk=risk.clone(),
                joint_clearance=clearance.clone(), joint_ducking=allowance.clone())
            self.saved_terms = {k:v.clone() for k,v in terms.items()}
            self.last_metrics.update(joint_phase=self.joint_phase.any(1).clone(),
                                     joint_dynamic=moving['episode_dynamic'].clone())
        if not same:
            self.previous_cover.copy_(cover)
        else:
            self.previous_cover[initial] = cover[initial]
