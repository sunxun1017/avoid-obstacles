"""Continuous native sphere motion; sensing samples a timestamped history."""

import mujoco
import torch

from embodied_learning.simulators.mjlab.cat_geometry import CollisionEnvelope

RADII = (.08, .10, .12)


def sphere_spec(radius):
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='ball')
    body.add_geom(name='collision', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=(radius, 0., 0.), rgba=(.9, .2, .15, 1.),
                  friction=(1., .005, .001))
    return spec


class DynamicSphereRuntime:
    def __init__(self, native, config):
        self.native, self.config = native, config
        n, dev = native.num_envs, native.device
        self.envelope = CollisionEnvelope.from_model(native.sim.mj_model, dev)
        self.position = torch.zeros(n, 3, device=dev)
        self.position[:, 2] = -100
        self.velocity = torch.zeros_like(self.position)
        self.target_velocity = torch.zeros_like(self.position)
        self.radius_id = torch.zeros(n, dtype=torch.long, device=dev)
        self.radius = torch.full((n,), .08, device=dev)
        self.episode_dynamic = torch.zeros(n, dtype=torch.bool, device=dev)
        self.moving = self.episode_dynamic.clone()
        self.trigger = torch.zeros(n, dtype=torch.long, device=dev)
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.position_history = self.position[None].repeat(6, 1, 1)
        self.velocity_history = self.velocity[None].repeat(6, 1, 1)
        self.timestamp_history = torch.zeros(6, n, device=dev)
        self.history_index = 0
        self.last_tick = -1
        self.substeps = 0
        self.active = False
        self.original_step = native.sim.step
        native.sim.step = self.step
        self.rows = torch.arange(n, device=dev)
        # Nominal upright head shape is computed once, independent of later
        # pelvis/head heights. The reference height is the approved 1.35 m.
        model = native.sim.mj_model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        pos = torch.tensor(data.geom_xpos, device=dev, dtype=torch.float32)[None]
        rot = torch.tensor(data.geom_xmat, device=dev, dtype=torch.float32).reshape(1,-1,3,3)
        centers = self.envelope.world_centers(pos, rot, pos.new_zeros(1,3))[0]
        select = self.envelope.groups == 0
        self.head_radii = self.envelope.radii[select]
        self.head_offsets = centers[select].clone()
        pelvis = torch.tensor(data.xpos[model.body('robot/pelvis').id], device=dev, dtype=torch.float32)
        self.head_offsets[:, :2] -= pelvis[:2]
        self.head_offsets[:, 2] += 1.35-self.head_offsets[:, 2].mean()

    def reset(self, ids):
        self.position[ids] = self.position.new_tensor([0., 0., -100.])
        self.velocity[ids] = 0
        self.target_velocity[ids] = 0
        self.episode_dynamic[ids] = False
        self.moving[ids] = False
        self.radius_id[ids] = 0
        self.radius[ids] = .08
        self.trigger[ids] = 0
        scenes = self.native.cat_state.manifest.scenes
        catalog = self.config.get('scenarios', {})
        for scene_id, scene in enumerate(scenes):
            slots = ids[self.native.cat_state.scene_ids[ids] == scene_id]
            entries = catalog.get(scene.id, [])
            if not len(slots) or not entries:
                continue
            # The controller consumes a prevalidated finite scenario catalog;
            # it does not improvise geometry or future paths during training.
            picks = torch.randint(len(entries), (len(slots),), device=self.position.device)
            table = self.position.new_tensor([e['position']+e['velocity']+[e['radius'], e['trigger']] for e in entries])
            selected = table[picks]
            self.position[slots] = selected[:, :3]
            self.target_velocity[slots] = selected[:, 3:6]
            self.radius[slots] = selected[:, 6]
            radius_id = (selected[:, 6:7]-self.position.new_tensor(RADII)[None]).abs().argmin(-1)
            if not torch.allclose(selected[:, 6], self.position.new_tensor(RADII)[radius_id], atol=1e-6, rtol=0):
                raise ValueError('Sphere catalog requires native radius .08, .10 or .12.')
            self.radius_id[slots] = radius_id
            self.trigger[slots] = selected[:, 7].long()
            self.episode_dynamic[slots] = True
            self.moving[slots] = self.trigger[slots] == 0
        self.delay[ids] = (torch.randint(6, (len(ids),), device=self.position.device)
                           if self.config['stage'] == 4 else 0)
        self.position_history[:, ids] = self.position[ids]
        self.velocity_history[:, ids] = 0
        self.timestamp_history[:, ids] = self.native.common_step_counter
        self.write_poses()

    def write_poses(self):
        for i in range(3):
            pose = self.position.new_zeros(len(self.position), 7)
            pose[:, 3] = 1
            selected = self.episode_dynamic & (self.radius_id == i)
            pose[:, :3] = torch.where(selected[:, None], self.position,
                                     self.position.new_tensor([0., 0., -100.]))
            pose[:, :3] += self.native.scene.env_origins
            self.native.scene[f'obstacle_dynamic_{i}'].write_mocap_pose_to_sim(pose, env_ids=self.rows)

    def begin(self):
        phases = self.native.cat_state.joint_phase
        for i in range(4):
            self.moving |= self.episode_dynamic & (self.trigger == i+1) & phases[:, i]
        self.active = True

    def step(self, *args, **kwargs):
        if self.active:
            dt = self.native.physics_dt
            target = self.target_velocity*self.moving[:, None]
            change = target-self.velocity
            change *= (2.*dt/change.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp_max(1)
            new_velocity = self.velocity+change
            self.position += .5*(self.velocity+new_velocity)*dt
            self.velocity.copy_(new_velocity)
            self.write_poses()
            self.substeps += 1
        return self.original_step(*args, **kwargs)

    def payload(self, root, navigation_rotation):
        tick = self.native.common_step_counter
        if tick != self.last_tick:
            self.history_index = (self.history_index+1) % 6
            self.position_history[self.history_index] = self.position
            self.velocity_history[self.history_index] = self.velocity
            self.timestamp_history[self.history_index] = tick
            self.last_tick = tick
        indices = (self.history_index-self.delay) % 6
        position = self.position_history[indices, self.rows].clone()
        velocity = self.velocity_history[indices, self.rows].clone()
        if self.config['stage'] == 4:
            # Refresh is cached by task state, so each control boundary consumes
            # one observation noise sample, independent of reward/obs reads.
            position += (torch.rand_like(position)*2-1)*.01
            velocity += (torch.rand_like(velocity)*2-1)*.05
        env = self.native
        geom_pos = env.sim.data.geom_xpos
        geom_rot = env.sim.data.geom_xmat.reshape(*geom_pos.shape[:2], 3, 3)
        cover = self.envelope.world_centers(geom_pos, geom_rot, env.scene.env_origins.to(geom_pos))
        upright = (navigation_rotation[:, None] @ self.head_offsets[None, :, :, None]).squeeze(-1)
        upright[:, :, :2] += root[:, None, :2]
        packet = dict(position=self.position, velocity=self.velocity,
                      observed_position=position, observed_velocity=velocity,
                      radius=self.radius, valid=self.episode_dynamic,
                      episode_dynamic=self.episode_dynamic,
                      age=(tick-self.timestamp_history[indices, self.rows])*.02)
        geometry = dict(centers=cover, radii=self.envelope.radii,
                        upright_head=upright, head_radii=self.head_radii)
        return packet, geometry

    def close(self):
        self.native.sim.step = self.original_step
        self.active = False
