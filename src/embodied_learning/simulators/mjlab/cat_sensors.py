"""mjlab-only acquisition; outputs the same world rays for depth and LiDAR.

The simulated sensor sees/occludes robot geometry before self returns are
discarded. Ground segmentation replaces the author's real-sensor ground filter;
only endpoints on the support plane are removed, not rays behind obstacles.
"""

import math
from dataclasses import dataclass

import torch

from embodied_learning.perception.rays import (
    RayMeasurements,
    rays_from_depth,
    rays_from_lidar,
)


@dataclass(frozen=True)
class LidarPatternCfg:
    azimuth_count: int = 128
    elevation_count: int = 32

    def generate_rays(self, mj_model, device):
        """360 degree instantaneous scan, elevation -45..45 degrees."""
        azimuth = torch.arange(self.azimuth_count, device=device) * (
            2 * math.pi / self.azimuth_count
        )
        elevation = torch.linspace(
            -math.pi / 4, math.pi / 4, self.elevation_count, device=device
        )
        el, az = torch.meshgrid(elevation, azimuth, indexing="ij")
        rays = torch.stack((el.cos() * az.cos(), el.cos() * az.sin(), el.sin()), -1)
        rays = rays.reshape(-1, 3)
        return torch.zeros_like(rays), rays


@dataclass(frozen=True)
class OnlineSensorConfig:
    source: str = "lidar"
    max_range: float = 5.0
    width: int = 64
    height: int = 48
    azimuth_count: int = 128
    elevation_count: int = 32

    def __post_init__(self):
        if self.source not in ("depth", "lidar"):
            raise ValueError("source must be depth or lidar.")
        if not math.isfinite(self.max_range) or self.max_range <= 0:
            raise ValueError("max_range must be finite and positive.")
        if any(
            type(n) is not int or n < 2
            for n in (
                self.width,
                self.height,
                self.azimuth_count,
                self.elevation_count,
            )
        ):
            raise ValueError("Sensor resolutions must be integers >= 2.")


def attach_online_sensor(cfg, sensor: OnlineSensorConfig):
    from mjlab.sensor import CameraSensorCfg, RayCastSensorCfg
    from mjlab.sensor.builtin_sensor import ObjRef

    if sensor.source == "depth":
        # MuJoCo camera -Z points forward (+X torso), camera +Y points up (+Z).
        camera = CameraSensorCfg(
            name="cat_depth",
            parent_body="robot/torso_link",
            pos=(0.08, 0.0, 0.4),
            quat=(0.5, 0.5, -0.5, -0.5),
            fovy=90.0,
            width=sensor.width,
            height=sensor.height,
            data_types=("depth", "segmentation"),
            use_textures=False,
            use_shadows=False,
        )
        cfg.scene.sensors = (*cfg.scene.sensors, camera)
    else:
        lidar = RayCastSensorCfg(
            name="cat_lidar",
            frame=ObjRef(type="site", name="head", entity="robot"),
            pattern=LidarPatternCfg(sensor.azimuth_count, sensor.elevation_count),
            ray_alignment="base",
            max_distance=sensor.max_range,
            exclude_parent_body=False,
        )
        cfg.scene.sensors = (*cfg.scene.sensors, lidar)


def filter_mapping_rays(rays, self_hits, ground_hits):
    """Self is no information; support ground clears to its measured surface."""
    valid = rays.valid_mask & ~self_hits
    return RayMeasurements(
        rays.origins, rays.endpoints, valid, rays.hit_mask & valid & ~ground_hits
    )


class CatSensorReader:
    """Snapshot an initialized mjlab 1.6.0 sensor on the simulation thread."""

    def __init__(self, env, config: OnlineSensorConfig):
        import mujoco

        self.env, self.config = env, config
        model = env.sim.mj_model
        body_names = [model.body(i).name for i in range(model.nbody)]
        self.robot_geom = torch.tensor(
            [body_names[int(b)].startswith("robot/") for b in model.geom_bodyid],
            device=env.device,
        )
        self.ground_geom = torch.tensor(
            [
                body_names[int(b)] == "terrain"
                and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
                for i, b in enumerate(model.geom_bodyid)
            ],
            device=env.device,
        )
        if config.source == "depth":
            from embodied_learning.simulators.mjlab.sensors import DepthCameraReader

            self.depth = DepthCameraReader(env, "cat_depth")

    def read(self):
        import mujoco

        cfg = self.config
        if cfg.source == "depth":
            frame = self.depth.read_frame()
            seg = self.env.scene["cat_depth"].data.segmentation.clone()
            rays = rays_from_depth(
                frame.depth,
                frame.intrinsics,
                rotation=frame.rotation,
                position=frame.position,
                max_depth=cfg.max_range,
                no_hit_mask=(seg[..., 0] == -1) & (seg[..., 1] == -1),
            )
            ids = seg[..., 0].flatten(1).long()
            is_geom = seg[..., 1].flatten(1) == int(mujoco.mjtObj.mjOBJ_GEOM)
        else:
            import warp as wp

            sensor = self.env.scene["cat_lidar"]
            data = sensor.data  # Updates the public buffers and cached beam pose.
            ranges = data.distances.clone().reshape(self.env.num_envs, -1)
            # mjlab 1.6.0 lacks public per-beam origins/directions and geom IDs.
            # Keep this pinned-version dependency inside the simulator adapter.
            origins = sensor._cached_world_origins.reshape(*ranges.shape, 3).clone()
            directions = sensor._cached_world_rays.reshape(*ranges.shape, 3).clone()
            rays = rays_from_lidar(
                origins,
                directions,
                ranges,
                max_range=cfg.max_range,
                no_hit_mask=ranges == -1,
            )
            ids = wp.to_torch(sensor._ray_geomid).clone().long()
            is_geom = ids >= 0
        valid_id = is_geom & (ids >= 0) & (ids < len(self.robot_geom))
        safe = ids.clamp(0, len(self.robot_geom) - 1)
        return filter_mapping_rays(
            rays, valid_id & self.robot_geom[safe], valid_id & self.ground_geom[safe]
        )
