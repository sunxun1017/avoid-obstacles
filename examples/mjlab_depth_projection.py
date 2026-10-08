"""Inspect the depth-to-dual-map pipeline in one native mjlab environment.

Run with the mjlab environment's Python. The camera is fixed above the world
origin, so this is a geometry probe rather than a deployable robot sensor.
Grid geometry is an explicit engineering interpretation, not paper defaults.
"""

import argparse

import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg

from embodied_learning.perception.depth import backproject_depth
from embodied_learning.perception.DualProjector import DualProjector
from embodied_learning.perception.geometry import world_to_yaw_local
from embodied_learning.simulators.mjlab.sensors import (
    DepthCameraReader,
    add_depth_camera,
    read_robot_pose,
)


def main() -> None:
    """Create a camera, reconstruct points, and report observed grid cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=3.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This rendering probe requires a CUDA device.")
    projector = DualProjector(
        1,
        height_range=(-1.5, 0.8),
        elevation_height_range=(-1.5, -0.2),
    )
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    cfg.scene.num_envs = 1
    add_depth_camera(
        cfg,
        CameraSensorCfg(
            name="depth_camera",
            pos=(0.0, 0.0, 3.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            fovy=60.0,
            width=32,
            height=24,
            data_types=("depth",),
        ),
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        env.reset()
        frame = DepthCameraReader(env).read_frame()
        world = backproject_depth(
            frame.depth,
            frame.intrinsics,
            rotation=frame.rotation,
            position=frame.position,
            max_depth=10.0,
        )
        position, quaternion = read_robot_pose(env)
        local = world_to_yaw_local(world, position, quaternion)
        result = projector.project(local, args.radius)
        assert result.elevation.shape == (1, 21, 17)
        assert result.distance.shape == (1, 13, 17)
        assert result.elevation_valid.any(), "No ground observed in this probe."
        flat = result.flattened()
        print(
            "Elevation:", result.elevation.shape, "MLP input:", flat.elevation.shape
        )
        print("Distance:", result.distance.shape, "MLP input:", flat.distance.shape)
        print("Observed elevation cells:", result.elevation_valid.sum().item())
        print("Observed distance cells:", result.distance_valid.sum().item())
    finally:
        env.close()


if __name__ == "__main__":
    main()
