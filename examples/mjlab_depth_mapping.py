"""Read a native mjlab depth frame, map rays, and query a voxel field.

The camera is fixed above the origin. The goal and query points are in
observed air above the robot; this checks geometry, not a navigation policy.
"""

import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg

from embodied_learning.perception.occupancy import OccupancyMapper
from embodied_learning.perception.rays import rays_from_depth
from embodied_learning.perception.voxel_field import build_voxel_field
from embodied_learning.simulators.mjlab.sensors import (
    DepthCameraReader,
    add_depth_camera,
)


def main() -> None:
    """Compose sensors, mapping and field construction at the application edge."""
    if not torch.cuda.is_available():
        raise RuntimeError("The mjlab camera example requires CUDA.")
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
    mapper = OccupancyMapper(
        1,
        (25, 25, 23),
        voxel_size=0.15,
        origin=(-1.8, -1.8, -0.15),
        device="cuda:0",
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        env.reset()
        frame = DepthCameraReader(env).read_frame()
        rays = rays_from_depth(
            frame.depth,
            frame.intrinsics,
            rotation=frame.rotation,
            position=frame.position,
            max_depth=3.5,
            # mjlab/MuJoCo Warp uses zero for confirmed ray misses.
            no_hit_mask=frame.depth == 0,
        )
        mapper.update(rays)
        state = mapper.snapshot()
        assert state.occupied.any() and state.known_mask.any()
        field = build_voxel_field(
            state.occupied[0].cpu().numpy(),
            known_mask=state.known_mask[0].cpu().numpy(),
            origin=tuple(state.origin[0].cpu().tolist()),
            voxel_size=state.voxel_size,
            goal=(0.0, 0.3, 2.1),
        ).to("cuda:0")
        positions = frame.depth.new_tensor([[[0.0, 0.0, 2.4]]])
        sample = field.sample(positions)
        assert sample.guidance_valid.all(), "Query neighborhood was not observed."
        print("Observed cells:", state.observed.sum().item())
        print("Occupied cells:", state.occupied.sum().item())
        print("Distance:", sample.sdf.cpu().tolist())
        print("Guidance:", sample.guidance.cpu().tolist())
        print("Validity:", sample.guidance_valid.cpu().tolist())
    finally:
        env.close()


if __name__ == "__main__":
    main()
