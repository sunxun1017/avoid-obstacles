"""Configure mjlab depth cameras and read their runtime depth buffers.

Camera configuration must be attached before constructing the environment.
The reader consumes an initialized environment and never creates a camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.sensor import CameraSensorCfg
    from torch import Tensor


@dataclass(frozen=True)
class DepthFrame:
    """Metric depth [B,H,W], K [B,3,3], and optical-to-world pose.

    Position is [B,3] in meters; rotation is [B,3,3]. All tensors are
    independent snapshots. Optical axes are X right, Y down, Z forward.
    """

    depth: Tensor
    intrinsics: Tensor
    position: Tensor
    rotation: Tensor


def read_robot_pose(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> tuple[Tensor, Tensor]:
    """Snapshot root world position [B,3] and orientation [B,4] in wxyz order.

    Call after reset/step, without advancing simulation during frame reads.
    """
    data = env.scene[entity_name].data
    return data.root_link_pos_w.clone(), data.root_link_quat_w.clone()


def add_depth_camera(
    env_cfg: ManagerBasedRlEnvCfg,
    camera_cfg: CameraSensorCfg,
) -> None:
    """Append a depth camera before constructing the environment.

    Use the official CameraSensorCfg to wrap an XML camera or create one.
    Existing sensors are preserved. Camera names and body references are
    resolved by mjlab during scene construction.

    Raises:
        ValueError: If depth is disabled or the sensor name is already used.
    """
    if "depth" not in camera_cfg.data_types:
        raise ValueError("The camera must enable data_types containing 'depth'.")
    if any(sensor.name == camera_cfg.name for sensor in env_cfg.scene.sensors):
        raise ValueError(f"Sensor {camera_cfg.name!r} is already configured.")
    env_cfg.scene.sensors = (*env_cfg.scene.sensors, camera_cfg)


class DepthCameraReader:
    """Read an existing camera after the environment has been initialized.

    Call env.reset() or env.step() before reading a frame. This reader does
    not advance the simulation. Depth uses mjlab's camera-plane convention.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str = "depth_camera",
    ) -> None:
        """Bind to a scene sensor; an unknown name raises KeyError."""
        self._sensor = env.scene[sensor_name]
        self._env = env
        self.sensor_name = sensor_name

    def read(self) -> Tensor:
        """Return an independent depth frame shaped [B, H, W].

        The float32 tensor stays on its original device. Cloning prevents
        later rendering or caller mutation from changing the shared buffer.
        Invalid/background values are preserved for downstream masking.

        Raises:
            RuntimeError: If the sensor does not provide depth data.
            ValueError: If the depth buffer is not shaped [B, H, W, 1].
        """
        depth = getattr(self._sensor.data, "depth", None)
        if depth is None:
            raise RuntimeError(f"Sensor {self.sensor_name!r} has no depth output.")
        if depth.ndim != 4 or depth.shape[-1] != 1:
            raise ValueError(
                f"Expected depth shape [B, H, W, 1], got {depth.shape}."
            )
        return depth.squeeze(-1).clone()

    def read_frame(self) -> DepthFrame:
        """Read depth, current calibration, and pose for one simulation state.

        Supports perspective FOV and intrinsic cameras, including per-world
        randomized focal parameters. Orthographic cameras are rejected.
        Call from the simulation thread without stepping during this method.
        """
        import torch

        depth = self.read()
        batch, height, width = depth.shape
        sim = self._env.sim
        idx = self._sensor.camera_idx
        if int(sim.mj_model.cam_projection[idx]) != 0:
            raise ValueError("Orthographic cameras are not supported.")

        fovy = sim.model.cam_fovy[:, idx].to(depth).expand(batch)
        intrinsic = sim.model.cam_intrinsic[:, idx].to(depth).expand(batch, -1)
        sensor_w, sensor_h = sim.mj_model.cam_sensorsize[idx]
        k = torch.eye(3, device=depth.device, dtype=depth.dtype).repeat(batch, 1, 1)
        if sensor_h != 0:
            # Match MuJoCo Warp's aspect-ratio adjustment of the sensor plane.
            aspect = width / height
            if aspect > sensor_w / sensor_h:
                sensor_h = sensor_w / aspect
            else:
                sensor_w = sensor_h * aspect
            k[:, 0, 0] = intrinsic[:, 0] * width / sensor_w
            k[:, 1, 1] = intrinsic[:, 1] * height / sensor_h
            k[:, 0, 2] = width / 2 - intrinsic[:, 2] * width / sensor_w
            k[:, 1, 2] = height / 2 - intrinsic[:, 3] * height / sensor_h
        else:
            focal = height / (2 * torch.tan(torch.deg2rad(fovy) / 2))
            k[:, 0, 0] = focal
            k[:, 1, 1] = focal
            k[:, 0, 2] = width / 2
            k[:, 1, 2] = height / 2

        position = sim.data.cam_xpos[:, idx].to(depth).clone()
        rotation = sim.data.cam_xmat[:, idx].to(depth).reshape(batch, 3, 3).clone()
        # MuJoCo camera axes: X right, Y up, Z backward.
        rotation[:, :, 1:] *= -1
        return DepthFrame(depth, k, position, rotation)
