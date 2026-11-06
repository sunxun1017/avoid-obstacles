# Depth camera input

This integration follows mjlab 1.6.0's official camera lifecycle. Configure
the camera before constructing the environment, then read after reset/step.

```python
from mjlab.sensor import CameraSensorCfg
from embodied_learning.simulators.mjlab.sensors import (
    DepthCameraReader,
    add_depth_camera,
)
from embodied_learning.perception.depth import backproject_depth

# cfg is the native task configuration. This example creates a fixed camera
# looking down from three meters above the world origin.
add_depth_camera(cfg, CameraSensorCfg(
    name="depth_camera",
    pos=(0.0, 0.0, 3.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    fovy=60.0,
    width=160,
    height=120,
    data_types=("depth",),
))

# Construct env from cfg using ManagerBasedRlEnv, then call env.reset().
reader = DepthCameraReader(env)
frame = reader.read_frame()
cloud = backproject_depth(
    frame.depth,
    frame.intrinsics,
    rotation=frame.rotation,
    position=frame.position,
    max_depth=5.0,
)
```

`frame.depth` is `[B, H, W]` metric optical-axis depth. Intrinsics are
`[B, 3, 3]`. Position `[B, 3]` and rotation `[B, 3, 3]` describe the optical
camera frame in world coordinates. The optical frame uses X right, Y down,
Z forward; MuJoCo camera Y and Z have the opposite signs.

`cloud.points` is `[B, H * W, 3]`, and `cloud.valid_mask` is `[B, H * W]`.
Zero, negative, non-finite, or out-of-range depth is invalid. Invalid points
are filled with zero: always apply the mask before projection. The output
remains a Torch tensor on the depth device. To use a NumPy-only projector,
explicitly transfer it with `cloud.points.detach().cpu().numpy()` and also
transfer the mask. GPU training should eventually use a tensor projector.

Omit rotation and position to obtain optical-frame points instead. World
points still need conversion to the robot's chosen local frame before an
ADAPT projection. Use world_to_yaw_local and DualProjector as described below.

Supported cameras are perspective pinhole cameras defined by FOV or
intrinsic parameters. Orthographic projection is rejected. Intrinsic
conversion matches MuJoCo Warp's sensor-plane aspect adjustment and its
pixel-center convention `(u + 0.5, v + 0.5)`. Call read_frame from the
simulation thread, without stepping concurrently. Snapshots are cloned.

The camera reader does not create or advance an environment. The application
owns environment creation, reset/step, and closing it in a finally block.

Sources:

- [Official mjlab camera guide](https://mujocolab.github.io/mjlab/v1.6.0/source/sensors/rgbd_camera.html)
- Installed `mujoco_warp/_src/render_util.py`: `compute_ray_dir` determines
  pixel rays and the intrinsic projection convention.
- Installed `mujoco_warp/_src/render.py`: the rendering kernel writes
  camera-plane depth and uses zero for ray misses.

Run the CPU geometry and reader tests with:

```bash
python -m pytest tests/test_depth.py tests/test_mjlab_sensors.py -q
```

These unit tests do not require an actual GPU renderer.

## Robot-local coordinates and projections

```python
from embodied_learning.perception.geometry import world_to_yaw_local
from embodied_learning.perception.DualProjector import DualProjector
from embodied_learning.simulators.mjlab.sensors import read_robot_pose

position, quaternion = read_robot_pose(env)
local = world_to_yaw_local(cloud, position, quaternion)
projector = DualProjector(
    env.num_envs,
    height_range=(-1.5, 0.8),
    elevation_height_range=(-1.5, -0.2),
)
result = projector.project(local, radius=3.0)
flat = result.flattened()
```

These height limits are example engineering choices in meters relative to
the robot root, not verified ADAPT parameters. Adjust them to robot height
and terrain. The yaw-local frame has X forward, Y left, Z up. It follows
heading but does not tilt with roll/pitch. It assumes world Z opposes gravity.
Each environment's root position is subtracted independently.

Elevation rows are X bins and columns Y bins. By default, X spans
`[-0.2 * radius, radius)` and Y spans `[-0.5 * radius, 0.5 * radius)`;
horizontal radius is also clipped. Each cell keeps maximum root-relative Z
within elevation_height_range. The filter avoids overhead contamination
but does not perform ground segmentation. Changing root height shifts Z.

Distance rows are height bins, columns azimuth bins in `[-45, +45)` degrees.
Each cell keeps the nearest horizontal XY distance within radius. Height
range stays fixed as radius changes. This follows the vertical cylindrical
surface in ADAPT Fig. 2. The paper text instead calls the 13 bins radial;
this conflict is unresolved. This implementation is an approved engineering
interpretation, not a claim of exact paper reproduction. Grid sizes alone
do not verify projection semantics.

Output shapes are `[B,21,17]` and `[B,13,17]`, with matching boolean masks.
All bin intervals are lower-inclusive and upper-exclusive. Invalid or
out-of-range points contribute nothing. Unknown elevation is zero and
unknown distance is radius; consumers must use masks to distinguish missing
observations from measured values. No cell is certified free merely because
it lacks a point. These masks add information beyond the paper's 578 scalars;
decide explicitly how the eventual policy consumes them.

The projector now accepts Torch PointCloud objects, replacing the earlier
NumPy preallocation stub. Calling flattened() before project() raises an
error. Each call returns a fresh frame, with no temporal accumulation.

Run a complete GPU geometry probe from the project root:

```bash
PYTHONPATH=src python examples/mjlab_depth_projection.py --radius 3
```

The fixed camera probes a small region near the world origin. For moving
robots, attach the camera to the robot using a correct body and mounting
pose. No policy or training loop is modified by this example.
The PYTHONPATH setting makes the src-layout package importable without an
editable install. If already installed with `python -m pip install -e .`,
the setting can be omitted.

Source for grid sizes and the unresolved projection description:
[ADAPT, Section III-B and Figure 2](https://arxiv.org/abs/2603.16328v1).

## G1Cat 公开源码任务

独立的 `cat_source_v1` 配置使用源码奖励、指令、步态、162/250 观测和 12 维腿部增量动作。旧论文势场与 source 场缓存有不同契约。训练与播放见 [G1Cat 公开源码训练](../../../../docs/cat_source_training.md)。
