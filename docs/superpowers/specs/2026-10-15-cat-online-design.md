# CAT online perception integration

The user approved implementation after reviewing the in-chat architecture and
confirmed CAT, not ADAPT. Preserve source training (precomputed fields); add a
decoupled sensor-driven deployment path with depth and LiDAR inputs. Work in the
existing feature checkout, preserving the user's dirty files. No automatic
commit, dependency upgrades, hardware control, or long training is needed.

## Source and boundaries

Pin Click-and-Traverse at `866ba392f1c1e84b92ad75fa66550f26e8af8e48`.
Reference `deploy/gx_loco_deploy/policies/g1_cat/pf.py` and
`deploy/scripts/exp_dis_pf/octomap_bridge.py`. The latter performs a 5-cube
closing with two iterations, isolated-voxel filtering, upward occupancy filling
above index 20 and downward filling below index 5. Its occupancy-only field
treats unobserved cells as non-obstacles. Preserve this explicit source mode;
report measured coverage separately. Do not call this confirmed free space.

Use existing Torch ray fusion as an alternative to OctoMap, with externally
supplied poses. It is not Fast-LIO2/SLAM. Use source CPU scikit-fmm for field
construction, Torch GPU-capable preprocessing and sampling. Do not silently
replace Eikonal distance with graph distance to claim a GPU implementation.

`perception/cat_deployment.py`: pure numerical preprocessing and source field
construction; center-based origins match existing map snapshots. Test against
functions extracted from the pinned source, with source SHA256 metadata.

`perception/online_field.py`: mapping and field lifecycle; independent per-env
state, explicit timestamps, update interval, atomic field replacement, reset,
goal change invalidation, stale-data rejection, and diagnostics. An external
occupancy snapshot can also be supplied, allowing another mapper later.

`simulators/mjlab/cat_sensors.py`: body-mounted depth or angular LiDAR rays.
Keep poses/world coordinates consistent, preserve occlusion, ignore robot hits
without clearing through them. Ground support is removed only from obstacle
classification; rays stop at the measured surface. No simulated sensors import
into the generic perception modules.

The task consumes a source-compatible field provider. Training still accepts
FieldBank. Online playback does not require or load field files, and must not
train on sensor-derived collision labels. Physical contacts are deployment
collision diagnostics/terminations. Missing/stale/out-of-grid fields produce
source stop/default field inputs and explicit availability diagnostics.

## Integration and compatibility

Add an explicit online-play CLI for project CAT checkpoints, not arbitrary
author ONNX files. Preserve 162/250 observations, 12 latent actions, tanh at the
existing environment boundary, joint order and PD semantics. Validate checkpoint
against the sensorless scene model and task contract, permitting only the
explicit switch from stored field identity to online perception at inference.
Training/resume loaders remain strict. Pure perception does not depend on tasks,
robot assets, ROS, Unitree SDK, RSL-RL, or mjlab.

Run sensors at a configurable cadence (default 10 Hz) with 50 Hz policy control.
The initial implementation is synchronous and fixed-grid; report measured timing
and do not claim guaranteed 10 Hz or large parallel online reconstruction.
Origins are stable world coordinates within each episode. Reset clears only
the selected environments. Out-of-grid goals fail explicitly; no fake target
clamping. Long-range exploration/rolling maps and real hardware transport are
outside this integration.

## Acceptance

- Pinned author function fixtures match preprocessing, sdf/bf/gf and sampling.
- Depth and LiDAR yield rays with correct hit/miss/invalid semantics.
- CPU tests cover occlusion, ground/robot filtering, scheduling, stale fields,
  goal changes, reset isolation, observation defaults and no offline cache use.
- CUDA tests cover mapping/preprocessing parity and both native sensor loops
  when available; skip with a clear reason otherwise.
- Existing training/observation/action tests retain their original semantics.
- CLI help, config validation, checkpoint compatibility and executable smoke
  commands are documented; no claims of trained navigation from random actions.
