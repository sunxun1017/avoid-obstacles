# CAT body envelope and braking implementation

The user authorized implementation after the 3,448-episode diagnosis, followed
by a fixed 1,000-episode evaluation. This plan records the selected implementation
and experimental boundaries; it does not claim the behavioral failures fixed.

## Requirements and constraints

- Preserve `cat_source_v1` defaults, observation fixtures, model identities and
  strict checkpoint loading/resume. Add an opt-in `cat_recovery_v1` profile.
- Keep the native G1 geometry, 12 actions, one tanh transform, incremental joint
  targets and explicit PD unchanged. Existing checkpoints remain immutable.
- This experiment uses the existing complete offline scene information. Exact
  AABB distance is an explicitly labeled offline provider, not online perception.
  Reject the research profile in online mode until measured coverage is defined.
- Keep the original success/contact/fall definitions for comparison. Conservative
  envelope overlap is a warning/learning signal, not a measured collision.
- Server code transfer uses Git only. Use isolated worktrees and the server's
  `.venv-field` Python layered on mjlab. Do not alter installed dependencies or
  interrupt other GPU jobs.
- Evaluate all designated first episodes, including failures, over the same
  20-second horizon and seed 20261019. Training uses 4096 environments when memory
  permits; 1000-episode evaluation uses 1000 environments, one episode each.

## Selected design

### Geometry

Cover all 33 native collision geoms (31 capsules, 2 spheres). Capsules use axial
centers with at most 4 cm spacing, enlarged by half the segment spacing. This
covers the volume, including between samples. Transform using native geom world
poses, subtract environment origins, and query exact scene-box signed distances.
Subtract cover radii and aggregate minima and associated gradients into the
existing eleven body groups. Replace research SDF/BF observation and reward
inputs with these group clearances; preserve source guidance generation. Invalid
geometry is not free space. Keep original eleven-point fields in source mode.

### Navigation and learning

Use a separate navigator with DRIVE, BRAKE, SETTLE and HOLD states. Its stopping
envelope depends on measured speed, delay and configured deceleration. Approach
velocity decreases continuously near the goal; a fixed one-second stop latch is
not used. Actual low velocity, angular velocity and foot support gate settling
and recovery, with distance hysteresis and dwell time. Recovery ramps commands
and restarts opposing foot phases; it cannot rearm while the robot is moving
rapidly away from the goal.

Actor and critic receive the same navigation velocity command. Append actual
local root velocity, local goal XY and four mode indicators to the actor
(162+9=171). Append local goal XY and modes to the critic (250+6=256). Observed
linear velocity requires an estimator for eventual real deployment.

Research rewards replace fixed world-X/stopped GF bonuses with command-consistent
progress, add non-saturating velocity errors and near-goal physical settling and
position costs, and use body-envelope clearance penalties. Research aggregation
allows negative values; termination costs discourage ending a bad episode early.
Original reward formulas and fixtures remain unchanged in source mode.

### Checkpoints

Add explicit PPO `--warmstart`, mutually exclusive with strict `--resume`.
Only an audited source-to-recovery migration is accepted: identical physical
model, scene/cache identities, joint/action order, architecture and unnormalized
networks. Copy old input columns and zero new columns; copy other actor/critic
weights strictly. Start fresh optimizer, iteration and RNG state; record parent
checkpoint SHA256 and contract in every output checkpoint.

## Implementation tasks

1. Geometry module and tests: `simulators/mjlab/cat_geometry.py` and
   `tests/test_cat_geometry.py`. Verify rotations, conservative coverage and
   existing native toe/torso counterexamples; all starting poses remain clear.
2. Navigation/reward module and tests: `tasks/cat_recovery.py` and
   `tests/test_cat_recovery.py`. Test the measured 2.2 m/s approach case, continuous
   approach to the goal, safe recovery, inactive/reset slots and reward shortcuts.
3. CLI/warmstart and tests: `cli/field_training.py`, `algorithms/ppo.py`, optional
   migration helper, and `tests/test_cat_recovery_loading.py`. Preserve strict
   resume and reject incompatible migrations.
4. Native/task integration and research config: `tasks/cat_traversal.py`,
   `simulators/mjlab/cat_traversal.py`, `configs/experiments/g1_cat_recovery.yaml`.
   Test profile isolation, actor/critic dimensions and consistent commands,
   pre-reset metrics, repeated-refresh idempotence and envelope invalidity.
5. Independent review, Git synchronization, GPU smoke, bounded warmstart training
   and exact-cohort evaluation. Record raw failures and unchanged baseline along
   with endpoint goal holding, physical contacts and source success. Do not
   promote a checkpoint based only on training reward or unit tests.

## Execution ledger

- Workspace: `/tmp/embodied-learning-cat-recovery`, branch
  `research/cat-geometry-braking`, base `6edff65`.
- Baseline targeted source tests: 18 passed, 1 native GPU test skipped locally.
- Tasks 1–3 delegated with disjoint file ownership; root owns task/native wiring.
- Geometry, navigation, migration and integration each require failing behavioral
  tests before implementation, then relevant regression tests and review.
- Local full-suite CUDA collection limitations are known; native checks run on
  zjut. No dependency installation is part of this task.
