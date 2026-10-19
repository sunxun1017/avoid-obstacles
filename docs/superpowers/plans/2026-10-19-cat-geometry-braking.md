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
rapidly away from the goal. After drift, safe recovery waits 0.4 s beyond
0.25 m and 1 s for smaller drift outside the 0.1 m goal radius. This avoids a
permanent stationary band between the goal and the larger recovery threshold.
The velocity reference must ramp down before entering stationary stance.

Actor and critic receive the same navigation velocity command. Append actual
local root velocity, local goal XY and four mode indicators to the actor
(162+9=171). Append local goal XY and modes to the critic (250+6=256). Observed
linear velocity requires an estimator for eventual real deployment.

Research rewards replace fixed world-X/stopped GF bonuses with command-consistent
progress, add non-saturating velocity errors and near-goal physical settling and
position costs, and use body-envelope clearance penalties. Research aggregation
allows negative values; termination costs discourage ending a bad episode early.
The first measured obstacle contact costs 5 reward units per episode even when
the old point-distance termination misses it; a failure termination costs 10.
Neither conservative overlap nor native contact changes the original horizon.
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
- Implementation and CPU behavioral tests are in place. Independent review
  caught and corrected a stationary dead band, a reference discontinuity on
  entering stance, missing pelvis/torso clearance costs, and nonfinite reward
  overflow. Invalid research transitions now return a finite failure cost.
- Native envelope queries cache scene boxes and grouping indices for the GPU
  hot path; the standalone checked query remains a CPU/reference oracle.
- Evaluation adds ever/final goal holding and first-arrival/final speeds without
  changing the historical success or contact definitions.
- Server parent checkpoints were checked read-only: wall15800 and hurdle3000
  share field identity `ef3177329abbff5b9c7b8d111696af9b22cdd3ba5a7880dbaec60439f852379b`
  and native model identity `5d86b8d275276e3bce6178f69943f1a2a064b493e865096dd6ffc465721e24e3`.
- Formal training/evaluation has completed (results below); unit tests and the
  GPU smoke alone do not establish recovered robot behavior.
- CPU regression sweep: 305 passed, 31 CUDA skipped, 1 CUDA-status assertion
  deselected; the unconditional CUDA demo `test_scan.py` was excluded. The later
  native-first-contact cost passed its red/green regression and a 38-test
  task/navigation/source regression run (2 CUDA skipped).
- Server GPU smoke: 3 native tests passed on GPU 5 (RTX 4090). GPU 2 (RTX 5090)
  could not run the installed PyTorch kernels, so no dependency changes were
  attempted. A 4096-environment wall15800 warmstart completed 3 PPO updates,
  approximately 1.9 s/update, and saved its strict recovery checkpoint.
- Reject research parameters beyond float32 representability (review found
  `ground=1e50` could otherwise produce invalid reward arithmetic); 44 related
  task/navigation/loading tests passed after the fix.
- `tools/experiments/run_cat_recovery.py` records an immutable Git/config/parent
  manifest, runs wall then hurdle training/evaluation sequentially, and stops on
  stage failure. Requested formal budget: 1000 PPO iterations × 4096 environments
  per specialist, followed by 1000 fixed first episodes per specialist.
- The final implementation is committed as `eebaee0` and integrated into the
  local main working branch. The CPU regression sweep on that tree completed
  with 308 passed, 31 CUDA skips and 1 CUDA-status assertion deselected; the
  unconditional CUDA demo `test_scan.py` remains excluded locally.
- Code reached zjut through Git, in the isolated worktree
  `/home/sunxun/embodied-learning-recovery-eebaee0`. The bounded experiment runs
  from that frozen commit under
  `/home/sunxun/embodied-learning/runs/cat-recovery-20261019-eebaee0`.
- Static geometry evidence is recorded in
  `docs/benchmarks/cat-recovery-geometry-2026-10-19.json`, with the same-prefix
  clearance figure. Three counterexamples retain positive old point SDFs while
  native robot geometry penetrates; all are warned by the cover. All 33 native
  collision geoms are covered, and all 12 tested starting/goal poses are clear.
  This establishes the sampled static geometry issue, not policy performance.
- Same-code baseline reevaluations each completed 1000 designated first episodes
  on the formal run's RTX 4090. Wall15800: 844 success, 58 physical-contact
  episodes, 2 fall flags; first-arrival mean speed 2.1095 m/s (901 arrivals), no
  goal holding. Hurdle3000: zero arrivals/success, 1000 physical-contact episodes,
  998 fall flags, no goal holding. Independent raw-record audits passed for both.
  Minor differences from the earlier baseline are retained rather than hidden;
  the current reevaluations will be the primary comparisons.
- Goal-hold requires the current position inside the goal and the preceding
  0.5 s of low measured speed; it does not require that entire interval inside
  the goal or exclude historical contact. Report it together with whole-episode
  safety. Arrival speeds are conditional on reaching; terminal speeds include
  early failures. Same seed does not make the changed profiles a paired trial,
  and this combined intervention is not an isolated geometry/braking ablation.
- Wall completed 1000 PPO updates with 4096 environments and 1000 first-episode
  evaluations. All 1000 loss records and checkpoint tensors are finite; the
  parent checkpoint is unchanged. Success regressed from 844/1000 to 291/1000,
  physical-contact episodes rose from 58 to 131, and goal holding remained zero.
  Conditional arrival speed fell from 2.1095 m/s (901 arrivals) to 0.3588 m/s
  (324 arrivals); this is not an overall navigation improvement. Among full
  20-second timeouts, terminal speed increased from 0.0835 to 0.1927 m/s.
- A diagnostic-only commit `2afd700` adds real navigator telemetry and the
  `settle_without_support` inference probe. It temporarily substitutes only the
  navigator support input on selected low-speed BRAKE calls with zero reference;
  real contact telemetry and success definitions remain unchanged. The frozen
  formal training/evaluation continues on `eebaee0`. Related CPU checks: 42 passed.
- Two new 64-episode wall diagnostics probe the settling-entry bottleneck.
  Baseline terminal modes: 62 BRAKE and 2 DRIVE, no SETTLE. Probe terminal modes:
  64 SETTLE, no HOLD. Both have zero physically settled goal holds and seven
  obstacle-contact episodes. In eight sampled trajectories, zero-reference
  baseline braking never combines speed below 0.15 m/s with real dual support;
  the probe permits stance but measured angular motion remains excessive.
  Removing the entry gate alone is therefore insufficient. Prefix trajectories
  differ before observed SETTLE entries, so success 19/64 versus 23/64 and the
  early failure difference are not treated as strict paired causal estimates.
- At all 14 first-contact events across those two cohorts the new body envelope
  is negative while the old eleven-point SDF remains positive. This confirms
  detection at the sampled contact frames; earlier warning time is not measured.
  These 128 mechanism-probe episodes are separate from the formal evaluation.
- The next behavioral gate is a measured low-speed/standstill and gait-transition
  curriculum before another navigation comparison. A controller reaching SETTLE
  is not equivalent to the learned policy maintaining balance. The support-input
  probe must not become a production shortcut. The current experiments remain
  at expert PPO validation; online near/far perception and CMPC integration are
  not established by this offline geometry experiment.
- Both specialists completed exactly 1000 PPO updates with 4096 environments,
  followed by 1000 designated first episodes each. The frozen driver completed
  all four stages at 2026-10-19 18:42:44 Asia/Shanghai. Same-code source-parent
  reevaluations add two further 1000-episode cohorts (4000 formal evaluations in
  total); the 3-update smoke and small mechanism probes are excluded.
- Hurdle/crouch improved from 0/1000 to 129/1000 safe successes and from 1000 to
  232 physical-contact episodes; goal arrivals rose to 153. Fall flags decreased
  from 998 to 395, source-collision flags rose from 5 to 28, and invalid flags
  remained zero. These outcomes may overlap. Goal holding remains zero, so this
  is partial navigation progress and not an accepted expert policy.
- Final safe CPU checkpoint audits confirmed 1000 finite loss records per
  specialist, completed_iterations=1000 and environment_steps=20000, finite
  parameter tensors and nonzero stored parameter changes. Parent hashes remain
  unchanged; evaluated checkpoint/config hashes match. The frozen server
  worktree remains clean. All result claims retain the original success and
  physical stopping thresholds.
- Independent raw-episode audits passed for all four formal cohorts, checking
  unique first-episode slots, counts, distributions and input hashes; derived
  success-and-hold and timeout/failure strata are recorded separately. The final
  evidence report is `docs/benchmarks/cat-recovery-evaluation-2026-10-19.json`.
- An additional 64-episode hurdle diagnostic records 4 successes, 10 obstacle
  contact episodes, 18 fall flags, 4 source-collision flags and no goal holds.
  All 18 fall flags use the head-height branch with positive up_z, but substantial
  body inclination, angular motion and downward velocity support dynamic loss
  of balance rather than stable crouching alone. Twelve have no recorded obstacle
  contact; substep contact or other external forces cannot be excluded. These
  diagnostics do not replace the formal 129/1000 success estimate.
