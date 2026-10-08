# G1 Field PPO and DAgger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended for this tightly coupled integration) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver portable Unitree G1 field-conditioned PPO specialists and an online DAgger generalist, with verified server training commands.

**Architecture:** Known declarative scene geometry produces physical obstacles and offline goal fields. A G1 mjlab task supplies body kinematics, paper field observations and rewards to RSL-RL PPO. Generic DAgger collects student-visited observations, routes frozen teachers by scene, aggregates labels and optimizes a student.

**Tech Stack:** Python >=3.10, Torch, NumPy/SciPy/scikit-fmm, mjlab 1.6.0, rsl-rl-lib 5.4.2, pytest/Ruff.

**Spec:** `docs/superpowers/specs/2026-10-08-g1-field-training-design.md`.

## Global Constraints

- No AMP, online mapping, sensor requirement or single-point navigation replacement.
- Default body schema is the published 11-point layout; document the paper's K=13 discrepancy, never invent missing points.
- Use paper equations 1--10 with explicitly documented singularity regularization; do not rename the upstream cosine-window reward as vMF.
- Core algorithms must not import tasks, robots, perception or simulators; compose at CLI boundaries.
- Scene-local metric geometry must match simulation geometry; support environment translations and goal/scene indices independently.
- Physics ground is support, not forbidden obstacle geometry.
- Strict checkpoint contracts include schema, joints/scales, observations, network, normalization and scene hashes.
- No user-file rollback: preserve modified `examples/depth_mapping_demo.py` and untracked `src/embodied_learning/README.md`.
- Short optimization verifies trainability, not convergence or reproduced paper success rates.
- Comments/docstrings English, public annotations, <=85-character lines, no eager simulator import in generic modules.

## Review Focus

- Reset slots must use new goal fields without contaminating reset-before rewards or other environments (Task 3).
- Unreachable/outside body samples cannot inject NaN, false free space or positive alignment reward (Tasks 1/3).
- Scene geometry and cache mismatches must fail before training rather than train on a different world (Task 2).
- Teachers must receive their original observation/normalization and label student-visited states in the correct scene (Tasks 4/5).
- Resumed DAgger must retain aggregated examples and scheduler/optimizer state, not silently restart behavior cloning (Task 5).

## File Ownership and Boundaries

| Files | Responsibility |
| --- | --- |
| `perception/field_observation.py`, `perception/field_reward.py` | Batched paper field evaluation, yaw conversion, stable vMF reward |
| `data/field_scenes.py`, `data/field_cache.py` | Declarative scene validation, geometry identity and offline fields |
| `robots/g1.py` | Ordered named body point and action schema |
| `tasks/field_traversal.py`, `simulators/mjlab/field_traversal.py` | Task composition, physical geometry, state/termination/reset adapter |
| `protocols/imitation.py`, `algorithms/ppo.py`, `policies/field_policy.py` | Trainable environment/teacher contracts, PPO backend and policy loading |
| `data/aggregation.py`, `algorithms/dagger.py` | Bounded persistent dataset and student-driven online DAgger |
| `cli/field_training.py`, `configs/experiments/g1_field.json` | Application composition and portable CLI/config |
| `docs/field_training.md`, `tests/test_field_*.py`, `tests/test_dagger.py` | Server guide, provenance and verification |

Paths in the table are relative to `src/embodied_learning/` except configs/docs/tests.

### Task 1: Paper field observation and reward

**Files:** Create `src/embodied_learning/perception/field_observation.py`, `field_reward.py` and `tests/test_field_math.py`.

**Interfaces:** `evaluate_body_field(sample: FieldSample, velocity: Tensor, base_priority: Tensor, *, parameters: FieldParameters) -> BodyFieldObservation`; output guidance `[B,K,3]`, distances `[B,K]`, valid `[B,K]`. `vmf_direction_reward(guidance: Tensor, velocity: Tensor, *, concentration_scale: float, valid_mask: Tensor) -> Tensor` returns `[B]`. `yaw_rotate_vectors(vectors: Tensor, quaternion_wxyz: Tensor) -> Tensor` returns the original vector shape in yaw-local coordinates.

- [x] Add failing tests named `test_vmf_matches_explicit_3d_density`, `test_vmf_zero_concentration`, `test_zero_velocity_not_aligned`, `test_vmf_large_concentration_finite`, `test_invalid_field_mask`, `test_root_priority` and `test_yaw_rotation`. Assert small-kappa limit `log C3=-log(4*pi)`, aligned > orthogonal > opposite, and no NaN for zero/large inputs.
- [x] Run `python -m pytest tests/test_field_math.py -q`; verify failure is missing new APIs.
- [x] Implement equations 1--5 using existing pure functions and equations 7--10 using stable log normalization; explicit empty-scene attraction and invalid-query masks. Keep field guidance distinct from physical velocity.
- [x] Run the new tests plus `tests/test_potential_field.py`; require all pass.
- [x] Commit the numerical deliverable and record formulas and regularization in its documentation.

### Task 2: Portable known scene assets and goal field cache

**Files:** Create `src/embodied_learning/data/field_scenes.py`, `field_cache.py`, `configs/experiments/g1_field.json`, `tests/test_field_scenes.py`.

**Interfaces:** `load_scene_manifest(path: Path) -> SceneManifest`; immutable scene records contain ID, boxes, grid bounds/resolution, legal starts/goals. `prepare_fields(manifest: SceneManifest, output_dir: Path) -> FieldCacheManifest`; `load_field_cache(path: Path, scene_manifest: SceneManifest, device: str) -> FieldBank`. `FieldBank.sample(positions: Tensor, scene_ids: Tensor, goal_ids: Tensor) -> FieldSample` uses scene-local positions.

- [x] Add failing tests `test_box_voxel_geometry`, `test_all_start_goal_pairs_reachable`, `test_scene_goal_routing`, `test_cache_rejects_geometry_change`, `test_cache_roundtrip` and `test_no_floor_obstacle`. Use two scenes with distinctly different obstacle coordinates; compare requested fields against direct samples.
- [x] Run `python -m pytest tests/test_field_scenes.py -q` and verify failure.
- [x] Implement float32 offline fields from declared box geometry, multiple candidate goals, hashes and portable `.npz` plus JSON metadata (no pickle). Include wall detour, passage and overhead/ground layouts; keep ground physics separate. Validate starts with G1 clearance and goals with query support/reachability. Reuse unchanged cache files.
- [x] Run scene tests, requiring physical scene descriptions and cache occupancy to derive from the exact same box records.
- [x] Commit the scene/cache deliverable.

### Task 3: Real G1 field-conditioned mjlab task

**Files:** Create `src/embodied_learning/robots/g1.py`, `tasks/field_traversal.py`, `simulators/mjlab/field_traversal.py`, `tests/test_field_task.py`.

**Interfaces:** `G1_BODY_SCHEMA` defines ordered source-backed named sites/body offsets (explicit adaptations for missing mjlab sites). `make_field_env(manifest: SceneManifest, bank: FieldBank, *, scene_ids: tuple[str, ...], num_envs: int, device: str, seed: int) -> FieldTrainingEnvironment`. Its protocol exposes native observations, action contract, per-slot scene IDs, reset/step, close and schema metadata.

- [x] Add failing tests `test_body_schema_order`, `test_multi_env_field_coordinates`, `test_reset_goal_isolation`, `test_reward_precedes_reset`, `test_goal_collision_fall_timeout`, `test_invalid_field_termination`, `test_support_ground_not_collision`, and `test_actor_contains_all_body_fields`.
- [x] Run task tests and verify missing implementation failure; keep GPU integration cases explicitly marked/skipped when CUDA unavailable.
- [x] Build physical boxes and G1 from native mjlab configs, derive kinematic point positions/velocities, compose yaw-local field observations and paper rewards with configurable locomotion terms. Configure contact/termination by obstacle category. Remove random twist commands and incompatible strong standing-pose rewards. Adapt env origin/scene assignment explicitly; avoid per-step CPU field construction.
- [x] Run actual CUDA reset/step with >=2 slots, verify finite observations/rewards and reset isolation. Check scene geometry against declared coordinates and test collision with a placed obstacle, not only numerical SDF.
- [x] Commit the working environment deliverable and record actual action/schema dimensions.

### Task 4: PPO specialists and strict frozen teachers

**Files:** Create `src/embodied_learning/protocols/imitation.py`, `algorithms/ppo.py`, `policies/field_policy.py`, `tests/test_field_ppo.py`.

**Interfaces:** `train_specialist(env: FieldTrainingEnvironment, config: PpoConfig, output_dir: Path, *, resume: Path | None = None) -> Path`; `load_teacher(checkpoint: Path, *, expected_contract: PolicyContract, device: str) -> ExpertPolicy`; `ExpertPolicy.predict(observations: TensorDict) -> Tensor`. Algorithms access only protocols and installed RSL-RL; simulator integration is supplied by the CLI.

- [x] Add failing tests `test_reject_schema_action_scene_mismatch`, `test_teacher_loads_normalizer`, `test_teacher_deterministic_frozen`, `test_ppo_update_and_resume` and `test_checkpoint_retains_contract`.
- [x] Run focused tests; verify missing implementation failure.
- [x] Integrate official RSL-RL PPO/runner APIs, local logging, saved immutable configuration and metadata. Validate full actor/critic/network contract and normalizer state on resume; inference is eval/no_grad. Support configurable env count/iterations and load compatible initial actors only by explicit configuration.
- [x] Execute two different scene specialist runs on actual CUDA, at least two rollout/update iterations each; inspect parameter changes, finite losses and checkpoint files. Load teachers and resume a further PPO update.
- [x] Commit the PPO/checkpoint deliverable.

### Task 5: Generic online DAgger with persistent aggregation

**Files:** Create `src/embodied_learning/data/aggregation.py`, `algorithms/dagger.py`, add student model support to `policies/field_policy.py`, create `tests/test_dagger.py`.

**Interfaces:** `AggregationBuffer(capacity: int, observation_dim: int, action_dim: int, device: str)` exposes `add`, `sample`, `state_dict`, `load_state_dict`. `train_dagger(env: ImitationEnvironment, student: StudentPolicy, experts: Mapping[int, ExpertPolicy], config: DaggerConfig, output_dir: Path, *, resume: Path | None = None) -> Path`; environment supplies student/expert observations and scene IDs through the protocol. Student prediction returns compatible deterministic joint actions; fit minimizes explicit MSE to teacher labels.

- [x] Add failing tests `test_labels_student_visited_states`, `test_teacher_scene_routing`, `test_labels_before_env_step`, `test_old_rounds_retained`, `test_beta_zero_executes_student`, `test_teacher_no_grad`, `test_capacity_bound`, `test_resume_buffer_optimizer_schedule`, `test_eval_never_uses_teacher` and `test_action_contract_mismatch` using tiny deterministic mock environments/experts.
- [x] Run `python -m pytest tests/test_dagger.py -q` and verify failure.
- [x] Implement labeled student rollouts, explicit beta mixture and decay, bounded aggregation with uniform/reservoir sampling, minibatch fitting and immutable observation snapshots. Preserve student/teacher normalization and ordered contracts; prevent labels from being paired with post-reset observations. Checkpoint buffer, model, optimizer, scheduler/round/seed state; evaluate student alone.
- [x] Run mock tests; then collect and optimize actual mjlab rollouts using the two Task 4 teacher checkpoints. Verify dataset growth, student parameter changes, teacher parameters unchanged, save/load and one further DAgger round.
- [x] Commit the generic DAgger and real integration evidence.

### Task 6: CLI, evaluation and server commands

**Files:** Create `src/embodied_learning/cli/field_training.py`, `docs/field_training.md`, `tests/test_field_cli.py`; modify `pyproject.toml`, main `README.md`, perception README and `third_party/NOTICE.md` only where needed.

**Interfaces:** `python -m embodied_learning.cli.field_training {prepare,ppo,dagger,evaluate}` accepts `--config`, `--output-dir`, `--device`, `--seed` and stage-specific `--scene`, `--num-envs`, `--iterations`, `--teachers`, `--resume`, `--checkpoint`. Teacher manifest maps actual scene IDs to saved compatible teacher checkpoint paths.

- [x] Add failing tests `test_cli_help_without_simulator`, `test_missing_teacher_fails_early`, `test_invalid_config_fails_early`, `test_relative_paths_portable` and `test_manifest_action_schema_checks`. Do not create a simulator for `--help` or malformed input.
- [x] Run CLI tests and verify missing entry failure.
- [x] Implement composition at the CLI, pinned compatible training extras, stage output directories and machine-readable metrics. Document installation, prepare, teacher runs, teacher manifest, DAgger, evaluation and resume with no local absolute paths. Annotate differences from source/paper and license provenance.
- [x] Run every documented smoke command with small actual counts; require saved PPO/DAgger artifacts to be accepted by evaluate/resume. Run Ruff check/format for changed Python files and all relevant existing plus new tests; inspect cached diff for accidental user-file edits.
- [x] Commit the completed CLI/documentation deliverable and provide the real long-training command sequence with its exact verified flags.

## Execution Handoff

This plan requires user review before implementation under the brainstorming/writing-plans workflow.
Recommended method: native execution in this session, because scene metadata, observations and policy contracts couple every stage; one independent whole-change review follows passing checks.
After approval, use the isolation/TDD/execution/verification skills, track each checked step, and persist until the objective's full training readiness audit passes.
