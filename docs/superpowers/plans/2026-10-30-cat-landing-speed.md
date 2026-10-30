# CAT Landing Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a controlled opt-in landing-speed reward pilot without changing historical defaults.
**Architecture:** A focused landing module owns analytic foot geometry, substep cost accumulation, configuration and strict Actor-transfer adaptation; controller and reward hooks remain inactive by default. A persistent isolated GPU driver runs matched baseline/candidate training and automatic evaluations.
**Tech Stack:** Python, PyTorch, MuJoCo/mjlab, existing PPO trainer.
**Spec:** docs/superpowers/specs/2026-10-30-cat-landing-speed-design.md

## Global Constraints

Preserve all old PD/reward/geometry/noise defaults, dimensions and strict resumes. No use ofGPU0. No physical robot operations. Keep user config edits and all experiment data. Pilot only: weight0.1, height0.05m, descent threshold0.5m/s; 32updates4096envs160controls/update, evaluations256+32episodes.

## Review Focus

- Neutral phase0 must not bypass the cost.
- A landing entirely between50Hz observations must contribute its2ms integral exactly once.
- Rotated capsules and env z offsets must produce true foot-bottom heights.
- Resetting one environment must not erase other environments' accumulated costs.
- Reward transfer must not permit unrelated PD/physics/observation changes or full-state resume.

### Task 1: Implement and validate opt-in landing cost and pilot

**Files:** Create src/embodied_learning/tasks/cat_landing_speed.py and tests/test_cat_landing_speed.py. Modify cat_control.py, cat_reward.py, cat_traversal.py, cat_hurdle.py, cat_actor_warmstart.py and train_cat_curriculum.py. Add isolated pilot worker and recorded benchmark outside defaults.

**Interfaces:** `landing_speed_cost(sole_height, vertical_velocity, *, height_m, speed_m_s)` returns positive[B] cost rate. `FootLandingState` computes exact sphere/capsule bottoms, accumulates per-substep cost integral, exposes cost_rate; controller resets control integrals and per-slot state. Protocol `foot_landing_speed` carries an exact validated schema. Same-stage `landing_speed_refinement` declares parent hash and only before/after landing schema.

- [ ] Write behavioral tests first: descent/speed/height gating including phase0, time-integral and reset, analytic rotated geometry, disabled historical reward identity, opt-in reward sign, strict transfer and invalid config/changed contracts.
- [ ] Run new tests and confirm missing-feature failure. Expected: FAIL.
- [ ] Implement minimal feature and wire physical-substep accumulation to existing rate aggregation. Expected: old defaults unchanged and new term exact.
- [ ] Run new tests and full pytest suite. Expected: all pass; report warnings.
- [ ] Request a fresh whole-change code review, fix important findings with tests, and verify again.
- [ ] Validate GPU preflight: actual PD unchanged, parent Actor exact, fresh identical Critic/Adam, only new reward/schema differs; finite loss and nonzero neutral-phase landing cost.
- [ ] Launch persistent matched32update pilot, automatic success and500Hz impact evaluations; retain scripts/status/data and report actual state. Expected: running or complete, no unsupported improvement claim.
- [ ] Integrate verified source into user's project preserving dirty config, commit and attempt authorized normal push. Record any network blocker.
