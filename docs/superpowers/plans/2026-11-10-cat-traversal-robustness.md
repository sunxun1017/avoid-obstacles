# CAT Traversal Robustness Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task by task.

**Goal:** Implement the user's approved collision, clearance, mixed curriculum and corridor/push changes together.
**Architecture:** Versioned task helper owns geometric costs/navigation; native simulator captures contacts before task refresh. A new curriculum recipe adds anchor scenes while using the existing strict evaluator.
**Tech Stack:** Python, PyTorch, MuJoCo/mjlab, pytest.
**Spec:** docs/superpowers/specs/2026-11-10-cat-traversal-robustness-design.md

## Global Constraints
Legacy contracts unchanged. Keep overhead posture gating, full push magnitude, evaluator226/256,600 updates and128x2 cohorts. No long training launch.

## Review Focus
- Early contact must survive clear final substep and partial auto-reset.
- Rear foot remains constrained after root/front foot passes; flat/high roof unaffected.
- Mixed scenes must match physical geometry, fields and checkpoint selection; gate must use target only.
- Targeted push must not duplicate timer push or leak into evaluation.
- Bootstrap is pinned and auditable; undeclared contract changes/resume fail.

### Task 1: Contact and geometric guidance
Files: tasks/cat_traversal_safety.py, tasks/cat_traversal.py, simulators/mjlab/cat_traversal.py, tasks/cat_substep_contacts.py; evaluation wrapper and tests.
- [x] Write and run failing tests for transient contact, zero terminal reward, obstacle foot cover clearance and corridor command.
- [x] Implement geometry helper and native substep accumulator; wire task terms and command observations.
- [x] Run targeted tests, existing posture/task/geometry/contact regressions.

### Task 2: Curriculum and phase pushes
Files: tasks/cat_curriculum.py, tasks/cat_phase_push_curriculum.py, tools/experiments/train_cat_curriculum.py, policies/cat_actor_warmstart.py and tests.
- [x] Write and run failing tests for stages/anchors, phase/reset/fallback, bootstrap strictness.
- [x] Implement v2 recipe, training-only mixing and first-push scheduling with frozen evaluator.
- [x] Run integration and transfer tests.

### Task 3: Verify and deliver
- [x] Run full project tests and relevant lint/compile checks.
- [x] Native short smoke if available; inspect pre-reset contact and checkpoint behavior.
- [x] Independent whole-change review, address important findings.
- [x] Package patch and before/after notes; report tests and limitations without claiming a measured success gain.

Verification: local full suite692 passed35 skipped; native13 passed; pinned real parent1 PPO update/5120 transitions, strict reload and field routing passed. Independent reviewer found no P1/P2. No formal course launched.
