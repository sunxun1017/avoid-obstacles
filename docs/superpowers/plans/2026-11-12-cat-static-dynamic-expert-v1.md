# CAT 静动联合专家 v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现已批准的联合专家方案，验证后在服务器启动有门槛的自动课程。

**Architecture:** 原 CAT 静态路径保持兼容；新增独立联合状态、移动球原生实体、191/279 观测和有符号奖励。复用原 PPO，增加显式 Actor 扩维初始化与静态动作锚定。独立运行协议负责 S0–S4、准入、评测、恢复和失败停机。

**Tech Stack:** Python、PyTorch、MuJoCo/mjlab/Warp、RSL-RL、pytest、SSH。

**Spec:** `docs/superpowers/specs/2026-11-12-cat-static-dynamic-expert-v1-design.md`

## Global Constraints

- 动作12维；原 tanh、目标角增量0.5、PD、20ms控制/2ms物理不变。
- Actor191、Critic279；父模型哈希0662d42a64df67e30ed0776554f8b203b5e2364db65f607938f0a2cca6f9fb88。
- 新实验，不覆盖父模型；旧162/250接口和旧评测不改变。
- 第一版单球8–12cm、0.2–0.8m/s；30%静态保持；关键阶段独立门槛。
- 先通过基线、单元验证和真实GPU短训练，再启动课程；不足或失败必须记录并停机，不能跳门槛。

## Review Focus

- 部分环境自动重置不得覆盖另一些环境的终止奖励、碰撞或任务记忆（Task2/3原生测试）。
- 接近胸部但不触及名义头部的球不解除前俯约束（Task1几何测试）。
- 物理子步内接触和20s终止罚分不可被控制降采样/GAE截断掩盖（Task3/4测试）。
- 初始Actor输出及动作分布须与父模型一致；其他控制语义差异必须拒绝迁移（Task4）。
- 未触发关键阶段、静态退化、未完成反事实威胁验证均不得自动晋级（Task5）。

### Task 1: 几何、任务记忆与奖励纯函数

**Files:** Create `tasks/cat_dynamic.py`, `tasks/cat_dynamic_reward.py` under `src/embodied_learning`; test `tests/test_cat_dynamic.py`.
**Interfaces:** `predict_sphere_clearance(centers, radii, velocities, obstacle_position, obstacle_velocity, obstacle_radius)`; `head_allowance(...)`; `JointMemory.update(...)`; `joint_reward(...)` return batched tensors.

- [ ] Write failing tests: moving-away vs approaching separation, full-body radii, chest exclusion, wait gate retains nominal-forward danger, progress cannot repeat, -25 failure outside cost clip, no-head posture remains penalized.
- [ ] Run `PYTHONPATH=src /home/sx/miniconda3/envs/mjlab/bin/python -m pytest tests/test_cat_dynamic.py -q` and inspect RED.
- [ ] Implement formulas from spec, finite/device/shape checks and 29-column append schema.
- [ ] Run targeted tests and commit the independently usable tensor layer.

### Task 2: 联合任务与原生运动实体

**Files:** Create `tasks/cat_dynamic_state.py`, `simulators/mjlab/cat_dynamic.py`; modify `simulators/mjlab/cat_traversal.py`; test `tests/test_cat_dynamic_native.py`.
**Interfaces:** `JointTaskState` extends `CatTaskState`; `DynamicSphereRuntime` supplies reset, control-boundary observation snapshots, 2ms pose updates, geometry covers; `make_cat_env(..., joint_config=None)` preserves old callers.

- [ ] Test old path unchanged, joint observations191/279, mocap advances every substep, real contact terminates, partial reset preserves transition reports, no central restoring command.
- [ ] Implement separate config and schema; attach movement before physics, contact after physics; finite-horizon timeout is terminal rather than truncation so -10 reaches PPO.
- [ ] Implement terrain/route completion, actual-support clearance masks, reward-scale calibration collector and delayed token/h from the same snapshot.
- [ ] Run tensor tests and opt-in native smoke; commit.

### Task 3: 场景与可行性准入

**Files:** Create `tasks/cat_dynamic_curriculum.py`, `evaluation/cat_dynamic_scenarios.py`; test `tests/test_cat_dynamic_curriculum.py`.
**Interfaces:** versioned stage definitions, deterministic sample manifests, `admit_scenario(...)` with reference rollout evidence; complete phase-state snapshot/restore.

- [ ] Test exact stage proportions, no static interpenetration, bounded acceleration, critical contact inside skill window, 0.8s lower reaction bound, serialization/replay and missing evidence rejection.
- [ ] Implement widened new layouts and preserved original nine layouts; sample full episodes and validated phase starts with approved fractions.
- [ ] Implement fixed candidate-controller feasibility rollouts, recording accepted and rejected counts without using evaluated joint policy to filter tests.
- [ ] Verify and commit.

### Task 4: Actor迁移与静态锚定

**Files:** Create `policies/cat_dynamic_warmstart.py`; modify `algorithms/ppo.py`, `algorithms/brax_ppo.py`; test `tests/test_cat_dynamic_training.py`.
**Interfaces:** explicit pinned migration162→191; static teacher auxiliary loss on static samples only; checkpoints retain full joint protocol and parent provenance.

- [ ] Test zero-column expansion preserves both mean and scale logits, mismatched action/PD contracts rejected, fresh critic/optimizer, static anchor gradients and no dynamic anchor, terminal -10 retained by GAE.
- [ ] Implement migration/auxiliary integration without relaxing existing strict checkpoint contracts; gamma0.995/lambda0.95.
- [ ] Test actual 1-update PPO, save/reload and action equality; commit.

### Task 5: 基线、评测与自动课程

**Files:** Create `evaluation/cat_dynamic_expert.py`, `tools/experiments/train_cat_dynamic.py`, `tools/experiments/run_cat_dynamic_curriculum.py`; test `tests/test_cat_dynamic_gates.py`.
**Interfaces:** atomic protocol/status/results; S0 same-parent nine-stage old evaluations; dynamic/critical evaluation; stage advancement only after two consecutive passes after600 updates, evaluation every200, stop at2400.

- [ ] Test per-terrain/critical thresholds, static≤2pp drop and≥95%, failed/missing cohorts cannot pass, no stage skips, resume parent/protocol hash validation.
- [ ] Implement S0 and s_g calibration, held-out seeds, separate critical-window counterfactual trace metrics and final novel seeds.
- [ ] Implement child-process stages, checkpoint recovery, atomic status, failure exit, retained logs; no recurring app automation.
- [ ] Verify scripts on temporary tiny runs, then run suite and commit.

### Task 6: 审阅、服务器验证与启动

- [ ] Dispatch one fresh read-only whole-branch reviewer per executing-plans; fix important findings with regression tests.
- [ ] Run full test suite and real native GPU integration; prepare immutable release with commit/hash/protocol.
- [ ] Synchronize into new server directory; verify parent hash, runtime versions and GPU availability by UUID.
- [ ] Run S0, admission and short PPO end-to-end. If any mandatory gate fails, record explicit failure and do not launch invalid training.
- [ ] Launch detached curriculum process only after validations; observe real optimizer progress and checkpoint output, then report run path/PID/GPU/stage.

## Execution record

User approved automatic implementation/training on 2026-11-12; no extra plan approval round required. Native worktree tool could not recognize the existing repository on its host, so a Git worktree fallback was created in sibling `work/cat-static-dynamic-v1` on branch of the same name. Baseline suite:692 passed,35 skipped. Server parent hash verified; RTX4090 UUID GPU-de0e690a-9b65-1acb-6bfe-fccce34ed281 initially free.
