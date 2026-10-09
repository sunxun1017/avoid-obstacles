# G1 Public-Source Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. 执行方式沿用此前用户选择：当前会话逐项实现。

**Goal:** 在 mjlab/PyTorch 中迁移固定版本 G1Cat 的奖励及其控制、观测依赖，交付新的训练与 play 命令。

**Architecture:** 将作者纯数学函数转换成批量 Tensor 运算，模拟器仅提供运动学、接触和执行力矩。新增独立 source task，保留旧任务以回放既有检查点；PPO/DAgger 通过版本化契约接入两种任务。

**Tech Stack:** Python 3.11、PyTorch 2.7.1、mjlab 1.6.0、RSL-RL 5.4.2、MuJoCo/Warp、NumPy/scikit-fmm、pytest。JAX 仅用于开发阶段生成作者函数参考数据。

**Spec:** `docs/superpowers/specs/2026-10-09-g1-public-source-training-design.md`

## Global Constraints

- 来源固定 `866ba392f1c1e84b92ad75fa66550f26e8af8e48` 的 G1Cat，包括 env_loco 继承实现和 train_ppo 权重覆盖。
- source task：actor=162、critic=250、action=12；11 身体点、23 观测关节、29 物理关节按名字映射。
- dt=0.02、sim_dt=0.002；增量目标=上一帧目标+0.5*action；PD/RFI/力矩裁剪按 constants。
- 权重、命令、相位、动作历史顺序及最终 clip(sum(weighted)*dt,0,10000) 遵循设计，禁止二次 dt。
- 不混入论文 log-vMF、额外存活/到达奖励，不擅自改用 far_baseline；缺少身体/接触数据明确报错。
- 保留 mjlab 模型、RSL-RL 后端和几何场景差异的说明，不宣称跨后端轨迹一致。
- 新配置 `configs/experiments/g1_cat_source.yaml`、新输出 `runs/g1-cat-source`；旧配置、检查点和用户未提交修改保留。
- 来源改编保留 Apache-2.0 与 NOTICE；不修改服务器源码、不推送、不启动长期训练，除非另有授权。
- 执行在隔离 worktree；Python 使用 `/home/sx/miniconda3/envs/mjlab/bin/python`，普通数学测试 CPU、集成测试 CUDA。

## Review Focus

1. 每步重复读取观测/奖励不得推进相位或历史；用跨两步及异步 reset 测试锁定时序（Task 2/5）。
2. 原环境占用其他 RSL-RL editable 路径时应先诊断导入路径，不修改 AMP 环境（Task 8）。
3. 旧缓存或旧教师与新契约混用必须拒绝，不能产生看似正常的训练（Task 3/6）。
4. 场景坐标平移和出界查询不能导致错误速度、越界索引或 NaN（Task 3/5）。
5. 单环境 play、训练中间 checkpoint 和无显示服务器须可运行；只推理、不改文件中的策略参数（Task 7）。

## 文件职责

- `tasks/cat_reward.py`：奖励输入契约、各奖励纯函数、加权聚合。
- `tasks/cat_command.py`：作者速度指令投影，无仿真依赖。
- `tasks/cat_gait.py`：相位、停步与控制步历史。
- `tasks/cat_observation.py`：作者字段排序、坐标与观测组装。
- `tasks/cat_traversal.py`：source task 逐环境状态、日志与组合，不改写旧 FieldTaskState。
- `simulators/mjlab/cat_control.py`：增量关节目标与逐物理步 PD/RFI。
- `simulators/mjlab/cat_traversal.py`：G1 模型、接触、运动学采集及原生环境工厂。
- `robots/g1.py`：新增 source 关节表、默认姿态、PD 与力矩表；保留已有常量。
- `protocols/imitation.py`、`policies/field_policy.py`、`algorithms/ppo.py`：新维度、critic 架构与 normalization 开关。
- `cli/field_training.py`：配置选择、契约校验、正式 play 入口。
- `tasks/field_assets.py`、`perception/voxel_field.py`：source 查询与缓存来源/版本；通用论文函数保留。

### Task 1: 作者参考数据和纯奖励数学

**Files:** Create `tools/generate_cat_reference.py`, `tests/fixtures/cat_source_reference.npz`, `tests/fixtures/cat_source_reference.json`, `src/embodied_learning/tasks/cat_reward.py`, `tests/test_cat_reward.py`; Modify `third_party/NOTICE.md`.

**Interfaces:** `CatRewardInputs` dataclass 在 cat_reward.py 定义全部奖励所需 Tensor：pelvis/torso rpy `[B,3]`、head_pos `[B,3]`、cmd_vel/root_vel/torso_ang_vel `[B,3]`、leg_rotation `[B,N,3,3]`、foot_pos/foot_vel `[B,2,3]`、pelvis_subtree_com `[B,3]`、feet_contact/gait_mask `[B,2]`、move_flag/foot_height `[B]`、joint_pos/vel/previous_vel/torques `[B,29]`、actions/last_actions/last_last_actions `[B,12]`、gf/bf `[B,11,3]`、sdf `[B,11]`、navigation_rotation `[B,3,3]`。`cat_reward_terms(inputs: CatRewardInputs, *, parameters: CatRewardParameters) -> dict[str, Tensor]` 返回原值 `[B]`；`aggregate_cat_reward(terms: dict[str, Tensor], scales: dict[str,float], *, dt: float) -> Tensor` 返回最终奖励。

- [x] 写 `test_cat_reward_matches_author_fixture`、`test_cat_reward_source_weights_and_single_dt`、`test_cat_field_masks_and_nan_handling`；断言每个原值/加权总值对照固定 fixture，float32 rtol=1e-5/atol=1e-6，并验证负和归零、极值封顶10000、dt恰好一次。
- [x] 运行 `python -m pytest tests/test_cat_reward.py -q`，确认缺实现导致 RED。
- [x] 实现 reference 生成器：输入固定 seed、读取固定提交源函数及 constants，调用作者函数生成近/远/内部、零速度、停步、crossed 等参考，记录 commit、输入和源文件 SHA256。不得重新手写一份奖励公式充当 oracle。将轻量输入/输出 fixture 纳入版本控制，运行时不需要 JAX。
- [x] 按设计奖励表移植实际 override 和继承函数，参数校验包含 shape/device/precision，更新来源许可记录。
- [x] 重跑上述测试 GREEN；只提交本任务文件：`feat(tasks): port source G1Cat reward mathematics`。

### Task 2: 源码指令与步态状态

**Files:** Create `tasks/cat_command.py`, `tasks/cat_gait.py`, `tests/test_cat_command_gait.py`; Extend Task 1 fixture generator。

**Interfaces:** `compute_cat_command(root_guidance: Tensor, guidance: Tensor, boundary: Tensor) -> Tensor` 消费 `[B,3]` 和 head/feet/hands 五点 `[B,5,3]`，返回 `[B,4]`。`CatGaitState(num_envs: int, device: str, *, dt: float=0.02)` 提供 `reset(env_ids: Tensor, frequency: Tensor, initial_phase: Tensor) -> None`、`advance(command: Tensor) -> Tensor`，返回源码修正后的 command；暴露 phase `[B,2]`、gait_mask `[B,2]`、stop_timestep `[B]` 和 last_command `[B,4]`。

- [x] 写 `test_cat_command_matches_author`、`test_move_flag_is_source_075`、`test_phase_stop_transition_matches_author_sequence`、`test_slot_reset_preserves_other_slots`；以 fixture 比较至少60步停止过程，断言随机相位两种初始化、mask ±0.6 边界与小命令阈值0.2。
- [x] 运行 `python -m pytest tests/test_cat_command_gait.py -q` 确认 RED。
- [x] 实现纯 Tensor 批量指令与 `_update_phase`，保留 .75 第一维和源码比较条件；reset 不更新其他槽位，不使用独立 GaitPlanner 的另一套逻辑。
- [x] 测试 GREEN；提交 `feat(tasks): port G1Cat command projection and gait state`。

### Task 3: 来源明确的场缓存与查询

**Files:** Modify `tasks/field_assets.py`, `perception/voxel_field.py`; Create `tests/test_cat_fields.py`; Extend `tests/test_field_scenes.py`。

**Interfaces:** 旧 `FieldBank.sample` 语义不变。新增 `FieldBank.sample_source(positions: Tensor, scene_ids: Tensor, goal_ids: Tensor) -> FieldSample`，返回 gf 原幅值、归一化 bf 可由 sdf_gradient 推导、sdf，按源码裁剪三线性查询；暴露 out_of_bounds 诊断。`prepare_fields(manifest, output_dir, *, source_profile: dict | None=None)` 可传入来源和场参数；旧调用仍有效。

- [x] 写 `test_source_field_matches_author_grid_fixture`、`test_source_query_clamps_without_cross_env_shift`、`test_cache_rejects_source_profile_drift`、`test_unknown_cache_not_silently_filled`，覆盖occupied、空图、地图外、双环境原点与近目标减速。
- [x] 跑上述测试 RED。
- [x] 对照 pf_modular 的有效默认参数和 BF/SDF/GF 构造；补 source_profile/cache identity，独立新缓存版本；使用原始 gf 生成 command，之后按来源归一化，不能调用论文力场代替。未知数据明确失败。
- [x] 本任务及 `tests/test_field_scenes.py tests/test_potential_field.py` GREEN；提交 `feat(perception): add source-profile field caching and sampling`。

### Task 4: 12 维动作、显式 PD 和运动学适配

**Files:** Modify `robots/g1.py`; Create `simulators/mjlab/cat_control.py`, `simulators/mjlab/cat_traversal.py`, `tests/test_cat_control.py`。

**Interfaces:** `CatJointController` 保存按名称映射的29维 targets 和12维 policy_actions；`process_action(actions: Tensor) -> None` 执行累加和soft-limit裁剪；`compute_torques(joint_pos: Tensor, joint_vel: Tensor, *, rfi: Tensor) -> Tensor` 产出29维裁剪力矩；`reset(env_ids: Tensor) -> None` 只重置指定环境。适配器 `read_cat_reward_inputs(env, state) -> CatRewardInputs` 按 Task 1 契约返回数据，缺失项不伪造零。

- [x] 写 `test_source_joint_order_and_incremental_targets`、`test_source_pd_torque_clip_each_substep`、`test_upper_body_default_hold`、`test_required_contacts_fail_fast`。断言两帧 action 的目标累计、source gains、力矩限制及异步reset。
- [x] CPU 控制测试 RED；运行不依赖 GPU 的名称/配置检查。
- [x] 配置source姿态/PD/软限位、12腿部动作、dt0.02/sim_dt0.002和显式力矩执行；不能保留第二套位置执行器PD。添加足端地面与指定自接触传感器、source腿部body集合、site和COM提取。
- [x] CPU测试 GREEN；CUDA上4个原生G1运行至少20控制步，检查所有输入形状、接触与实际力矩，无双重PD；提交 `feat(mjlab): add source G1Cat control and kinematics`。

### Task 5: 观测、状态时序、奖励聚合与终止

**Files:** Create `tasks/cat_observation.py`, `tasks/cat_traversal.py`, `tests/test_cat_task.py`; Extend `simulators/mjlab/cat_traversal.py`。

**Interfaces:** `CatTaskState` 组合Task1/2/3状态；`reset(env_ids: Tensor) -> None`、`advance_before_physics(actions: Tensor) -> None`、`refresh_after_physics(env) -> None`、`observations() -> TensorDict`、`reward() -> Tensor`、`outcomes() -> dict[str,Tensor]`。`make_cat_env(manifest, bank, *, scene_ids: tuple[str,...], num_envs: int, device: str, seed: int, options: dict) -> FieldTrainingEnvironment` 与旧工厂使用同一协议。

- [x] 写 `test_source_observation_fixture_162_250`、`test_reward_history_matches_author_step_sequence`、`test_reads_do_not_advance_state`、`test_terminal_reward_precedes_reset`、`test_source_collision_grace_49_50`、`test_ever_reached_goal_latched_until_episode_end`，验证延迟每5步刷新、噪声范围、导航系、phase字段和跨回合历史隔离。
- [x] 测试 RED。
- [x] 拼source actor/critic字段；保留真实G1Cat更新顺序和历史语义。逐reset采样PD/RFI、频率和初始phase；按source实现additive push而非直接覆写速度。
- [x] 将全部 source terms 作为一组总奖励以避免 RewardManager 二次dt；日志输出原值、加权值、clip结果与非有限次数。源码终止、50步豁免、timeout1000，goal仅评估不终止；box实际接触另记。
- [x] CPU时序测试和CUDA4环境episode/reset测试GREEN；提交 `feat(tasks): compose source G1Cat observations rewards and outcomes`。

### Task 6: 策略契约、PPO/DAgger兼容与独立配置

**Files:** Modify `protocols/imitation.py`, `policies/field_policy.py`, `algorithms/ppo.py`, `algorithms/dagger.py`, `cli/field_training.py`; Create `configs/experiments/g1_cat_source.yaml`, `tests/test_cat_training.py`; Extend `tests/test_field_ppo.py`, `tests/test_dagger.py`, `tests/test_field_cli.py`。

**Interfaces:** `PolicyContract` 新增有默认值的 `critic_hidden_dims: tuple[int,...] | None`，旧反序列化保持有效；normalization=False实际选择identity。`PpoConfig` 支持critic_hidden_dims、normalization、gamma等显式参数，旧默认保留。CLI根据配置 `task_profile: "cat_source_v1"` 选择Task5工厂和source缓存；没有该字段仍走旧路径。preflight按已注册schema校验，不硬编码173/29。

- [x] 写 `test_source_contract_rejects_old_teacher_and_cache`、`test_asymmetric_networks_and_no_normalization`、`test_source_ppo_update_resume`、`test_source_dagger_preserves_frozen_teachers`，断言新162/250/12、actor256/128/64、critic512/256/128、gamma.97/lr3e-4/rollout20/epochs4和参数更新。
- [x] 测试 RED。
- [x] 更新契约、网络与factory；source归一化关，teacher/student一致；为ground/lateral/overhead提供1/1/1默认且source行走预训练可设0/0/0。配置完整记录来源、gait、noise、PD/RFI、push、term_threshold及场参数。
- [x] CUDA上三个场景分别2次PPO更新、1次resume、冻结教师后2轮DAgger与1轮resume；旧单测回归GREEN。新输出限定 `runs/g1-cat-source`；提交 `feat(training): integrate source contracts PPO and DAgger`。

### Task 7: 正式play和可信评估

**Files:** Modify `cli/field_training.py`, `algorithms/dagger.py`; Create `tests/test_cat_play.py`; Update `tests/test_field_cli.py`。

**Interfaces:** CLI新增 `play --checkpoint PATH --viewer native|viser --port 8765 --num-envs 1`；其余配置/场景/device参数沿用evaluate。`play_policy(env: FieldTrainingEnvironment, policy: ExpertPolicy, *, viewer: str, port: int) -> None` 可放在cli独立模块 `cli/field_play.py`；viewer采用 `policy.predict`，Viser仅绑定127.0.0.1。

- [x] 写 `test_play_loads_intermediate_source_checkpoint`、`test_play_no_parameter_or_optimizer_updates`、`test_play_closes_env_on_error`、`test_evaluation_reports_ever_goal_and_final_distances`；断言play与evaluate使用同一确定性策略，旧checkpoint按旧profile可回放，source错误契约拒绝。
- [x] 测试 RED。
- [x] 实现play、checkpoint校验和try/finally资源清理；可关闭evaluation扰动但保留training contract原定义，在结果记录evaluation overrides。评估报告episode outcomes、ever-goal、安全最小/最终距离和停步比例；无完成episode返回null率。
- [x] CLI和mock viewer测试GREEN；实际native或Viser启动并确认G1会动、HTTP可访问、退出关闭环境；提交 `feat(cli): add source policy playback and outcome evaluation`。

### Task 8: 总体验证、来源说明与服务器命令

**Files:** Update `docs/field_training.md`, `src/embodied_learning/perception/README.md`, `src/embodied_learning/simulators/mjlab/README.md`, `third_party/NOTICE.md`; Create `docs/cat_source_training.md`。

- [x] 运行完整 `python -m pytest tests -q` 和修改Python文件的ruff check/format；恢复仅由测试生成的terrain_scan.npy，不触碰用户改动。记录实际通过、跳过和warning。
- [x] 为服务器写入可复制命令：独立环境RSL导入路径检查、source prepare、三场景ppo、dagger、evaluate、play与SSH转发；用GPU UUID或PCI顺序明确第5卡，避免误选5090。测试错误import路径诊断，不卸载用户AMP包。
- [x] 文档区分作者源码公式、当前三个场景、mjlab模型/RSL优化器差异；给出source函数→迁移模块映射和对照测试证据。列出停止阈值、目标判据、实际时序的限制。
- [x] 根据执行技能要求完成一次整分支独立审阅；修复实际问题后重跑相关测试。提交文档和必要修复，不推送、合并或开始长训练。
- [x] 交付新的prepare/ppo/play命令、验证结果和迁移范围；报告短smoke只能证明管线可运行，不能证明绕障成功。

## 执行前自检

设计中的源码数学、动作、观测、时序、缓存、终止、扰动、PPO差异、DAgger、play与来源均有任务；Task1定义输入，Task2定义状态，Task3定义采样，Task4/5消费它们，Task6/7沿协议接入。旧任务不被新默认静默覆盖。五项Review Focus均有具体测试归属。阶段结束后逐项勾选，测试和提交必须有实际工具输出。
