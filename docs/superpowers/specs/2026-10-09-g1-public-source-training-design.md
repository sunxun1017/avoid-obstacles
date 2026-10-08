# G1 公开源码训练逻辑迁移设计

日期：2026-10-09

状态：设计已完成，等待书面设计审阅；尚未实施。

## 目标和已确认范围

用户要求按 Click-and-Traverse 作者公开源码实现，而不是继续组合论文
log-vMF 与自行添加的行走奖励。沿用当前项目的 mjlab/PyTorch、PPO specialist
和 DAgger 管线，保留模块解耦；最终交付可以在服务器训练、评估和 play 的命令。
此前生成的策略和缓存保留，新方案使用独立的配置和输出目录。

这是公开训练逻辑到另一仿真/学习后端的迁移，不承诺与作者 JAX/MJX/Brax
训练产生相同轨迹、学习曲线，也不承诺直接加载作者发布的策略权重。

## 固定来源

基准提交：`866ba392f1c1e84b92ad75fa66550f26e8af8e48`。

- [env_cat.py](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/cat_ppo/envs/g1/env_cat.py)：G1Cat 奖励、速度指令、相位更新、控制、观测、终止。
- [env_loco.py](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/cat_ppo/envs/g1/env_loco.py)：继承的足端接触、打滑、关节和力矩成本。
- [constants.py](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/cat_ppo/envs/g1/constants.py)：12 个动作关节、23 个观测关节、29 关节默认姿态、PD 和力矩限制。
- [train_ppo.py](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/train_ppo.py)：ground/lateral/overhead 对场奖励权重的覆盖。
- [pf_modular.py](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/procedural_obstacle_generation/pf_modular.py)：progressive guidance、SDF 与 boundary field。

选定 G1Cat，不混用 G1CatPri、G1CatLoco、G1CatLong 中的重写。
`_re_gf0_far_baseline` 虽然存在，但 G1Cat 的 `_get_reward` 调用的是 `_re_gf0`；
默认迁移实际调用版本。以后若修改这些行为，必须另起配置并注明差异。

## 架构与数据流

```text
场景几何和目标 → CPU 离线 gf/bf/sdf 缓存
  → mjlab 提供 11 个身体点、足端接触、关节与刚体运动学
  → 源码场查询、速度指令、步态状态
  → 162 维 actor / 250 维 critic
  → RSL-RL PPO 输出 12 维腿部动作
  → 增量关节目标与 PD 力矩 → mjlab 仿真
  → 按源码状态更新顺序计算各奖励项和终止标志
```

纯 Tensor 数学放在 `tasks/cat_reward.py`、`tasks/cat_gait.py` 和
`tasks/cat_command.py`，不能依赖 mjlab 或 JAX。`tasks/field_traversal.py`
负责逐环境状态与组合；mjlab adapter 负责读取运动学、接触和应用控制。
通用 perception 中保留论文函数，公开源码训练路径不再调用它们。
PPO、DAgger 继续依赖协议和观测契约，不直接读取机器人内部状态。

## 场、速度指令和坐标

使用 progressive guidance 的 `gf`（含速度幅值）、归一化 SDF 梯度 `bf` 与
度量 `sdf`，而不是 `evaluate_body_field` 的论文优先级力场。
扩展 FieldBank 的来源标识和缓存版本；不能把旧缓存静默当成新配置。
默认体素、场参数以作者导出配置为基准，场覆盖必须包含全部身体点。
当前三个场景仍由 JSON 几何构造，并在文档中说明它们不是作者发布的数据集。

所有向量计算明确使用世界/场景坐标或作者的 pelvis 航向导航坐标。
逐环境平移原点要从位置中减去；不能对速度减去位置原点。
`compute_cmd_from_rtf` 按源码保留以下顺序：

1. pelvis 的原始 gf 水平分量乘 0.7。
2. 使用 head、feet、hands 的 bf/gf，对不满足的边界约束计算 delta。
3. 对 delta 求平均并加入原速度。
4. `[1,vx,vy,0]` 整体乘 0.75；速度范数小于 0.2 时输出 stop command。

注意第一维因整体缩放而为 0.75。保留源码比较 `==1.0` 与 `>0.5` 的实际差异，
不把 move_flag 擅自二值化。reward 使用源码在后续步骤归一化后的身体点 gf；
指令使用归一化前的 gf。actor 使用源码延迟查询路径和导航系变换。

场查询保留源码边界裁剪行为，附带 out-of-bounds 诊断；不继续沿用旧训练的
“任意身体点缺少测地距离就 invalid”的判定。未知或损坏缓存直接报错。
近目标减速和 stop command 可能在 0.1 米目标判定前停止，应报告真实停止距离，
不能为了成功率随意修改源码速度阈值。

## 步态和逐环境状态

采用 G1Cat `_update_phase`，不是另一个独立的 `utils/gait_planner.py`。
控制 dt=0.02，物理 dt=0.002；reset 时频率 U[1.3,1.5]，抬脚目标 0.07 米，
初始相位在 `[0,pi]`、`[pi,0]` 间随机选择。按源码 wrap 相位，cos 相位与
±0.6 阈值生成 stance/transition/swing 标志；保留停止倒计时、last_command
和 stance_phase 的真实更新顺序。

保存 motor_targets、last_act、last_last_act、last_joint_vel、phase、phase_dt、
gait_mask、foot_height、command/last_command、停止倒计时、步数、延迟 odometry
以及 PD/RFI 随机参数。reset 只修改指定槽位；缓存刷新不得重复推进相位或历史。
奖励不得使用 reset 后的身体状态来解释 reset 前的动作。

特别保留源码在 `_get_reward` 前更新动作历史的时序，数值测试覆盖它对
smoothness_action 的影响；不根据变量名推断原作者意图再修改实现。

## 奖励及权重

| 名称 | 基础权重 | 来源与语义 |
|---|---:|---|
| tracking_orientation | 2 | pelvis/torso roll、torso pitch 和 head 高度条件 |
| tracking_root_field | 1 | exp(-4 * 水平速度平方误差) |
| body_motion | -0.5 | 偏离命令方向的水平运动与 torso roll/pitch 角速度 |
| body_rotation | 1 | 双腿导航系旋转轴误差与 yaw 命令衰减 |
| foot_contact | -1 | 根据 gait_mask 比较足端实际/目标接触 |
| foot_clearance | -15 | 仅惩罚摆动脚低于目标高度 |
| foot_slip | -0.5 | 期望支撑阶段的完整足端线速度平方 |
| foot_balance | -30 | pelvis subtree COM 与双脚中心误差及窄足距放大 |
| straight_knee | -30 | sum(max(0.1-knee_angle,0)) |
| foot_far | 0 | 保留源码项，但不擅自启用 |
| smoothness_joint | -1e-6 | 观测关节速度与离散加速度成本 |
| smoothness_action | -1e-3 | 源码 action 的零/一/二阶成本及实际历史更新顺序 |
| joint_limits | -1 | soft joint limit 越界量之和 |
| joint_torque | -1e-4 | 全部执行器力矩平方之和 |

场奖励使用作者 `_re_gf0` 和 `_re_sdf`：

- gf：归一化向量余弦，乘 sigmoid(40*(tau-sdf)) 和 5；按源码 crossed mask
  置为 4，之后对每组身体点求均值。head/hands tau=0.5，feet tau=0.3。
- sdf：`-mean(20*softplus((0.05-sdf)/0.02))`。
- feetgf/feetdf 对应 ground；handsgf/handsdf/kneesdf/shldsdf 对应 lateral；
  headgf/headdf 对应 overhead。独立源码配置默认三个参数为 1，按照作者训练脚本
  建议启用全部障碍类型；无障碍行走阶段显式设置为 0，并记录该配置。
- crossed 的停步/支撑脚/身体点 x>1.5 条件按源码逐组实现。x 是去掉环境平移
  的场景坐标；这一硬编码适用范围会写入配置说明，不能泛化声称适用任意目标。

每个 reward NaN 替换为 0；加权总和乘 dt 后 clip 到 [0,10000]。
只在一处乘 dt，避免 mjlab RewardManager 再次缩放。记录各项原值、加权值、
最终裁剪奖励，以及非有限值诊断，不能把裁剪后高奖励当作任务成功。
源码路径不使用 vmf_weight、alive_weight、额外 goal/collision/fall 奖励。

## 机器人控制和观测契约

继续使用 mjlab G1 模型，但按照 constants 的关节名称映射，策略动作变成
12 个腿部关节；物理机器人依然有 29 关节。motor_targets 每控制步累加
0.5*action，再限制到源码 soft limits；其余关节保持作者默认姿态。
根据 constants 的 kp/kd/torque_limit 施加每物理步 PD，加上源码 RFI 并裁剪力矩。
不能沿用 native position actuator 后又声称显式 PD 与作者一致。
模型的质量、惯性、足端形状和自接触过滤若与源码 XML 不同，应报告差异；
本次不打包原作者机器人 mesh，也不宣称动力学逐轨迹一致。

actor=162、critic=250、action=12。严格按 `_get_obs` 和 23 关节名称表拼接；
actor 含命令、腿部目标、步态相位、77 维 gf/bf/sdf，critic 加 noiseless
运动学、身体点状态、接触与随机参数。噪声、5 控制步 odometry 延迟刷新、
字段排序和 nan_to_num 按来源实现并测试。

检查点必须记录 task/source commit、奖励/场/观测版本、关节映射、控制设置、
source scales、模型标识与场缓存哈希。移除 CLI 中 173/29 硬编码，替换为
注册的契约校验。旧 checkpoint 不可 resume 或作为新 DAgger 教师；必须给出
清晰 mismatch。旧配置和旧回放路径仍可选择，避免覆盖原训练结果。

## 终止、评估与扰动

源码终止条件：pelvis upvector.z<0、head.z<0.7、指定足/小腿自接触、
身体点 sdf 小于负碰撞阈值、qpos/qvel NaN；接触条件前 50 步豁免。
默认 threshold=0.04。timeout=1000 控制步。地面接触是步态输入，不作为
障碍碰撞直接终止；源码没有目标到达奖励或目标到达终止。

目标距离/是否进入当前 0.1 米范围保留为额外评估信息，不参与 reward/done。
累计每个回合“曾经安全到达”的标志，到回合结束报告成功率；同时输出
最小距离、结束距离、停步占比和各类终止。明确此成功判据属于本项目，
不能冒充作者 benchmark。场景内 box 接触另行记录，以区分点 SDF 与实体碰撞。

按 env_cat 默认启用观察噪声、reset PD/RFI 随机和 5--10 秒 additive XY push，
幅度 0.1--1；play/evaluate 提供明确的无扰动模式用于观察，训练设置写入契约。
ENABLE_RANDOMIZE=False 对应不额外启用作者 MJX 模型域随机化，不能把这个
全局开关错误地解释为关闭 env_cat 内的 PD/RFI/push。

## 学习后端和交付

保留 RSL-RL PPO，source 配置对齐 gamma=0.97、lr=3e-4、entropy=0.01、
gae=0.95、clip=0.2、rollout=20、4 次更新，actor 256/128/64、critic 512/256/128、
关闭 observation normalization。Brax batch 与 RSL minibatch 的定义不同，
在文档中单列，不能宣称优化器或采样完全相同。默认环境数按服务器容量设置，
不盲目照搬作者 32768。DAgger 的契约更新支持不同 actor/critic 架构与新维度。

新增独立 `g1_cat_source.json` 与输出 `runs/g1-cat-source`。提供 prepare/ppo/
dagger/evaluate/play 命令；play 可用 native viewer 或服务器 localhost Viser，
通过 SSH 转发观看。当前临时 `/tmp/embodied_field_play.py` 不作为正式入口。
迁移代码保留 Apache-2.0 来源和修改说明，更新 NOTICE，原 MIT 不覆盖第三方条款。

## 验收

1. 固定输入对照作者函数输出，验证全部奖励原值、权重、dt、裁剪、命令和相位；
   覆盖 near/far/inside、零速度、zero command、crossed、stop、50 步、异步 reset。
2. 使用作者函数生成的固定参考数据对照 PyTorch；float32 使用明确的容差，
   不用“复制另一份公式”代替独立来源参考。JAX 仅为开发 oracle，运行时不依赖。
3. actor/critic 的字段、维度、坐标、延迟刷新与噪声范围逐项检查；动作名称、
   累积目标、力矩限制和上肢保持状态检查；缺传感器或关节时明确失败。
4. CPU 数学测试、相关回归检查与 CUDA 原生 G1 集成；逐项核对奖励没有二次 dt。
5. 三个场景各做小规模 PPO 更新、保存/恢复、教师冻结和 DAgger smoke；
   参数确实变化、loss 有限、旧 contract 被拒绝、play 实际驱动正确检查点。
6. 在指定 RTX 4090 上提供完成测试的命令；短 smoke 不宣称已经学会绕障。

## 设计自检

- 选定唯一 G1Cat 版本，继承与 override 明确；没有混用 far_baseline 或论文 vMF。
- 控制、命令、相位、观测是奖励依赖，均纳入迁移；不是只替换一个 reward 函数。
- 旧训练、用户本地修改与正在看的回放保留；不启动长期训练、推送或修改服务器代码。
- 每个后端差异有明确边界；没有把迁移描述成作者原生训练的完整复现。
- 没有待定参数、未定义字段或以高奖励代替到达指标的验收项。
