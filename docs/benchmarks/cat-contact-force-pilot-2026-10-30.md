# 基线保留与实测触地力单项奖励试验

日期2026-10-30，状态：两组已开始正式训练，尚无效果结论。

## 基线的含义

选用上一轮原奖励基线第128轮模型：
`/home/sunxun/embodied-learning/runs/cat-landing-speed-stage4-pilot-20261030-01/baseline/continuation/update-0128/latest.pt`

SHA256：`22505bcb518ef36eb3b064e3ec7e7eed2f510c02154bec098a6e769ef3903e1f`。

用户明确要求已发现并修正的错误不得改回。实现从当前代码2804134开始，只增加测力奖励开关，没有整仓回滚。动作历史、势场插值、物理子步采样时序、缓冲区检查、严格模型契约等既有修复保留；旧落地速度实验源码/记录保留，但两组均不启用其奖励。名义PD仍hip pitch100/2、knee200/4、ankle30/2，保持当前低杆+头顶横梁难度。用户未提交的g1_cat_source.yaml配置修改保持原样。

## 论文与源码依据

- [legged_gym](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py) 的 `_reward_feet_contact_forces` 提供超过阈值的线性力惩罚；基础配置未默认启用此项，不能写成所有legged_gym策略均使用。
- [IsaacLab](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/rewards.py) 的 `contact_forces` 先取每脚历史力峰值，再对超阈值部分求和。
- [Unitree RL Lab H1](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h1/velocity_env_cfg.py) 配置中启用了500N阈值及负权重；本次查阅的G1配置没有相同力项，不能把H1参数冒充G1默认值。
- [Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking](https://icra-2024-humanoid.github.io/docs/assets/papers/robust-walking.pdf) 用单脚接触及宽限期改善行走，并讨论了站立时强求双脚接触可能妨碍恢复迈步。该项针对步态，与直接抑制冲击的目标不同，本轮不叠加接触数量或固定双脚支撑奖励。

## 本轮唯一新增项

每个20ms控制周期，对每脚的实际脚—地净接触力向量，在10个2ms子步中取模长峰值。正代价为：

`sum_feet(relu(peak_force / robot_weight_N - 3))`

权重-0.2，仍由既有奖励聚合器按控制dt积分。3倍整机体重和0.2权重是本次选定的试验参数，尚未证明最优；不是照搬其他机器人100N或500N阈值。当前模型整机体重约327.08N，阈值约981N/脚。正常支撑低于阈值不扣分，不因一直触地而额外得分，不硬裁执行力矩/动作，也不依赖步态相位。

只给原左右脚地面接触传感器增加force输出。独立编译原始模型，逐项验证非传感器物理数组及求解选项未变，再沿用经核实的原物理身份；新增奖励记录在独立版本化契约中，不能借此跳过动力学、PD、几何或观测变更。每次物理步之后采样、刷新传感器缓存、保留控制周期峰值；不增加forward或随机数抽样。新奖励与旧落地速度项禁止静默叠加，完整续训仍要求契约一致。

## 已验证

- 独立审查未发现Critical/Important问题，审查者另跑56项针对性测试通过。
- 隔离副本全量546通过/34跳过，随后补充的两项定向测试通过；主项目最终全量582项通过，1条既有MULTICCD警告，详见main-tests.log。
- GPU预检两组原PD、初始q/root/随机倍率哈希匹配，除力奖励外契约一致。
- 正式初始化Actor哈希匹配，fresh Critic哈希匹配，优化器均为空；正式训练损失有限。
- 对固定父策略的300个控制周期、32环境预检，新项在250个slot-control上确实降低聚合后的奖励，合计下降1.854；过渡相位存在非零超力代价。因此没有仅添加函数却未进入总奖励、或全部被正奖励裁剪掩盖的情况。该验证不代表训练已经改善。

## 运行计划

根目录：`/home/sunxun/embodied-learning/runs/cat-contact-force-stage4-pilot-20261030-01`。

GPU1运行baseline，GPU5运行force020，均用UUID固定设备，不使用GPU0。两组同从上述基线128继承Actor、fresh Critic/Adam，再新增128轮；每32轮保存和评估。4096环境、160控制步/轮，各83,886,080环境转移。首段Actor-only，后续完整恢复PPO状态。

每个32/64/96/128阶段评估256局成功率（种子20261030）及32局500Hz冲击/身体/关节曲线（种子20261026）；最终第128轮再评估256局独立种子20261031。没有自动晋级或替换旧最优模型。

查看根目录driver-status.json和各组status.json；阶段结果在各组history.json、chunks/update-XXXX/evaluation-seed20261030/evaluation.json、impact-evaluation/analysis.json；完整结果在results-summary.json。baseline-selection.json记录“回到基线”的模型与保留修复的约束。
