# 当前难度PD重训启动记录（2026-10-27）

用户要求：hip_pitch140/3、knee180/4，当前难度从上一难度重新开始，不使用本难度已有训练产物初始化。此次训练已启动，尚未完成，效果尚未评估。

## 实际参数与场景

| 关节 | 标称Kp | 标称Kd | 随机PD开启时Kp范围 | 随机PD开启时Kd范围 |
|---|---:|---:|---:|---:|
| 左右hip_pitch |140|3|105–175|2.25–3.75|
| 左右knee |180|4|135–225|3–5|

双踝保持30/2；其他关节PD沿用父模型。原观测噪声、推力、随机PD、RFI0.375、脚部GF进度门控、原动作累加/裁剪和奖励保持。未启用此前失败的全身速度惩罚，也未加入尚在讨论的目标越界惩罚。

本次指定增益使用新实验protocol.json的controller_pd显式覆盖，未更改机器人全局历史默认值。common训练入口已支持记录并校验该覆盖。

Stage4：横杆高0.12m、厚0.24m；横梁底面1.20m；横梁到横杆净间隔0.70m。

## 父模型与初始化

父模型为stage3/update416，上一难度评估最优（249/256，97.265625%），与原先进入stage4时选用的父模型一致。

父检查点：`/home/sunxun/embodied-learning/runs/cat-feetgf-progress-stage1-20261026-01/stage3-session-gpu0/chunks/update-0416/latest.pt`

SHA256：`b3c15301a1f7d1eb0d9f58633cc6e9f534796b0ea8bffaf00f42e162c89eb925`

新Actor精确等于父Actor；Critic重新初始化，优化器初始状态为空，当前难度更新计数从0开始。没有从stage4/update480、stage4/update611或奖励试训模型继承。

新增controller_pd_transfer声明明确父哈希、父实验/协议哈希、源stage3、目标stage4、变更前后全部29关节增益。只适配PD兼容性副本，保留机器人动力学、观测、奖励、阶段及场景校验；full resume仍不允许改变PD。原始父检查点未改写。

## 训练与验证

使用GPU6（4090，UUID `GPU-35960ade-feb7-18fb-1e03-2d17a6af7e7f`），4096环境，160步/更新，共611更新，约400,424,960条环境转换；每32更新在seed20261026下评估256回合，手动固定stage4不自动升难度。

训练结束后从门控评估中选最优检查点，再以独立seed20261027评估，并追加保留观测噪声/RFI但关闭推力的诊断。

GPU32环境冒烟训练及实际控制器增益核验通过，仅双髋pitch和双膝发生指定变化，机器人动力学指纹不变。正式4096环境Actor初始化再次精确核对父stage3/update416，Critic/空优化器核验通过；已观察到4轮PPO更新且损失均有限。session/status.json中的completed_updates在整个32更新块完成后才更新，暂为0不代表没有开始优化。

CPU测试518通过、34个CUDA测试跳过；GPU冒烟与正式启动检查另外执行。

## 运行位置

新实验：`/home/sunxun/embodied-learning/runs/cat-hip140-knee180-stage4-restart-20261027-01`

总状态：`/home/sunxun/embodied-learning/runs/cat-hip140-knee180-stage4-restart-20261027-01/driver-status.json`

分块训练状态：`/home/sunxun/embodied-learning/runs/cat-hip140-knee180-stage4-restart-20261027-01/stage4-session-gpu6/status.json`

独立源码快照和哈希清单用于训练过程中逐次校验，避免本机后续代码改动影响正在运行的训练。旧stage4与历史奖励试验目录保留作对照，本次不继承它们。
