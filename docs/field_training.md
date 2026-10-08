# G1 势场 PPO 与 DAgger 训练

这条入口使用已知场景和离线目标场训练 G1，不读取相机、不进行在线建图，也不使用 AMP。
先训练每个场景的 PPO specialist，再让学生访问多个场景，由对应教师标注并进行 DAgger 数据聚合。
短训练已验证数据流、优化、保存和恢复；避障能力与收敛需要长训练和独立评估。

## 安装和准备

从上传后的项目根目录运行，建议 Python 3.11 和支持 CUDA 的 NVIDIA GPU。

```bash
python -m pip install -e '.[field-training]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m embodied_learning.cli.field_training --help
python -m embodied_learning.cli.field_training prepare --output-dir runs/g1-field
```

`field-training` 锁定已验证的 Torch 2.7.1、mjlab 1.6.0、RSL-RL 5.4.2、
MuJoCo/MuJoCo Warp 3.11.0 和 Warp 1.16.0。安装由 pip 正常构建隔离处理。
Torch CUDA、GPU 驱动及显存应匹配；本机验证为 RTX 4060 Laptop 8GB。
`prepare` 在 CPU 上运行：检查默认 G1 姿态在所有起点/目标处的实际碰撞几何和身体点范围，
然后为每个目标构造体素 SDF、测地场及梯度。它需要训练依赖，但不需要 CUDA。
地面 z=0 作为物理支撑，不放入禁入障碍；网格向下延伸到 z=-0.2，容纳脚部查询。
这项检查证明初始姿态合法，不证明完整身体能够通过所有通道。

配置位于 `configs/experiments/g1_field.json`，含绕墙、窄通道、低横梁/地面障碍三类场景。
物理盒体与体素盒体来自同一几何记录。体素精度仍有离散误差，不能替代真实接触检测。
`runs/g1-field/fields/manifest.json` 和 `.npz` 文件可以随训练产物搬到服务器，也可重新生成。
更改几何后须使用新的输出目录；哈希不同会拒绝已有缓存。

## 正式训练命令

以下是一套完整命令，`1024` 个环境是工程起点。显存不足可以降低 `--num-envs`，
策略契约不会因为并行环境数改变。迭代数用于长训练，不保证达到原论文成功率。

```bash
python -m embodied_learning.cli.field_training prepare --output-dir runs/g1-field

for scene in wall_detour passage hurdle_crouch; do
  python -m embodied_learning.cli.field_training ppo \
    --output-dir runs/g1-field --scene "$scene" \
    --device cuda:0 --num-envs 1024 --iterations 30000
done

python -m embodied_learning.cli.field_training dagger \
  --output-dir runs/g1-field --teachers runs/g1-field/teachers.json \
  --device cuda:0 --num-envs 1024 --iterations 1000

python -m embodied_learning.cli.field_training evaluate \
  --output-dir runs/g1-field --checkpoint runs/g1-field/generalist/latest.pt \
  --device cuda:0 --num-envs 128 --steps 5000
```

PPO 自动维护 `teachers.json`，例如：

```json
{
  "wall_detour": "specialists/wall_detour/latest.pt",
  "passage": "specialists/passage/latest.pt",
  "hurdle_crouch": "specialists/hurdle_crouch/latest.pt"
}
```

清单中的相对路径以**清单所在目录**为基准。教师必须是对应场景训练的 checkpoint，
且场景哈希、身体点、关节顺序、尺度和观测都兼容。DAgger 可使用不同网络结构的教师，
各教师保留自己的归一化统计；学生结构由 `--hidden-dims` 指定。
教师训练质量决定蒸馏上限，建议先对每个教师独立评估，再开始长 DAgger。

默认 PPO 为官方 RSL-RL PPO：24 步 rollout、5 epochs、4 minibatches、
ELU 网络 `(512,256,128)`、actor/critic 观测归一化、本地 TensorBoard。
不会要求 W&B 登录。每 100 次更新保存一个 PPO checkpoint，最终为 `latest.pt`。

DAgger 默认每轮采集 64 步，监督更新 100 次，batch 512，CPU 聚合容量 200000。
每步先克隆当前观测、按场景路由教师标注，再执行学生或混合动作；标签不会来自重置后的状态。
`beta` 从 1 线性衰减，在第 20 轮之后为 0，可用 `--beta-start 0` 从首轮纯学生采集。
缓冲区均匀采样所有保留的轮次；满时淘汰最老数据。这是有界 FIFO 聚合，不是 reservoir。
它支持 TensorDict 中独立的 actor/critic 组；本任务两者目前相同。

## 先跑短验证

这套缩小参数的命令已在真实 CUDA mjlab 中运行。输出目录应为空或来自相同配置。
只说明训练管线可运行，生成的短训练教师不适合据此判断避障质量。

```bash
python -m embodied_learning.cli.field_training prepare --output-dir runs/g1-smoke

for scene in wall_detour passage hurdle_crouch; do
  python -m embodied_learning.cli.field_training ppo \
    --output-dir runs/g1-smoke --scene "$scene" --num-envs 4 \
    --iterations 2 --rollout-steps 4 --hidden-dims 32 16 \
    --epochs 1 --mini-batches 1
done

python -m embodied_learning.cli.field_training ppo \
  --output-dir runs/g1-smoke --scene wall_detour --num-envs 2 \
  --iterations 1 --rollout-steps 4 --hidden-dims 32 16 \
  --epochs 1 --mini-batches 1 \
  --resume runs/g1-smoke/specialists/wall_detour/latest.pt

python -m embodied_learning.cli.field_training dagger \
  --output-dir runs/g1-smoke --num-envs 2 --iterations 2 \
  --rollout-steps 4 --updates 2 --batch-size 8 --capacity 100 \
  --hidden-dims 32 16 --beta-start 0

python -m embodied_learning.cli.field_training dagger \
  --output-dir runs/g1-smoke --num-envs 2 --iterations 1 \
  --rollout-steps 4 --updates 2 --batch-size 8 --capacity 100 \
  --hidden-dims 32 16 --beta-start 0 \
  --resume runs/g1-smoke/generalist/latest.pt

python -m embodied_learning.cli.field_training evaluate \
  --output-dir runs/g1-smoke --num-envs 2 --steps 8 \
  --checkpoint runs/g1-smoke/generalist/latest.pt

python -m embodied_learning.cli.field_training evaluate \
  --output-dir runs/g1-smoke --scene wall_detour --num-envs 2 --steps 8 \
  --checkpoint runs/g1-smoke/specialists/wall_detour/latest.pt
```

恢复时 `--iterations` 表示**新增**更新/轮数，网络、rollout、优化参数和任务配置必须与原 run 一致。
DAgger 保存聚合数据、优化器、学习率调度、归一化、已完成轮数、教师哈希及 Torch 随机状态。
恢复重新开始物理 episode，不恢复瞬时 MuJoCo 物理状态，所以不承诺与未中断 rollout 逐位相同。
修改教师权重后，已有 DAgger resume 会拒绝；请冻结教师文件，或开始新的蒸馏 run。
Torch checkpoint 使用 Python 训练状态，只加载自己训练或可信来源的文件。

## 可选训练噪声、扰动和课程

配置 `task.training`，默认全部关闭：

```json
{
  "observation_noise": false,
  "push_velocity": 0.0,
  "push_interval_s": 2.0,
  "field_warmup_steps": 0
}
```

`observation_noise=true` 启用原生 G1 actor 本体状态噪声；critic 和身体势场不加噪声。
`push_velocity>0` 使用官方 mjlab interval event，周期性随机设置根部 X/Y 速度到
`[-push_velocity,+push_velocity]` m/s；`push_interval_s` 指定周期。
`field_warmup_steps>0` 将 vMF 和沿场速度 shaping 的权重从 0 线性升到配置值，
保留存活、安全和限位项，作为先稳定再加强场引导的基础奖励课程；它不更改障碍几何。
全局 control step 随 PPO/DAgger checkpoint 保存和恢复，不因恢复而重新进入课程起点。
这些设置也进入策略契约，更改后须建立新 run。它们是工程选项，不声称复刻作者完整课程。

## 输出与调试

- PPO：`specialists/<scene>/latest.pt`、`model_<iteration>.pt`、`losses.jsonl`、TensorBoard（含目标距离和各终止标记）。
- DAgger：`generalist/latest.pt`、`round_<round>.pt`、`metrics.jsonl`。
- 评估：`evaluation.json` 含奖励、已结束 episode 数和 goal/collision/fall/invalid/timeout。
  未结束的 episode 不计成功率；若没有结束 episode，相应 rate 为 null。
  这些独立终止标记可以同时为真，成功要求无碰撞、跌倒或无效场。

`evaluate` 对学生只调用学生；加载单个 PPO 教师评估时用 `--scene` 选择其场景。
默认随机目标和起点、固定 seed=42，支持 `--seed`。

```bash
python -m pdb -m embodied_learning.cli.field_training ppo \
  --output-dir runs/g1-debug --scene wall_detour --num-envs 2 --iterations 1
```

先运行 `prepare --output-dir runs/g1-debug`。常用入口：
`simulators/mjlab/field_traversal.py:make_field_env`、`tasks/field_traversal.py:FieldTaskState`、
`algorithms/ppo.py:train_specialist`、`algorithms/dagger.py:train_dagger`。

## 论文、源码和迁移差异

依据 [论文 arXiv:2601.16035v2](https://arxiv.org/html/2601.16035v2)，
公式 1--5 用于势场及身体优先级，公式 6 用于多点观测，公式 7--10 用于 vMF log-likelihood，
III-B2 给出 PPO specialists → DAgger generalist。身体点参考作者
[固定提交 866ba392](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/tree/866ba392f1c1e84b92ad75fa66550f26e8af8e48)。

- 论文 K=13，公开代码为 11 点。本项目使用头、骨盆、躯干、双脚、双手、双膝、双肩共 11 点，
  不补造未定义点。新增 site 偏移来自作者 `g1_mjx_feetonly_torque.xml`：头相对 torso z=0.4，
  膝点位于 knee_link 原点，肩点位于 shoulder_pitch_link 原点。
- G1 使用 mjlab 原生 29 关节位置控制、原生 PD 和 action scale；作者动作布局/力矩控制不同。
  checkpoint 记录有序 action_names、joint_names、尺度、默认偏移和动作裁剪 ±5。
- actor/critic 当前均为 173 维：96 维本体状态/上一动作，33 维 yaw-local 势场，
  33 维身体点速度，11 维障碍距离。使用全部身体点，不把势场转换成根部速度命令。
- 默认方向项严格计算 raw `sum(log C3 + kappa * dot(mu, velocity_unit))`。
  零速度的方向点积为零，不被当作正确对齐。稳定小/大 kappa 归一化避免 NaN。
- 基础行走 shaping 是工程配置，并非作者奖励的逐项复刻：根部沿引导的速度、存活、目标到达、
  障碍接触、跌倒、无效场、轻微倾斜、关节力平方代价、关节限位和动作平滑。
  默认存活项 30 抵消 11 点均匀密度的负基线，避免仅因每步负奖励而偏好提前结束。
  无强制站立姿态奖励，允许下蹲；参数全部位于配置 `task.field` 与 `task.rewards`。
  原生奖励整体按 control dt 缩放；goal 判定为根部 XY 距离 <0.1m，跌倒为倾角 >70° 或根高 <0.35m。
- SDF 小距离用 `min_distance=0.02m` 切线延拓；优先级指数限制在 [-50,50]。
  空场 sdf=inf，仅公式计算用显式 1m 距离以保留吸引，返回 sdf 仍为 inf。
  观测 sdf 裁剪到 [-1,10]m；无效/不可达身体点终止 episode，不能用它获得方向奖励。
- 已知几何、CPU 离线场和 GPU 查询替代原论文实际传感器/定位流程；本实现不集成 Fast-LIO2 或 OctoMap。
  不使用原论文规模的 3D-FRONT 数据、不复现成功率数字。
- 每个 batch slot 具有独立 scene_id、goal_id 和平移原点；mocap 盒体选择当前场景，
  非当前场景盒体放至 z=-100m，避免共享模型限制。场求解不在训练 step 内执行。

来源与许可见 [third_party/NOTICE.md](../third_party/NOTICE.md)；数值模块说明见
[势场说明](potential_fields.md)。通用算法不导入任务、机器人、感知或仿真器，CLI 负责组装。
