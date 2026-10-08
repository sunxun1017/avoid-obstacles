# G1Cat 公开源码版本训练

本入口迁移 Click-and-Traverse 固定提交 `866ba392f1c1e84b92ad75fa66550f26e8af8e48` 的 **G1Cat**，包含 env_loco 继承方法和 train_ppo 的势场权重覆盖。它独立于旧的 `g1_field.json` 论文公式实验；旧 checkpoint 不能作为新任务的教师或恢复点。

这仍然是在已知盒体场景上做离线势场训练。没有接入相机、在线建图、Fast-LIO2、OctoMap 或 AMP。mjlab 的 G1 模型和 RSL-RL 的优化器与作者 MJX/Brax 实现不同，不能声称跨后端轨迹或训练结果一致。

## 入口与配置

入口是 `src/embodied_learning/cli/field_training.py`。新配置 `configs/experiments/g1_cat_source.json` 选择 `cat_source_v1`，默认输出 `runs/g1-cat-source`。不指定 `--config` 仍保留旧实验默认值，所以以下命令都显式选择新配置。

| 数据/控制 | source task |
|---|---|
| actor / critic / action | 162 / 250 / 12 |
| 物理关节 / 观测关节 / 身体点 | 29 / 23 / 11，按名字映射 |
| actor / critic 网络 | 256,128,64 / 512,256,128，ELU |
| 观测归一化 | 关闭 |
| 控制 / 仿真周期 | 0.02 / 0.002 秒 |
| 动作 | 12 个腿关节目标增量，上帧目标 + 0.5 × action，再裁剪 |
| 执行 | 每个物理步显式 PD、随机 RFI、源码力矩限制；上身保持默认姿态 |
| PPO | lr=3e-4、gamma=.97、rollout=20、epochs=4 |
| 步态/扰动 | 频率 U[1.3,1.5]，两种初相位，PD U[.75,1.25]，RFI，5–10秒 additive push |
| 场网格 | 0.04 米，goal_radius=.12、max_speed=.6、5 voxel 投影 |

配置的 `task.ground/lateral/overhead` 默认都是 1，对应作者 train_ppo 覆盖。可以复制配置并设为 0/0/0 进行无身体势场项的行走预训练；root tracking 和基本运动奖励仍存在。修改配置应使用新的输出目录及相应检查点契约。

`source_semantics` 是记录固定实现常量的说明字段，不是调参接口。可以调节的任务项是 `task`，优化项是 `ppo`。场景体素精度由每个场景 `voxel_size` 指定。服务器显存不够时先减小 `--num-envs`。

## 服务器环境与第 5 卡

不要卸载服务器其他项目的 RSL-RL。此前同名包实际来自 AMP 的 editable 目录，虽然版本元数据相同却没有 `rsl_rl.models`。先激活 mjlab，再在项目中建立独立环境：

```bash
conda activate mjlab
cd ~/embodied-learning
python -m venv --system-site-packages .venv-field
source .venv-field/bin/activate
python -m pip install --ignore-installed --no-deps 'rsl-rl-lib==5.4.2'
python -m pip install -e '.[field-training]'
python -c 'import sys, rsl_rl; from rsl_rl.models import MLPModel; print(sys.executable); print(rsl_rl.__file__)'
```

最后一行必须能导入 `MLPModel`，且 RSL 路径应来自 `.venv-field`。仅查看 `pip show` 的版本不够。若 scikit-fmm 构建失败，需要看错误前面的 Meson 编译器报错；后续 `No module named skfmm` 是安装失败的结果。

按你之前给的服务器设备，第 5 卡 UUID 是 `GPU-de0e690a-9b65-1acb-6bfe-fccce34ed281`。UUID 可避免 CUDA 编号排序与 nvidia-smi 序号不同导致选错卡。先核对当前机器：

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv
export CUDA_VISIBLE_DEVICES=GPU-de0e690a-9b65-1acb-6bfe-fccce34ed281
python -c 'import torch; print(torch.cuda.get_device_name(0))'
```

随后程序中的 `cuda:0` 指向这个可见 GPU。不会启动五张卡。

## 准备、训练和恢复

```bash
python -m embodied_learning.cli.field_training prepare \
  --config configs/experiments/g1_cat_source.json

for scene in wall_detour passage hurdle_crouch; do
  python -m embodied_learning.cli.field_training ppo \
    --config configs/experiments/g1_cat_source.json \
    --scene "$scene" --device cuda:0 \
    --num-envs 1024 --iterations 30000 || break
done
```

`prepare` 在 CPU 上构造每个场景每个目标的源版 SDF/BF/GF，缓存有场景哈希及 source_profile。地图和场版本不匹配会拒绝加载。默认场景是本项目的绕墙、窄通道、横梁与矮障碍，不是作者全部训练地形。

先用 `--num-envs 4 --iterations 2 --rollout-steps 4 --epochs 1 --mini-batches 1` 验证服务器管线，再开启长训练。短测试和正式训练参数不同，应使用独立 `--output-dir`，不能将短测试目录直接用于正式恢复。

正常训练恢复保留原先配置，`--iterations` 表示额外更新次数：

```bash
python -m embodied_learning.cli.field_training ppo \
  --config configs/experiments/g1_cat_source.json \
  --scene wall_detour --device cuda:0 --num-envs 1024 --iterations 10000 \
  --resume runs/g1-cat-source/specialists/wall_detour/latest.pt
```

训练会写入 `teachers.json`。三个教师准备好后：

```bash
python -m embodied_learning.cli.field_training dagger \
  --config configs/experiments/g1_cat_source.json \
  --device cuda:0 --num-envs 1024 --iterations 1000
```

DAgger 是本项目的 PyTorch 数据聚合与监督优化实现；教师只进行确定性推理，学生不使用 critic 的特权观测作为网络输入。

## 播放和评估

本地有显示窗口时：

```bash
python -m embodied_learning.cli.field_training play \
  --config configs/experiments/g1_cat_source.json \
  --checkpoint runs/g1-cat-source/specialists/wall_detour/latest.pt \
  --scene wall_detour --device cuda:0 --num-envs 1 --viewer native
```

无显示服务器用 `--viewer viser --port 8765`。Viser 只绑定 `127.0.0.1`。在本地另开终端：

```bash
ssh -N -L 8765:127.0.0.1:8765 zjut
```

打开 `http://127.0.0.1:8765`，Ctrl+C 退出播放。`--steps 500` 可做有限步播放测试。中间 `model_*.pt` 和 `latest.pt` 都可加载，不需要等待训练完成。学生播放只需将 checkpoint 换为 `runs/g1-cat-source/generalist/latest.pt`，可多次传 `--scene`。

```bash
python -m embodied_learning.cli.field_training evaluate \
  --config configs/experiments/g1_cat_source.json \
  --checkpoint runs/g1-cat-source/generalist/latest.pt \
  --device cuda:0 --num-envs 128 --steps 2000 --no-perturbations
```

`--no-perturbations` 在 play/evaluate 中关闭观测噪声、push、PD/RFI 随机化，不改变检查点的训练契约。evaluate 在输出中记录 override。评估记录已完成回合的 ever-goal、最终/最小 XY 距离以及逐步停步比例。未完成任何回合时回合成功率为 null，不能把逐帧 goal 布尔值当成成功率。

源码的停止指令阈值是 0.2；额外评估 goal 判据是 XY 距离 < 0.1 米。策略可能提前停止而达不到该判据，goal 不是训练奖励也不终止回合。

## 源码对应关系与时序

| 作者函数/数据 | 本项目迁移 |
|---|---|
| env_cat / env_loco 的 `_get_reward` 及各子函数 | `tasks/cat_reward.py` |
| `compute_cmd_from_rtf` | `tasks/cat_command.py` |
| `_update_phase` | `tasks/cat_gait.py` |
| `_get_obs` | `tasks/cat_observation.py` |
| constants 与 `torque_step` | `robots/g1.py`、`simulators/mjlab/cat_control.py` |
| step/reset/终止与历史 | `tasks/cat_traversal.py`、mjlab `cat_traversal.py` |
| pf_modular / sample_field | `perception/voxel_field.py` 的 source 函数 |

总奖励是 **clip(sum(源码权重 × 原值) × 0.02, 0, 10000)**，没有论文 log-vMF、额外 alive/goal 奖励。基础项包括 root 速度指令跟踪、姿态、步态、脚部滑动/高度、动作/力矩/关节速度和平衡等。日志 `CatReward/*` 是原值，`CatRewardDiagnostic/nonfinite_terms` 记录源码 NaN 清理前的非有限项数，`CatWeighted/*` 是加权值，`Cat/clipped_reward` 是最终值。

保留作者真实动作历史更新：先 last_last_act←last_act，再 last_act←当前 action，然后计算奖励；因此 smoothness 的一阶动作差为零。没有擅自使用源码里未被调用的 far_baseline 变体。

观测每五步更新延迟根位姿；当前身体点通过延迟位姿变换查询场。reset 观测不推进相位，同一控制步的多次读不推进状态。奖励和终止读取复位前状态，回合结束后 reset 对应槽位，日志保留终止步结果。

源码碰撞使用身体点负 SDF 与指定自接触，并有前 50 步豁免；物理盒体接触额外记录为诊断。跌倒、非法状态、1000 步 timeout 终止，goal 不终止。mjlab 使用 nconmax=128/njmax=512，以避免单环境播放的自动容量不足。

源版采样保留作者 X/Z corner-weight 排列与索引裁剪行为，即使它不同于标准三线性插值。空障碍地图以 +inf SDF、零 BF 处理，属于明确的工程扩展。旧论文场采样函数保持原语义，不能与 source 缓存混用。

## 验证与限制

开发期 `tools/generate_cat_reference.py` 直接抽取并执行固定提交的作者函数，生成奖励、指令、60+步停步过程、观测和场参考；fixture 记录文件 SHA256。运行时不依赖 JAX。原作者 Apache-2.0 许可保留在 `third_party/licenses/Click-and-Traverse-LICENSE`。

本机 CUDA 完整回归 159 项通过，包含三个场景各两次 PPO 更新及恢复、DAgger 两轮及恢复、教师不变、162/250 观测对照、部分复位和终止奖励。正式 0.04 米网格 prepare、两次 PPO、单环境 Viser 500 步及 HTTP200 已验证。已有 HFIELD/CAPSULE MULTICCD 警告与基线相同。

这些证据只证明数值对照和训练/播放管线可运行，**不证明 G1 已经学会绕障或达到论文成功率**。模型接触几何、质量/惯量、后端积分/约束求解、PPO minibatch 和 DAgger 优化方式仍存在差异。

非法位姿仅在场索引路径使用有限占位值，原始运动学仍用于奖励诊断，invalid 标志终止对应槽位；未知终止距离不计入平均值。prepare 的起点/目标检查使用 source 默认关节姿态。
