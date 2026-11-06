# Click-and-Traverse 势场：源码与论文对应

本项目将势场作为 `perception` 的另一种表示，与 ADAPT 双投影并列。
双投影把点云变成二维图；势场从三维环境地图计算身体关键点附近的距离和运动提示。

## 来源与版本

- 笔记对应论文 [Collision-Free Humanoid Traversal in Cluttered Indoor Scenes](https://arxiv.org/abs/2601.16035v2)，公式参考 v2。
- [官方场构建代码 pf_modular.py](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/procedural_obstacle_generation/pf_modular.py)。
- [官方环境场查询 env_cat.py](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/cat_ppo/envs/g1/env_cat.py)。
- 固定参考 commit：`866ba392f1c1e84b92ad75fa66550f26e8af8e48`。
- 上游 Apache-2.0 许可证随项目保存在 [third_party/licenses](../third_party/licenses/Click-and-Traverse-LICENSE)。

`voxel_field.py` 改编场构建算法，增加有效性掩码、输入检查和 Torch 批量查询，删除绘图与上游配置依赖。
论文公式独立实现于 `potential_field.py`。这不是完整训练系统或论文指标的复现。

## 官方代码怎样做

1. `occupied[X,Y,Z]` 中 True 是禁止接触的占据区域。
2. `skfmm.distance` 对占据/空闲之间的隐式零面求 SDF：外面正、里面负，单位 m。
3. 在目标附近放一个小球种子，把障碍屏蔽，再通过 Fast Marching 求绕障距离 `T`。
4. 远离障碍时使用 `g=-grad(T)`；靠近障碍时逐渐使用切向分量。

设 `n` 为 SDF 梯度的单位方向，`r` 为影响半径：

```text
g_tangent = g - dot(g, n) * n
t = clip(max(sdf, 0) / r, 0, 1)
w = 1 - t²(3 - 2t)
g_mixed = (1 - w) * g + w * g_tangent
```

障碍内部使用向外法向。最终归一化并乘速度，速度在目标 XY 距离小于 0.3 m 时按三次幂降低。
这是官方 `make_guidance_field_progressive` 的工程方案，不能说成单纯的 `-grad(U_att+U_rep)`。
SDF 原始梯度用于论文链式法则，单位法向用于官方切向投影，二者没有混用。

本项目默认：目标球半径为 1.2 个栅格、投影影响半径为 5 个栅格、远处速度为 0.6 m/s。
目标球按栅格大小设置，区别于上游默认固定 0.12 m；这些参数都可以显式修改。
球内 `T` 的负值被截为零，因此它表示到目标球边界的距离，而不是精确到目标点的距离。

## 论文公式怎样做

`paper_potential` 接收已求出的 `T`、`grad(T)`、`d` 和 `grad(d)`：

```text
U_att = eta * T
U_rep = xi/2 * (1/d - 1/d0)²   （0 < d <= d0）
F = -eta * grad(T) + xi * (1/d - 1/d0) / d² * grad(d)
```

`d > d0` 时排斥势和排斥方向均为零。
论文在 `d=0` 有奇点，本实现从 `min_distance`（默认 0.02 m）起用势函数的切线向内线性延拓。
这样势能和引导向量都有限，穿入障碍后仍向外，且引导保持为该延拓势函数的负梯度。
这项正则化属于项目的工程选择。

`prioritize_guidance` 对身体部位加权：

```text
urgency = lambda * max(-dot(grad(d), velocity), 0.5) * exp(-d / distance_scale)
F_H = base_priority * urgency * normalize(F)
```

默认 `distance_scale=1 m`，指数保护范围 `[-50,50]`；`F=0` 时输出零。
基础权重由调用者传入，如根部 1、其他部位 0.5。`K` 可以是 13，也可以按机器人实际配置。
这是身体运动的观测/奖励提示，不是直接施加的物理力或可执行的全身动作。

## 地图与坐标约定

- 标量图 `[X,Y,Z]`，向量图 `[X,Y,Z,3]`；注意不是图像常用的 ZYX 顺序。
- 原点是索引 `[0,0,0]` 的栅格顶点位置，索引 `[i,j,k]` 的位置是 `origin + voxel_size*[i,j,k]`。
- 目标、关键点位置、场梯度、关键点速度必须在同一坐标系，位置单位 m、速度单位 m/s。
- 场与公式仅接受 `float32` 或 `float64`；半精度会造成近障碍公式溢出和几何边界精度损失，不支持直接以 `float16`/`bfloat16` 查询。策略网络如需半精度，应在处理有效掩码后转换观测。
- `sample` 输入 `[num_envs,K,3]`；标量与掩码输出 `[num_envs,K]`，方向输出 `[num_envs,K,3]`。
- 一个场对象表示一张地图，多个 batch 可以在同一地图查询。不同环境的地图不同，应分别建场；当前没有打包不同地图的批量构建器。
- 地面是否占据取决于接触语义：给脚的支撑面与给头部的禁止接触面可能不同。

## 未知区域、不可达与插值

三线性查询要求八个角点都满足相应条件；地图外的点不会被夹到边界冒充有效观测。

| 掩码 | 意义 |
| --- | --- |
| `distance_valid` | 八个角点都在已知区域，可使用 SDF 和梯度 |
| `geodesic_valid` | 八个角点都是可达的已知自由空间，可使用目标距离和梯度 |
| `guidance_valid` | 八个角点都是可达自由空间或已知占据，可使用引导向量 |

没有障碍的完全已知地图中，SDF 是 `+inf`，距离有效，法向为零。
论文公式接口要求有限输入；这种情况下只保留吸引项，或在调用前将 SDF 设为 `d0`、法向设零。
未知/越界输出用零填充，必须连同掩码使用；不可达位置的地图目标距离保留 `+inf`，查询值则填零并置 `geodesic_valid=False`。

与上游不同，本实现没有用最大有限距离填满不可达区域；目标距离梯度仅使用可达邻居的一侧/中心差分，避免障碍填充值产生错误吸引方向。
目标球若隔墙覆盖其他连通分量，只在目标所在的连通分量播种。

`known_mask=None` 表示调用者已提供完整地图，适合仿真真值和人工场景。
真实深度/射线数据需要先建立地图，只有有依据的空间才可设为已知：

```text
传感器 -> 统一坐标系 -> 占据图 + known_mask -> 建场 -> 身体关键点查询
```

未知区域阻止路径传播，并保守地视作 SDF 的阻挡边界；它们自身不能当已知障碍查询。
点云只提供表面点，不能自动确定障碍内部与所有空闲空间。
现在可以使用 `OccupancyMapper` 进行射线空闲清空和多帧占据融合，见 [传感器建图说明](sensor_mapping.md)。
建图器与势场构建器相互独立，调用者显式传入占据/已知掩码并决定重建时机；尚未接入训练任务。

## 安装与运行

```bash
# 使用你的 mjlab 环境，从项目根目录运行。
/home/sx/miniconda3/envs/mjlab/bin/python -m pip install -e '.[potential-fields]'
PYTHONPATH=src /home/sx/miniconda3/envs/mjlab/bin/python examples/potential_field_demo.py
```

`build_voxel_field` 使用 CPU NumPy/scikit-fmm/SciPy，适合地图初始化或更新时建场。
`field.to(device, dtype=...)` 将地图一次搬到 Torch CPU/CUDA，再重复 `sample`，保持数据类型和设备一致。
该建场器不是实时 GPU SLAM 或逐步重建接口；地图分辨率与尺寸会影响时间和内存。

```python
import numpy as np
import torch

from embodied_learning.perception.voxel_field import build_voxel_field

occupied = np.zeros((31, 31, 5), dtype=bool)
occupied[15, :24, :] = True  # 墙的开口在 +Y 方向。
field = build_voxel_field(occupied, goal=(2.5, 1., .2), voxel_size=.1)
positions = torch.tensor([[[1., 1., .2]]])
sample = field.sample(positions)
observation = sample.guidance.flatten(1)
validity = sample.guidance_valid  # 策略需要使用这一掩码。
```

本次数值验证覆盖绕墙方向、断连空间、未知区域、插值轴顺序、边界和论文势能梯度。
这些测试不验证实际机器人平衡、动作可行性或论文通过率。
