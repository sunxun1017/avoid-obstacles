# 从传感器测量到占据地图

模块之间只传数据：

```text
mjlab / Isaac Lab / 实机适配器：测量与位姿
             ↓
rays.py：通用 RayMeasurements
             ↓
occupancy.py：OccupancyMapper.update
             ↓
OccupancySnapshot：占据、已知区域、原点、分辨率
             ↓ 调用者显式组合
voxel_field.py：build_voxel_field → 身体关键点查询
```

`rays.py` 与 `occupancy.py` 都不导入仿真器。
建图器不调用传感器、势场构建器或策略；环境步进、更新频率和重置由应用/任务负责。
这里实现已知位姿下的占据建图，不包含定位、位姿优化或回环。

## 输入：射线比点云多了哪些信息

`RayMeasurements` 保存 `origins`、`endpoints`，形状都是 `[B,N,3]`，单位 m；
`valid_mask`、`hit_mask` 是 `[B,N]` 布尔值。
所有测量必须在稳定的 `world` 坐标系，不能直接传随机器人移动的 `yaw_local` 点。
每条射线可以有不同起点，适用于扫描时移动的传感器，但逐射线位姿校正由调用者提供。

| valid | hit | 更新方式 |
| --- | --- | --- |
| False | 任意 | 不更新，也不推断自由空间 |
| True | True | 沿途空闲，终点占据，终点之后不更新 |
| True | False | 只清空到明确给出的有效探测终点 |

非有限坐标、零长度射线也不更新。
点云里的“缺失点”不等于“该方向一路空闲”。

### 射线扫描与点云

已在世界系的表面命中点可以用：

```python
from embodied_learning.perception.rays import rays_from_point_cloud

rays = rays_from_point_cloud(world_cloud, sensor_origins)
mapper.update(rays)
```

`sensor_origins` 支持 `[B,3]` 或 `[B,N,3]`。点云有效点作为命中，无效点不会清空空间。
若扫描器提供确认未命中的射线，应直接构造 `RayMeasurements`，
把已知起点与探测终点传入并置 `valid=True, hit=False`；不能用点云无效点的零坐标替代终点。

### 深度图

```python
from embodied_learning.perception.rays import rays_from_depth

rays = rays_from_depth(
    depth, intrinsics, rotation=optical_to_world,
    position=camera_position, max_depth=5.0,
    no_hit_mask=confirmed_no_return,
)
mapper.update(rays)
```

深度、内参和位姿约定沿用 `backproject_depth`：像素中心、光轴 Z 深度、光学系 X 右 Y 下 Z 前。
`max_depth` 是光轴 Z 上限，不是每条斜射线的球形量程。
有限正深度超过上限时，射线只清空到上限，不在那里虚构障碍。
没有显式 `no_hit_mask` 的 NaN/inf/非正深度不更新；
只有传感器确认的未命中才可用 `no_hit_mask=True` 清空到上限。
此掩码必须按具体传感器的背景值定义生成。通用层不认识 mjlab 的背景值。

## 地图状态与坐标

初始化时必须指定环境数、XYZ 尺寸、体素间距，可以给每个环境单独的地图原点：

```python
from embodied_learning.perception.occupancy import OccupancyMapper

mapper = OccupancyMapper(
    num_envs=1, shape=(41, 41, 31), voxel_size=0.1,
    origin=(-2.0, -2.0, 0.0), device="cuda:0",
)
```

每个环境有独立的 `[X,Y,Z]` 网格；原点支持 `[3]` 或 `[B,3]`。
索引 `[i,j,k]` 的**体素中心**位于 `origin + voxel_size*[i,j,k]`，
与势场查询节点坐标相同；体素边界是中心两侧各半个间距。
整个地图的下边界是 `origin-voxel_size/2`，上边界不包含在地图中。
射线只更新与线段有正长度交集的体素；只接触边/角的格子不作为已探测自由空间。
DDA 每次从边界重新计算绝对穿越时刻，避免增量累加漂移；射线参数 `[0,1]` 上使用 `8*eps(dtype)` 的容差。
容差内的短片段保守地不清空，近同时穿越视为同一次边角穿越。

建图器把射线裁到地图包围盒：传感器可以位于地图外，地图外的命中不会添加到边缘格子。
地图是固定尺寸、固定原点；没有滚动窗口或自动扩容。
输入和地图使用相同的设备及 `float32`/`float64` 精度，不隐式搬运到 CPU。

## 多帧融合、未知与冲突

射线使用三维 DDA 遍历网格；占据证据用 log-odds 累积。
初始化 `L=0`、`observed=False`；默认：

| 参数 | 值 |
| --- | --- |
| 每次命中证据 | +0.85 |
| 每次空闲证据 | -0.4 |
| 截断范围 | [-2.0, 3.5] |
| 空闲分类 | L <= -0.2 |
| 占据分类 | L >= +0.2 |

阈值之间是“已观测但不确定”，仍不向路径规划器承诺自由空间。
这些参数是项目工程默认值，可配置，并非传感器标定结果。
同一 `update` 代表一个扫描周期：每个体素仅加一次命中或空闲证据，命中优先。
这与 [OctoMap 官方测量更新说明](https://octomap.github.io/octomap/doc/classoctomap_1_1OccupancyOcTreeBase.html) 的去重/冲突处理原则一致；这里使用自行实现的稠密 Torch 网格，不复制 OctoMap 源码。

连续帧可以改变旧判断。例如旧障碍位置后来被清空射线穿过，空闲证据会逐步抵消命中证据。
没有时间衰减、移动物体跟踪或自动遗忘，未再次观测的旧证据会保留。
也没有地面、自身机器人或语义过滤；允许接触的地面应由任务选择合适的障碍建模方式。

`snapshot()` 返回独立数据：

- `log_odds[B,X,Y,Z]`：累计证据。
- `observed[B,X,Y,Z]`：曾接受命中/空闲证据。
- `occupied[B,X,Y,Z]`：已观测且达到占据阈值。
- `known_mask[B,X,Y,Z]`：已观测且可以分类为空闲或占据。
- `origin[B,3]`、`voxel_size`：地图元数据。

无观测与证据不确定的位置都满足 `known_mask=False`；`occupied=False` 不能单独证明自由空间。
修改快照不会改变实时地图。每个 episode 结束时应调用 `reset`：

```python
mapper.reset()  # 全部环境。
mapper.reset(env_ids)  # 指定环境，int64 [M]，在地图设备上。
```

重置不修改原点，重定位和地图生命周期由调用者决定。

## 显式连接势场

```python
from embodied_learning.perception.voxel_field import build_voxel_field

snapshot = mapper.snapshot()
env_index = 0
field = build_voxel_field(
    snapshot.occupied[env_index].cpu().numpy(),
    known_mask=snapshot.known_mask[env_index].cpu().numpy(),
    origin=tuple(snapshot.origin[env_index].cpu().tolist()),
    voxel_size=snapshot.voxel_size,
    goal=goal_in_world,
).to(mapper.device)
```

势场构建是 CPU Fast Marching，调用者显式选择何时重建和数据搬运。
不同环境分别构建自己的场；旧场是旧地图的快照，不会随 `mapper.update` 自动改变。
目标必须在已知空闲区域，否则现有建场器会拒绝；不要把未知空间强行标为已知来让它运行。
稀疏射线可能产生较稀疏的已知区域；这里没有膨胀自由空间来补洞。
占据栅格中心作为势场采样节点，SDF 边界仍是离散占据/自由样本之间的隐式零面，精度受分辨率限制。

## 运行示例与验证

从项目根目录运行，无仿真器示例用两个不同位姿观察同一个平面：

只使用射线与建图模块可安装 `pip install -e '.[mapping]'`，只依赖 Torch。
以下连接势场的示例还需要 `pip install -e '.[potential-fields]'` 中的 CPU 求解依赖。

```bash
PYTHONPATH=src /home/sx/miniconda3/envs/mjlab/bin/python examples/depth_mapping_demo.py
PYTHONPATH=src /home/sx/miniconda3/envs/mjlab/bin/python examples/depth_mapping_demo.py --device cuda
```

真实 mjlab 摄像机示例：

```bash
PYTHONPATH=src /home/sx/miniconda3/envs/mjlab/bin/python examples/mjlab_depth_mapping.py
```

它使用固定向下相机，查询机器人上方已观测空气中的测试点，验证真实深度与地图的几何一致性；不是行走策略。
mjlab/MuJoCo Warp 的零深度未命中约定由示例转换为 `no_hit_mask`，不放进通用建图器。

测试覆盖命中与遮挡、确认未命中、无效测量、去重、命中优先、多帧不确定状态、射线裁剪、负向/对角遍历、单环境重置、快照隔离及 CUDA 一致性。
另用独立的逐体素线段/AABB 交集算法核对随机射线的 DDA 结果，并覆盖精确边角接触与密集深度 CPU/CUDA 一致性。
