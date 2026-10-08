<!--
 * @Author: sunxun sx2728977548@163.com
 * @Date: 2026-09-24 14:15:12
 * @LastEditors: sunxun sx2728977548.com
 * @LastEditTime: 2026-09-24 14:16:12
 * @FilePath: /embodied-learning/src/embodied_learning/perception/README.md
 * @Description: 
-->
# perception

目前主要是想要放点云预处理和ADAPT双投影等表示

| 表示 | 入口 | 输入 | 输出 |
| --- | --- | --- | --- |
| 深度反投影 | `depth.backproject_depth` | 深度、内参、可选相机位姿 | 带有效掩码的点云 |
| ADAPT 双投影 | `DualProjector.project` | yaw-local 点云、感知半径 | 高程图、距离图及观测掩码 |
| 通用射线 | `rays_from_depth` / `rays_from_point_cloud` | 测量、标定和传感器位姿 | 世界系起点、终点及有效/命中掩码 |
| 占据建图 | `OccupancyMapper.update` | 通用射线 | 逐环境占据、已知区域与证据快照 |
| Click-and-Traverse 势场 | `voxel_field.build_voxel_field` | 三维占据图、已知区域、目标 | 距离场、绕障距离及引导场 |
| 身体关键点查询 | `VoxelPotentialField.sample` | 同一地图坐标系的 `[B,K,3]` 位置 | 距离、梯度、引导方向及有效掩码 |
| 论文势函数与优先级 | `paper_potential` / `prioritize_guidance` | 有效场样本、速度、基础权重 | 势能与带权方向 |

学习与运行说明见 [势场说明](../../../docs/potential_fields.md)。
传感器建图见 [建图说明](../../../docs/sensor_mapping.md)，支持 Torch CPU/CUDA。
势场构建在 CPU 上进行，身体查询支持 Torch CPU/CUDA；这些通用模块均不依赖仿真器。

多身体点论文观测与稳定 vMF 奖励见 `field_observation.py`、`field_reward.py`。
G1 的已知场景 PPO/DAgger 训练见 [训练指南](../../../docs/field_training.md)，
不要求传感器建图；通用感知模块与策略、机器人和仿真器保持解耦。
