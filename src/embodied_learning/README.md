<!--
 * @Author: sunxun sx2728977548@163.com
 * @Date: 2026-09-28 19:14:23
 * @LastEditors: sunxun sx2728977548.com
 * @LastEditTime: 2026-09-28 20:11:56
 * @FilePath: /embodied-learning/src/embodied_learning/README.md
 * @Description:
-->
# 架构

先来一个整体流程

```text
mjlab 扫描传感器
    ↓
读取命中点 + 有效标记 + 机器人位姿
    ↓
把世界坐标点转换到约定的机器人局部坐标
    ↓
DualProjector.project(points, valid_mask, radius)
    ├── 高程投影 → (num_envs, 21, 17)
    └── 距离投影 → (num_envs, 13, 17)
    ↓
分别展平 → 两个 MLP
```

这里是把点云的读取放进 `simulators/mjlab/`，双投影计算留在 `perception`，这样以后更换点云来源时
可以复用投影代码

两张投影图也可以由一份点云构建，前提是点云覆盖了需要的地面和障碍，可以把两种点云生成方式（例如射线扫描和深度图反投影）

```text
src/embodied_learning/
├── simulators/mjlab/
│   ├── adapter.py
│   └── sensors.py          # 读取射线结果、深度图和传感器位姿
└── perception/
    ├── point_cloud.py      # 点云数据结构
    ├── depth.py            # 深度图反投影
    └── DualProjector.py    # 点云 → 两张投影图
```

我们是把射线扫描和深度图反投影做成两个点云来源，统一接到 `DualProjector`

 路径            原始输入                      转为点云的方法
━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 射线扫描        射线命中位置、有效标记        直接读取命中点；你已经在使用 hit_pos_w
──────────────  ────────────────────────────  ──────────────────────────────────────────────
 深度图反投影    深度图、相机内参、相机位姿    将像素还原为相机坐标中的三维点，再转换坐标系
