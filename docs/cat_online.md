# CAT 在线点云与势场

目标是 Click-and-Traverse（CAT）。在线入口由真实的仿真传感器回波建立地图，
不读取 `fields/manifest.json`，也不调用场景盒子的体素化函数生成感知地图。
场景几何仍用于仿真物理和传感器成像；导航目标来自配置。

训练和在线感知解耦：原 CAT PPO/DAgger 继续使用预计算场；在线环境只用于推理，
训练器会拒绝它。没有增加数万环境同时建图的假设，也没有改变原训练接口。
作者训练源码也读取预计算的 `sdf.npy / bf.npy / gf.npy`；源码依据和训练命令见
[公开源码训练](cat_source_training.md)。这与部署阶段通过点云在线建图是两个阶段。

## 运行

使用已有的 `field-training` 环境；这里验证过的 Python 是
`/home/sx/miniconda3/envs/mjlab/bin/python`。从项目根目录执行：

```bash
# 无需 prepare，无需策略文件：只验证传感器/建图/观测，零动作不代表导航成功。
python -m embodied_learning.cli.cat_online --sensor lidar --smoke --steps 150
python -m embodied_learning.cli.cat_online --sensor depth --smoke --steps 150

# 使用本项目 cat_source_v1 训练出的 PPO 或 DAgger 检查点。
python -m embodied_learning.cli.cat_online \
  --sensor lidar --checkpoint runs/g1-cat-source/generalist/latest.pt \
  --scene wall_detour --viewer native --steps 1000
```

默认配置为 `configs/experiments/g1_cat_source.yaml`，默认一个环境、CUDA、无窗口。
可以用 `--sensor depth` 切换来源，或用 `--viewer viser --port 8765` 打开本地网页视图。
CLI 默认 `--execution async --field-backend cuda --mapping-backend warp`。
`--update-period 0.1` 按单调墙钟限制实际扫描采集，正常情况下每 0.1 秒采集并尝试提交一次，后台进程处理最新请求；
无窗口控制循环以 0.02 秒为目标周期，并报告错过周期次数。异步不等于硬实时。
`--max-age 0.3` 从**原始传感器采集时间**计算有效期，建图完成不会重置年龄。
结束时输出有效射线数、测量/占据体素数、场版本、真正重建次数、丢弃的旧扫描、
场年龄、后台耗时和控制耗时。首次 CUDA 初始化期间使用原来的停止默认值。

需要作者 CPU FMM 基准时显式选择：

```bash
python -m embodied_learning.cli.cat_online --sensor lidar --smoke --steps 6 \
  --execution sync --field-backend source --mapping-backend torch
```

Python 工厂 `make_cat_online_env` 仍默认 `sync/source/torch`，保持原调用兼容；
使用 `execution="async", field_backend="cuda", mapping_backend="warp"` 启用新路径。
同步路径沿用仿真时钟，异步路径使用墙钟；两者的周期/年龄不能混着解释。

CAT 的奖励/终止读取和 mjlab 步末读取共用同一采集调度，同一状态不重复渲染或射线扫描。
物理前向计算与接触检查仍每个控制步执行。重置或目标变化会立即补采一次；
原生传感器按整批执行，但只有到期/变化槽位的测量进入地图，其他槽位保持原周期。
采集时间戳保留在提交和发布过程中；后台忙导致丢帧不会让采集退化为每个控制步执行。

`--checkpoint` 与 `--smoke` 必选其一。Smoke 使用零潜动作，只验证链路；没有提供
训练好的策略，也没有通过短 smoke 测试验证避障成功率。

在线加载只放行“离线场 → 测量场”和传感器附件的显式变化。机器人、物理参数、
场景哈希、身体点、关节/动作顺序、162/250 维观测仍严格比较。为避免相机改变
编译模型哈希导致误判，额外编译不含新增传感器的基准模型进行检查。检查通过后，
推理关闭观测噪声、增益/力矩扰动和推扰，并使用检查点保存的观测归一化设置及统计量。原训练/恢复加载器没有放宽。
本入口不直接加载作者硬件部署 ONNX。

## 模块边界

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 传感器适配 | `simulators/mjlab/cat_sensors.py` | 深度与标定/位姿、球面 LiDAR 回波 → 世界坐标射线 |
| 点云射线 | `perception/rays.py` | 区分命中、确认无回波、无效数据；两种输入共用 `RayMeasurements` |
| 占据融合 | `perception/occupancy.py`、`occupancy_warp.py` | Torch 参考遍历或 Warp 每束并行 DDA；log-odds、已测量掩码 |
| CAT 数值 | `perception/cat_deployment.py` | 作者占据预处理和 SDF/BF/GF 建场 |
| GPU 势场 | `perception/eikonal_warp.py`、`cat_gpu.py` | CUDA Eikonal 迭代、源版 CAT 混合与速度曲线 |
| 异步进程 | `perception/field_worker.py`、`async_field.py` | 有界共享内存、最新扫描、独立复位/目标代次、非阻塞消费 |
| 生命周期 | `perception/online_field.py` | 更新节流、目标变化、过期、独立复位、场查询 |
| 任务适配 | `tasks/cat_online.py` | 稳定场景坐标、CAT 查询接口、停止默认值、诊断 |
| 仿真组装 | `simulators/mjlab/cat_online.py` | 注入传感器与在线场，复用 CAT 控制器和观测 |
| 推理入口 | `cli/cat_online.py`、`policies/cat_online.py` | 检查点校验、执行、可视化 |

纯 `perception` 模块不导入 mjlab、机器人、策略、ROS 或硬件 SDK。
真实设备接入可给 `rays_from_depth` 提供标定/深度/外部定位，或给
`rays_from_lidar` 提供每束的世界起点/方向/距离，再调用 `OnlineFieldMap.update`。
只有点云时可使用 `rays_from_point_cloud`；没有的点不能当作已确认自由射线。
也可以用 `publish_snapshot` 接入外部占据建图器，快照时间必须是新测量时间。

## 与作者源码的对应

固定提交：[`866ba392`](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/tree/866ba392f1c1e84b92ad75fa66550f26e8af8e48)。
参考文件：[`octomap_bridge.py`](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/deploy/scripts/exp_dis_pf/octomap_bridge.py)、
[`pf.py`](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/blob/866ba392f1c1e84b92ad75fa66550f26e8af8e48/deploy/gx_loco_deploy/policies/g1_cat/pf.py)。

- 保留 5×5×5、两次迭代闭运算，孤立体素去除，以及索引 20 以上向上填充和
  索引 0–4 向下填充；在线任务要求 0.04 m 分辨率。
- `source` 后端保留源版 FMM；两后端保留有符号距离梯度、渐进投影混合、
  最大场速度 0.6 m/s、目标种子
  半径 0.12 m；空障碍地图也保留作者的两个角落占据点。
- 源版占据场把未知区域当作非障碍。这里保留其数学语义，但单独报告测量覆盖，
  不把未观测区域标成已测量自由空间。
- 字段原点统一为首个体素中心，作者的格子边界原点需要加半格。
- 沿用项目已有的源版插值顺序，包括作者 X/Z 角点与权重排列的不一致；这是
  保持源码一致性的选择。精确网格边界处仍可能出现不连续和浮点敏感性。

与作者系统不同的部分：Torch 固定体素融合替换 OctoMap；仿真提供定位，不实现
Fast-LIO2、SLAM、旋转扫描运动补偿或 ROS 传输。深度来自真实渲染深度，LiDAR
来自瞬时 360° 射线扫描（默认 128×32，俯仰 −45° 至 45°）。自身几何先参与
遮挡，再丢弃自身回波，避免穿透身体清空后方；地面用仿真分割筛除，只清空到
实测地面，不穿透地面。硬件接入需要替换这一地面/自身分类器。

## GPU、频率与边界

后台进程拥有占据地图、形态学处理和势场构建；前台只提交测量、尝试读取已完成场、
查询身体点。使用 `spawn`，不 fork CUDA；共享内存只有一份待处理输入和一份输出。
前台对输入/输出锁只尝试获取，忙则跳过；新请求可以覆盖旧的待处理扫描。
每个环境携带独立 episode/goal 代次，旧结果不能复活 reset 前的地图或旧目标。
正常关闭、初始化失败和 worker 异常都会回收进程和共享内存。
这适合单机器人或少量在线环境，没有宣称支持数万份独立在线地图。

`source` 后端保留 scikit-fmm 作为数值基准。`cuda` 后端是自行实现的并行 Eikonal
活动集迭代，用二阶迎风差分、子网格界面初始化和 CUDA graph；**不是作者原来的
heap-FMM，也不是逐位等价实现**。每次新场重新初始化，包含移除障碍的情况；
超出迭代预算则拒绝发布。默认停止阈值为 1e-4 个体素的单调更新量，预算为
`2 * sum(shape) + 64` 次迭代；它不是对真实连续距离或源 FMM 误差的数学保证。
复杂迷宫可能超过预算，错误会进入诊断并使该环境使用停止默认值。

每次有效新扫描都更新覆盖和测量时间。只有处理后占据、目标和原点都不变时复用
已构建场；`versions` 统计成功测量发布，`build_counts` 统计实际势场构建。
缓存复用不能用来证明新地图达到 10 Hz。CPU 后端移入进程只解除控制等待，
不会把约 0.8 秒的 FMM 计算变成 0.1 秒；保持 0.3 秒有效期时慢结果仍会过期。

测量结果与复现命令见下方性能记录；CUDA 与仿真共用 GPU，控制仍有调度抖动。

同批场景需共用网格原点、形状和分辨率；每环境的地图独立。当前使用配置里的
5 米级固定范围，没有滚动窗口或自动探索。越界/被占据目标会显式使该环境的场
失效；当前位置和延迟里程计位置的查询共同决定有效性：任一身体点越界、场过期
或尚无有效地图时，actor/critic 都获得停止命令和默认场特征。
这表示导航输入停止，不是紧急制动控制器。碰撞终止使用仿真物理接触，不用不完整
感知图当作碰撞真值。

## 验证与复现

`tests/fixtures/cat_online_reference.*` 记录固定源文件 SHA256，以及从作者函数
抽取的预处理/SDF/BF/GF 数值。`tools/generate_cat_online_reference.py` 在重生成前
验证哈希，不在运行时下载或导入作者的 ROS 工程。

```bash
python -m pytest tests/test_cat_deployment.py tests/test_online_field.py \
  tests/test_cat_online_sensors.py tests/test_cat_online_policy.py \
  tests/test_cat_online_native.py tests/test_cat_sensor_schedule.py \
  tests/test_cat_online_task.py tests/test_occupancy.py -q
```

原生测试需要 CUDA；覆盖两个传感器、禁止离线场加载/体素化、单环境复位隔离和
用实际无传感器模型生成的检查点进行在线加载。其他测试覆盖源函数数值、过期、
无效/越界扫描、目标变化、射线距离/遮挡语义、训练入口拒绝在线环境。

异步/GPU 新增测试还覆盖 mailbox 忙、worker 退出、未来时间戳、旧代次发布、
源版射线遍历对照、解析平面距离、不可达区域、障碍移除和求解不收敛。
完整回归排除原有 `tests/test_scan.py`：该文件是收集时立即运行 CUDA 并写文件的演示脚本。

```bash
python -m pytest tests --ignore=tests/test_scan.py -q
python tools/benchmark_cat_online.py --sensor lidar --repeats 20 \
  --output /tmp/cat-lidar.json
python tools/benchmark_cat_online.py --sensor depth --repeats 20 \
  --output /tmp/cat-depth.json
```

基准使用 `wall_detour` 的真实仿真传感器扫描、完整 126×126×56 网格和 0.04 m
体素。静止扫描回放，目标 Y 在 ±0.02 m 之间变化，强制每次重新建场；另外单独
测同地图/同目标缓存，以及处理后地图平移一格的合成障碍变化（只计建场）。
异步测试先单独测首次初始化，再预热 5 次，以 10 Hz 提交、50 Hz 轮询 20 次。
这是单环境计算/传输基准，不是移动机器人闭环导航成功率或最坏延迟证明。

## 2026-10-15 性能记录

本机 RTX 4060 Laptop 8 GB，Python 3.11.15、Torch 2.7.1+cu126、Warp 1.16.0。
以下每项 20 次，5 次预热；CPU 和 GPU 使用相同点云与目标序列。数值为毫秒。

| 项目 | LiDAR P50 / P95 | 深度 P50 / P95 |
| --- | ---: | ---: |
| 原 Torch 融合 + CPU 源版新建场 | 885.1 / 899.2 | 885.5 / 913.7 |
| Warp 融合 + GPU 新建场 | 55.9 / 59.2 | 44.7 / 47.9 |
| GPU 改变障碍后建场（不含融合/形态学） | 38.8 / 41.2 | 29.4 / 32.3 |
| 异步后台全流程（含输入读取/结果传输） | 64.0 / 70.2 | 52.0 / 63.9 |
| 前台提交或轮询查询 | 0.73 / 12.64 | 1.07 / 10.40 |
| 场发布被读取时的测量年龄 | 90.0 / 91.6 | 68.4 / 89.0 |

两种来源均在约 2 秒计时窗口内接收 **20/20 次新场，约 10 Hz，无丢帧**。
这组受测负载达到 100 ms 新场预算。首次 worker 发布另计约 0.84 / 0.64 秒，
此外还有约 1 秒进程启动和仿真初始化；不能把预热数据解释为启动即实时。
温度、功耗、CUDA 与仿真竞争会影响结果，上表不是最坏延迟保证。

同一完整地图对照 CPU 源版：SDF 误差 P99 约 5.4 / 4.9 mm，目标距离误差 P99
均约 24.4 mm、最大 44.2 mm；非零场向量方向误差 P95 约 2.83° / 2.91°，
最大约 32.7° / 33.8°；可达掩码无差异。方向最大偏差不可忽略，尤其在梯度
切换附近，仍需用训练策略做闭环成功率验证；数值测试不替代此验证。

完整数据（含缓存单独计时和启动时间）：
[LiDAR JSON](benchmarks/cat-realtime-2026-10-15-lidar.json)、
[深度 JSON](benchmarks/cat-realtime-2026-10-15-depth.json)。

默认 LiDAR CLI 另跑 150 步 smoke：最终场有效、无 worker 错误；完整仿真单步
平均 35.35 ms，最大 85.36 ms，150 步中 123 步错过 20 ms 控制预算。
**因此当前不能宣称完整系统已达 50 Hz**：上面的 10 Hz 是建图/传输基准结果，
前台仍承担仿真、CAT 状态/观测查询及共享 GPU 的开销。Smoke 没有运行训练策略，
也没有证明闭环避障。详见 [smoke 记录](benchmarks/cat-realtime-2026-10-15-smoke.json)。

最终回归：**223 项通过（53.14 秒）**，Ruff 和空白检查通过。独立审查发现并修复了
多环境错峰提交被二次节流、初始化失败共享内存清理和过期场验收漏洞；对应测试
均先复现失败再通过。改动保留在当前工作区，未提交。
