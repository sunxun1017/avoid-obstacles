# CAT 单项落地速度奖励试验

2026-10-30，状态：已启动训练，尚无效果结论。旧 PD：hip pitch100/2、knee200/4、ankle30/2。

## 用户要求延长至128轮（2026-10-30）

原32轮两组已完成训练及评估，随后用户要求增加轮数。现两组均扩展为128轮，每组总计83,886,080环境转移；从第32轮检查点完整恢复Actor、Critic和Adam，按32轮一段续训到64/96/128，每段分别评估256局成功率及32局500Hz冲击。保留所有中间模型和原32轮结果。

为保持严格续训校验，原protocol.json及其SHA不改写；新增extended-budget.json是预算扩展的依据。根目录driver-status.json和各组extension-status.json是当前128轮任务状态；原status.json只代表32轮已完成。最终结果为extended-results-summary.json及根目录extended-comparison.json。不自动晋级。

32轮阶段性数据：baseline成功216/256=84.375%，landing010成功220/256=85.9375%；越障窗口内逐局单脚峰值均值8.463BW→6.094BW，越障后1秒角速度1.173→1.163rad/s。峰值有下降迹象，越障后摆动尚无明显改善；不把小幅成功率差异当成已验证优势。

## 唯一干预

协议可选项 `foot_landing_speed`，默认不存在、历史奖励与契约保持不变。首轮参数由实施者选定，尚未验证为最优：

```json
{"version":"sole_descent_substep_v1","height_m":0.05,"speed_m_s":0.5,"weight":0.1}
```

每脚正代价为 `clamp(1-sole_height/0.05,0,1)*relu(-foot_site_vz-0.5)^2`，两脚求和，权重-0.1。足底高度来自已启用原生球/胶囊碰撞几何的最低点，考虑完整几何旋转与环境地面高度。当前只支持已验证的零高度平地任务。上抬、离地>=5cm、下落速度<=0.5m/s时均为0；不依赖步态掩码，包含相位0。

在每次物理步之后读取派生状态、按2ms积分，再除以20ms控制周期得到奖励率。已有奖励聚合再乘控制dt，得到精确的一次积分代价。控制开始清积分，单个环境重置只清对应条目，重复刷新不重复计费。没有增加物理forward、随机数抽样、策略观测维度或训练传感器。

独立审查指出最初放在apply_actions前采样会读到[0,0,2,...,16]ms，重复周期起点并漏掉末尾18ms。已修正为post-sim.step，真实CPU MuJoCo回归验证[0,2,...,18]ms各一次，并验证仅最后子步触发的代价不会漏计，RED→GREEN通过。

## 公平对照

- 两组同从旧stage4最优update480继承Actor，父文件SHA256：`b127bcf7bf7613460f7252694612dfb291e4c63ace4fb292b6d8c82f11bf4a83`。
- fresh Critic、空Adam；正式Actor/Critic哈希相同。预检32环境的q/root/PD/RFI倍率哈希相同。
- 每组32updates、4096env、160controls/update，即20,971,520环境转移；只选最终第32轮，无中途挑选最优造成的选择偏差。
- baseline保持全部旧奖励；landing010只增加上述一项。旧PD、势场、噪声、RFI、推扰、当前低杆+头顶横梁难度不变。
- GPU1跑baseline，GPU5跑landing010，均使用UUID固定设备，不用GPU0。
- 基线第一次预检因跨实验Actor继承缺少显式父模型声明而终止，尚未开始训练。补齐before=None/after=None声明后重启，保留失败记录；未修改任何基线奖励或模型参数。
- 新奖励预检确认最大控制周期代价率11.409，1949个slot-control代价非零，步态过渡相位累计正代价15.727。契约除该奖励项外逐项相同。

## 自动评估与数据

训练完成后，每组自动运行：

1. 256局完整扰动成功率评估，种子20261030。
2. 32局首回合500Hz触地力、机身和固定前4环境关节记录，种子20261026；输出落脚峰值、50ms冲量、入地速度、两脚点越过低杆后的1秒角速度，以及终点阶段运动。

按无碰撞成功率、障碍接触、冲击和越障后稳定性共同判断，不自动升级难度或替换旧最优模型。训练会独立于聊天持续运行并完成评估。

远程根目录：`/home/sunxun/embodied-learning/runs/cat-landing-speed-stage4-pilot-20261030-01`。

每组记录：`status.json`、`preflight.json`、`train/losses.jsonl`、`train/latest.pt`、`evaluation-seed20261030/evaluation.json`、`impact-evaluation/analysis.json`、`results-summary.json`。完整源码清单逐文件校验，每个子任务前核对哈希。

## 整合与验证

开关已接入项目通用训练入口和严格Actor-only迁移，完整PPO续训不能借此绕过契约。用户原有未提交的g1_cat_source.yaml修改保留。测试包括几何最低点、下降/高度门限、子步积分、相位0奖励接线、局部重置、无默认变更、无相关变化迁移及严格完整续训。主项目最终全量测试结果和提交信息以本次最终答复为准。
