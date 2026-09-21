# 【重要】V12 教师示范与残差 RL 训练：启动指南与实现记录

更新日期：2026-09-17。代码 schema：`can_teacher_intent_v1`。

## 1. 你下一步怎么运行

在工程根目录 PowerShell 中运行：

```powershell
& '.\RL PPO\START_TEACHER_RL.ps1' -Mode Train -Updates 100
```

该命令从 `RL PPO/teacher_runs/bc/bc_best.pt` 初始化，每次 PPO 更新采集两个完整抓抛回合，共训练至第 100 次更新。采集、BC 和短 PPO 验证由本次搭建工作先行完成，实际完成结果见本文末尾。

中断后恢复：

```powershell
& '.\RL PPO\START_TEACHER_RL.ps1' -Mode Resume -Updates 100
```

`Updates` 是最终更新编号，不是额外更新次数。恢复使用最新完整更新的网络、优化器、随机数状态和下一回合种子。中断时尚未提交的部分更新重新采集；不能在更新中间恢复 MuJoCo 状态。

完整训练输出目录是 `RL PPO/teacher_runs/ppo_full/`。它与旧 V11 的 checkpoint 不兼容。

## 2. 本次搭建的文件

| 文件 | 用途 |
|---|---|
| `teacher_rl/env.py` | 包装 CAN 教师，提供 reset/step、10 维动作意图、阶段掩码、实际落地判定和物理子步统计 |
| `teacher_rl/model.py` | 观测 Actor、privileged Critic、训练集归一化、原子 checkpoint |
| `teacher_rl/data.py` | 教师与 DAgger 采集、原子 NPZ/JSON、整回合划分、阶段平衡和质量权重 |
| `teacher_rl/__main__.py` | collect / bc / dagger / evaluate / ppo 命令 |
| `START_TEACHER_RL.ps1` | 从任意工作目录启动的 PowerShell 入口 |
| `tests/test_teacher_rl.py` | 动作边界、Actor 真值隔离、checkpoint、PPO 和环境测试 |

教师源代码来自 `../umarm-mjx-rl-catch-and-place-pivot/mjx_experiments/umarm_can/`，克隆提交为 `fb12ab5638d21b33bdc51ba71484b772d669d380`。

环境包装器自动定位工程根目录并设置 `UMARM_CAN_REPO`，不需要手动配置外部导入路径。使用当前工程已有 `.venv`，无需 GPU 或 JAX。Torch 默认单线程，教师采集支持多进程。

## 3. 训练方式与边界

执行链路：

```text
带噪声和延迟的传感器观测
  -> Actor 输出 10 维动作意图，30 Hz
  -> 相对当前教师意图施加有界修正
  -> 原 PID / 逆动力学 / 压力分配 / 环绕驱动，150 Hz
  -> 原 CAN 数字孪生及软 weld 抓取
  -> 实际球飞行与地面接触评分
```

BC 模仿的是高层动作意图。抓取位置标签来自在线球轨迹预测，阶段、共收缩压力和目标半径也会影响标签。该版本保留经典控制器和解析释放器；它不是脱离教师的独立端到端抓抛网络。BC 离线误差降低也不等于机器人闭环成功率提高，必须实际评估。

教师很多参数本身是常数，这些维度的模仿问题较简单。它们在 PPO 中成为可优化的参数；不能把常数模仿损失下降当成复杂运动技能已被全部学会。

状态机为 approach / capture / settle / spinup / release / flight。每个回合从接球一直运行到落地或任务超时，未设置旧版本那种长期卡在 Stage 2A 的课程升级门槛。

## 4. 默认任务

| 参数 | 默认值 |
|---|---|
| 球质量 / 半径 | 0.5 kg / 0.04 m |
| 来球距离 / 初速度 | 2 m / 5 m/s |
| 发射随机性 | 速度 ±3%，方向锥 1 度，位置每轴 ±2 cm |
| 目标距离 / 方位角 | 1.5 m / 20 度 |
| 环绕目标半径 | 0.62 m |
| 夹爪安装 roll | 45 度 |
| 关节反馈 / 控制频率 | 240 Hz / 150 Hz |
| 接球速度匹配比例 | 名义到达速度的 0.3 |
| 闭爪时间 | 69 ms 提前量 |
| 返回 / 稳定 / 蓄能 | 0.3 s 减速、1 s 回中，最多 6 s 稳定，8 s 蓄能 |
| 释放搜索 | 0.1 s 前瞻，最低预测落点误差 |
| 命中阈值 | 15 cm 和 30 cm 分开统计 |

该教师分布的名义来球仍瞄准初始抓取中心，再加入扰动；当前数据不能证明机械臂已经学会大偏移追球。默认训练先复现已经可运行的任务，然后再扩展来球偏移、目标方位和质量。

`t_arr_known_s` 使用已知的名义发射信息，再由观测修正；本版本并非完全未知发射条件的视觉接球。

## 5. 网络与动作

Actor：`82 -> 256 SiLU -> 256 SiLU -> 10`。

Critic：`113 -> 256 SiLU -> 256 SiLU -> 1`。

Actor 的 82 维包括：关节角 12、关节速度估计 12、压力反馈 24、球相对位置及速度估计 6、夹爪位置及速度估计 6、目标 xy 2、阶段 one-hot 6、时钟/可见/持球标志 4、上一动作意图 10。

Critic 额外使用 31 维真实关节角、关节速度、球位置、球速度和当前约束力。Actor 的计算明确切除这些真值。

注意：Actor 的持球标志、状态机阶段仍依赖仿真抓取事件；它不是已经验证可直接部署的纯视觉策略。实机需要用夹爪反馈或可靠的持球检测替换该事件来源。

归一化均值和标准差只由训练集计算；PPO 中固定，checkpoint 保存。标准差下限 0.05，归一化输入裁剪到 ±10。

| 动作维度 | 含义 | authority=1 时相对教师最大修正 |
|---|---|---|
| 0–2 | 抓取位置意图 xyz | 每轴 ±15 mm |
| 3 | 接球速度匹配比例 | ±0.05 |
| 4 | 共收缩压力 | ±1 psi |
| 5 | 环绕半径 | ±3 cm |
| 6 | 环绕最大切向驱动力参数 | ±1 N |
| 7 | 径向阻尼 | ±1 |
| 8 | 释放方位偏置 | ±1 度 |
| 9 | 闭爪提前量 | ±5 ms |

默认 `authority=0.25`，实际边界为表中四分之一。阶段掩码关闭无关动作；这是学习时的动作权限，并非改变机械臂物理限位。扩大权限会改变闭环行为，需要重新验证。

第一版提供共收缩压力优化，尚未实现独立的方向刚度椭圆学习。也没有实现策略主动压缩稳定阶段和蓄能阶段时长。

## 6. 数据集

正式目录：`teacher_runs/dataset_v1/`。

每个完整回合有一个 NPZ 和一个 JSON 完成记录。文件名由种子、任务配置、源代码及资产 hash、checkpoint hash、混合比例和扰动参数共同确定。重复相同采集命令跳过已完成回合；没有完成记录的临时文件不参与训练。

保存全部回合，包括失败，避免只保留容易样本。种子能被 5 整除的整个回合归入 validation，其余归入 train。同一发射种子的不同配置也保留在同一分组。训练时 phase 2/3/5 每五帧保留一帧，再使用阶段平衡采样，避免长时间回中和绕圈淹没接球窗口。

样本权重：15 cm 命中为 1、抓取成功为 0.5、抓取失败为 0.25；约束峰值超过 400 N 时进一步降权。这是演示采样权重，不代表失败回合被当成成功。

NPZ 内容：

- `observations`：113 维，前 82 维是 Actor 输入。
- `teacher_actions` / `actions` / `applied_actions`：教师标签、网络或混合提议、实际经过边界限制的动作。
- `masks` / `phases` / `rewards` / `dones`：有效动作维度、阶段、奖励和终止标志。
- `reference_q_qd_qdd`：当时接球/稳定规划器的参考状态，抛球阶段保留的此项不代表环绕控制参考。
- `ball_estimate`：球 Kalman 位置和速度；不可见时为零，由 observation 中的可见标志区分。
- `trace150`：150 Hz 细轨迹，列顺序为时间1、阶段1、测量关节12、估计关节速度12、压力反馈24、原始压力命令24、提前补偿后实际命令24、真值关节12、真值关节速度12、球位置3、球速度3、weld 力1、夹爪接触力1，共130列。

JSON 包括源版本指纹、数字孪生拟合文件来源、事件时刻、成功标志、首次地面接触误差、接触力峰值/冲量、约束力峰值、释放后碰撞、关节峰值和压力积分代理。

压力积分是用气/控制强度代理，不是电能或真实气动功。手指接触力和 weld 约束力分别记录。子步统计用于峰值和冲量，不仅在30 Hz或150 Hz采样点统计。

早期调试目录 `teacher_runs/pilot/` 和 `teacher_runs/dataset/` 不作为正式训练输入。

## 7. 重新采集、BC、DAgger

从工程根目录：

```powershell
& '.\RL PPO\START_TEACHER_RL.ps1' -Mode Collect -Episodes 96 -Workers 4
& '.\RL PPO\START_TEACHER_RL.ps1' -Mode BC
```

增加学生访问状态上的教师标注：

```powershell
& '.\RL PPO\START_TEACHER_RL.ps1' -Mode DAgger -Episodes 24 -Workers 4
```

默认种子 2001 起，每个 30 Hz 决策以 beta=0.5 选教师，否则选 BC 学生；两种情况都用当前观测重新计算教师意图作为标签。它是高层意图的在线 DAgger，底层经典控制器仍运行。

在 `RL PPO` 目录细调 BC，然后用新的 BC checkpoint 开一轮 PPO：

```powershell
..\.venv\Scripts\python.exe -m teacher_rl bc --data teacher_runs/dataset_v1 --init teacher_runs/bc/bc_best.pt --out teacher_runs/bc_dagger --epochs 30
..\.venv\Scripts\python.exe -m teacher_rl ppo --data teacher_runs/dataset_v1 --init teacher_runs/bc_dagger/bc_best.pt --out teacher_runs/ppo_dagger --updates 100
```

受扰动教师采集示例：

```powershell
..\.venv\Scripts\python.exe -m teacher_rl collect --seed 3001 --episodes 24 --perturb 0.05 --workers 4 --out teacher_runs/dataset_v1
```

BC checkpoint 和数据集固定后再开始一轮 PPO。不要在同一轮 PPO 运行或恢复前覆盖 BC anchor、往数据集追加 DAgger 回合；恢复会校验 anchor hash、数据清单 hash、任务配置、PPO 参数和评估种子。数据或参数改变后，使用 `--init` 和新的 `--out` 开新实验。

需要更多示范时，可把 Collect 的 `-Episodes` 增加到512；种子仍从401开始，已完成的96回合会跳过。采集完成后重新训练 BC，并另开 PPO 输出目录。

## 8. PPO 与柔顺性阶段

PPO 默认：learning rate `3e-5`、clip `0.10`、gamma `0.999`、GAE lambda `0.98`、3 个优化 epoch、minibatch 128、梯度裁剪 0.8、target KL `0.01`、entropy coefficient `0.0001`。

每次 PPO 更新同时加入训练集 BC loss（系数1）和相对冻结 BC 策略的 Gaussian KL（系数0.05），抑制遗忘。探索 log_std 初始 -3，限于 [-4,-1.5]；动作使用 tanh 分布和对应 log probability。

奖励：首次抓住 +5、主动释放 +5、落点平滑奖励最大20、15 cm命中额外20、未完成任务终止 -10，加小的残差正则。首次实际地面接触才算落点，不用预测命中替代结果，也不在弹跳后重新选最好落点。

成功率优先阶段 `quality_weight=0`。稳定后可开新运行加入约束峰值、接触冲量和压力积分惩罚：

```powershell
..\.venv\Scripts\python.exe -m teacher_rl ppo --init teacher_runs/ppo_full/ppo_best.pt --anchor teacher_runs/bc/bc_best.pt --data teacher_runs/dataset_v1 --out teacher_runs/ppo_quality --quality-weight 0.1 --authority 0.25 --updates 100
```

这一实现是可调质量惩罚加独立评估选模，不是具有成功率数学保证的约束优化算法。训练中的成功率仍可能下降，应比较独立评估结果。选模先按15 cm命中率、30 cm命中率、释放率、抓取率排序，全部相同时才选约束峰值更低者。未用训练回报选择 best。

完整训练每5次更新评估固定种子90001–90006。另用未参与选模的测试种子91001起：

```powershell
& '.\RL PPO\START_TEACHER_RL.ps1' -Mode Evaluate -Episodes 24
```

每回合和每次更新会输出进度。训练会保存最新完整更新，checkpoint 使用临时文件后替换。质量阶段恢复需要继续提供相同 `--quality-weight`、任务参数和 `--anchor`。

## 9. 必须保留的物理解释

现在的抓取是进入50 mm范围且闭合延迟满足后建立软 weld。它有承载与放气模型，但仍不是完整三指主动接触抓取。前50 ms的冲击窗口允许短暂超过168 N承载值，所以日志可能出现382 N甚至更高峰值而没有判定掉球。

默认目标方向和45度夹爪安装角是一组经过调试的组合；扩大目标方位之前要检查释放时球与手指是否再次碰撞。当前结果也不能直接与旧 V11 的25 g球、较短任务和较小抓取区域成功率作数值对比。

默认保留自动抓取半径，不偷偷收紧物理条件。未来向真实接触迁移应另建任务版本、记录变化并重新验证。

## 10. 本次实际完成和验证结果

### 10.1 正式教师数据

已完成种子401至496，共96个完整回合，文件保存在 `teacher_runs/dataset_v1/`，约122 MB。

| 指标 | 实测 |
|---|---|
| 训练 / 验证回合 | 77 / 19 |
| 抓取并主动释放 | 95/96，98.96% |
| 落点15 cm内 | 93/96，96.88% |
| 落点30 cm内 | 95/96，98.96% |
| 完成落地的95回合平均误差 | 8.05 cm |
| 完成落地的最大误差 | 17.41 cm |
| 全96回合平均 weld 峰值 | 413.50 N，包含失败回合的0值 |
| 最大 weld 峰值 | 489.71 N |
| 原始30 Hz决策样本 | 47,057 |
| 下采样后训练 / 验证样本 | 10,463 / 2,566 |

失败分析：412未抓到球；427和431成功抓抛，但误差分别为17.27 cm和17.41 cm。它们全部保留，没有从成功率统计中删除。

本批球与手指接触力统计为0，不能解释成无冲击：主要载荷通过软 weld 传递，约束峰值仍超过400 N。这也是后续质量优化必须同时关注约束力的原因。

数据来源指纹：`82d0ce0c3a82f7f85de6099c2308023f1e319f5c95dac47b4939f7403df2399e`。

### 10.2 BC 预训练和闭环对比

正式 BC 已完成60个 epoch，最佳 checkpoint 位于 `teacher_runs/bc/bc_best.pt`，选中第57个 epoch，阶段加权验证 MSE 为 `0.00014004`。该损失使用归一化动作单位，不是米或投掷落点误差。

采用相同的独立种子90001至90006，真实运行完整抓抛：

| 控制方式 | 抓取 / 释放 / 15 cm命中 | 平均落点误差 | 平均 weld 峰值 |
|---|---|---|---|
| 原教师，零残差 | 6/6 / 6/6 / 6/6 | 6.32 cm | 424.17 N |
| BC + 教师底层控制 | 6/6 / 6/6 / 6/6 | 7.49 cm | 423.84 N |

结果文件：`teacher_runs/teacher_baseline_eval.json`、`teacher_runs/bc/evaluation.json`。

结论是学生在小样本评估中保住了可用的成功率，不是超过教师；平均投掷误差反而略大。6个回合也不足以证明广泛泛化。后续 PPO 将重复使用这些种子选模，因此最终测试应另用91001起的种子。

### 10.3 PPO、DAgger 和测试

- 正式 BC 上的 PPO 短验证输出：`teacher_runs/ppo_verified/`。先完成第1次更新，再通过 `--resume` 恢复并完成第2次更新；训练种子连续为10001、10002，没有从头重跑。
- 两个训练回合均命中15 cm，误差4.58 cm、3.01 cm。固定评估种子90001在初始、第1次、第2次更新后的误差为8.45 cm、8.26 cm、10.54 cm。只说明训练与恢复链路可运行，不能据此声称 RL 已提升效果。
- DAgger 验证输出：`teacher_runs/dagger_smoke/`。种子2001、2002，教师/学生混合比例0.5，两回合均成功，误差0.60 cm和10.04 cm。该目录仅验证采集链路，未加入正式96回合数据集，也未用于正式 BC。
- `..\.venv\Scripts\python.exe -m pytest tests -q`：37项通过，包括动作边界、Actor额外真值隔离、原子记录与重复采集、网络保存读取、PPO更新和环境观测。
- 旧 V11 训练代码和外部 CAN 教师源代码未由本框架修改。调试目录 `bc_smoke`、`ppo_smoke`、`ppo_verified` 均不是完整训练目录。

### 10.4 接下来做什么

直接执行本文第一条 Train 命令即可。正式数据和 BC checkpoint 已准备好，不需要先重新采集或重跑 BC。完整训练目录 `teacher_runs/ppo_full/` 尚未启动。

这版目标是先把已有经典控制成功行为变成可继续优化的学习基线；成功率、投掷精度、约束冲击和共收缩压力分开评估。扩大来球偏移、脱离教师独立控制、真实手指接触与方向柔顺性，仍是后续任务，不是此次已经完成的能力。
