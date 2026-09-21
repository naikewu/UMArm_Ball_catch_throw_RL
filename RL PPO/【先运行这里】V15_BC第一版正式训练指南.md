# V15 BC 第一版正式训练指南

日期：2026-09-18

## 1. 这一版要学到什么

V15 BC 学习的是传感器观测到 teacher 高层动作意图的映射，不是直接学习阀门压力、力矩或一条完全独立的端到端抓抛轨迹。

- Actor 输入 82 维可观测状态：关节位置与速度估计、24 路压力、球的相对位置与速度、末端状态、目标 XY、当前阶段、时钟/可见/持球标志和上一动作。
- Actor 输出 10 维归一化 intent：三维拦截偏移、接球速度匹配、共收缩压力、绕转半径、驱动力、阻尼、投掷方位和闭合意图。
- 训练覆盖来球速度 4.5-5.5 m/s、发射距离 1.7-2.3 m、目标距离 1.3-1.7 m、目标方位 10-30 度，并包含发射角度和位置扰动。
- 网络应学会根据观测阶段切换接球、稳定、绕转和释放前的 intent，并在未见过的参数组合上接近 V15 teacher 的闭环表现。

底层 PID、简单逆动力学、解析释放器、动作安全限幅仍然运行。V15 的目标距离条件化绕转半径也是固定 teacher 配方的一部分，由 `ImprovedTeacherEnv` 执行；当前标签没有把它变成一个可脱离该环境独立运行的神经网络能力。因此，这一版的准确表述是“BC 学会在 V15 控制框架内复现 teacher 高层意图”。

## 2. 正式数据与质量

默认数据目录：`teacher_runs/v15_improved/dataset_1000_formal`

- 1000 个完整回合，800 个训练回合、200 个验证回合。
- 原始 452,281 个决策步；按阶段下采样后为 104,147 个训练样本和 26,232 个验证样本。
- 抓取/释放 90.1%，15 cm 命中 89.4%，30 cm 命中 90.0%。
- 成功落地回合平均误差 5.15 cm；抓取后条件 15 cm 命中率为 894/901 = 99.22%。
- 最弱参数区间的 15 cm 命中率为 86.0%。

训练入口会先校验 schema、teacher source hash、可达性采集器 hash、场景清单 hash、配方、1000 个 episode JSON/NPZ 及 split。旧 V13/V14 数据、配方混合、缺失回合和 student/DAgger 轨迹都会被拒绝。

## 3. 第一版参数

| 参数 | 设置 |
|---|---:|
| Actor | `82 -> 256 SiLU -> 256 SiLU -> 10` |
| Epochs | 60 |
| Batch size | 256 |
| Optimizer | Adam |
| Learning rate | `3e-4`，固定 |
| Gradient clip | 1.0 |
| 随机种子 | 20260927 |
| CPU threads | 4 |
| 模型选择 | 验证集加权 masked MSE 最低 |
| Checkpoint | 每轮写 `bc_latest.pt`，改善时写 `bc_best.pt` |

只训练 Actor；Critic 不参与 BC。观测均值和标准差只由训练 split 计算，标准差下限为 0.05。

长时间的 settle、spinup 和 flight 阶段沿用每 5 帧保留 1 帧的下采样。训练采样先让每个存在有效动作的阶段获得相同基础权重，再使用轨迹质量权重：15 cm 命中为 1.0、抓到但未命中为 0.5、漏接为 0.25；异常高 weld 峰值会进一步降权。失败样本不会删除，因为它们仍包含来球覆盖和接球边界信息。

## 4. 启动训练

在工程根目录 PowerShell 中运行：

```powershell
& '.\RL PPO\START_V15_BC.ps1' -Mode Train
```

默认输出：`RL PPO/teacher_runs/v15_improved/bc_v1_formal`

主要文件：

- `bc_config.json`：数据契约、网络与训练参数。
- `bc_metrics.jsonl`：每轮训练 MSE、验证 MSE、动作 RMSE 和分阶段 MSE。
- `bc_best.pt`：验证指标最佳的 checkpoint。
- `bc_latest.pt`：最后一轮 checkpoint。
- `bc_summary.json`：最佳 epoch、最佳验证损失和耗时。

已有正式输出时脚本会停止，不会覆盖。重新试验超参数时应指定新目录，例如：

```powershell
& '.\RL PPO\START_V15_BC.ps1' -Mode Train `
  -OutputDir 'teacher_runs/v15_improved/bc_v1_lr1e4' `
  -LearningRate 1e-4
```

## 5. 训练后闭环评估

训练完成后运行：

```powershell
& '.\RL PPO\START_V15_BC.ps1' -Mode Evaluate -Workers 8
```

评估使用 seed 92001 起、design seed 20260928 的 200 个全新可达 Latin-hypercube 场景。每个场景都让 V15 teacher 与 `bc_best.pt` 配对运行，输出到 `bc_v1_formal/evaluation_200/comparison.json`。

第一版通过条件：

- BC 抓取率相对 teacher 下降不超过 3 个百分点。
- BC 15 cm 命中率相对 teacher 下降不超过 5 个百分点。
- BC 成功落地平均误差不超过 7 cm。
- 四个泛化变量所有区间中，最差区间 15 cm 命中率不低于 80%。

验证 MSE 只是动作模仿误差，不能代替闭环抓取和落点评估。只有 `comparison.json` 中 `acceptance.passed=true`，才把本 checkpoint 作为后续 DAgger 或残差 PPO 的正式初始化。

## 6. 已完成结果与下一步

BC 已完成 60 轮，最佳为第 54 轮。200 个新场景中，BC 抓取/释放 87.5%、15 cm 命中 86.5%、平均落点误差 4.925 cm；Teacher 对应为 88.0%、86.5%、4.882 cm。最差参数区间 BC 为 74%、Teacher 为 76%，原有绝对 80% 门槛未通过，`acceptance.passed=false` 继续保留。

经讨论，使用此 checkpoint 启动受限残差 RL 实验，检验成功率和柔顺性是否还能改善；这不等于把 BC 的绝对泛化验收改为通过。具体操作见 [V15 残差 RL 训练指南](【先运行这里】V15_残差RL训练指南.md)。
