# V22 轨迹包络采集与在线 Teacher

## 目的

V21 证明校正后的释放判断能够改善落点预测，但旧甩动轨迹只能覆盖少量已接球场景。V22 扩大驱动力和轨迹半径，使实际落点覆盖目标距离，再用严格分层验证后的模型决定在线释放时机。

全部在线决策只使用 24 帧传感器拟合状态、关节状态、轨迹状态和目标。真实离手状态与真实落点只作为离线标签。

## 已完成的校准流程

```powershell
& '.\RL PPO\START_V22_TRAJECTORY_ENVELOPE.ps1' -Mode Pilot
& '.\RL PPO\START_V22_TRAJECTORY_ENVELOPE.ps1' -Mode Collect
& '.\RL PPO\START_V22_TRAJECTORY_ENVELOPE.ps1' -Mode Fit
```

正式采集包含 6 条轨迹、6 个释放相位、每单元 8 条，共 288 条。Fit 使用按轨迹和相位分层的五折验证，并要求每个单元至少 4 条可用数据。

## 下一步：20 场景 Screen

当前 `smoke_v1` 未选出合格候选，因此暂时不要运行正式 Screen。以下命令保留给 V23 轨迹选择修正完成后的正式筛选。

仅当 `release_calibration_v22.json` 中 `accepted=true` 时运行：

```powershell
& '.\RL PPO\START_V22_TEACHER.ps1' -Mode Screen
```

Screen 同时评估 V15 BC 和 6 个 V22 轨迹候选。只有候选不回退接球、释放和 15 cm 命中，并通过关节、安全、连续交接与落点质量门槛，才生成 `selected_config.json`。

## 之后：80 场景独立验证

仅在 Screen 生成 `selected_config.json` 后运行：

```powershell
& '.\RL PPO\START_V22_TEACHER.ps1' -Mode Validate
```

只有 Validate 生成 `qualified_teacher.json` 后才能重新采集 BC 数据；此阶段仍不能直接启动 PPO。
