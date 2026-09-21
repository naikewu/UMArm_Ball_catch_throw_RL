# V25 数字孪生预览与 PPO：五阶段运行说明

V25 是一条独立管线，不覆盖 V15、V23 或 V24。V15 仍负责接近和接球；V25 只在成功交接后控制甩动轨迹。正式产物位于 `teacher_runs/v25_twin_residual/`。

## 已实施的结构修正

1. **完整数字孪生快照**：保存并恢复 MuJoCo 数据、可变模型参数、24 个 CAN 节点、阀门事件、控制器、估计器、释放器和随机数状态。验证要求从同一动态交接点恢复后逐位一致，不能用容差掩盖状态遗漏。
2. **真实中途动作标签**：每个 0.5 秒决策点从同一快照分别推演 9 个力/半径联合动作。旧 1200 回合只训练初始轨迹选择，不再充当中途动作标签。
3. **可学习的观测和动作**：观测加入当前校准落点、二维有符号落点误差、当前误差、误差变化、OOD 余量和剩余释放时间。动作是 `{-1,0,+1} × {-1,0,+1}`，步长分别为 0.05 和 0.0225；完全无效的边界动作会被屏蔽。
4. **释放权独立**：PPO 没有 release head。150 Hz 校准释放守卫始终拥有最终释放权，低频动作不会再错过单帧释放机会。
5. **强制候选 PPO**：训练时没有 V15 退出动作。每个候选回合与同场景 V15 配对，初始轨迹直接获得完整的“候选效用−V15效用”，不经过低层动作序列折扣；动态动作按真实 0.5 秒间隔计算 GAE。
6. **独立保守门控**：候选 Actor 达标后，用最终候选模型重新采集 200 个配对场景并训练 bootstrap ensemble。只有效用差下置信界和成功率下置信界都达标时才接管。
7. **拒绝假成功**：零接管、update 0、候选未达标、开发 screen 未通过，均不会产生最终合格文件。80 场景独立测试还检查最差参数分箱。

## 运行前检查

在 PowerShell 中进入目录：

```powershell
Set-Location "E:\Python\UMArm_koopman_compliance_control_espproject\RL PPO"
```

确认以下既有资产存在：

```powershell
Test-Path "teacher_runs/v15_improved/bc_v1_formal/bc_best.pt"
Test-Path "teacher_runs/v23_contextual/warm_kernel_v1/release_calibration_v23_kernel.json"
Test-Path "teacher_runs/v23_contextual/combined_dataset_inventory_v2.json"
```

三项都应返回 `True`。

## 第一阶段：验证快照并采集数字孪生分支

```powershell
.\START_V25_TWIN_PPO.ps1 -Mode verify
.\START_V25_TWIN_PPO.ps1 -Mode preview -Workers 8
```

第一条命令必须输出：

```text
"exact_bitwise_replay": true
"max_absolute_error": 0.0
```

第二条命令默认采集 48 个场景，在每个动态决策点推演 9 个未来分支。主要产物：

- `stage1_verify/verification.json`
- `stage1_preview/preview_contract.json`
- `stage1_preview/preview_dataset.json`

## 第二阶段：蒸馏数字孪生教师

```powershell
.\START_V25_TWIN_PPO.ps1 -Mode pretrain
```

该阶段同时使用两类数据：

- 已有 1200 回合固定轨迹数据：只训练 9 个初始力/半径配置；
- 第一阶段的真实动态分支：训练 9 个中途联合动作。

产物为 `stage2_pretrain/policy_init.pt`。该网络默认约 1 万参数以内，且没有释放动作头。

## 第三阶段：强制候选 on-policy PPO

```powershell
.\START_V25_TWIN_PPO.ps1 -Mode train -Workers 8
```

默认配置是 60 次更新、每次 64 个新场景、每 5 次更新做 40 场景强制候选验证。每一批 rollout 只用于紧接着的一次更新，所以这是 on-policy PPO。每次更新保存：

- `update_NNNN_transitions.npz`：观测、下一观测、动作、有效动作 mask、奖励、优势、实际参数变化和 episode id；
- `update_NNNN.json`：配对结果和 PPO 统计；
- `candidate_best.pt`：当前最好的强制候选诊断模型；
- `candidate_qualified.pt`：默认自动准入仍要求已接球后的释放率不低于 90%、已释放后的 15 cm 命中率不低于 90%，且安全和接球检查通过。

若训练被中断，从同一合同恢复：

```powershell
.\START_V25_TWIN_PPO.ps1 -Mode train -Workers 8 -Resume
```

如果训练结束后没有 `candidate_qualified.pt`，先审计所有固定验证，不得只降低一个百分比便发布候选。只有候选同时满足安全、接球不退化、释放后命中率不低于 90%，并在至少 25% 的已接球验证场景中实现“安全命中且配对效用优于 V15”时，才可以降级为“仅允许收集独立门控数据”。这不代表候选可以部署；全新 200 场景门控采集、40 场景 hybrid screen 和 80 场景独立验收仍必须通过。

本轮 update 25 经审计采用上述门控准入：释放率 22/34=64.71%，释放后命中 20/22=90.91%，安全正收益场景 9/34=26.47%，安全与接球检查通过。`candidate_qualification.json` 明确记录 `accepted_for_deployment=false`，其 `candidate_qualified.pt` 仅用于第四阶段验证门控能否识别这 9 类有利上下文。

## 第四阶段：训练独立 V15/候选门控

只有存在可读取、哈希匹配的 `candidate_qualification.json` 和 `candidate_qualified.pt` 后运行：

```powershell
.\START_V25_TWIN_PPO.ps1 -Mode gate -Workers 8
```

该阶段使用 200 个此前未用过的场景重新运行最终候选和冻结 V15，按完整场景划分训练/验证集。产物为：

- `stage4_gate/paired_dataset.json`
- `stage4_gate/gate.pt`
- `stage4_gate/gate_contract.json`

门控不看 9 个动作中某一类的 softmax 概率，而是估计候选相对 V15 的效用差和安全成功概率，并使用 ensemble 下置信界。

## 第五阶段：开发 screen 与一次独立验收

先运行固定 40 场景 screen：

```powershell
.\START_V25_TWIN_PPO.ps1 -Mode screen -Workers 8
```

它同时检查：候选实际接管覆盖率至少 25%、零接管直接失败、接球/释放/15 cm 命中相对 V15 不退化、平均落点误差不恶化超过 5 mm、接管场景释放时间平均缩短至少 20%、无持续停顿并满足关节安全。

只有生成 `stage5_screen/screen_passed.json` 后，才运行一次 80 场景独立测试：

```powershell
.\START_V25_TWIN_PPO.ps1 -Mode evaluate -Workers 8
```

最终只有全部检查通过才生成：

```text
teacher_runs/v25_twin_residual/stage5_evaluation_80/V25_ACCEPTED.json
```

缺少这个文件表示 V25 尚未被接受；训练完成或 checkpoint 存在本身不代表成功。

## 可选冒烟测试

正式运行前可以检查整条命令和文件合同：

```powershell
.\START_V25_TWIN_PPO.ps1 -Mode all -Workers 2 -Smoke
```

冒烟模式数据很少，结果没有统计意义，也不能进入独立 80 场景验收。它使用独立目录 `teacher_runs/v25_twin_residual_smoke/`，不会污染正式产物。

## 一次执行全部正式阶段

```powershell
.\START_V25_TWIN_PPO.ps1 -Mode all -Workers 8
```

建议第一次仍按五个阶段逐项运行并检查产物。`all` 会在任何合同不匹配或验收失败处立即停止，不会越过失败阶段继续发布模型。
