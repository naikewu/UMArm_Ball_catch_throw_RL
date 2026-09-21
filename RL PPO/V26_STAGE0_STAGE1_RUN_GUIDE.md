# V26 Stage 0–4 运行说明

V26 当前开放四个经过测试的前置阶段：

- `verify`：验证数字孪生从动态决策状态推演到终局时可以精确重复；
- `oracle`：在200个独立场景中，分别完整运行V15和9个固定力/半径轨迹，判断当前动作空间是否包含足够多的可释放解。
- `mpc_pilot`：在100个全新场景中，从接球后的当前观测状态建立快照，在线推演9条终局轨迹，选择并实际执行最优轨迹。
- `hybrid_validate`：把精确V15回退作为第十个高层动作，在另一批100个全新场景中验证混合Teacher。
- `collect`：在混合Teacher通过后采集至少1000个全新独立接球状态及九动作终局标签。

PPO和静态Actor入口尚未开放。V25的`screen`也不应继续运行。

## 第一步：verify

在PowerShell中执行：

```powershell
Set-Location -LiteralPath "E:\Python\UMArm_koopman_compliance_control_espproject\RL PPO"
.\START_V26_TWIN_MPC_PPO.ps1 -Mode verify
```

正常耗时约15至30秒。输出必须同时包含：

```text
"passed": true
"exact_terminal_replay": true
```

完成后先检查`teacher_runs/v26_twin_mpc_ppo/stage0_verify/verification.json`，再决定是否运行第二步。

## 第二步：oracle

只有正式verify通过后才运行：

```powershell
.\START_V26_TWIN_MPC_PPO.ps1 -Mode oracle -Workers 8
```

该阶段执行200个场景，每个场景包含1个V15基线和9个完整候选轨迹。按5场景smoke实测，正式任务预计约1.5至2小时，具体取决于CPU并行竞争。每个场景完成后都会立即写入`stage1_oracle/episodes`；用相同命令重新启动时会跳过已有场景。

正式oracle已经完成，结果如下：

- 200个场景中V15接球172个；
- oracle释放169/172，`release_given_capture=98.26%`；
- 释放后15 cm命中168/169，`hit15_given_release=99.41%`；
- 安全正收益103/172，`safe_positive_fraction=59.88%`；
- `capture_nonregression=true`、`safety=true`、`action_space_feasible=true`。

正式oracle的验收条件为：

- `release_given_capture >= 0.90`；
- `hit15_given_release >= 0.90`；
- `safe_positive_fraction >= 0.25`；
- `capture_nonregression=true`；
- `safety=true`。

只有`action_space_feasible=true`才实施因果MPC测试和后续Actor/PPO。当前已经通过。

## 第三步：因果终局MPC pilot

执行：

```powershell
.\START_V26_TWIN_MPC_PPO.ps1 -Mode mpc_pilot -Workers 8
```

该阶段使用100个与V25及第二步oracle均不重叠的新场景。策略只读取接球后的当前观测状态；它保存一份已验证的数字孪生快照，把9个固定力/半径轨迹分别推演到释放或终局，然后恢复同一快照并实际执行选中的轨迹。它不读取场景seed、未来真实状态或事后落点。

正式报告写入`teacher_runs/v26_twin_mpc_ppo/stage2_causal_mpc_pilot/pilot_report.json`。必须检查：

- `release_given_capture >= 0.90`；
- `hit15_given_release >= 0.90`；
- `safe_positive_fraction >= 0.25`；
- `exact_twin_prediction=true`；
- `capture_nonregression=true`、`release_nonregression=true`、`hit15_nonregression=true`、`safety=true`；
- `qualified_as_causal_mpc=true`。

当前源码的5场景独立smoke已经通过：V15与MPC均接球5/5、释放5/5并命中15 cm 5/5；5/5数字孪生终点预测与实际执行逐项一致；安全正收益覆盖3/5，接球、释放和命中均不低于V15。相同启动命令再次运行会直接读取5个已有episode并生成相同报告，已验证中断续跑合同。该结果验证实现，正式性能由100场景pilot决定。

正式pilot已完成：V26接球84、释放82、命中80，V15接球84、释放84、命中83。V26的条件释放率97.62%、条件命中率97.56%、安全正收益覆盖51.19%和84/84孪生复现均通过，但释放与命中非退化失败，因此`qualified_as_causal_mpc=false`。失败场景的九条分支均没有安全15 cm命中，PPO不能修复这个动作空间上限。

## 第四步：精确V15回退混合Teacher独立验证

执行：

```powershell
.\START_V26_TWIN_MPC_PPO.ps1 -Mode hybrid_validate -Workers 8
```

该阶段使用从`33200001`开始的100个新种子，与V25、oracle和因果MPC pilot均不重叠。十个高层动作是`V15精确回退 + 9条V26轨迹`。门控只读取当前接球状态与九条digital twin终局预测：只要存在安全15 cm命中就执行最优V26轨迹，否则保留V15。

正式报告写入`teacher_runs/v26_twin_mpc_ppo/stage3_hybrid_validation/validation_report.json`。必须同时满足：

- 接球、释放、15 cm命中均不低于同场景V15；
- 接球后释放率和释放后命中率均不低于90%；
- V26接管率不低于50%，安全正收益覆盖不低于25%；
- 平均落点误差相对V15退化不超过5 mm；
- V26预测执行一致、V15回退逐项一致；
- 无夹持破坏、最大关节角不超过36度；
- `qualified_hybrid_teacher=true`。

5场景smoke完成了4次V26接管并全部命中，1个未接球场景保持V15；小样本落点均值比V15多6.14 mm，因此没有用smoke冒充正式通过。已额外重放已知困难场景`33100047`：九条分支预测0次命中，高层动作正确选择0号V15回退，最终释放、命中且与V15结果逐项一致。19项单元与回归测试通过，相同smoke命令可直接复用episode。

在`hybrid_validate`正式通过前，1000状态采集、Actor蒸馏与PPO入口继续锁定。曾尝试直接用现有172个接球状态训练39维到9动作的静态选择器，但36个隔离验证状态仅达到77.78%释放、释放后78.57%命中，说明动作结果很多并不等于独立状态足够。

正式`hybrid_validate`已经通过。100个场景中，V15和混合V26均接球、释放、命中15 cm 81/81；V26接管80次、精确V15回退1次，接管率98.77%；安全正收益48/81（59.26%）；平均落点误差4.296 cm，优于V15的4.431 cm；81/81预测或回退逐项复现，最大关节角34.72度且无夹持破坏。`qualified_hybrid_teacher=true`，所有11项检查均为true。

## 第五步：采集至少1000个独立接球状态

执行：

```powershell
.\START_V26_TWIN_MPC_PPO.ps1 -Mode collect -Workers 8
```

正式合同使用1300个全新场景，种子从`33300001`开始。按当前约81%的接球率，预计得到约1050个独立接球状态。每个接球状态保存：

- 39维交接实测上下文；
- 九条digital twin终局分支标签；
- 一个V15回退加九个V26轨迹的高层动作；
- 实际执行结果和同场景V15结果；
- 释放、命中、终局误差、释放时间、效用、关节及安全信息。

数据按场景固定划分为训练和验证分区，正式门槛为总接球状态不少于1000、训练状态不少于800、验证状态不少于180、终局标签完整、预测/回退精确、种子唯一且安全。结果写入`teacher_runs/v26_twin_mpc_ppo/stage4_dataset/collection_report.json`，最终必须有`dataset_ready=true`。任务可用同一命令断点续跑。

5场景采集smoke得到3个独立接球状态和27条终局标签；数据完整性、预测复现、种子与安全检查通过。相同命令重跑直接复用全部episode。smoke的`dataset_ready=false`符合预期，因为它不满足正式的800/180数量门槛。

正式数据合同通过后，下一阶段才生成十动作Actor预训练入口；Actor先学习V15回退与九个V26轨迹选择，并同时预测释放、命中、误差、时间和风险。Actor在隔离验证集通过后，才开放单决策on-policy PPO。

## 已完成的启动前验证

- V26单元测试与V25回归测试：15项通过；
- 终局快照重复：逐位一致；
- V26正式oracle种子从`33000001`开始，与已登记的4228个V25合同种子无重叠；
- 5场景独立重放smoke：5/5释放、5/5命中15 cm、安全正收益覆盖80%；
- 两次独立重放的45条动作结果和5个最佳动作完全一致。

曾尝试在交接状态建立前共享快照以缩短运行时间，但部分场景的后续分支错误重复首个动作。该结果已隔离到`stage1_oracle_invalid_prehandover_snapshot_20260921`，正式程序坚持从相同种子独立重放每条轨迹。
