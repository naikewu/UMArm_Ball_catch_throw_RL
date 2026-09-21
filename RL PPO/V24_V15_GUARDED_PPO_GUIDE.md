# V24：V15 保底的受限 on-policy PPO

更新日期：2026-09-20

## 修改目标

旧版 `can_v15_anchor_dynamic_ppo_v1` 在接球后无条件切换到新轨迹控制器。它的 update 0 在40个配对场景中只命中22个，而V15命中33个；PPO随后还把终局回报交给释放窗口之后的无效动作，并让数量远多于初始动作的动态样本主导梯度。因此旧版即使完成100次更新，也没有一个checkpoint通过选模。

V24使用新schema `can_v15_anchor_dynamic_ppo_v2`，保留旧实验目录，输出到 `teacher_runs/v15_anchor_dynamic_ppo_v2/`。

## 已完成的修改

1. 初始动作从9个轨迹参数扩为“精确V15 passthrough + 9个轨迹参数”。确定性评估只有在某个接管动作概率达到0.55时才切换控制器，否则重新运行完整V15闭环，保证保底结果确实等于V15。
2. 预训练同时读取1080条动作结果和120条V15配对结果。V15效用增加5分保守接管余量；新动作必须有清楚优势，模型才学习接管。
3. 98个有效接球上下文仍按场景分成79个训练、19个留出。除初始选择外，每个3×3网格点还用其实际相邻动作结果生成一步动态监督，预训练动态Actor向更优的相邻力/半径移动。
4. 取消低频Actor对释放候选的阻断。校准释放器仍以150 Hz检查时间窗、OOD、关节角和8 cm误差条件；PPO只调整轨迹。这样不会因为0.5秒动作周期错过只持续一个控制周期的安全释放候选。
5. 动态决策到18秒释放窗口结束即停止。窗口后的动作不再进入rollout，终局奖励只分配给最后一个仍可能影响结果的动作。
6. 加入仅由可观测量计算的进度奖励，包括历史最小校准误差、候选就绪和关节余量变化。正式选模仍只看实际命中、释放、安全和配对非退化，进度奖励不能替代验收。
7. 初始动作和动态动作分别标准化 advantage，Actor loss、熵和锚定KL按两个阶段等权计算，修复旧版约1:30样本比例导致的梯度失衡。
8. 正式训练开始前先做40场景 update-0 配对检查。若初始策略没有通过V15任务非退化与安全检查，训练在0回合处停止，不再浪费完整训练预算。
9. 每次update从16个新episode增加到32个，100次更新共3200个严格on-policy episode。每批数据只用于紧接着的一次PPO更新，不进入经验回放。

## 已完成验证

- 32项相关测试通过。
- 预训练合同确认读取1080条候选动作结果、120条V15结果，79/19按场景拆分。
- 19个留出上下文的确定性选择全部回退V15，配对非退化率为100%。这只是预训练保护检查，不是正式独立成绩。
- 最终5回合smoke的 update 0 与V15完全一致：接球5、释放5、15 cm命中5，已通过配对安全和非退化检查并发布smoke `ppo_best.pt`。
- smoke训练产生95个on-policy样本，其中5个初始动作、90个动态动作；完成4轮PPO epoch，近似KL为 `9.37e-6`，无KL提前停止。update 1保持5/5命中，选模仍保留更稳妥的update 0。

上述结果证明修正后的训练闭环、V15保底、动态探索和选模均按设计工作。正式3200回合训练尚未运行，因此不能据此声称最终策略已经超过V15。

## 运行命令

在 `RL PPO` 目录执行正式训练：

```powershell
.\START_V15_ANCHORED_PPO.ps1 -Mode Train -Workers 12
```

脚本会复用已经生成且与当前源码匹配的 `teacher_runs/v15_anchor_dynamic_ppo_v2/pretrain/policy_init.pt`，先做40场景配对保护检查，然后执行100次更新、每次32个新episode。

训练被正常中断后，从同一合同恢复：

```powershell
.\START_V15_ANCHORED_PPO.ps1 -Mode Resume -Workers 12 -Updates 100 -EpisodesPerUpdate 32
```

只有存在通过配对选模的 `ppo_best.pt` 后，才运行80个全新场景的独立测试：

```powershell
.\START_V15_ANCHORED_PPO.ps1 -Mode Evaluate -Workers 12
```

主要结果位于：

- `teacher_runs/v15_anchor_dynamic_ppo_v2/formal/training_summary.json`
- `teacher_runs/v15_anchor_dynamic_ppo_v2/formal/validation_*.json`
- `teacher_runs/v15_anchor_dynamic_ppo_v2/formal/ppo_best.pt`
- `teacher_runs/v15_anchor_dynamic_ppo_v2/formal/evaluation_80/comparison.json`

## 正式运行结果（2026-09-20）

正式训练已完成100次更新和3200个on-policy回合，文件完整，但没有学出超过V15的新确定性策略。`training_summary.json`记录`best_update=0`；`ppo_best.pt`与预训练模型参数相同，全部21次固定配对验证都由35个已接球回合执行精确V15 passthrough，新轨迹接管数始终为0。验证保持35/40释放、35/40命中15 cm和4.725 cm平均落点误差，因此安全保护成功，PPO改善未成功。

训练期间实际随机探索了1142次新轨迹接管，其中806次释放、707次命中15 cm，命中率61.9%、平均效用13.22；V15 passthrough共1621次，1591次命中15 cm，命中率98.1%、平均效用31.36。update 100在98个已有接球上下文上的V15平均概率已从45.3%升至72.3%，九个新动作总概率降至27.7%。这说明PPO根据回报学会了更多回退V15，而不是学会稳定接管。

`evaluation_80/comparison.json`尚未生成。当前最佳checkpoint只是update 0回退模型，运行80场景只会再次评估V15，不能验证PPO收益。下一轮不应直接降低0.55门槛或延长同一训练；需要把动作接口改为V15后半程闭环上的有界残差，并把二元残差启用门和条件残差动作分开建模。
