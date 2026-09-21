# V21 校正后在线 Teacher 筛选

## 在线边界

V21 Teacher加载已通过分层五折验证的`release_calibration_stratified.json`。在线释放只使用24帧传感器拟合球状态、机器人关节与轨道状态、目标和阀门延迟，不读取仿真真实离手状态或真实落点。

释放决策限制在标定覆盖的交接后2.0--4.0秒，并且只评估标定中出现过的立即排气候选。特征任一维超过训练分布4.5个标准差、关节超过候选阈值、预测落点未进入候选容差时均拒绝释放。

## 第一步：20场景筛选

从项目根目录运行：

```powershell
& '.\RL PPO\START_V21_TEACHER.ps1' -Mode Screen
```

默认运行同组BC和5个校正后Teacher候选，各20个全新场景，共120条仿真。输出目录是`RL PPO/teacher_runs/v21_calibrated_teacher/screen_v1/`。

只有候选在抓住、释放、15 cm命中、落点误差、接触质量、最大36度关节偏转、即时交接和标定覆盖范围上全部通过，才写出`selected_config.json`。没有该文件就不得运行Validate、采集BC数据或启动PPO。

## 第二步：80场景独立验证

仅在Screen生成`selected_config.json`后运行：

```powershell
& '.\RL PPO\START_V21_TEACHER.ps1' -Mode Validate
```

通过80个全新场景及任务参数分箱门槛后才写出`qualified_teacher.json`。该文件表示Teacher可以用于重新采集连续抓抛BC数据，并不表示可以跳过新BC直接训练PPO。
