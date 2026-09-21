# 此预览数据不可用于V25预训练

该批48场景、926决策的数据通过了文件和快照合同，但分支评分读取了整个episode的历史最小校准误差，而不是每条分支从当前快照开始的未来最小误差。

结果是580个决策的所有有效分支具有完全相同的误差，动作变化惩罚使`hold`被错误选中。该问题已修复；目录被保留仅用于审计。正式`pretrain`必须读取重新生成的`teacher_runs/v25_twin_residual/stage1_preview/preview_dataset.json`。
