# ZhiXing 夹爪专用优化目录

更新：2026-10-02。当前服务器纯CPU；上游冻结基线 `n0_20260930_r1` 保持不变。

**当前状态：安全教师已在诊断试块任务中通过5次回归和10个新增初态，真实LIBERO物体与总体适配尚未验收。**

## 当前入口

1. [安全教师路径实现与验证记录](安全教师路径_实现与验证记录_20261002_r1.md)：工具坐标、外形、带载安全高度、可达姿态、阶段保护、结果与边界。
2. [最终汇总](evidence/safe_path_summary_20261002_r1.json)。
3. [回归证据检查](evidence/safe_path_regression_verify_20261002_r2/verification.json)与[新增初态证据检查](evidence/safe_path_validation_verify_20261002_r2/verification.json)。
4. [证据索引r3](evidence/证据索引_20261002_r3.md)。

教师为 `scripts/safe_teacher_r8.py`，运行入口 `scripts/run_safe_teacher_r9.py`，独立检查 `scripts/verify_safe_teacher_r2.py`。版本清单见 `maintenance/safe_teacher_20261002_r1/`。

服务器仓库根目录运行（使用全新批次名）：

```bash
python3 agent-work-space/zhixing_gripper_solve/scripts/run_safe_teacher_r9.py \
  --batch 新的唯一批次名 --seeds 47100,47101,47103 --workers 2
```

该入口执行诊断试块教师，保存数值证据；不生成RGB/LeRobot训练数据，不代表已切换生产采集器。

## 前一轮结论与历史

- [带载测量与分阶段定位](带载测量与分阶段定位_20261001_r1.md)：证明持续夹持力，定位原教师碰篮、推移篮子后仍向旧坐标投放的问题。文中的“后续优化未实施”描述的是当时状态，现由本轮教师记录更新。
- [测量口径与运行环境](测量口径与运行环境_20261001_r2.md)：力的单位、原运行时异常、隔离CPU环境和带载诊断协议。
- 旧教师诊断入口 `loaded_diagnostics_r5.py` / `run_loaded_matrix_r4.py` 仅用于重现基线；安全教师使用上方新入口。
- `safe_teacher_r1`至`r7`、早期验证批次均为开发证据；不与最终r8统计混合，不删除失败样本。
- 2026-10-01已按授权清理12个错误/误导文件，清单位于 `maintenance/cleanup_20261001_r1/`。

新试验使用新目录，不覆盖既有证据；受管文件变化须建立新版本。试块教师成功不放行真实物体闸门、规模训练、Spatial正式评测或实机部署。
