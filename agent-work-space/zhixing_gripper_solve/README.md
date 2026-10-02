# ZhiXing夹爪专用优化目录

更新：2026-10-02；当前服务器纯CPU；上游冻结基线保持不变。

当前：**实际LIBERO罐10/10、碗8/10，尚未通过本阶段两任务门槛**。这是两个任务的脚本教师验证，整体机械臂适配、π0.5微调和Spatial十任务×20次正式验收仍未完成。

1. [当前实现与验证记录](真实碗罐几何接入_实现与验证记录_20261002_r1.md)。
2. [预注册方案r7](真实碗罐_实施与验收方案_20261002_r7.md)、[独立测试汇总](evidence/real_objects_summary_20261002_r3.json)。
3. [控制配置A/B执行记录](控制配置_AB执行与定位记录_20261002_r1.md)、[教师开发定位](真实碗罐_开发定位补充_20261002_r1.md)。
4. [此前诊断试块教师记录](安全教师路径_实现与验证记录_20261002_r1.md)。

当前候选：`scripts/real_object_teacher_r20.py`；启动器：`scripts/run_real_control_ab_r7.py`；检查：`scripts/verify_real_object_r6.py`。使用已有`eco65_no_extra_pd`隔离运行时配置；生产默认配置未切换。所有早期版本和失败批次仅作历史证据，不与当前独立测试合并统计。

服务器仓库根目录复现时须使用新批次名并显式指定profile：

```bash
python3 agent-work-space/zhixing_gripper_solve/scripts/run_real_control_ab_r7.py \
  --batch 新的唯一批次名 --cases can,bowl --seeds 52100 \
  --profiles eco65_no_extra_pd --workers 2
```

该入口保存数值证据，不生成训练数据。受管文件必须建立新版本；冻结完整性记录见`maintenance/real_objects_20261002_r1/`。原始证据已包括失败样本；不覆盖旧结果。
