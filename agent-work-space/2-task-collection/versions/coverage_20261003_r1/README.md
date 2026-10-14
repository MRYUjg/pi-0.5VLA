# 场景覆盖采集版本

版本coverage_20261003_r1；旧pilot_20261003_r1保持冻结。当前方案见[场景覆盖方案](场景覆盖方案.md)，15个场景与划分以`scenes.json`为准。

本版新增位置/偏航/篮子初态变化，在reset原生采样器返回阶段应用；执行中不注入物体或机械臂状态。保留原r20教师、控制器配置、真实动作记录方式及r6物理门限，新增初始化非支撑碰撞和静置漂移检查。

冻结后，在服务器仓库根目录运行新的唯一批次：

```bash
python3 agent-work-space/2-task-collection/versions/coverage_20261003_r1/scripts/pipeline.py \
  --batch 新的唯一批次名 --workers 2
```

默认按预登记9个train＋3个validation场景执行。可用`--scene-ids train_00,train_01`指定子集；test场景由入口禁止采集。场景ID、seed、split、实际静置后位姿、采样器原始/覆盖后参数、清单哈希写入HDF5和审计记录。状态/图像/动作契约沿用上一版：20Hz、双256RGB、8维具名state、7维原样OSC动作、T+1观测。

`preflight.py`用于初始化验证，输出不构成训练数据；`expert_probe.py`用于无图像专家开发回归，不计入正式试采分母。`run.py/run_r2.py/run_r3.py`保留为继承辅助入口，覆盖采集使用`run_coverage.py`及pipeline。

每条必须通过原物理审计、实际动作回放、时序/字段/图像检查和真实退出码检查才可接受。`dataset_manifest.json`报告每次尝试；按scene split生成独立训练/验证清单，禁止按帧拆分。旧试采数据不自动混入。本轮不加载策略或训练权重。

冻结工具增加`.bddl`依赖指纹；`scenes.json`作为本版本源文件一起归档。版本文件不可覆盖；后续更改场景、采集或判定均须新建版本。相机、光照、材质、物体质量未随机化；当前覆盖不等于整体LIBERO分布覆盖，也不是Spatial正式验收。
