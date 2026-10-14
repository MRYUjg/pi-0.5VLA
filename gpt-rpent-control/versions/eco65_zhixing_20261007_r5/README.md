# ECO65＋ZhiXing 既有部署配置复验 r5

复用r4已通过实际链路/独立审计的原生DeepSeek＋pi0.5＋RPent薄适配器。正式批次Spatial 10×1、seed0、官方初态index0、10步静置、220步动作预算、相同模型/图像/状态/动作/成功判定/MP4条件。

本版唯一控制差异：显式复用1.3及2-task-collection既有eco65_no_extra_pd profile。初始化和10步静置继续legacy，首个模型动作前切换该既有profile，与专家采集既有时序一致。未调整OSC增益、力矩、动作尺度、摩擦/质量、接触、夹爪速度或生产默认值。r4 legacy正式结果单独保留，不覆盖。

先执行10步真实模型链和独立审计，再运行正式10任务。失败不替换，不补抽初态。预先锁定源码；正式批次只运行一次。保存20份MP4及全部原始证据，和legacy批次逐任务比较初态/首个预测及最终结果。PD配置差异不能混作单独改变模型的实验。

state姿态保持原LIBERO评测robot0_eef_quat；后两位按名字取Left_1_Joint/Right_1_Joint，与采集的夹爪编码一致；不声称完整state8与采集姿态定义等价。参见reports/eco65_zhixing_20261007_r1/观测编码说明_20261007_r1.md。

复现每次使用新run-id：

```bash
cd /mnt/data/vla-policy/openpi/gpt-rpent-control
python3 versions/eco65_zhixing_20261007_r5/workflow.py --run-id 新批次名 --workers 2
```

本轮只测试原模型在当前配置的仿真闭环。多初态统计、完整RPent工具集、LLM自主恢复增益、微调及实机适配未验收。旧r1–r4源码、开发失败与Panda基线保留。
