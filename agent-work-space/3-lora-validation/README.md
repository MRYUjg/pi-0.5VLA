# ECO65＋ZhiXing 小数据 LoRA 验证

工作目录：`/mnt/data/vla-policy/openpi/agent-work-space/3-lora-validation`。

**当前状态：300 步小数据训练与保存/重载/续训通过；附加闭环 0/3，任务适配尚未通过。**

本目录承接 `2-task-collection` 已验收的单任务场景覆盖数据与 LeRobot 转换结果，使用新服务器 RTX 3090 24GB 实际验证 π0.5 的 LoRA 微调。训练代码、归一化、运行结果与检查点独立保存；原采集、控制器和审核版本不改。

- 完成记录：[小数据 LoRA 微调验证](reports/小数据LoRA微调验证_完成记录_20261004_r1.md)
- 训练方案：`versions/lora_20261004_r1/方案.md`
- 训练入口：`versions/lora_20261004_r1/scripts/launch.py`
- 正式重载入口：`versions/reload_cache_20261004_r1/run.py`（与训练共享编译缓存；原冷编译失败保留）
- 附加闭环检查：`versions/rollout_20261004_r1/`
- 本轮训练输出：`runs/smoke_train_r2/`
- 真实退出码、显存采样与完整日志：`runs/smoke_train_r2_execution/`
- 冻结清单和源代码归档：`releases/`
- 输入完整性校验、环境与完成记录：`reports/`

小数据训练验证不等于高成功率适配验收。最终目标仍需独立的任务闭环测试支持，不能用训练 loss 代替成功率。
