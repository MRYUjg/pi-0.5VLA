# 控制器验证工作区

固定适配方向：**ECO65 + PandaGripper**。这是独立的、无模型的仿真测试，不连接实体机器人，不启动策略服务，不修改现有机器人配置或控制源码。

## 运行

在服务器 `/mnt/data/vla-policy/openpi` 执行：

```bash
bash agent-work-space/controller_validation_20260927/run.sh --smoke --output smoke_results
bash agent-work-space/controller_validation_20260927/run.sh
bash agent-work-space/controller_validation_20260927/run.sh --contact-ablation --output contact_ablation
docker run --rm --network=none --cpus=2 \
  -v /mnt/data/vla-policy/openpi:/app -w /app libero \
  /bin/bash -c 'source /.venv/bin/activate && python /app/agent-work-space/controller_validation_20260927/report.py'
```

Docker 使用已有 `libero` 镜像、CPU 仿真、关闭渲染和网络，不需要 GPU。正式复跑建议使用 `--output results_<新名称>` 保留历史记录；`report.py` 默认读取 `results`。

## 产物

- `RESULTS.md`：指标表、方法和边界；`CONCLUSIONS.md`：结果分析。
- `CONTACT_ABLATION.md`：基座—第一连杆碰撞排除的成对因果验证。
- `contact_ablation/`：每项独立进程复核的碰撞消融数据；`contact_ablation_first_run/`：退出释放内存异常的首轮数据存档。
- `results/summary.json`：64 项详细结果，逐项写盘。
- `results/metrics.csv`：便于筛选的汇总表。
- `results/*_*.csv`：每 50 ms 的末端实际位置、真实控制目标和位姿误差。
- `results/*_*.json`：关节限位、每仿真子步力矩饱和、机器人接触对等。
- `results/fixed_tracking.png`、`results/native_response.png`：响应曲线。
- `smoke.log`、`run.log`：运行记录。

固定目标测试使用 OSC 的绝对 set_pos/set_ori 参数持续保持相同目标，避免把零增量错误当作保持原目标。原生增量测试则保留评测接口的行为。额外 PD 关闭同时将该项比例和阻尼系数归零，OSC 本身的姿态零空间控制不变。

每对 PD 开/关测试均断言 reset 的全部 qpos 相同；每仿真子步断言预测的限幅力矩等于执行器 ctrl。2 mm / 2° 阈值仅用于本次筛查；通过不代表能完成抓取，也不代表实际硬件已验证。
