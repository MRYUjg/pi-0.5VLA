# ECO65 + ZhiXing 多任务采集：最终项目存档

归档版本：`final_20261005_r1`。按用户要求，将当前工作结果作为本阶段最终版，合并说明并清理旧代码、旧视频及过程 Markdown。服务器目录为 `/mnt/data/vla-policy/openpi/agent-work-space/4-multi-task-collection`。

**本阶段成果已归档；高成功率策略适配尚未验收。** 罐入篮新增 16 条合格示范；Spatial 十任务首次开发试采中教师成功 6/10，不能视为 π0.5 策略成功率。没有新增微调，旧 LoRA 闭环 0/3 结论不变。

## 最终结果

| 内容 | 最终结果 | 原始证据 |
|---|---|---|
| 罐入篮扩展 | train 12 条 / 6992 帧，validation 4 条 / 2313 帧；16/16 通过采集、回放和审计 | [数据清单](runs/expansion_train_val_r1/dataset_manifest.json) |
| LeRobot 转换 | 9305 帧、18610 张双相机图像全量一致；没有训练 | [转换验证](lerobot_exports/expansion_lerobot_r1/validation.json) |
| 与原批次来源汇总 | train 21 条 / 12218 帧，validation 7 条 / 4034 帧；仅来源索引，未合并数据集或重算归一化 | [来源索引](reports/combined_sources_20261004_r1.json) |
| Spatial 首次试采 | task 0—9、seed 57100—57109，每任务一次；5584 个教师动作；成功 task 0/2/3/5/6/8 | [结构审计](runs/spatial_pilot_20261004_r1/structural_audit.json) |
| Spatial 证据检查 | 物理证据一致性 10/10、结构检查 10/10；有动作的 8 条回放全部通过，误差为 0；task 4/9 无教师动作，回放跳过 | [最终回放](runs/spatial_replay_20261004_r1/summary.json) |
| 视频 | 罐任务 1 段、Spatial 每任务 1 段，共 11 段；完整解码和 SHA256 已验证 | [视频索引](videos/spatial_pilot_20261004_r2/视频索引.md) · [浏览页面](videos/spatial_pilot_20261004_r2/index.html) |

罐任务训练 seed 为 56100—56111，验证 seed 为 56200—56203；原保留测试 seed 55300—55302 未采集为示范。覆盖仍为约 50mm 邻域，不能代表完整原生任务分布。原始 HDF5 保留 T+1 观测；LeRobot 仅导出与 T 个动作配对的观测。转换器检查长度 50 的动作块并不改变模型训练 horizon。

Spatial 数据尚未放行训练；失败轨迹和视频保留作诊断。视频为真实脚本教师采集，含独立的准备阶段画面；不是模型推理录像。

## 唯一目录索引

| 目录 | 保留内容 |
|---|---|
| [versions](versions) | 最终依赖的五个功能模块，详见下表 |
| [tools](tools) | 统一运行入口及最终归档校验器 |
| [runs](runs) | 原始 HDF5、物理记录、过程退出码、初始化预检和最终回放证据 |
| [lerobot_exports](lerobot_exports) | 已验证的罐任务 LeRobot 数据和转换证据 |
| [reports](reports) | 最终汇总、场景覆盖、来源索引和隔离检查 |
| [videos/spatial_pilot_20261004_r2](videos/spatial_pilot_20261004_r2) | 唯一保留的视频版本，11 段 MP4 及元数据、预览 |
| [archive/final_20261005_r1](archive/final_20261005_r1) | 文件 SHA256 清单、外部依赖清单、删除台账和最终核验 |

`versions` 内的日期是实际源码组件标识，不是待选历史版本。保留原路径是因为教师证据记录了源码绝对路径及 SHA256，最终回放也直接导入采集模块。两套 vendor 虽相同，分别被罐任务和 Spatial 证据引用，故保留。以下各功能只保留最终有效实现：

| 功能 | 固定组件 | 说明 |
|---|---|---|
| 罐任务采集、动作回放及审计 | [expansion_20261004_r1](versions/expansion_20261004_r1) | 受控初始化场景、r20 教师、r6 审计 |
| LeRobot 转换及全量读取验证 | [lerobot_expansion_20261004_r1](versions/lerobot_expansion_20261004_r1) | 固定引用已验收罐任务批次 |
| Spatial 采集及物理证据审核 | [spatial_pilot_20261004_r1](versions/spatial_pilot_20261004_r1) | 原生任务映射和准备阶段图像；旧 replay/video/batch 已删除 |
| Spatial 最终动作回放 | [spatial_replay_20261004_r1](versions/spatial_replay_20261004_r1) | 对齐第一步准备动作之后的渲染器创建时点 |
| 最终视频导出 | [spatial_video_20261004_r2](versions/spatial_video_20261004_r2) | 双视角、原生 prompt、真实成功/失败标签及完整解码 |

## 未解决问题与已解决问题

最终失败诊断见 [failure_analysis.json](runs/spatial_pilot_20261004_r1/failure_analysis.json)。这些问题在本存档中仍未修复：

| task | 根因定位 | 后续方向 |
|---|---|---|
| 1 | close 时右指垫碰到旁边小碗，首次非目标法向力 5.211N | 完整接近与闭合路径避碰，选择空闲侧碗沿 |
| 4 | 212 候选中 16 个双端 IK 通过，仍存在张开接近路径碰撞，最大预测穿透 37.469mm；原生抽屉已打开 | 验证避开柜体的侧向接近和取出路径 |
| 7 | 最终物体坐标原点水平距离 30.535mm，原生判据要求严格小于 30mm；放置末端包围盒误差 2.204mm 不代表原生成功 | 针对原生落点、姿态和释放漂移留足余量 |
| 9 | 191 候选中 1 个双端 IK 通过，但张开下降路径碰碗壁，预测穿透 11.800mm | 扩展抓取侧、姿态和路径搜索；不能据此断言机械臂不可达 |

回放时点问题已解决：采集准备图像引入的首次离屏渲染会触发 `sim.forward()`，旧回放创建渲染器太晚，产生状态/RGB 偏差。最终回放在第一步准备动作之后创建渲染器，检查首张准备图像，保持原误差阈值。8 条有动作轨迹全部精确重现。旧错误回放程序已删除；原始失败 JSON/log 作为问题证据留在 runs 中，不作为当前验收结果。

后续需要任务感知的初始化审计，区分合法叠放与实际穿插，并逐项修复教师、完成每任务 3—5 场景回归后再确定正式采集量与 split。最终策略验收目标仍为 Spatial 十任务各 20 次独立测试、总成功率 ≥90%，保留 Panda 对照。本次仅整理归档，没有修改控制器、夹爪参数、教师或判定阈值。

## 校验与使用

在服务器本目录执行只读完整核验：

```bash
python3 tools/verify_archive.py
```

[manifest.json](archive/final_20261005_r1/manifest.json) 是本最终版唯一有效的文件清单；`archive/final_20261005_r1/verification.json` 是归档交付时的核验结果。[cleanup_ledger.json](archive/final_20261005_r1/cleanup_ledger.json) 列出被删除或替代文件的路径、大小、SHA256 和原因。旧冻结脚本、旧 releases 和过程 Markdown 已移除；历史 JSON 中的旧路径与旧校验通过记录表示当时事实，不是当前校验入口。

统一入口 [tools/run.py](tools/run.py) 默认仅打印命令，添加 `--execute` 执行，保留原容器参数与进程退出码。输入/输出使用 `/app/agent-work-space/4-multi-task-collection/...` 容器路径。查看最终 Spatial 回放参数：

```bash
python3 tools/run.py --execute spatial-replay -- --help
```

原始数据受冻结清单保护；归档内已有输出不可重复覆盖。若复现或继续开发，应建立新的工作版本和输出目录，完成后另行归档。单次采集不是完整批次审计：需要保留真实退出码和 teacher.status，再做物理审核、回放及结构检查。罐任务批次入口 `run_coverage.py` / `validate.py` 原样保留；已删除依赖旧冻结包的 pipeline。Spatial `structural_audit.py` 是原审计源码的等字节归档，其输出固定且采用独占创建，不能在本冻结批次上直接重跑。

依赖本仓库 `robosuite`、`third_party/libero`、`libero:latest` 容器、`1.3-zhixing_gripper_solve/runtime/py310_mj326` 及 `2-task-collection/runtime/lerobot_py310_r1`；并非可脱离仓库单独运行的数据包。外部源码和运行时依赖指纹见 [dependencies.json](archive/final_20261005_r1/dependencies.json)。归档时服务器为 A100 40GB。外部资产、运行时及其他任务目录未清理。

服务器保留完整原始数据。本地工作区只同步源码、说明、审计和视频，未复制大体积 HDF5/NPZ/Parquet；本地不能代替服务器执行全量归档核验。
