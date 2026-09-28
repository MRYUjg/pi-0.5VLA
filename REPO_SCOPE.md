# 仓库范围说明（第一版）

本仓库为 **ECO65 + ZhiXing 夹爪适配**工作的第一版快照，取自服务器
`/mnt/data/vla-policy/openpi` 的当前状态（2026-09-28）。

## 包含

- `agent-work-space/` —— 全部分节点文档、脚本、结果 JSON、审核与复验记录
  （`任务方案.md`、`N0-N4_节点审核记录`、`N0-N4_返工复核与优化建议`、
   `措施A/B/C2/C3_执行记录` 等）
- `robosuite/`、`third_party/libero/`、`examples/`、`scripts/`、`src/`、`my-robot/` —— 代码树
- **改动过的资产**（用 `git add -f` 强制加入，因为它们位于被 `.gitignore` 排除的 `assets/` 下）：
  - `robosuite/models/assets/robots/ECO65-B/`（含 `robot.xml`：基座↔link_1 自碰撞排除）
  - `robosuite/models/assets/grippers/zhixing_gripper.xml`（eef 前移至 0.20254）
  - `robosuite/models/assets/grippers/zhixing_gripper.urdf`
  - `robosuite/models/assets/grippers/meshes/zhixing_gripper/`

## 未包含（体积原因）

| 项 | 体积 | 原因 |
|---|---|---|
| `agent-work-space/**/raw/` | 4.3 GB | 采集数据（`*.npz`），单个文件最大 133 MB，超 GitHub 100 MB 硬限 |
| `agent-work-space/**/audit_stage/` | 475 MB | raw 的硬链接暂存 |
| `agent-work-space/eval/` | 48 MB | 评测视频（`*.mp4`） |
| `libreo_raw_data/` | 143 MB | 第三方数据集（含 113 MB tfrecord） |
| `ECO_data/` | 136 MB | 数据集压缩包 |
| `robosuite.zip` | 187 MB | 压缩包 |
| `4.28-openpi仿真复现.pptx` | 1 MB | 演示文稿 |
| `robosuite/models/assets/`（其余） | ~590 MB | 上游网格库，可由上游仓库重建 |
| `third_party/libero/**/assets/` | 416 MB | 上游 LIBERO 场景资源，可由上游仓库重建 |

> 因此**从本仓库直接 clone 不能独立运行**：需要补回上游 openpi / LIBERO 的网格资源。
> 如需把数据也纳入版本管理，建议用 Git LFS 另建数据仓库，不要混在代码仓库里。

## 已知状态（截至第一版）

- 措施 A/B/C 已执行；**N4 数据全部标记为「待验收 / 诊断证据」，不得进入训练集**。
- 措施 C 全矩阵在单一冻结版本（`eco65_no_extra_pd` + speed 0.01）下 **PASS**。
- 教师仍因 transport/place 目标点不可达而超时，**措施 D 未开始**。
- 详见 `agent-work-space/zhixing_adaptation_20260927/措施C3_带载运输定因_执行记录_20260928.md`
  及 `N0-N4_返工复核与优化建议_20260928.md`。
