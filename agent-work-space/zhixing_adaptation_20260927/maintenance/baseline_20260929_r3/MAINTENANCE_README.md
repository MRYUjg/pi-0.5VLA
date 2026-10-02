# maintenance/baseline_20260929_r3 — 台账与操作说明

本目录**不在受管范围内**（`n0_freeze.catalog` 只覆盖 `robosuite/`、`third_party/liberolibero`、节点 `scripts/`、`configs/` 与节点根部 `*.md`）。因此这里的文件不会进入任何 manifest，也不会改变冻结判定。

## 文件

| 文件 | 说明 |
|---|---|
| `deletion_plan.json` | 本次清理的唯一查询入口：113 条，含被删路径、删除前 SHA256、字节数与退役理由 |
| `cleanup_deleted.json` | 实际删除结果（113 条）与移除的空目录 |
| `raw_before.json` / `raw_after.json` / `raw_preservation.json` | `raw/` 46 条轨迹在清理前后的逐一 SHA256 对照；`identical_before_after = true` |
| `FINAL_VERSION.json` | 最终基线版本（`n0_20260929_r4`）的 manifest / integrity / archive 哈希、指针一致性、相对 r3 的变更、清理汇总 |

## 在服务器上重建/校验冻结（必须用容器）

服务器系统 `python3` 为 **3.8.10**，不支持 `Path.is_relative_to`，`n0_freeze.py` 会直接崩溃。必须在 `libero:latest` 容器内执行：

```bash
cd /mnt/data/vla-policy/openpi
docker run --rm --network none -v "$PWD:/app" -w /app --entrypoint /bin/bash libero:latest \
  -c "cd /app && python3 agent-work-space/zhixing_adaptation_20260927/scripts/n0_freeze.py verify --check-archive"
```

`manifests/freezes/<version>/` 为 `root:root 0700`；以 `vipuser` 手工核对哈希需 `sudo`，容器内以 root 运行则不受限。

## docker shim（新建版本时才需要）

`n0_freeze.py create` 的 `capture_environment()` 会调用 `docker image inspect` 与 `docker run`，而 libero 容器内没有 docker CLI。建立 r3/r4 时使用了一个只读 shim：

- 位置：`/shim/docker`（`755`），调用记录：`/shim/calls.log`；
- 只实现两条被调用的命令，`docker run` 分支直接以容器自身 `/.venv/bin/python` 执行探针；
- 返回的镜像 Id 与真实 `libero:latest` 一致：`sha256:1ac69906b0b877c9fe1c2964794aa5b90112850da01967a445658e41a958f284`；
- **shim 内的镜像元数据是硬编码的**：镜像一旦重建，须同步更新 shim 中的 `Id` / `Created`，否则环境记录会失真。

建立新版本的完整命令（r4 用的是 `--retire-list` 省略版）：

```bash
docker run --rm --network none -v "$PWD:/app" -w /app -v /shim:/shim \
  -e PATH=/shim:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --entrypoint /bin/bash libero:latest -c "cd /app && python3 \
  agent-work-space/zhixing_adaptation_20260927/scripts/n0_freeze.py create --version n0_YYYYMMDD_rN"
```

前置条件：`configs/frozen_protocol.json` 的 `freeze_version` 必须等于新版本号；删除受管源必须提供 `--retire-list`，且每条都要与前一版 manifest 的 SHA256 绑定并写明理由。

## 保留策略

`manifests/freezes/` 现有三个冻结包（r2 / r3 / r4，各约 438 MB）。r2 已非当前版本亦非直接前身，如需按「仅保留当前＋前身」收敛可删除 r2 包；其 manifest 已作为 `n0_20260929_r3/predecessor_manifest.json` 保留，删包不影响比对溯源。
