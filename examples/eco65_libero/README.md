# ECO65 + RethinkGripper LIBERO Benchmark

使用 ECO65 六轴机械臂 + RethinkGripper 夹爪运行 LIBERO 任务套件。

**复用 libero 的 Docker 镜像**，仅通过 compose 覆盖启动命令。
不构建额外镜像。

## 文件说明

| 文件 | 作用 |
|------|------|
| `main.py` | ECO65 推理脚本，结构与 `examples/libero/main.py` 一致 |
| `patch_libero.py` | 容器启动时运行，替换 LIBERO 源码中的 Panda -> ECO65 |
| `compose.yml` | 引用 `examples/libero/Dockerfile` 构建，覆盖 CMD |

## 与 libero 的差异

- Docker 镜像: **完全相同** (`examples/libero/Dockerfile`)
- runtime 启动命令: 先 `patch_libero.py` 再 `main.py`
- state 维度: ECO65 发送 7 维，`Normalize` 自动截断 `stats[:7]`
- 无需手动补零

## 启动

```bash
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./checkpoints/pi05_libero"
export CLIENT_ARGS="--task-suite-name libero_spatial"
docker compose -f examples/eco65_libero/compose.yml up
```

## 局限性

- 复用 Panda 归一化统计量，未微调，成功率会低于 Panda 原生
- ECO65 必须在 robosuite 中正确注册（含 OSC_POSE 控制器）
