#!/usr/bin/env bash
# 在 libero 容器里运行初始条件对齐工具（需要 GPU/EGL 渲染腕部图像；用 compose 服务以继承正确环境）
set -euo pipefail
cd /mnt/data/vla-policy/openpi
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/mnt/data/vla-policy/caches/openpi}
SUB=${1:-measure}; shift || true
docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
  /bin/bash -c "source /.venv/bin/activate && python -u /app/agent-work-space/initial_pose_alignment_20260927/align_initial_pose.py $SUB $*"
