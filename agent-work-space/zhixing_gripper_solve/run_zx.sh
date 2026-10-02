#!/bin/bash
# 本目录脚本的统一运行器（宿主机侧）。
#   ./run_zx.sh <脚本名> [参数...]
# 例：./run_zx.sh zhixing_aperture_metric.py --out /app/agent-work-space/zhixing_gripper_solve/evidence/aperture_$(date +%Y%m%d).json
set -eu
SCRIPT="${1:?用法: ./run_zx.sh <脚本名> [参数...]}"; shift || true
REPO=/mnt/data/vla-policy/openpi
SOLVE=/app/agent-work-space/zhixing_gripper_solve
cd "$REPO"
exec docker run --rm --network none -v "$REPO:/app" -w /app -e MUJOCO_GL=osmesa \
  --entrypoint /bin/bash libero:latest -c "
    source /.venv/bin/activate
    export PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero
    python $SOLVE/scripts/$SCRIPT $*"
