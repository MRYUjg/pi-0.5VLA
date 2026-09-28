#!/bin/bash
# N2-v2 重测编排：在容器内顺序运行全部测量，每个步骤单独落日志。
# 不修改任何历史 results/*.json —— 全部输出到 results/n2v2_20260928/。
set -uo pipefail
ROOT=/app/agent-work-space/zhixing_adaptation_20260927
SC=$ROOT/scripts
OUT=$ROOT/results/n2v2_20260928
LOG=$OUT/logs
mkdir -p "$LOG"
cd "$ROOT"
source /.venv/bin/activate

echo "host_date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -c "import mujoco,robosuite,numpy;print('mujoco',mujoco.__version__,'robosuite',robosuite.__version__,'numpy',numpy.__version__)"

step () {
  local name="$1"; shift
  echo "=== [$name] $* ==="
  "$@" >"$LOG/$name.log" 2>&1
  local rc=$?
  echo "[$name] exit=$rc log=$LOG/$name.log"
  tail -n 6 "$LOG/$name.log"
  echo
}

step tcp_geometry python "$SC/n2v2_run_redirected.py" --which tcp_geometry --out "$OUT"
step pad_face     python "$SC/n2v2_run_redirected.py" --which pad_face     --out "$OUT"
step init_pose    python "$SC/n2v2_run_redirected.py" --which init_pose    --out "$OUT"
step audit        python "$SC/audit_tcp_camera.py" --out "$OUT"
step settled      python "$SC/n2v2_settled_probe.py" --out "$OUT"

echo "=== outputs ==="
ls -la "$OUT"
echo "=== done ==="
