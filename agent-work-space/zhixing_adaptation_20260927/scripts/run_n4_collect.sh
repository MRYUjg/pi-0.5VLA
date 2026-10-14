#!/bin/bash
# N4 采集入口：按契约试采。**默认不删除任何已有数据**。
#
# 用法（服务器仓库根目录）：
#   docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
#     /bin/bash -c "source /.venv/bin/activate && bash /app/.../run_n4_collect.sh 5 16 n4v2"
#
# 位置参数：WANT(成功条数) MAXATT(最大尝试数) TAG(标签，同时决定文件名前缀)
# 可选环境变量：SUITE(默认 libero_object) TASK_ID(默认 0)
#
# [R5 修复] 原实现第 10 行为 `rm -rf "$ROOT/raw/success"/* "$ROOT/raw/failure"/*`，
# 在 Python 的"拒绝覆盖"检查之前执行，导致每次采集都会清空全部既有成果，
# 换 tag 也不保护旧文件（审核记录 R5）。现改为：
#   1) 不删除任何数据；
#   2) tag 冲突时**拒绝运行**（非 0 退出），避免覆盖；
#   3) 删除数据必须走独立、显式的 scripts/clean_n4_data.sh。
set -euo pipefail
ROOT="${ROOT:-/app/agent-work-space/zhixing_adaptation_20260927}"
WANT="${1:-5}"; MAXATT="${2:-16}"; TAG="${3:-n4}"
SUITE="${SUITE:-libero_object}"; TASK_ID="${TASK_ID:-0}"

echo "[run_n4] tag=$TAG want=$WANT max_attempts=$MAXATT suite=$SUITE task=$TASK_ID"

# [R5] 拒绝覆盖：同 tag 同 suite 同 task 已有数据时直接退出。
shopt -s nullglob
existing=("$ROOT/raw/success/${TAG}_${SUITE}_t${TASK_ID}_"*.npz \
          "$ROOT/raw/failure/${TAG}_${SUITE}_t${TASK_ID}_"*.npz)
shopt -u nullglob
if [ "${#existing[@]}" -gt 0 ]; then
  echo "[run_n4] 拒绝运行：tag=$TAG 下已存在 ${#existing[@]} 个 episode 文件，例如：" >&2
  printf '  %s\n' "${existing[@]:0:3}" >&2
  echo "[run_n4] 请改用新 tag（推荐），或先显式执行：bash scripts/clean_n4_data.sh <tag>" >&2
  exit 2
fi

mkdir -p "$ROOT/raw/success" "$ROOT/raw/failure"
cd "$ROOT"
exec python scripts/n4_collect.py --suite "$SUITE" --task-id "$TASK_ID" \
  --want "$WANT" --max-attempts "$MAXATT" --tag "$TAG"
