#!/bin/bash
# [R5] 显式数据清理工具（与采集入口分离）。
#
# 默认只**列出**匹配的 episode 文件，不删除；必须显式加 --yes 才真正删除。
#
# 用法：
#   bash scripts/clean_n4_data.sh <tag>            # 预览：列出将被删除的文件
#   bash scripts/clean_n4_data.sh <tag> --yes      # 真正删除
#   bash scripts/clean_n4_data.sh <tag> --yes --include-debug   # 连同 raw/_debug 一起
#
# 审核记录 R5 要求："清理应作为单独显式操作。" 本脚本即该操作。
set -euo pipefail
ROOT="${ROOT:-/app/agent-work-space/zhixing_adaptation_20260927}"

TAG="${1:-}"; shift || true
CONFIRM=0; INCLUDE_DEBUG=0
for a in "$@"; do
  case "$a" in
    --yes) CONFIRM=1 ;;
    --include-debug) INCLUDE_DEBUG=1 ;;
    *) echo "[clean] 未知参数：$a" >&2; exit 64 ;;
  esac
done

if [ -z "$TAG" ]; then
  echo "用法：bash scripts/clean_n4_data.sh <tag> [--yes] [--include-debug]" >&2
  echo "      <tag> 为 episode 文件名前缀，例如 n4、n4v2。" >&2
  echo "      **不带 --yes 时只预览，不删除。**" >&2
  exit 64
fi

shopt -s nullglob
targets=("$ROOT/raw/success/${TAG}_"*.npz "$ROOT/raw/failure/${TAG}_"*.npz)
if [ "$INCLUDE_DEBUG" -eq 1 ]; then
  targets+=("$ROOT/raw/_debug/${TAG}_"*.npz)
fi
shopt -u nullglob

if [ "${#targets[@]}" -eq 0 ]; then
  echo "[clean] tag=$TAG 无匹配文件，无需清理。"
  exit 0
fi

echo "[clean] tag=$TAG 匹配 ${#targets[@]} 个文件："
printf '  %s\n' "${targets[@]}"

if [ "$CONFIRM" -ne 1 ]; then
  echo
  echo "[clean] 预览模式：未删除任何文件。确认无误后加 --yes 重新执行。"
  exit 0
fi

rm -f -- "${targets[@]}"
echo "[clean] 已删除 ${#targets[@]} 个文件。"
