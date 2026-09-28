#!/bin/bash
# 把 rsync 中断后残留的临时件 .<name>.XXXXXX 归一化为正式名
# 用法: normalize_temp.sh <dir> [name]   （不给 name 则处理目录下所有临时件）
D="$1"; N="$2"
if [ -n "$N" ]; then
  for t in "$D"/."$N".*; do
    [ -f "$t" ] || continue
    if [ -f "$D/$N" ]; then rm -f "$t"; else mv "$t" "$D/$N"; fi
  done
else
  for t in "$D"/.*; do
    [ -f "$t" ] || continue
    b=$(basename "$t"); b=${b#.}; b=${b%.??????}
    if [ -f "$D/$b" ]; then rm -f "$t"; else mv "$t" "$D/$b"; fi
  done
fi
