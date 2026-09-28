#!/bin/bash
# 启动完整一套：openpi_server（策略服务） + runtime（LIBERO/ECO65 客户端）
set -u
cd /mnt/data/vla-policy/openpi
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
COMPOSE="docker compose -f examples/libero/compose.yml"
LOG=/mnt/data/vla-policy/openpi/agent-work-space/logs

echo "=== [1] 启动策略服务 ==="
export SERVER_ARGS="--env LIBERO"
$COMPOSE up -d openpi_server
echo "等待 8000 端口就绪（最多 15 分钟）..."
for i in $(seq 1 90); do
  if ss -ltn 2>/dev/null | grep -q ':8000'; then echo "port 8000 up (${i}0s)"; break; fi
  sleep 10
done
ss -ltn | grep ':8000' || echo "!! 8000 仍未监听"

echo "=== [2] 策略服务日志（最近 40 行）==="
$COMPOSE logs --tail=40 openpi_server

echo "=== [3] 启动 LIBERO/ECO65 客户端 ==="
export CLIENT_ARGS="--args.host 127.0.0.1 --args.port 8000 --args.task-suite-name libero_spatial --args.num-trials-per-task 1 --args.num-steps-wait 10 --args.video-out-path /app/agent-work-space/eval/docker_M2"
$COMPOSE up -d runtime
echo "客户端已启动（后台）"
sleep 20
$COMPOSE logs --tail=30 runtime
