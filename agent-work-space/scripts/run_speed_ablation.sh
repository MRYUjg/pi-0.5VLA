#!/bin/bash
# 夹爪开合速度消融：ZhixingGripper speed=0.01（对齐 PandaGripper）
set -u
cd /mnt/data/vla-policy/openpi
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
export SERVER_ARGS='--env LIBERO'
export CLIENT_ARGS=''
LOG=/mnt/data/vla-policy/openpi/agent-work-space/logs

echo "=== [1] 启动策略服务 ==="
docker compose -f examples/libero/compose.yml up -d openpi_server 2>&1 | tail -2
for i in $(seq 1 40); do
  ss -ltn 2>/dev/null | grep -q ':8000' && { echo "port 8000 up (${i}0s)"; break; }
  sleep 10
done
ss -ltn | grep -q ':8000' || { echo "!! 服务端未就绪"; exit 1; }

run() {
  tag=$1; robot=$2; gripper=$3; speed=${4:-}
  echo "############ $tag : $robot + $gripper  speed=${speed:-default} ############"
  if [ -n "$speed" ]; then EARG="-e ZHIXING_GRIPPER_SPEED=$speed"; else EARG=""; fi
  docker compose -f examples/libero/compose.yml run --rm --no-deps $EARG runtime \
    /bin/bash -c "source /.venv/bin/activate && python /app/agent-work-space/scripts/eval_matrix.py \
        --robot $robot --gripper $gripper --tag $tag \
        --task-suite-name libero_spatial --num-trials-per-task 1 --seed 7 \
        --out-root /app/agent-work-space/eval" 2>&1 | tee $LOG/eval_$tag.log | \
    grep -E 'robot_model|==>'
  echo "############ $tag 结束 ############"
}

# 用户要求的那一格
run F_eco65_zhixing_speed001 ECO65 ZhixingGripper 0.01
# 顺手加的对照格：同样的 speed 改动装在 Panda 上，用来判断"速度"是不是唯一原因
run G_panda_zhixing_speed001 Panda ZhixingGripper 0.01
echo "SPEED_ABLATION_DONE"
