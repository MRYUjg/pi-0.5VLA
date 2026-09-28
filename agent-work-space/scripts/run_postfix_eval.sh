#!/bin/bash
# 修复基座自碰撞后的 π0.5 评测：ECO65 + PandaGripper，以及 ECO65 + ZhixingGripper 对照
set -u
cd /mnt/data/vla-policy/openpi
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
LOG=/mnt/data/vla-policy/openpi/agent-work-space/logs

run() {
  tag=$1; robot=$2; gripper=$3
  echo "############ $tag : $robot + $gripper ############"
  docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
    /bin/bash -c "source /.venv/bin/activate && python /app/agent-work-space/scripts/eval_matrix.py \
        --robot $robot --gripper $gripper --tag $tag \
        --task-suite-name libero_spatial --num-trials-per-task 1 --seed 7 \
        --out-root /app/agent-work-space/eval" 2>&1 | tee $LOG/eval_$tag.log | \
    grep -E 'robot_model|gripper =|==>'
  echo "############ $tag 结束 ############"
}

run F_eco65_panda_contactfix    ECO65 PandaGripper
run G_eco65_zhixing_contactfix  ECO65 ZhixingGripper
echo "POSTFIX_EVAL_DONE"
