#!/bin/bash
# 三格对照评测：同一脚本 / 同 seed / 同任务集，只改 robot + gripper
set -u
cd /mnt/data/vla-policy/openpi
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
LOG=/mnt/data/vla-policy/openpi/agent-work-space/logs
TRIALS=${TRIALS:-1}

run() {
  tag=$1; robot=$2; gripper=$3
  echo "############ $tag : $robot + $gripper (trials=$TRIALS) ############"
  docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
    /bin/bash -c "source /.venv/bin/activate && python /app/agent-work-space/scripts/eval_matrix.py \
        --robot $robot --gripper $gripper --tag $tag \
        --task-suite-name libero_spatial --num-trials-per-task $TRIALS --seed 7 \
        --out-root /app/agent-work-space/eval" 2>&1 | tee $LOG/eval_$tag.log | \
    grep -E '^\S*INFO:root:\[|robot_model|gripper =|==>' 
  echo "############ $tag 结束 ############"
}

run A_eco65_zhixing ECO65 ZhixingGripper
run B_eco65_panda    ECO65 PandaGripper
run C_panda_panda    Panda PandaGripper
echo "MATRIX_DONE"
