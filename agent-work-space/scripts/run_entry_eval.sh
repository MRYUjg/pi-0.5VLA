#!/bin/bash
# 用真正的默认入口 examples/libero/main.py 跑两种夹爪，验证分派与实际效果
set -u
cd /mnt/data/vla-policy/openpi
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
LOG=/mnt/data/vla-policy/openpi/agent-work-space/logs

run() {
  g=$1; tag=$2
  echo "############ main.py --args.gripper $g ############"
  docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
    /bin/bash -c "source /.venv/bin/activate && python examples/libero/main.py \
        --args.host 127.0.0.1 --args.port 8000 \
        --args.gripper $g \
        --args.task-suite-name libero_spatial --args.num-trials-per-task 1 \
        --args.num-steps-wait 10 --args.seed 7 \
        --args.video-out-path /app/agent-work-space/eval/entry_$tag/videos" 2>&1 \
    | tee $LOG/entry_$tag.log | grep -E 'Gripper in use|Caught exception|Total success rate|Total episodes'
}
run PandaGripper  panda
run ZhixingGripper zhixing
echo "ENTRY_EVAL_DONE"
