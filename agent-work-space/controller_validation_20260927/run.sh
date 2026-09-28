#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
docker run --rm --network=none --cpus=2 \
  -v "$ROOT:/app" -w /app \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  libero /bin/bash -c 'source /.venv/bin/activate && python -u /app/agent-work-space/controller_validation_20260927/validate_controller.py "$@"' audit "$@"
