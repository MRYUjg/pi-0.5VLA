#!/bin/bash
# D7: 镜像验证（GPU / robosuite 本地副本 / EGL 渲染）
set -u
cd /mnt/data/vla-policy/openpi
echo "=== [D7] 镜像验证 $(date -Is) ==="

echo "--- 镜像列表 ---"
docker images

echo "--- 1) libero 镜像：Python/robosuite/夹爪注册 ---"
docker run --rm --gpus all \
  -v "$PWD":/app \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero \
  libero /bin/bash -lc 'source /.venv/bin/activate && python - <<PY
import sys
import robosuite, mujoco, libero
print("python     :", sys.version.split()[0])
print("robosuite  :", robosuite.__file__, robosuite.__version__)
print("mujoco     :", mujoco.__version__)
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.robots import REGISTERED_ROBOTS
print("ZhixingGripper:", "ZhixingGripper" in GRIPPER_MAPPING)
print("ECO65        :", "ECO65" in REGISTERED_ROBOTS)
import importlib.metadata as md
for p in ["torch","numpy","robosuite","mujoco","bddl","robomimic"]:
    try: print(f"{p:12s}:", md.version(p))
    except Exception as e: print(f"{p:12s}: (missing)")
PY'

echo "--- 2) libero 镜像：EGL 无头渲染 ---"
docker run --rm --gpus all -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  libero /bin/bash -lc 'source /.venv/bin/activate && python -c "
import mujoco
m = mujoco.MjModel.from_xml_string(\"<mujoco><worldbody><geom size=\\\".1\\\"/></worldbody></mujoco>\")
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64); r.update_scene(d)
print(\"EGL render OK:\", r.render().shape)"'

echo "--- 3) openpi_server 镜像：JAX/PyTorch ---"
docker run --rm --gpus all openpi_server /bin/bash -lc 'source /.venv/bin/activate && python -c "
import jax, torch, openpi.policies.libero_policy as lp
print(\"jax   :\", jax.__version__, jax.devices())
print(\"torch :\", torch.__version__, torch.cuda.is_available())
print(\"libero_policy:\", lp.__file__)"'

echo "=== [D7] 完成 $(date -Is) ==="
