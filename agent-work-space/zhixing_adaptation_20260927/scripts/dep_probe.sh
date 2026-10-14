#!/bin/bash
# 容器内依赖版本探针（供 n0_fingerprint_current.py 调用）。
# 单独成文件是为了绕开 docker compose + /bin/bash -c 的多层引号嵌套
# ——旧实现在嵌套引号里把 `python` 变成了未定义名字，导致
# container_versions.ok=false（审核记录 §3.1）。
set -u
source /.venv/bin/activate 2>/dev/null || true
python - <<'PY'
import json, platform, sys
out = {"python": platform.python_version(), "executable": sys.executable}
for mod in ("numpy", "mujoco", "robosuite", "torch", "jax", "libero"):
    try:
        m = __import__(mod)
        out[mod] = getattr(m, "__version__", "unknown")
    except Exception as e:                      # noqa: BLE001
        out[mod] = "IMPORT_FAIL: %s" % type(e).__name__
print("DEP_PROBE_JSON " + json.dumps(out, ensure_ascii=False))
PY
