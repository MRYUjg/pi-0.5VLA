#!/usr/bin/env python3
"""D4c: 补齐 D4b 漏掉的 RUN 段替换（上次的判据 'build-constraint' 被 COPY 行误命中）"""
import pathlib

df = pathlib.Path("/mnt/data/vla-policy/openpi/examples/libero/Dockerfile")
t = df.read_text(encoding="utf-8")

old_run = """RUN uv pip sync /tmp/requirements.txt /tmp/requirements-libero.txt /tmp/openpi-client/pyproject.toml \\
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \\
    --extra-index-url https://download.pytorch.org/whl/cu113 \\
    --index-strategy=unsafe-best-match"""
new_run = """# [server-local] third_party/libero 里 robosuite 未固定版本，显式钉到 1.4.1（与 examples 侧一致）
RUN sed -i -E 's/^[[:space:]]*robosuite[[:space:]]*$/robosuite==1.4.1/' /tmp/requirements-libero.txt && grep -n robosuite /tmp/requirements-libero.txt
# [server-local] 加 uv 缓存挂载 + 构建期 setuptools 约束，避免重试时重复下载 ~2.5GB 轮子
RUN --mount=type=cache,target=/root/.cache/uv \\
    uv pip sync /tmp/requirements.txt /tmp/requirements-libero.txt /tmp/openpi-client/pyproject.toml \\
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \\
    --extra-index-url https://download.pytorch.org/whl/cu113 \\
    --index-strategy=unsafe-best-match \\
    --build-constraint /tmp/build-constraints.txt"""

if "--build-constraint" in t:
    print("[skip] RUN 段已包含 --build-constraint")
else:
    assert old_run in t, "未找到待替换的 uv pip sync 段"
    t = t.replace(old_run, new_run, 1)
    df.write_text(t, encoding="utf-8")
    print("[patched] RUN 段已替换")

print("\n=== 当前安装段 ===")
lines = df.read_text(encoding="utf-8").splitlines()
for i, ln in enumerate(lines, 1):
    if 44 <= i <= 62:
        print(f"{i:3d}: {ln}")
