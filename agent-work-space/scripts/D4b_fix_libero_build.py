#!/usr/bin/env python3
"""D4b: 修复 libero 镜像在 Python 3.8 下的 sdist 构建失败
根因: build 隔离环境解析到不支持 py3.8 的新版 setuptools（PEP585 语法在 3.8 报 TypeError）。
修复: 用 --build-constraint 把构建期 setuptools 钉在 <76；顺带给 uv 加 cache mount 让重试不必重新下载。"""
import pathlib

ROOT = pathlib.Path("/mnt/data/vla-policy/openpi")

# 1) 构建期约束文件
cst = ROOT / "examples/libero/build-constraints.txt"
cst.write_text(
    "# [server-local] 构建隔离环境用的约束：setuptools>=76 起不再支持 Python 3.8，\n"
    "# 会在 sdist 构建时因 PEP585 语法报 \"'type' object is not subscriptable\"。\n"
    "setuptools<76\n"
    "wheel<0.46\n",
    encoding="utf-8",
)
print(f"[written] {cst.relative_to(ROOT)}")

# 2) Dockerfile 打补丁
df = ROOT / "examples/libero/Dockerfile"
t = df.read_text(encoding="utf-8")

old_copy = "COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml"
new_copy = (
    "COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml\n"
    "COPY ./examples/libero/build-constraints.txt /tmp/build-constraints.txt"
)
if "build-constraints.txt" not in t:
    t = t.replace(old_copy, new_copy, 1)

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
if "build-constraint" not in t:
    assert old_run in t, "未找到待替换的 uv pip sync 段"
    t = t.replace(old_run, new_run, 1)

df.write_text(t, encoding="utf-8")
print("[patched] examples/libero/Dockerfile")
for i, ln in enumerate(t.splitlines(), 1):
    if any(k in ln for k in ("build-constraint", "cache,target", "sed -i -E", "COPY ./examples/libero/build")):
        print(f"  {i}: {ln.strip()[:130]}")
