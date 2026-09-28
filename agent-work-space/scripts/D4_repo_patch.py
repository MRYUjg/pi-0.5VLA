#!/usr/bin/env python3
"""D4: 构建前的仓库修补（compose 编码/代理、Dockerfile apt+pip 源、uv.lock 镜像 URL、uv 版本）"""
import pathlib
import re
import shutil

ROOT = pathlib.Path("/mnt/data/vla-policy/openpi")
APT_PATCH = (
    "RUN set -eux; \\\n"
    "    for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do \\\n"
    "      if [ -f \"$f\" ]; then \\\n"
    "        sed -i -E 's@https?://([a-z0-9.-]+\\.)?archive\\.ubuntu\\.com@https://mirrors.aliyun.com@g; "
    "s@https?://security\\.ubuntu\\.com@https://mirrors.aliyun.com@g' \"$f\"; \\\n"
    "      fi; \\\n"
    "    done\n"
)


def patch(path: pathlib.Path, fn):
    t = path.read_text(encoding="utf-8")
    new = fn(t)
    if new != t:
        path.write_text(new, encoding="utf-8")
        print(f"[changed] {path.relative_to(ROOT)}")
    else:
        print(f"[unchanged] {path.relative_to(ROOT)}")


# ---- 1) examples/libero/compose.yml: UTF-16 -> UTF-8, 删除写死的代理 ----
p = ROOT / "examples/libero/compose.yml"
raw = p.read_bytes()
if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
    shutil.copy2(p, p.with_suffix(".yml.utf16.bak"))
    p.write_bytes(raw.decode("utf-16").encode("utf-8"))
    print("[changed] examples/libero/compose.yml  (UTF-16 -> UTF-8)")

text = p.read_text(encoding="utf-8")
lines = [ln for ln in text.splitlines() if "127.0.0.1:7891" not in ln]
if len(lines) != len(text.splitlines()):
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[changed] examples/libero/compose.yml  (删除 {len(text.splitlines()) - len(lines)} 行代理环境变量)")
else:
    print("[unchanged] examples/libero/compose.yml  (无代理行)")


# ---- 2) examples/libero/Dockerfile: apt 源 + pip 源 ----
def libero_df(t: str) -> str:
    if "mirrors.aliyun.com" not in t:
        t = t.replace("RUN apt-get update && \\", APT_PATCH + "\nRUN apt-get update && \\", 1)
    old = ("uv pip sync /tmp/requirements.txt /tmp/requirements-libero.txt /tmp/openpi-client/pyproject.toml "
           "--extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match")
    new = ("uv pip sync /tmp/requirements.txt /tmp/requirements-libero.txt /tmp/openpi-client/pyproject.toml \\\n"
           "    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \\\n"
           "    --extra-index-url https://download.pytorch.org/whl/cu113 \\\n"
           "    --index-strategy=unsafe-best-match")
    if old in t:
        t = t.replace(old, new, 1)
    return t


patch(ROOT / "examples/libero/Dockerfile", libero_df)


# ---- 3) scripts/docker/serve_policy.Dockerfile: apt 源 + pip 源 + uv 版本 ----
def serve_df(t: str) -> str:
    if "mirrors.aliyun.com" not in t:
        t = t.replace(
            "RUN apt-get update && apt-get install -y git git-lfs linux-headers-generic build-essential clang",
            APT_PATCH + "\nRUN apt-get update && apt-get install -y git git-lfs linux-headers-generic build-essential clang",
            1,
        )
    if "UV_DEFAULT_INDEX" not in t:
        t = t.replace("ENV UV_LINK_MODE=copy",
                      "ENV UV_LINK_MODE=copy\n# [server-local] 使用清华 PyPI 镜像（服务器直连 pypi.org 仅 0.58 MiB/s）\nENV UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple",
                      1)
    # uv.lock 是 version=1 revision=3（由 uv 0.12.x 写出），镜像内置的 0.5.1 无法解析
    t = t.replace("ghcr.io/astral-sh/uv:0.5.1", "ghcr.io/astral-sh/uv:0.12.15")
    return t


patch(ROOT / "scripts/docker/serve_policy.Dockerfile", serve_df)


# ---- 4) uv.lock: pypi.org / files.pythonhosted.org -> 清华镜像（配合 --frozen）----
lock = ROOT / "uv.lock"
t = lock.read_text(encoding="utf-8")
if "tuna.tsinghua" not in t:
    shutil.copy2(lock, ROOT / "uv.lock.orig")
    n1 = t.count("https://pypi.org/simple")
    n2 = t.count("https://files.pythonhosted.org")
    t = t.replace("https://pypi.org/simple", "https://pypi.tuna.tsinghua.edu.cn/simple")
    t = t.replace("https://files.pythonhosted.org", "https://pypi.tuna.tsinghua.edu.cn")
    lock.write_text(t, encoding="utf-8")
    print(f"[changed] uv.lock  (registry {n1} 处, wheel URL {n2} 处 -> TUNA)")
else:
    print("[unchanged] uv.lock  (已是镜像 URL)")


# ---- 5) 汇总 ----
print("\n=== 修补后关键行 ===")
for f in ["examples/libero/Dockerfile", "scripts/docker/serve_policy.Dockerfile"]:
    print(f"--- {f} ---")
    for i, ln in enumerate((ROOT / f).read_text(encoding="utf-8").splitlines(), 1):
        if any(k in ln for k in ("mirrors.aliyun", "tuna.tsinghua", "index-url", "astral-sh/uv", "UV_DEFAULT_INDEX")):
            print(f"  {i}: {ln.strip()[:120]}")
print("\n--- examples/libero/compose.yml 前 30 行 ---")
print("\n".join((ROOT / "examples/libero/compose.yml").read_text(encoding="utf-8").splitlines()[:30]))
