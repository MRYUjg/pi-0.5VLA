#!/usr/bin/env python3
"""D4d: 修复 P0 编码阻断 —— src/openpi/policies/libero_policy.py 是 UTF-16LE，CPython 无法解析，
导致 --env LIBERO / --policy.config pi05_libero 全部起不来（容器内外都一样）。
同时给 compose 的 openpi_server 加 PYTHONPATH=/app/src 作为 import openpi 的兜底。"""
import pathlib
import shutil

ROOT = pathlib.Path("/mnt/data/vla-policy/openpi")

# ---- 1) libero_policy.py: UTF-16 -> UTF-8 ----
p = ROOT / "src/openpi/policies/libero_policy.py"
raw = p.read_bytes()
if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
    shutil.copy2(p, ROOT / "agent-work-space/logs/libero_policy.py.utf16.bak")
    text = raw.decode("utf-16")
    p.write_bytes(text.encode("utf-8"))
    print(f"[fixed] libero_policy.py UTF-16 -> UTF-8（{len(raw)} -> {p.stat().st_size} B，备份在 logs/）")
else:
    print("[skip] libero_policy.py 已是 UTF-8")
import ast
ast.parse(p.read_text(encoding="utf-8"))
print("[ok] ast.parse 通过")

# ---- 2) compose: openpi_server 加 PYTHONPATH（编辑器兜底 import openpi）----
c = ROOT / "examples/libero/compose.yml"
t = c.read_text(encoding="utf-8")
if "/app/src" not in t:
    t = t.replace(
        "      - OPENPI_DATA_HOME=/openpi_assets\n      - IS_DOCKER=true",
        "      - OPENPI_DATA_HOME=/openpi_assets\n      - IS_DOCKER=true\n      # [server-local] 兜底：容器内 uv run 若跳过 sync，import openpi 仍可用\n      - PYTHONPATH=/app/src",
        1,
    )
    c.write_text(t, encoding="utf-8")
    print("[patched] examples/libero/compose.yml  (+PYTHONPATH=/app/src)")
else:
    print("[skip] compose 已有 /app/src")

# ---- 3) 全局扫描还有没有其它 UTF-16 源文件 ----
bad = []
for f in ROOT.rglob("*"):
    if not f.is_file() or "__pycache__" in str(f) or ".git/" in str(f):
        continue
    if f.suffix.lower() in {".py", ".yml", ".yaml", ".json", ".toml", ".txt"}:
        head = f.read_bytes()[:2]
        if head in (b"\xff\xfe", b"\xfe\xff"):
            bad.append(str(f.relative_to(ROOT)))
print("[scan] 仍是 UTF-16 的文件:", bad or "无")
