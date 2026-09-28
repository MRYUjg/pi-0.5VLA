#!/usr/bin/env python3
"""扫描仓库内所有 .py 的语法，报告无法解析的文件。"""
import ast
import pathlib
import sys

ROOT = pathlib.Path("/app")
roots = sys.argv[1:] or ["."]
skip_dirs = {"__pycache__", ".git", "node_modules"}
files = []
for r in roots:
    for f in (ROOT / r).rglob("*.py"):
        if any(part in skip_dirs for part in f.parts):
            continue
        files.append(f)

bad = []
for f in sorted(files):
    raw = f.read_bytes()
    try:
        ast.parse(raw)
    except SyntaxError as e:
        bad.append((str(f.relative_to(ROOT)), f"{e.msg} (line {e.lineno})"))
    except ValueError as e:
        bad.append((str(f.relative_to(ROOT)), f"{type(e).__name__}: {e}"))

print(f"python {sys.version.split()[0]} | 扫描 {len(files)} 个 .py 文件")
if bad:
    print(f"发现 {len(bad)} 个语法错误文件：")
    for p, msg in bad:
        print(f"  !! {p}: {msg}")
else:
    print("全部通过 ✅")
