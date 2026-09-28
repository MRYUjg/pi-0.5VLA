#!/usr/bin/env python3
"""把公开 GCS 前缀镜像到本地目录（匿名可读，无需 gsutil/凭证）。

用法（在能快速访问 GCS 的机器上运行，例如本地 Mac）：
    python3 mirror_gcs_prefix.py openpi-assets checkpoints/pi05_libero/ ~/openpi-assets/checkpoints/pi05_libero/

特性：
  * 可重复执行：按对象大小校验，已完整下载的文件跳过，损坏/不完整的重新下载
  * 断点续传：大文件先写入 <name>.part，用 HTTP Range 从断点继续，完成后原子改名
  * 自动重试：网络抖动导致的截断会重试（默认 8 次，指数退避）
"""
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

RETRIES = 8
CHUNK = 1 << 20


def list_prefix(bucket: str, prefix: str) -> list[tuple[str, int]]:
    """列出 bucket 下某前缀的所有 (key, size)（分页）。"""
    marker, out = "", []
    while True:
        url = (
            f"https://storage.googleapis.com/{bucket}"
            f"?prefix={urllib.parse.quote(prefix)}&max-keys=1000&marker={urllib.parse.quote(marker)}"
        )
        xml = urllib.request.urlopen(url, timeout=60).read().decode()
        keys = re.findall(r"<Key>([^<]+)</Key>", xml)
        sizes = re.findall(r"<Size>(\d+)</Size>", xml)
        out += list(zip(keys, (int(s) for s in sizes)))
        if "<IsTruncated>true</IsTruncated>" not in xml or not keys:
            return out
        marker = keys[-1]


def fetch(url: str, target: pathlib.Path, expected: int) -> None:
    """下载 url 到 target，支持断点续传；完成后校验大小。"""
    part = target.with_name(target.name + ".part")
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= expected:
            break
        headers = {"User-Agent": "curl/8.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=120)
            mode = "ab" if have and resp.status == 206 else "wb"
            if mode == "wb":
                have = 0
            with open(part, mode) as fh:
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
        except Exception as exc:  # noqa: BLE001
            got = part.stat().st_size if part.exists() else 0
            print(f"    ! 第 {attempt}/{RETRIES} 次中断（{got}/{expected} B）: {str(exc)[:90]}", flush=True)
            time.sleep(min(2 ** attempt, 60))
            continue
        if part.stat().st_size >= expected:
            break
        print(f"    ! 第 {attempt}/{RETRIES} 次不完整（{part.stat().st_size}/{expected} B），续传", flush=True)
        time.sleep(min(2 ** attempt, 60))

    got = part.stat().st_size if part.exists() else 0
    if got != expected:
        raise RuntimeError(f"{target.name}: 只拿到 {got}/{expected} B")
    part.replace(target)


def mirror(bucket: str, prefix: str, dest: str) -> None:
    dest_path = pathlib.Path(dest).expanduser()
    objects = list_prefix(bucket, prefix)
    total = sum(s for _, s in objects)
    print(f"{len(objects)} objects / {total / 1024**3:.2f} GiB -> {dest_path}", flush=True)
    for i, (key, size) in enumerate(objects, 1):
        rel = key[len(prefix):] if key.startswith(prefix) else key
        target = dest_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == size:
            print(f"[{i}/{len(objects)}] skip {rel} ({size} B)", flush=True)
            continue
        print(f"[{i}/{len(objects)}] {rel} ({size / 1048576:.1f} MiB)", flush=True)
        t0 = time.time()
        fetch(f"https://storage.googleapis.com/{bucket}/{urllib.parse.quote(key)}", target, size)
        dt = time.time() - t0
        print(f"    done in {dt:.0f}s ({size / 1048576 / max(dt, 1e-9):.2f} MiB/s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(2)
    mirror(sys.argv[1], sys.argv[2], sys.argv[3])
