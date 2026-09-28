#!/bin/bash
# D3: NVIDIA Container Toolkit + docker data-root 迁到 /mnt/data/docker + GPU 直通验证
set -eu
export DEBIAN_FRONTEND=noninteractive
echo "=== [D3] 开始 $(date -Is) ==="

echo "--- 添加 NVIDIA Container Toolkit 源 ---"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "--- 安装 nvidia-container-toolkit ---"
apt-get update 2>&1 | tail -3
apt-get install -y nvidia-container-toolkit 2>&1 | tail -6
nvidia-ctk --version

echo "--- 配置 docker 的 nvidia runtime ---"
nvidia-ctk runtime configure --runtime=docker

echo "--- 合并 data-root=/mnt/data/docker 与日志轮转 ---"
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/etc/docker/daemon.json")
cfg = json.loads(p.read_text()) if p.exists() and p.stat().st_size else {}
cfg["data-root"] = "/mnt/data/docker"
cfg["log-driver"] = "json-file"
cfg["log-opts"] = {"max-size": "50m", "max-file": "3"}
p.write_text(json.dumps(cfg, indent=2) + "\n")
print(p.read_text())
PY
mkdir -p /mnt/data/docker

echo "--- 重启 docker ---"
systemctl restart docker
sleep 6
docker info 2>/dev/null | grep -E "Docker Root Dir|Storage Driver|Cgroup Driver|Default Runtime|Runtimes" -A2

echo "--- 预拉基础镜像 + GPU 直通验证（同时为构建预热镜像层）---"
docker run --rm --gpus all nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 nvidia-smi

echo "--- 落盘检查 ---"
du -sh /mnt/data/docker 2>/dev/null
df -h / /mnt/data | tail -2
echo "=== [D3] 完成 $(date -Is) ==="
