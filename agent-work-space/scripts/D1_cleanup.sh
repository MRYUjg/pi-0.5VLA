#!/bin/bash
# D1: 清理与 NVIDIA Container Toolkit 冲突的 snap docker，以及 Ubuntu 仓库的 docker.io
set -u
echo "=== [D1] 开始 $(date -Is) ==="
echo "--- 移除前状态 ---"
snap list 2>/dev/null | grep -i docker || echo "(no docker snap)"
dpkg -l 2>/dev/null | grep -E "^ii\s+(docker\.io|containerd|runc)" | awk '{print $2, $3}'

echo "--- 预演 apt 移除（确认不会连带删除无关包）---"
DEBIAN_FRONTEND=noninteractive apt-get remove -s docker.io containerd runc 2>&1 | grep -E "^Remv|^The following" | head -20

echo "--- 1) 停止并移除 snap docker ---"
snap stop docker 2>&1 || true
snap remove --purge docker 2>&1 || true

echo "--- 2) 停止并禁用 apt docker.io 服务 ---"
systemctl stop docker.service docker.socket containerd.service 2>&1 || true
systemctl disable docker.service docker.socket containerd.service 2>&1 || true

echo "--- 3) 移除 apt docker.io ---"
DEBIAN_FRONTEND=noninteractive apt-get remove -y docker.io containerd runc 2>&1 | tail -8
DEBIAN_FRONTEND=noninteractive apt-get autoremove -y 2>&1 | tail -5

echo "--- 移除后校验 ---"
ps aux | grep -E "dockerd|containerd" | grep -v grep || echo "NO_DOCKER_DAEMON_OK"
ls -la /var/run/docker.sock 2>&1 || echo "NO_DOCKER_SOCK_OK"
which -a docker 2>&1 || echo "NO_DOCKER_BIN"
snap list 2>/dev/null | grep -i docker || echo "NO_DOCKER_SNAP_OK"
echo "=== [D1] 完成 $(date -Is) ==="
