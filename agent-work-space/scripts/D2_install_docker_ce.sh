#!/bin/bash
# D2: 从阿里云镜像源安装 docker-ce（官方 download.docker.com 实测仅 0.2 MiB/s）
set -eu
echo "=== [D2] 开始 $(date -Is) ==="

echo "--- 安装前置包 ---"
export DEBIAN_FRONTEND=noninteractive
apt-get update 2>&1 | tail -3
apt-get install -y ca-certificates curl gnupg 2>&1 | tail -3

echo "--- 添加 docker 官方 GPG key（阿里云镜像）---"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo "--- 添加 docker-ce apt 源（focal）---"
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu focal stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update 2>&1 | tail -5

echo "--- 安装 docker-ce + compose 插件 ---"
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 2>&1 | tail -12

echo "--- 加入 docker 组并开机自启 ---"
usermod -aG docker vipuser
systemctl enable --now docker
sleep 3

echo "--- 校验 ---"
docker version 2>&1 | head -20
echo "--- compose ---"
docker compose version
echo "--- hello-world ---"
docker run --rm hello-world 2>&1 | tail -6
echo "--- daemon.json（应为空或不存在）---"
cat /etc/docker/daemon.json 2>/dev/null || echo "(no daemon.json)"
echo "--- docker root dir ---"
docker info 2>/dev/null | grep -E "Docker Root Dir|Storage Driver|Cgroup"
echo "=== [D2] 完成 $(date -Is) ==="
