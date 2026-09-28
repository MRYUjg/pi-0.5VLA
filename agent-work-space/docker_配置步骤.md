# Docker 镜像配置步骤（参考官方 `docs/docker.md`）

| 项目 | 内容 |
| --- | --- |
| 目标 | 在 `galaxy-mobile`（223.109.239.30:20920，用户 `vipuser`）上按官方 `docs/docker.md` 装好 Docker + NVIDIA 运行时，并构建 openpi 的镜像 |
| 参考 | 仓库内 `docs/docker.md`（与 GitHub main 分支一致，已比对）、`scripts/docker/install_docker_ubuntu22.sh`、`scripts/docker/install_nvidia_container_toolkit.sh` |
| 代码目录 | `/mnt/data/vla-policy/openpi` |
| 文档状态 | **配置方案，尚未执行**。文中所有环境与网络数据为 2026-09-26 实测 |
| 关键结论 | sudo 可用；**snap 版 docker 正在运行，官方文档明确它与 NVIDIA Container Toolkit 不兼容，必须先移除**；服务器上没有任何 docker 镜像/容器，移除无损失 |

---

## 0. 官方要求 ↔ 服务器现状

| 官方 `docs/docker.md` 要求 | 服务器现状【实测】 | 结论 |
| --- | --- | --- |
| 安装 Docker Engine（官方 apt 源） | 装的是 Ubuntu 仓库的 `docker.io 20.10.21` **且** snap `docker 29.8.0` 同时在跑 | 需重装：清理后装 `docker-ce` |
| **必须 rootless 模式** | 当前是 rootful（systemd + `/var/run/docker.sock`） | 见 §2 决策点 1：建议 rootful，理由见下 |
| 必须装 NVIDIA Container Toolkit | 未安装 | 需安装 |
| **不能用 snap 版 docker**（会拿不到 `libnvidia-ml.so`） | snap docker 29.8.0 **正在运行并占用 socket** | **必须移除** |
| 不能用 Docker Desktop | 未安装 | ✓ 无需处理 |
| Ubuntu 22.04 可用便捷脚本 | 系统是 Ubuntu 20.04.4（focal） | 脚本逻辑通用（用 `$VERSION_CODENAME`），但需按 §4 调整 |
| `docker compose -f examples/<name>/compose.yml up --build` | `docker compose` **插件不存在**（只有 snap 的 `/snap/bin/docker-compose` v5.5.1） | 装 `docker-compose-plugin` 后可用 |

**关于"必须 rootless"**：官方文档正文要求 rootless，但官方自己提供的便捷脚本 `scripts/docker/install_docker_ubuntu22.sh` 装的是 **rootful** docker-ce 并把用户加进 `docker` 组。本机建议跟随便捷脚本（rootful），原因：
1. `examples/libero/compose.yml` 使用了 `privileged: true`，rootless 模式下该选项无法真正生效（容器能力被限制在用户命名空间内）；
2. rootless + NVIDIA 运行时需要额外配置（`nvidia-ctk` 的 rootless 模式、`/etc/subuid`、systemd user session + linger）；
3. 这是单用户开发服务器（`/mnt/data` 下只有几个项目目录），rootful + docker 组是常规做法。
若必须严格 rootless，见 §2 的备选清单。

---

## 1. 侦察结论（已完成的只读检查）

```text
# 权限
uid=1000(vipuser) groups=...,27(sudo),...        → 在 sudo 组；sudo 用同一个密码验证通过 ✅

# docker 现状
/usr/bin/docker          20.10.21（apt docker.io）
/snap/bin/docker         29.8.0（snap，正在运行，占用 /var/run/docker.sock）
/usr/bin/dockerd         root  pid 1167  -H fd://              （apt 版）
/snap/docker/3613/bin/…  root  pid 5170  --data-root=/var/snap/docker/…（snap 版）
/var/run/docker.sock     srw-rw---- root:docker  → vipuser 不在 docker 组，故 permission denied
docker compose           不存在（只有 snap 的 docker-compose v5.5.1）
docker ps -a / images    空 —— 两个 daemon 都没有任何镜像与容器
/var/lib/docker          244K       /var/snap/docker/…  372K   → 移除 snap docker 不会丢东西
```

网络实测（服务器侧）：

| 目标 | 速率 | 用途 |
| --- | --- | --- |
| `registry-1.docker.io` / `auth.docker.io` | 可达，**拉层 3.06 MiB/s** | 拉基础镜像 ✅ 不需要镜像加速器 |
| `ghcr.io` | 可达（HTTP 401 = 正常） | 拉 `ghcr.io/astral-sh/uv` ✅ |
| `download.docker.com` | 可达但仅 **0.20 MiB/s** | 装 docker-ce 很慢 |
| `mirrors.aliyun.com/docker-ce` | 可达，**1.36 MiB/s** | ✅ 改用这个装 docker-ce |
| `nvidia.github.io`（toolkit 源 + deb） | 可达，**0.41 MiB/s** | 装 NVIDIA toolkit（约 50 MB，2~3 分钟） |
| `archive.ubuntu.com` | 仅 **0.20 MiB/s** | 容器内 `apt-get install` 会极慢 → 必须换镜像 |
| `mirrors.aliyun.com/ubuntu` | **2.23 MiB/s** | 构建时 apt 源 |
| `pypi.org` 直连 | 0.58 MiB/s | 构建时 pip 源 |
| `pypi.tuna.tsinghua.edu.cn` | **3.32 MiB/s** | 构建时 pip 源 |
| `download.pytorch.org` | **2.21 MiB/s** | torch cu113 轮子，够快，可保留 |
| 服务器 7890/7891 端口 | **无监听** | compose 里写死的 `http_proxy=127.0.0.1:7891` 必须删掉 |

磁盘：`/` 可用 136 G，`/mnt/data` 可用 350 G。

---

## 2. 决策点（执行前需要确认）

**决策 1：rootful（推荐）还是 rootless（文档字面要求）**
- 推荐 rootful：`docker-ce` + `usermod -aG docker vipuser`。
- rootless 备选额外需要：`sudo apt install uidmap`、`/etc/subuid`&`/etc/subgid` 中给 `vipuser` 分配 65536 个 ID、`loginctl enable-linger vipuser`、`dockerd-rootless-setuptool.sh install`，并接受 `privileged: true` 失效、NVIDIA 运行时另行配置。

**决策 2：docker 数据盘放哪里**
- `/`（默认 `/var/lib/docker`）可用 136 G，镜像构建后约需 30~35 G，够用。
- `/mnt/data/docker` 可用 350 G，更宽裕，但需要写 `/etc/docker/daemon.json` 的 `data-root`（改动前该文件只有 nvidia runtime 配置，合并即可）。
- 建议：**迁到 `/mnt/data/docker`**（与项目数据同盘，避免以后镜像多了把根盘写满）。

**决策 3：是否允许修改仓库内的 Dockerfile / compose（构建必需）**
服务器网络条件下，**不改 Dockerfile 几乎必然失败或耗时数小时**（apt 0.2 MiB/s）。需要四处修改，见 §5。

---

## 3. 步骤 D1：清理冲突的 Docker

```bash
# 1) 停掉并移除 snap 版 docker（官方文档要求；已确认无镜像/容器）
sudo snap stop docker
sudo snap remove docker

# 2) 停掉 apt 版 docker.io 与其 socket
sudo systemctl stop docker.service docker.socket containerd.service
sudo systemctl disable docker.service docker.socket containerd.service
sudo apt-get remove -y docker.io containerd runc
sudo apt-get autoremove -y

# 3) 确认没有残留 daemon 与 socket
ps aux | grep -E "dockerd|containerd" | grep -v grep || echo "no docker daemon ✅"
ls -la /var/run/docker.sock 2>&1
```

> ⚠️ 这一步会移除 snap docker。已核实两个 daemon 都没有镜像/容器（数据目录分别只有 372 K / 244 K），且 `snap list` 里的 `lxd` 是独立 snap，不受影响。若仍有顾虑，可先执行 `sudo docker ps -a`（当前 socket 属于 snap daemon）确认。

## 4. 步骤 D2：安装 docker-ce（用阿里云镜像源）

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# GPG key（走阿里云镜像，避免 download.docker.com 的 0.2 MiB/s）
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 官方 apt 源（focal = Ubuntu 20.04）
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu focal stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 免 sudo 使用 docker（重新登录后生效）
sudo usermod -aG docker vipuser
sudo systemctl enable --now docker

# 当前 SSH 会话还没继承新组，先用 sudo 验证；或执行 newgrp docker
sudo docker version
sudo docker compose version
sudo docker run --rm hello-world
```

**注意**：装完 `docker-ce` 后应确认 snap 版没有回来（`which -a docker`、`docker context ls` 只应有 `default`）。

## 5. 步骤 D3：安装 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**（决策 2 = 迁盘时）在 `nvidia-ctk` 之后合并 daemon.json**：

```bash
sudo python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/etc/docker/daemon.json")
cfg = json.loads(p.read_text()) if p.exists() else {}
cfg["data-root"] = "/mnt/data/docker"
cfg["log-driver"] = "json-file"
cfg["log-opts"] = {"max-size": "50m", "max-file": "3"}
p.write_text(json.dumps(cfg, indent=2))
print(p.read_text())
PY
sudo mkdir -p /mnt/data/docker
sudo systemctl restart docker
docker info | grep -E "Docker Root Dir|Runtimes|Default Runtime"
```

**验证 GPU 直通**（官方文档的核心验收点）：

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
# 期望输出与宿主机 nvidia-smi 一致：RTX 3090 / 24576 MiB / driver 535.129.03
```

## 6. 步骤 D4：构建前的仓库修补（**必做**）

原因：容器内构建要走外网，服务器上 `archive.ubuntu.com` 只有 0.20 MiB/s、PyPI 直连 0.58 MiB/s，且 compose 里写死了本机不存在的代理。

### 补丁 1：`examples/libero/compose.yml` 是 UTF-16，compose 无法解析

```bash
cd /mnt/data/vla-policy/openpi
cp examples/libero/compose.yml examples/libero/compose.yml.utf16.bak
python3 - <<'PY'
import pathlib
p = pathlib.Path("examples/libero/compose.yml")
p.write_bytes(p.read_bytes().decode("utf-16").encode("utf-8"))
print("rewritten as UTF-8")
PY
```

### 补丁 2：删除写死的 `http_proxy/https_proxy=127.0.0.1:7891`

`examples/libero/compose.yml` 的 `runtime` 与 `openpi_server` 两个服务里各有两行。服务器没有 7891 监听，保留会让容器内所有外网请求（pip、`gs://` 下载、HF）直接失败。

```bash
sed -i '/http_proxy=http:\/\/127.0.0.1:7891/d; /https_proxy=http:\/\/127.0.0.1:7891/d' examples/libero/compose.yml
grep -n "proxy" examples/libero/compose.yml || echo "proxy 已清理 ✅"
```
（`examples/libero_eco65`、`examples/eco65_sim` 的 compose 同样有，若之后要用一并清理。）

### 补丁 3：两个 Dockerfile 的 apt 源换成阿里云

在 `examples/libero/Dockerfile` 与 `scripts/docker/serve_policy.Dockerfile` 的 `RUN apt-get update` **之前**插入：

```dockerfile
RUN set -eux; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do \
      if [ -f "$f" ]; then \
        sed -i -E 's@https?://([a-z0-9.-]*\.)?archive\.ubuntu\.com@https://mirrors.aliyun.com@g; \
                   s@https?://security\.ubuntu\.com@https://mirrors.aliyun.com@g' "$f"; \
      fi; \
    done; \
    cat /etc/apt/sources.list 2>/dev/null || true
```

### 补丁 4：pip / uv 源换成清华 TUNA

**`examples/libero/Dockerfile`**（第 46 行左右）：

```dockerfile
RUN uv pip sync /tmp/requirements.txt /tmp/requirements-libero.txt /tmp/openpi-client/pyproject.toml \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --extra-index-url https://download.pytorch.org/whl/cu113 \
      --index-strategy=unsafe-best-match
```
（`download.pytorch.org` 实测 2.21 MiB/s，够快，保留官方源即可。）

**`scripts/docker/serve_policy.Dockerfile`**：在 `uv sync` 前加 `ENV UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`。
但该 Dockerfile 用的是 `uv sync --frozen`，而 `uv.lock` 里记录了 **2245 条 `https://files.pythonhosted.org` 与 324 条 `https://pypi.org/simple`**，`--frozen` 会按锁文件里的 URL 下载（0.58 MiB/s，6~8 GB → 3~4 小时）。两个方案：

- **方案 ①（推荐，改动最小）**：把锁文件里的 URL 整批改写成清华镜像（TUNA 的 `/packages/...` 路径与 `files.pythonhosted.org` 一致，文件哈希相同），并让 Dockerfile 里的默认源与之一致：
  ```bash
  cp uv.lock uv.lock.orig
  sed -i 's#https://pypi.org/simple#https://pypi.tuna.tsinghua.edu.cn/simple#g; \
          s#https://files.pythonhosted.org#https://pypi.tuna.tsinghua.edu.cn#g' uv.lock
  grep -c "tuna.tsinghua" uv.lock     # 期望 ≈2569
  ```
  保留 `--frozen`。风险：uv 若判定"锁文件与索引配置不一致"而报错 → 改用方案 ②。
- **方案 ②（保底）**：去掉 `--frozen`，并把 `--mount=type=bind,source=uv.lock,target=uv.lock` 改成 `...,rw`（或改成 `COPY`），让 uv 按清华源重新解析。副作用：仓库里的 `uv.lock` 会被改写（先备份）。

### 补丁 5（很可能需要）：升级 `uv` 镜像版本

`uv.lock` 头部是 `version = 1  revision = 3`，而 `scripts/docker/serve_policy.Dockerfile` 用的是 `ghcr.io/astral-sh/uv:0.5.1`。`revision = 3` 是较新 uv 写入的，0.5.1 可能无法解析。若构建时报锁文件版本/字段错误，把两处 `0.5.1` 换成较新版本（例如 `ghcr.io/astral-sh/uv:0.8.15`），或直接 `uv:latest` 并记录实际版本。

## 7. 步骤 D5：预置 π0.5 权重缓存（避免容器内下载失败）

容器内 `--env LIBERO` 默认加载 `gs://openpi-assets/checkpoints/pi05_libero`，会落到 `$OPENPI_DATA_HOME`（compose 里挂成 `/openpi_assets`）。两个原因必须**事先在宿主机放好**：

1. 服务器直连 GCS 只有 0.39 MiB/s，11.58 GiB 要 8 小时以上；
2. `download.py:108-120` 优先用 `gsutil`，容器里没有 `gsutil` 就回退 `gcsfs`，而该桶用 gcsfs 匿名访问**可能因鉴权失败**。

**缓存路径必须严格匹配 `download.py:60` 的规则** `$OPENPI_DATA_HOME / <netloc> / <path>`：

```text
$OPENPI_DATA_HOME/storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/{params,assets}
```

```bash
# 本地 Mac：镜像 GCS 前缀（脚本见 agent-work-space/scripts/mirror_gcs_prefix.py）
python3 agent-work-space/scripts/mirror_gcs_prefix.py \
  openpi-assets checkpoints/pi05_libero/ ~/openpi-assets/checkpoints/pi05_libero/

# 上传到宿主机缓存目录
rsync -avP --partial -e "ssh -p 20920" ~/openpi-assets/checkpoints/pi05_libero/ \
  vipuser@223.109.239.30:/mnt/data/vla-policy/caches/openpi/storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/

# 服务器校验：必须 16 个文件、约 11.58 GiB
find /mnt/data/vla-policy/caches/openpi/storage.googleapis.com/openpi-assets/checkpoints/pi05_libero -type f | wc -l
```
宿主机 `OPENPI_DATA_HOME` 就取 `/mnt/data/vla-policy/caches/openpi`（compose 用 `${OPENPI_DATA_HOME:-~/.cache/openpi}` 传入）。

> 缓存失效逻辑（`download.py:198-214`）的正则锚定在相对路径开头，而实际相对路径以 `storage.googleapis.com/` 开头，**不会命中失效规则**，因此手工放置的缓存会被直接复用。✅

## 8. 步骤 D6：构建镜像

```bash
cd /mnt/data/vla-policy/openpi
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
export MUJOCO_GL=egl

# 只构建不启动（会构建 libero 与 openpi_server 两个镜像）
sudo docker compose -f examples/libero/compose.yml build 2>&1 | tee agent-work-space/logs/D6_build.log

sudo docker images
```

镜像来源对照：

| 镜像 | Dockerfile | 内容 | 预估体积 |
| --- | --- | --- | --- |
| `libero` | `examples/libero/Dockerfile` | Python 3.8 + robosuite 1.4.1 + LIBERO + torch 1.11 cu113 + mujoco 3.2.3 | ~10–12 GB |
| `openpi_server` | `scripts/docker/serve_policy.Dockerfile` | Python 3.11.9 + `uv sync`（JAX cuda12 + torch 2.7.1）+ transformers 补丁 | ~12–15 GB |
| 基础层 | `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` | 两个镜像共用，只拉一次（压缩 1.93 GiB） | ~6 GB |
| 工具 | `ghcr.io/astral-sh/uv:0.5.1` | COPY 出 uv/uvx | ~20 MB |

**时间预算**（按实测速率）：基础镜像 ≈11 min；libero 的 apt ≈5 min + pip（torch 1.9 GB @2.2 MiB/s）≈20 min；openpi_server 的 apt ≈3 min + uv sync（5–6 GB @3.3 MiB/s）≈30 min。**首次构建合计约 70–90 分钟**，磁盘峰值约 35 GB。

## 9. 步骤 D7：验证与启动

```bash
cd /mnt/data/vla-policy/openpi
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi

# 1) GPU 直通
sudo docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi

# 2) libero 镜像：确认本地 robosuite 生效 + EGL 渲染可用
sudo docker run --rm --gpus all -v $PWD:/app -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero \
  libero /bin/bash -lc 'source /.venv/bin/activate && python -c "
import robosuite, mujoco; print(robosuite.__file__, robosuite.__version__, mujoco.__version__)
import mujoco; m=mujoco.MjModel.from_xml_string(\"<mujoco><worldbody><geom size=\\\".1\\\"/></worldbody></mujoco>\"); d=mujoco.MjData(m); mujoco.mj_forward(m,d); r=mujoco.Renderer(m,64,64); r.update_scene(d); print(\"EGL OK\", r.render().shape)"'

# 3) 启动一套（服务端 + LIBERO 客户端）
export SERVER_ARGS="--env LIBERO"
export CLIENT_ARGS="--args.host 127.0.0.1 --args.port 8000 --args.task-suite-name libero_spatial --args.num-trials-per-task 1 --args.video-out-path /app/agent-work-space/eval/docker_M2"
sudo docker compose -f examples/libero/compose.yml up
```

注意事项：
- `$PWD` 必须是仓库根目录（compose 用 `$PWD:/app`）。
- compose 里只有 `depends_on`、没有 healthcheck，客户端可能先于服务端就绪 → 先单独起 `openpi_server`，等出现监听日志后再起 `runtime`。
- 视频/图片写入容器内 `/app/agent-work-space/...`，即宿主机 `/mnt/data/vla-policy/openpi/agent-work-space/...`。
- 容器内跑的是 `examples/libero/main.py`（ECO65 + 知行夹爪主入口），与原生 venv 方案完全一致。

## 10. 回滚

```bash
# 拆除容器与镜像（保留缓存）
sudo docker compose -f examples/libero/compose.yml down
sudo docker rmi libero openpi_server

# 完全回到装 docker 之前
sudo systemctl stop docker docker.socket containerd
sudo apt-get remove -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo apt-get autoremove -y
sudo snap install docker      # 若要恢复 snap 版

# 恢复被改写的 compose / Dockerfile / uv.lock
cd /mnt/data/vla-policy/openpi
cp examples/libero/compose.yml.utf16.bak examples/libero/compose.yml   # 注意：备份是 UTF-16
cp uv.lock.orig uv.lock
```

## 11. 风险与未决项

| # | 风险 | 影响 | 处置 |
| --- | --- | --- | --- |
| 1 | 移除 snap docker | 若他人依赖 snap docker 会受影响 | 已确认无镜像/容器；执行前再向同事确认一次 |
| 2 | `uv.lock` 为 `revision = 3`，镜像内 uv 0.5.1 可能不兼容 | 构建报错 | 补丁 5：升级 uv 镜像 tag |
| 3 | `--frozen` 与镜像源冲突 | 构建从 pypi.org 慢速下载（3~4 h） | 方案 ① 改写锁文件 URL；失败转方案 ② |
| 4 | Dockerfile 改动偏离上游 | 后续同步上游需重新打补丁 | 每处改动加注释标注 `# [server-local]` |
| 5 | 基础镜像 12.2.2 与宿主驱动 535 | 兼容（535 支持 CUDA 12.2） | 已核对 ✅ |
| 6 | `privileged: true` + rootful | 生效 | 与官方便捷脚本一致 |
| 7 | 构建期间中断 | BuildKit 缓存可续 | 重跑同一条 `docker compose build` |
| 8 | 磁盘 | 峰值约 35 GB | 决策 2 迁到 `/mnt/data/docker` |
| 9 | 容器内 `gs://` 下载（gcsfs 鉴权） | 服务端起不来 | D5 预置缓存，或改用 `--policy.dir /openpi_assets/...` 显式本地路径 |

## 12. 与"原生 venv 方案"的关系

Docker 方案与 `openpi_复现方案.md` 的原生 venv 方案**目标一致、可互为回退**：

| | Docker（本文件） | 原生 venv |
| --- | --- | --- |
| 环境隔离 | 镜像自带 Python 3.8 / 3.11.9，最接近官方推荐 | 自建两个 venv + 离线 uv/Python 3.11 |
| 网络依赖 | 构建期需要外网（可换镜像解决） | 安装期需要外网（同样换镜像） |
| 前置权限 | **需要 sudo**（装 daemon + toolkit） | 不需要 sudo |
| 运行命令 | `docker compose up` | 两条终端命令 |
| 失败代价 | 需重建镜像 | 重装 venv |

**建议**：Docker 作为主路线（官方推荐、环境更稳），原生 venv 作为回退；两者共用同一份权重缓存目录与同一套命令行参数，评测结果可直接对比。

---

# 13. 实施记录（2026-09-26 实际执行）

## 13.1 各步骤结果

| 步骤 | 结果 |
| --- | --- |
| D1 清理 | snap docker 29.8.0（`--purge`）+ apt docker.io 20.10.21 移除，无镜像/容器损失 |
| D2 安装 | `docker-ce 28.1.1` + `containerd.io 1.7.27` + `compose v2.35.1`（阿里云 docker-ce 源）；`vipuser` 加入 docker 组，已可免 sudo |
| D3 NVIDIA | `nvidia-container-toolkit 1.20.1`；`daemon.json` 合并 `data-root=/mnt/data/docker` + 日志轮转；`docker run --gpus all nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 nvidia-smi` 容器内可见 RTX 3090 ✅ |
| D4 仓库修补 | 5 处补丁全部生效（见 13.3） |
| D5 权重 | Mac 下载 11.58 GiB（16 文件）→ 上传到 `/mnt/data/vla-policy/caches/openpi/openpi-assets/checkpoints/pi05_libero`，逐文件字节比对一致 ✅ |
| D6 构建 | `openpi_server` 12.7 GB、`libero` 9.41 GB，均构建成功 |
| D7 验证 | 镜像验证 + 仿真冒烟 + **完整一套跑通** ✅ |

## 13.2 端到端结果

```
openpi_server:  Restoring checkpoint 6.2 GiB / 9.87 s → norm stats OK → server listening on 0.0.0.0:8000
runtime:        Task suite: libero_spatial → 10 个任务 × 1 次 rollout，全部产出视频
                单 episode 约 25~35 s；Total episodes: 10；Total success rate: 0.0
产物:           agent-work-space/eval/docker_M2/ 下 20 个 mp4（10 rollout × agentview/wrist）
GPU:            服务端常驻 18.5 GB
```

**成功率 0/10 是预期结果，不是故障**，原因在方案 §8.3 已说明：

1. 权重是 `pi05_libero`（为 **Panda** 微调），仿真里是 **ECO65 + 知行夹爪**，属跨本体迁移；
2. `discrete_state_input=False` → 策略只以**图像 + 语言**为条件，而 ECO65 的相机位姿/视角与训练分布不同；
3. `examples/libero/main.py:97-101` 在 ECO65 分支跳过 `set_init_state`，改用随机初始化，与官方 benchmark 的固定初始状态不一致。

要拿到有意义的数字，需要走复现方案 M5（用数据微调）或先做相机/初始状态对齐，不能把 0/10 当作模型能力结论。

## 13.3 过程中发现并修复的真实缺陷

| # | 文件 | 症状 | 处理 |
| --- | --- | --- | --- |
| 1 | `src/openpi/policies/libero_policy.py` | **UTF-16LE**，CPython 无法解析 → `--env LIBERO` 完全起不来 | 转 UTF-8（4420 B），备份 `logs/libero_policy.py.utf16.bak` |
| 2 | `robosuite/utils/mjcf_utils.py` | 被误注入 `"site1", "site2"`，`new_element(tag="site", "site1", "site2", name=...)` **语法错误** → robosuite 无法 import | 用 robosuite 1.4.1 官方 wheel 原文件还原；全量 diff 确认 robosuite 其余仅 4 处有意改动 |
| 3 | 同上 | `MUJOCO_NAMED_ATTRIBUTES` 缺 `site1/site2` → `<connect>` 约束 site 引用未加前缀 → MuJoCo 报 `unknown element 'Left_Support_tip'` | 正确补上 `site1`/`site2`（原被改坏的文件里正是这个修补的半成品） |
| 4 | `examples/libero/compose.yml` | UTF-16 + 写死 `http_proxy=127.0.0.1:7891`（服务器无此端口） | 转 UTF-8 并删除代理行 |
| 5 | `examples/libero/Dockerfile` | py3.8 下 build 隔离环境解析到过新 setuptools，sdist 构建报 `'type' object is not subscriptable` | 新增 `build-constraints.txt`（`setuptools<76`）+ `--build-constraint`；顺带加 uv cache mount |
| 6 | `scripts/docker/serve_policy.Dockerfile` | `uv:0.5.1` 读不了 `uv.lock`（`revision = 3`） | 升到 `ghcr.io/astral-sh/uv:0.12.15`；`uv.lock` 内 2569 处 URL 换清华镜像 |

另有两个**我自己踩的坑**（已修正，记录备查）：

- `gs://` URL 的 netloc 是 **`openpi-assets`** 而不是 `storage.googleapis.com`，所以权重缓存必须放在
  `$OPENPI_DATA_HOME/openpi-assets/checkpoints/...`。一开始放错目录，容器退回 gcsfs 下载并报
  `gsutil not found, falling back to gcsfs`。
- `docker compose up -d <service>` 会在**其它服务环境变量变化时重建它们**。某次只导出了 `CLIENT_ARGS`
  没导出 `SERVER_ARGS`，导致服务端被重建、退回默认 `--env ALOHA_SIM` 去下载别的权重。
  **每次 compose 操作都要带齐 `OPENPI_DATA_HOME` / `SERVER_ARGS` / `CLIENT_ARGS`。**

## 13.4 日常启停

```bash
cd /mnt/data/vla-policy/openpi
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
export SERVER_ARGS="--env LIBERO"
export CLIENT_ARGS="--args.host 127.0.0.1 --args.port 8000 --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 --args.num-steps-wait 10 \
  --args.video-out-path /app/agent-work-space/eval/<run_name>"

docker compose -f examples/libero/compose.yml up -d        # 起服务端 + 客户端
docker compose -f examples/libero/compose.yml logs -f runtime
docker compose -f examples/libero/compose.yml down          # 停
```

客户端容器退出后不会自动重启（`Total episodes` 打印完即 Exit 0）；服务端容器常驻，重跑评测前
建议先 `down` 再 `up`，或单独 `up -d runtime`（注意 13.3 的 env 陷阱）。
