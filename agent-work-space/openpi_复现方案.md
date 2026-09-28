# OpenPI π0.5 + ECO65 / ZhiXing 服务器复现方案 v1

| 项目 | 内容 |
| --- | --- |
| 目标服务器 | `galaxy-mobile` / `223.109.239.30:20920` / 用户 `vipuser` |
| 代码位置 | `/mnt/data/vla-policy/openpi`（已于 2026-09-26 上传：2697 个文件 / 1.27 GiB，校验一致） |
| 本 agent 工作目录 | `/mnt/data/vla-policy/openpi/agent-work-space` |
| 依据代码快照 | 本地 `/Users/yuyu/Documents/openpi`（含 ECO65 / ZhiXing 适配改动） |
| 编写日期 | 2026-09-26 |
| 文档状态 | 实施设计。**本方案中的命令均尚未在服务器上执行过**，服务器环境与网络数据为当日实测。 |

**证据分级**（全文沿用）：

| 标记 | 含义 |
| --- | --- |
| 【实测】 | 2026-09-26 在服务器或本地 Mac 上真实执行并观察到的结果 |
| 【代码】 | 直接阅读仓库源码/配置确认的事实（附文件:行号） |
| 【推断】 | 沿调用链推导，未运行验证 |
| 【待核验】 | 需要实际运行日志/硬件才能确认 |

---

## 0. 结论摘要（先看这一节）

1. **复现路径分 6 个阶段**，先做「官方 π0.5 LIBERO 权重 + ECO65/知行 仿真闭环」，再补数据与微调。这样在拿到训练数据之前就能验证整条 ECO65 适配链路。
2. **有 3 个必须在动手前处理的问题：**
   - **P0 阻断**：`src/openpi/policies/libero_policy.py` 是 **UTF-16LE** 编码，CPython 无法解析；而 `src/openpi/training/config.py:22` 会 import 它 → **当前副本上 `--env LIBERO` / `--policy.config pi05_libero` 根本起不来**。【实测】
   - **带宽瓶颈**：服务器国际出口实测 **GCS 0.39 MiB/s、HF 0.23 MiB/s、PyPI 直连 0.58 MiB/s**。π0.5 权重 **11.58 GiB** 直连下载需 8～13 小时。必须走「**本地 Mac 下载 → rsync 上传**」（本地↔服务器实测 **3.4 MB/s**，Mac 侧 GCS 5.6 MiB/s）。
   - **显存上限**：单卡 RTX 3090 24 GB。官方 README 标注全参微调需 >70 GB，**本机只能做 LoRA 微调**（README 标注 LoRA >22.5 GB，仍需实测与调小 batch）。
3. **Docker 路线当前不可用**：docker CLI 存在但守护进程无权限（`permission denied`），且 `sudo` 需要密码【实测】。本方案全部采用**原生 venv** 路线（官方 README 的 "Without Docker" 路径）。
4. Python 环境需要自己装：服务器只有系统 Python 3.8.10，**没有 pip 模块、没有 conda/uv/cmake**【实测】。用 uv 管理两个隔离环境：模型侧 Python 3.11、仿真侧 Python 3.8。
5. 装依赖请用**清华 TUNA 镜像（实测 3.32 MiB/s，比直连快 6 倍）**，PyTorch cu113 轮子用**阿里云 pytorch-wheels 镜像**。

---

## 1. 事实基线

### 1.1 服务器环境【实测】

| 项目 | 实测值 | 对本方案的影响 |
| --- | --- | --- |
| 操作系统 | Ubuntu 20.04.4 LTS，kernel 5.4.0-216-generic，x86_64 | 官方声明主要测试 Ubuntu 22.04，20.04 可用但属未支持组合，需记录 |
| GPU | NVIDIA GeForce RTX 3090，24576 MiB，driver 535.129.03（支持 CUDA 12.2） | JAX `jax[cuda12]==0.5.3` 可用（驱动 ≥525 即可，CUDA 运行时由 pip 轮子提供） |
| CUDA Toolkit | 仅 `/usr/local/cuda-11.8`，`nvcc` 不在 PATH | 不影响：JAX/PyTorch 都走 pip 自带 CUDA 运行时 |
| CPU / 内存 | 8 核 / 23 GiB（可用约 20 GiB） | 训练 dataloader `num_workers` 不宜超过 2～4 |
| 磁盘 | `/mnt/data` 1000 G，可用 **350 G**；`/` 197 G，可用 136 G | 充足；所有大文件放 `/mnt/data` |
| Python | 仅 `/usr/bin/python3.8`（3.8.10），**无 pip 模块** | 必须自带 uv；模型侧需自备 Python 3.11 |
| 工具链 | 有 git / g++ 9.4 / make；**无 conda、mamba、uv、cmake** | 见 §5.1 离线安装 uv + Python 3.11 |
| Docker | CLI 存在，守护进程 `permission denied`；`sudo` 需要密码 | **放弃 Docker 路线**，改用原生 venv |
| EGL 渲染 | `libEGL_nvidia.so.0 → 535.129.03`；`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` 存在；`libEGL.so.1`、`libGL.so.1` 存在；**无** `libOSMesa`/`libglfw`/`libglew` | MuJoCo 走 `MUJOCO_GL=egl` 无头渲染，前提已满足，无需 apt |
| 其他目录 | `/mnt/data/vla-policy/` 下另有 `GPT-as-Policy`；`/mnt/data` 有 navsim 等既有项目 | 端口/磁盘占用需注意共存 |

### 1.2 网络实测（服务器侧，2026-09-26）【实测】

| 目标 | 速率 | 备注 |
| --- | --- | --- |
| 清华 TUNA PyPI 镜像 | **3.32 MiB/s** | ✅ 首选 pip 源 |
| 阿里云 pytorch-wheels 镜像 | 可达（HTTP 200/206，含 `torch-1.11.0+cu113-cp38` 轮子） | ✅ cu113 torch 首选 |
| 本地 Mac → 服务器 rsync | **3.4 MB/s**（≈3.24 MiB/s） | 单流；4 路并发实测无增益 |
| `storage.googleapis.com`（openpi-assets） | 0.39 MiB/s | ⚠️ 大权重不要直连下载 |
| `huggingface.co` | 0.23 MiB/s | ⚠️ 数据集不要直连下载 |
| `files.pythonhosted.org`（PyPI 直连） | 0.58 MiB/s | 仅作兜底 |
| `github.com` release 资源 | 0.08 MiB/s | ⚠️ 尽量本地下载后上传 |

**带宽结论**：凡是 >100 MB 的资产，一律「**Mac 下载 → rsync 上传**」。Mac 实测 GCS 5.6 MiB/s。

### 1.3 已在服务器上的代码基线【实测】

| 内容 | 数量/大小 |
| --- | --- |
| 项目文件 | 2697 文件 / 1,361,536,053 B（1.27 GiB），与本地逐字节一致（md5 抽检通过） |
| `robosuite/`（本地改造版 robosuite 1.4.1） | 1050 文件 / 595 MB |
| `third_party/libero/`（LIBERO 含 assets/bddl/init_files） | 1117 文件 / 426 MB，其中 assets 404 MB、585 文件 |
| `ECO_data/` | 294 文件 / 135 MB，chunk-000（136 行 CSV）、chunk-001（151 行 CSV） |
| `libreo_raw_data/` | 19 文件 / 143 MB，**只含 1/16 个 TFRecord 分片** |
| 未上传 | `.git`（1.1 GB）、`robosuite.zip`（187 MB，与已解压目录重复）、`__pycache__`、`*.pyc` |

**LIBERO 评测所需的 bddl_files / init_files / assets 已随代码上传**，无需额外下载。

### 1.4 需要获取的目标资产

| 资产 | 来源 | 大小 | 用途 |
| --- | --- | --- | --- |
| `pi05_libero` checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` | **11.58 GiB**（16 objects） | 官方 LIBERO 微调权重，阶段 M1–M3 直接用 |
| `pi05_base` checkpoint | `gs://openpi-assets/checkpoints/pi05_base` | **11.59 GiB**（29 objects） | 阶段 M5 微调初始权重 |
| LeRobot LIBERO 数据集 | HF `physical-intelligence/libero` | **21.31 GiB**（996 文件） | 阶段 M4/M5（选项 A） |
| RLDS 原始数据 | HF `openvla/modified_libero_rlds` | **9.53 GiB**（其中 `libero_spatial_no_noops` ≈2.1 GiB） | 阶段 M4/M5（选项 B） |

checkpoint 目录结构（已在服务器上确认要镜像的键）【实测】：

```text
checkpoints/pi05_libero/
├── assets/physical-intelligence/libero/norm_stats.json   # 归一化统计量（推理必需）
└── params/                                               # orbax/JAX 参数（14 个对象）
```

---

## 2. 复现目标与验收口径

| 阶段 | 目标 | 产物 | 通过判据 | 前置 |
| --- | --- | --- | --- | --- |
| **M0** | 代码与工具链就绪 | 两个 venv、修复后的代码 | 模型环境 `import openpi` 成功；仿真环境 `import robosuite` 指向**本地副本** | — |
| **M1** | π0.5 服务可加载并推理 | 8000 端口 WebSocket 服务 + 日志 | 客户端拿到 `(10, 7)` 动作块，无 NaN | M0 + pi05_libero 权重 |
| **M2** | ECO65+知行 仿真闭环冒烟 | 1 段 rollout 视频 + 日志 | 环境能跑完整 episode 不崩；图像/状态/动作形状正确 | M1 |
| **M3** | 官方权重基线评测 | `libero_spatial` 10 任务 × N 次成功率 | 记录成功率、视频、每步日志（**注意初始化方式与官方不同，见 §8.3**） | M2 |
| **M4** | 数据管线打通 | LeRobot 数据集 + `norm_stats.json` | `compute_norm_stats.py` 成功；episode/frame 数与数据源一致 | 数据上传 |
| **M5** | LoRA 微调 + 对照 | checkpoint + 前后评测对比 | 训练 loss 正常下降；显存不 OOM；评测成功率可比较 | M3 + M4 |
| **M6** | 离线 ECO 数据推理（可选：量化准备） | `inferred_actions.npy`、报告 | `ECO_data` 客户端能连通并保存动作 | M1 |

**验收总原则**：每个阶段都保存「命令 + 完整 stdout/stderr + 环境快照（`nvidia-smi`、`pip freeze`）+ 视频/数值产物」，统一放在 `agent-work-space/` 下按阶段建目录。

---

## 3. 关键约束与风险

| ID | 风险 | 依据 | 规避动作 |
| --- | --- | --- | --- |
| R1 | `libero_policy.py` UTF-16 导致服务端无法 import | 【实测】`compile()` 报 `source code string cannot contain null bytes` | §6 第一步修复为 UTF-8 |
| R2 | 国际带宽 0.2–0.6 MiB/s | 【实测】 | 大资产一律 Mac 下载 + rsync；pip 用 TUNA 镜像 |
| R3 | 单卡 24 GB，全参微调需 >70 GB | 官方 README | 阶段 M5 只用 LoRA（`gemma_2b_lora` + `gemma_300m_lora`），batch 从 4～8 起试 |
| R4 | 无 sudo、无 docker 守护进程权限 | 【实测】 | 原生 venv；不装 apt 包（EGL 依赖已就绪） |
| R5 | `import robosuite` 必须命中本地改造版，否则 zhixing/ECO65 全部失效 | 【代码】`robosuite/__init__.py` 即包本体 | 所有客户端命令都显式设置 `PYTHONPATH=$PWD:$PWD/packages/openpi-client/src:$PWD/third_party/libero`，并**每次都打印 `robosuite.__file__` 自检** |
| R6 | 状态语义偏差：8 维 state 里两个夹爪值是旋转关节角，不是开口宽度；`pi05_libero` 的两个配置都 `discrete_state_input=False`（状态数值不进网络条件） | 【代码】`examples/libero/main.py:143-149`、`src/openpi/training/config.py:745` | 结果是"图像+语言"条件的结果，写报告时必须注明；不要据此推断状态通道有效 |
| R7 | 初始化方式与官方 benchmark 不同：ECO65 分支用 `env.reset()` 随机摆放，跳过 Panda 的 `set_init_state` | 【代码】`examples/libero/main.py:97-101` | 报告里标注"随机初始化"；若要严格对齐 benchmark，需另做工作（见 §8.3） |
| R8 | 全局关节 PD 项影响所有 SingleArm | 【代码】`robosuite/robots/single_arm.py:252-259` | 记录为实验条件；若要对照，需做成开关并单独实验 |
| R9 | 本地数据不完整：LeRobot 目录只有 2/432 个 episode（`meta/info.json` 却声明 432），TFRecord 只有 1/16 分片 | 【实测】 | M4 必须先补齐数据或重新转换，禁止用这 2 个 episode 得出结论 |
| R10 | `examples/libero_eco65/main.py` 构造 12 维 state，与 8 维 norm stats 不兼容（`Normalize` 会广播失败） | 【代码】`main.py:133-138`、`src/openpi/transforms.py:137-139` | **不要用该入口**；统一用 `examples/libero/main.py` |
| R11 | 其他入口的坑：`eco65_sim`/`eco65_libero` 强制 RethinkGripper（非知行）；`eco65_lift` 腕部图像全零且 `ignore_done=True` 恒跑满步数 | 【代码】`register_eco65.py:31`、`env.py:32/36/64-67` | 阶段 M2/M3 只用 `examples/libero/main.py` |
| R12 | 端口 8000 可能与既有项目（navsim 等）冲突 | 【待核验】 | 启动前 `ss -ltnp | grep 8000`；冲突则 `--port 8001` 并同步客户端 |

---

## 4. 服务器目录规划

```text
/mnt/data/vla-policy/
├── openpi/                          # 代码（已就绪）
│   └── agent-work-space/            # 本 agent 工作目录
│       ├── openpi_复现方案.md        # 本文档
│       ├── logs/                    # 每个阶段的运行日志
│       ├── scripts/                 # 辅助脚本（镜像、自检、评测包装）
│       ├── eval/                    # 评测视频与结果 JSON
│       └── reports/                 # 阶段性报告
├── toolchain/                       # uv、CPython 3.11 standalone、离线轮子
├── venvs/
│   ├── openpi-py311/                # 模型/训练/服务环境（Python 3.11）
│   └── libero-py38/                 # 仿真环境（Python 3.8）
├── assets/
│   ├── openpi-assets/checkpoints/   # 镜像的 π0.5 权重
│   └── datasets/                    # LeRobot 数据集 / RLDS 原始数据
├── caches/                          # uv / HF / openpi 缓存
└── libero-config/config.yaml        # LIBERO_CONFIG_PATH 指向这里
```

训练产物沿用项目默认相对路径（仓库本身就在 `/mnt/data`，无需改配置）：
`/mnt/data/vla-policy/openpi/assets/`（norm stats）、`/mnt/data/vla-policy/openpi/checkpoints/`（训练输出）。

```bash
# 一次性建目录
mkdir -p /mnt/data/vla-policy/{toolchain,venvs,assets/openpi-assets/checkpoints,assets/datasets,caches,libero-config}
mkdir -p /mnt/data/vla-policy/openpi/agent-work-space/{logs,scripts,eval,reports}
```

---

## 5. 阶段 M0：工具链与两个 Python 环境

### 5.1 安装 uv 与 Python 3.11（离线优先）

服务器直连 GitHub release 只有 0.08 MiB/s，因此**在 Mac 上下载后上传**（约 40 MB，几秒钟）：

```bash
# ===== 本地 Mac =====
mkdir -p ~/openpi-toolchain && cd ~/openpi-toolchain
curl -fLO https://github.com/astral-sh/uv/releases/download/0.5.1/uv-x86_64-unknown-linux-gnu.tar.gz
curl -fLO https://github.com/astral-sh/python-build-standalone/releases/download/20250409/cpython-3.11.12+20250409-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
rsync -avP -e "ssh -p 20920" ~/openpi-toolchain/ \
  vipuser@223.109.239.30:/mnt/data/vla-policy/toolchain/
```

```bash
# ===== 服务器 =====
cd /mnt/data/vla-policy/toolchain
tar -xzf uv-x86_64-unknown-linux-gnu.tar.gz
mkdir -p ~/.local/bin && cp uv-x86_64-unknown-linux-gnu/uv ~/.local/bin/ && chmod +x ~/.local/bin/uv
mkdir -p /mnt/data/vla-policy/toolchain/python3.11
tar -xzf "cpython-3.11.12+20250409-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz" \
    -C /mnt/data/vla-policy/toolchain/python3.11 --strip-components=1
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
export PATH=$HOME/.local/bin:$PATH
uv --version
/mnt/data/vla-policy/toolchain/python3.11/bin/python3.11 -V   # 期望 Python 3.11.12
```

**兜底（直连安装）**：`curl -LsSf https://astral.sh/uv/install.sh | sh`，约 5～10 分钟。

### 5.2 模型环境（Python 3.11）

```bash
cd /mnt/data/vla-policy/openpi
export PATH=$HOME/.local/bin:$PATH
export UV_PROJECT_ENVIRONMENT=/mnt/data/vla-policy/venvs/openpi-py311
export UV_LINK_MODE=copy
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_CACHE_DIR=/mnt/data/vla-policy/caches/uv

# 1) 先固定解释器
uv venv --python /mnt/data/vla-policy/toolchain/python3.11/bin/python3.11 "$UV_PROJECT_ENVIRONMENT"

# 2) 备份锁文件后同步（允许按镜像重新解析；见下方说明）
cp -n uv.lock uv.lock.pypi-backup
GIT_LFS_SKIP_SMUDGE=1 uv sync --no-dev

# 3) 安装本项目与客户端
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 4) 打 PyTorch 用的 transformers 补丁（官方 README 要求）
cp -r src/openpi/models_pytorch/transformers_replace/* \
      "$UV_PROJECT_ENVIRONMENT"/lib/python3.11/site-packages/transformers/
```

**关于 `--frozen` 的取舍（重要）**【待核验】：
`uv.lock` 里记录的是 `https://pypi.org/simple`。若加 `--frozen`，uv 可能仍按锁文件里的 pypi.org URL 下载（实测直连仅 0.58 MiB/s，整个环境要 1～2 小时）。上面的写法让 uv 按 TUNA 镜像重新解析，**代价是 `uv.lock` 会被改写**（已用 `uv.lock.pypi-backup` 备份）。若希望完全不动锁文件，可先试 `GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev` 并观察下载 URL，再决定是否改用镜像。

预计下载量 6～8 GB（`jax[cuda12]` 的 NVIDIA 轮子 + `torch==2.7.1` + transformers），按 3.32 MiB/s 约 **30～45 分钟**。

**后续所有 `uv run` 都加 `--no-sync`**，避免误触发重新解析和下载。

### 5.3 仿真环境（Python 3.8）

```bash
cd /mnt/data/vla-policy/openpi
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/mnt/data/vla-policy/caches/uv
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

uv venv --python 3.8 --python-preference only-system /mnt/data/vla-policy/venvs/libero-py38
source /mnt/data/vla-policy/venvs/libero-py38/bin/activate

# 避免 third_party 里未固定的 robosuite 解析到 1.5.x（会拉 mink/qpsolvers/mujoco>=3.3，非本方案所需）
sed 's/^[[:space:]]*robosuite[[:space:]]*$/robosuite==1.4.1/' \
    third_party/libero/requirements.txt > /tmp/req-libero.txt

uv pip sync examples/libero/requirements.txt /tmp/req-libero.txt \
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu113/ \
  --index-strategy=unsafe-best-match

uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
```

说明：
- `torch==1.11.0+cu113` 通过阿里云 `--find-links` 提供（已实测该页含 `torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl`）。LIBERO 的 `get_task_init_states` 会用 `torch.load`，所以仿真环境确实需要 torch。
- `robosuite==1.4.1` 是从 PyPI 装的纯 Python 包，运行时会被本地 `robosuite/` 目录**遮蔽**（见 §5.4 自检）。它的作用是提供依赖与元数据。
- 本地代码从未 import `egl_probe`【实测 grep 为空】，所以不需要 cmake，也不会触发源码编译。

### 5.4 环境自检（每次开工先跑）

```bash
cd /mnt/data/vla-policy/openpi

# --- 模型环境 ---
source /mnt/data/vla-policy/venvs/openpi-py311/bin/activate
python - <<'PY'
import jax, torch, openpi
print("jax:", jax.__version__, jax.devices())
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
import openpi.policies.libero_policy as lp   # 修复编码后必须能 import
print("libero_policy:", lp.__file__)
PY
```

```bash
# --- 仿真环境（关键：确认 robosuite 指向本地副本）---
source /mnt/data/vla-policy/venvs/libero-py38/bin/activate
export PYTHONPATH=$PWD:$PWD/packages/openpi-client/src:$PWD/third_party/libero
python - <<'PY'
import robosuite, mujoco, libero
print("robosuite:", robosuite.__file__, robosuite.__version__)   # 期望 .../openpi/robosuite/__init__.py 1.4.1
print("mujoco:", mujoco.__version__)                              # 期望 3.2.3
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.robots import REGISTERED_ROBOTS
print("ZhixingGripper registered:", "ZhixingGripper" in GRIPPER_MAPPING)
print("ECO65 registered:", "ECO65" in REGISTERED_ROBOTS)
PY
```

期望输出（判定标准）：
- `robosuite.__file__` = `/mnt/data/vla-policy/openpi/robosuite/__init__.py`（**若指向 site-packages 则说明 PYTHONPATH 丢了，知行夹爪不会生效**）
- `ZhixingGripper registered: True`、`ECO65 registered: True`

```bash
# --- 无头 EGL 渲染自检（M2 的前置）---
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python - <<'PY'
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>")
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64); r.update_scene(d)
print("EGL render OK:", r.render().shape)
PY
```

若 EGL 失败，依次尝试：`MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0`、`PYOPENGL_PLATFORM=egl`、最后 `MUJOCO_GL=osmesa`（**服务器未装 libOSMesa，需要 apt，当前无 sudo → 大概率不可用**，届时应向上游申请装包或改用 Docker）。

---

## 6. 阶段 M0.5：代码修复清单

> 原则：**只修真正阻断运行的项**；行为类问题先记录、后实验，不在复现前改动语义。

### P0 — 必须修，否则起不来

**F1. `src/openpi/policies/libero_policy.py`：UTF-16LE → UTF-8**【实测】

```bash
cd /mnt/data/vla-policy/openpi
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/openpi/policies/libero_policy.py")
raw = p.read_bytes()
text = raw.decode("utf-16")           # 已确认可完整解码为 100 行有效 Python
p.write_bytes(text.encode("utf-8"))
print("rewritten:", p, p.stat().st_size, "bytes")
PY

# 验证
python3 -c "import ast,pathlib; ast.parse(pathlib.Path('src/openpi/policies/libero_policy.py').read_text('utf-8')); print('libero_policy.py OK')"
grep -c "LiberoInputs\|LiberoOutputs" src/openpi/policies/libero_policy.py   # 期望 >0
```

修复前建议留底：`cp src/openpi/policies/libero_policy.py agent-work-space/logs/libero_policy.py.utf16.bak`

### P1 — 影响工具链/可读性，建议修

| 编号 | 文件 | 问题【实测】 | 处理 |
| --- | --- | --- | --- |
| F2 | `examples/libero/compose.yml` | UTF-16LE + BOM，`docker compose` 无法解析 | 本方案不用 Docker；若将来用，先 `iconv -f UTF-16 -t UTF-8` |
| F3 | `src/openpi/training/config.py`、`examples/libero/main.py`、`robosuite/models/grippers/zhixing_gripper.py` 等 15 个文件 | UTF-8 **带 BOM**（CPython 可正常解析，已实测 `examples/libero/main.py` 编译通过） | 可不处理；如需干净 diff 再统一去 BOM |
| F4 | `examples/eco65_libero/patch_libero.py:86` | 语法错误，脚本不可运行 | 不使用该脚本；本方案主链路不需要它 |
| F5 | `examples/libero_eco65/compose.yml:10` | Dockerfile 路径拼写为 `libero_coe65` | 不使用 Docker |
| F6 | `examples/eco65_lift/compose.yml:25` | 启动的是 `examples/eco65_sim/main.py` | 不使用该 compose |

### P2 — 语义/工程问题，只记录不改（会影响结论解释）

| 编号 | 位置 | 事实 | 复现时如何处置 |
| --- | --- | --- | --- |
| S1 | `examples/libero/main.py:178` | `done` 未按 episode 初始化，异常早退时可能 `NameError`，成功判定只看最后一步 | 运行日志里保留 traceback；不要盲信汇总成功率 |
| S2 | `examples/libero/main.py:180-190` | 视频文件名不含 episode 序号，同任务同结果互相覆盖 | 评测时按 episode 改名归档（见 §8.4） |
| S3 | `examples/libero/main.py:122-124` | 每步用 cv2 覆盖写 `panda_agentview.png` / `panda_wrist.png` 到 CWD | 从项目根启动，事后取最新两张即可；或改成写 `agent-work-space/` |
| S4 | `robosuite/models/grippers/zhixing_gripper.py:27-30` + `robosuite/robots/single_arm.py:181` | `init_qpos` 只有 1 个值却广播到 6 个夹爪关节，与 5 条 equality 约束不自洽 | 阶段 M2 观察 reset 后前几步是否抖动；记录现象 |
| S5 | `robosuite/robots/single_arm.py:252-259` | 全局关节 PD（`kp=3.0`），对所有 SingleArm 生效、非零空间投影 | 作为固定实验条件记录；对照实验需另做开关 |
| S6 | `examples/libero_eco65/main.py:133-138` | 12 维 state 与 8 维 norm stats 不兼容 | **禁用该入口** |

---

## 7. 阶段 M1：模型服务

### 7.1 获取 π0.5 checkpoint（本地下载 → 上传）

在 **Mac** 上创建一个不依赖 gsutil/凭证的镜像脚本：

```bash
mkdir -p /mnt/data/vla-policy/openpi/agent-work-space/scripts   # 服务器路径；本地对应 agent-work-space/scripts
```

```python
# agent-work-space/scripts/mirror_gcs_prefix.py
#!/usr/bin/env python3
"""把公开 GCS 前缀镜像到本地目录（匿名可读，无需 gsutil/凭证）。可重复执行，已存在且非空的文件会跳过。"""
import pathlib, re, sys, urllib.parse, urllib.request


def list_prefix(bucket: str, prefix: str) -> list[str]:
    marker, keys = "", []
    while True:
        url = (f"https://storage.googleapis.com/{bucket}"
               f"?prefix={urllib.parse.quote(prefix)}&max-keys=1000&marker={urllib.parse.quote(marker)}")
        xml = urllib.request.urlopen(url, timeout=60).read().decode()
        batch = re.findall(r"<Key>([^<]+)</Key>", xml)
        keys += batch
        if "<IsTruncated>true</IsTruncated>" not in xml or not batch:
            return keys
        marker = batch[-1]


def mirror(bucket: str, prefix: str, dest: str) -> None:
    dest_path = pathlib.Path(dest).expanduser()
    keys = list_prefix(bucket, prefix)
    print(f"{len(keys)} objects under gs://{bucket}/{prefix} -> {dest_path}", flush=True)
    for i, key in enumerate(keys, 1):
        rel = key[len(prefix):] if key.startswith(prefix) else key
        target = dest_path / rel
        if target.exists() and target.stat().st_size > 0:
            print(f"[{i}/{len(keys)}] skip {rel}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(keys)}] {rel}", flush=True)
        urllib.request.urlretrieve(
            f"https://storage.googleapis.com/{bucket}/{urllib.parse.quote(key)}", target)


if __name__ == "__main__":
    mirror(sys.argv[1], sys.argv[2], sys.argv[3])
```

```bash
# ===== 本地 Mac：下载（GCS 实测 5.6 MiB/s，11.58 GiB 约 35 分钟）=====
cd /Users/yuyu/Documents/openpi/agent-work-space/scripts
python3 mirror_gcs_prefix.py openpi-assets checkpoints/pi05_libero/ ~/openpi-assets/checkpoints/pi05_libero/
# 阶段 M5 需要初始权重时再执行：
# python3 mirror_gcs_prefix.py openpi-assets checkpoints/pi05_base/ ~/openpi-assets/checkpoints/pi05_base/

# 完整性自检：对象数应为 16（pi05_libero）/ 29（pi05_base）
find ~/openpi-assets/checkpoints/pi05_libero -type f | wc -l
du -sh ~/openpi-assets/checkpoints/pi05_libero        # 期望 ≈11.58 GiB

# ===== 上传（3.4 MB/s，约 60 分钟）=====
rsync -avP --partial -e "ssh -p 20920" \
  ~/openpi-assets/checkpoints/pi05_libero/ \
  vipuser@223.109.239.30:/mnt/data/vla-policy/assets/openpi-assets/checkpoints/pi05_libero/
```

```bash
# ===== 服务器：校验 =====
find /mnt/data/vla-policy/assets/openpi-assets/checkpoints/pi05_libero -type f | wc -l   # 期望 16
du -sh /mnt/data/vla-policy/assets/openpi-assets/checkpoints/pi05_libero                 # 期望 ≈11.58 GiB
ls /mnt/data/vla-policy/assets/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json
```

**不使用 `gs://` 路径**：`create_trained_policy` 会对本地路径短路（`src/openpi/shared/download.py:52-56`），直接传绝对路径即可，避免缓存目录与慢速下载。

### 7.2 启动策略服务

```bash
cd /mnt/data/vla-policy/openpi
source /mnt/data/vla-policy/venvs/openpi-py311/bin/activate
export PATH=$HOME/.local/bin:$PATH
export UV_PROJECT_ENVIRONMENT=/mnt/data/vla-policy/venvs/openpi-py311
export UV_CACHE_DIR=/mnt/data/vla-policy/caches/uv
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

ss -ltnp 2>/dev/null | grep -w 8000 || echo "port 8000 free"

uv run --no-sync scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /mnt/data/vla-policy/assets/openpi-assets/checkpoints/pi05_libero \
  --port 8000 2>&1 | tee /mnt/data/vla-policy/openpi/agent-work-space/logs/M1_serve_policy.log
```

启动耗时预计 1～3 分钟（JAX 首次编译），显存占用约 9～12 GB【待核验】。

### 7.3 验证（不依赖仿真环境）

```python
# agent-work-space/scripts/ping_policy.py
import numpy as np
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000)
print("metadata:", client.get_server_metadata())
img = np.zeros((224, 224, 3), dtype=np.uint8)
out = client.infer({
    "observation/image": img,
    "observation/wrist_image": img,
    "observation/state": np.zeros(8, dtype=np.float32),
    "prompt": "pick up the black bowl between the plate and the ramekin and place it on the plate",
})
actions = np.asarray(out["actions"])
print("actions:", actions.shape, actions.dtype)
print("finite:", np.isfinite(actions).all())
print("first row:", np.round(actions[0], 4))
```

```bash
# 用仿真环境跑（已装 openpi-client）
source /mnt/data/vla-policy/venvs/libero-py38/bin/activate
cd /mnt/data/vla-policy/openpi
python agent-work-space/scripts/ping_policy.py
```

**通过判据**：`actions.shape == (10, 7)`、`finite: True`、夹爪维（第 7 维）取值落在 [-1, 1]。

---

## 8. 阶段 M2/M3：ECO65 + 知行 仿真闭环与评测

### 8.1 主入口（唯一推荐）

`examples/libero/main.py` 是本副本中**唯一**同时满足以下条件的入口【代码】：

- `_get_libero_env()` 显式 `OffScreenRenderEnv(robots=["ECO65"], ...)`（`main.py:217`）→ 走 ECO65 默认夹爪 **ZhixingGripper**
- state 为 8 维：`eef_pos(3) + quat2axisangle(eef_quat)(3) + robot0_gripper_qpos[[1,5]](2)`（`main.py:143-149`）
- 图像：`agentview_image` / `robot0_eye_in_hand_image` → 旋转 180° → `resize_with_pad(224)` → uint8
- 动作为 7 维，`replan_steps=5` 的 deque 执行

其余入口（`libero_eco65` / `eco65_sim` / `eco65_libero` / `eco65_lift`）分别是 12 维 state、Rethink 夹爪或非评测用途，**本阶段不使用**。

### 8.2 LIBERO 配置与启动

```bash
mkdir -p /mnt/data/vla-policy/libero-config
cat > /mnt/data/vla-policy/libero-config/config.yaml <<'EOF'
benchmark_root: /mnt/data/vla-policy/openpi/third_party/libero/libero/libero
bddl_files: /mnt/data/vla-policy/openpi/third_party/libero/libero/libero/bddl_files
init_states: /mnt/data/vla-policy/openpi/third_party/libero/libero/libero/init_files
datasets: /mnt/data/vla-policy/openpi/third_party/libero/datasets
assets: /mnt/data/vla-policy/openpi/third_party/libero/libero/libero/assets
EOF
```
（以上 5 个路径均已确认存在【实测】。若不建该文件，LIBERO 的 `__init__` 会走交互式 `input()` 提问而卡住。）

```bash
cd /mnt/data/vla-policy/openpi
source /mnt/data/vla-policy/venvs/libero-py38/bin/activate
export PYTHONPATH=$PWD:$PWD/packages/openpi-client/src:$PWD/third_party/libero
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH=/mnt/data/vla-policy/libero-config

# 开工自检（必须看到本地 robosuite 路径）
python -c "import robosuite; print(robosuite.__file__)"

# M2 冒烟：单任务单次
python examples/libero/main.py \
  --args.host 127.0.0.1 --args.port 8000 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 --args.num-steps-wait 10 --args.seed 7 \
  --args.video-out-path /mnt/data/vla-policy/openpi/agent-work-space/eval/M2_smoke \
  2>&1 | tee /mnt/data/vla-policy/openpi/agent-work-space/logs/M2_smoke.log
```

CLI 前缀说明【实测】：tyro 会加参数名前缀，**必须写 `--args.xxx`**；写成 `--task-suite-name` 会报 `Unrecognized options` 并以 exit code 2 退出。

### 8.3 偏差说明（写进报告，不要省略）

| 偏差 | 说明 |
| --- | --- |
| 初始化方式 | ECO65 分支执行 `env.reset()`（随机摆放物体），**跳过** `set_init_state(initial_states[episode_idx])`【代码 `main.py:97-101`】。官方 benchmark 用的是固定 pruned init 状态（Panda 时代预计算）。→ 本方案的成功率**不能**与官方 π0.5 LIBERO 数字直接对比 |
| 机器人/夹爪 | MountedECO65【推断：桌面任务构造器会给机器人名加 `Mounted` 前缀】+ ZhixingGripper（6 关节、1 执行器、`dof=1`） |
| 状态语义 | 两个夹爪值取自 `Left_Support_Joint` / `Right_Support_Joint`（旋转关节角），不是指端开口宽度 |
| 状态是否进网络 | `pi05_libero` / `pi05_libero_custom` 都是 `discrete_state_input=False`，π0.5 也无连续状态 token → **状态数值不是动作预测的条件输入**，预测条件主要是图像 + 语言【代码】 |
| 控制层 | LIBERO `ControlEnv` 显式 `load_controller_config(default_controller="OSC_POSE")` → 实际用 `osc_pose.json`（每维 [-1,1]，平移 ±0.05，旋转 ±0.5，kp=150），**不是** `controllers/config/robots/default_eco65.json` |
| 额外 PD | `single_arm.py:252-259` 的关节 PD 对所有 SingleArm 生效 |
| 严格对齐 benchmark（可选后续工作） | 需要为 ECO65 生成属于自己的 init states，或在客户端重建 `qpos/qvel/物体位置/控制器目标/夹爪 current_action` 后再恢复 |

### 8.4 正式评测（M3）与记录规范

```bash
# libero_spatial 10 个任务 × 每任务 10 次（先小规模，确认稳定后再上 50 次）
python examples/libero/main.py \
  --args.host 127.0.0.1 --args.port 8000 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 10 --args.seed 7 \
  --args.video-out-path /mnt/data/vla-policy/openpi/agent-work-space/eval/M3_pi05_libero \
  2>&1 | tee /mnt/data/vla-policy/openpi/agent-work-space/logs/M3_pi05_libero.log
```

由于视频文件名不含 episode 序号（S2），评测**每个 episode 结束后**立刻归档：

```bash
# 简易归档循环（在另一个终端按需执行；或把 main.py:180-190 的路径改成含 episode 序号）
d=/mnt/data/vla-policy/openpi/agent-work-space/eval/M3_pi05_libero
i=1; while :; do sleep 30; for f in "$d"/*.mp4; do [ -e "$f" ] || continue; \
  b=$(basename "$f" .mp4); [ -e "$d/../M3_archive/${b}_$(date +%s).mp4" ] || { mkdir -p "$d/../M3_archive"; cp "$f" "$d/../M3_archive/${b}_$(date +%H%M%S).mp4"; }; done; done
```

`main.py` 会在日志末尾打印：

```text
Total success rate: <float>
Total episodes: <int>
```

**记录规范**：每个任务的成功次数/总次数、使用的 checkpoint、`robosuite.__file__`、`MUJOCO_GL`、seed、是否随机初始化、耗时、显存峰值（`nvidia-smi --query-gpu=memory.used --format=csv -l 5`）。

单任务耗时估算：`max_steps = 220`（libero_spatial）+ `num_steps_wait = 10`，每 5 步一次推理。若单次推理 ~100 ms【待核验】，单 episode 约 1 分钟量级；10 任务 × 10 次 ≈ 2 小时【待核验】。**先用 `--args.num-trials-per-task 1` 测单次耗时再外推**。

---

## 9. 阶段 M4：数据管线

> 现阶段本地数据**不可用于训练**：LeRobot 目录只有 2/432 episode（`meta/info.json` 却声明 432，加载会不一致），TFRecord 只有 1/16 分片（27/432 episode）。必须补齐。

### 9.1 三种数据方案

| 方案 | 数据源 | 体积 | 适配配置 | 适用 |
| --- | --- | --- | --- | --- |
| **A（推荐先做）** | HF `physical-intelligence/libero`（LeRobot 格式，4 个套件合并） | 21.31 GiB / 996 文件 | `pi05_libero`（`repo_id` 与之完全一致） | 复现官方 π0.5 LIBERO 基线 |
| **B（贴合本项目）** | HF `openvla/modified_libero_rlds` → 本地脚本转 LeRobot | 9.53 GiB（`libero_spatial_no_noops` ≈2.1 GiB） | `pi05_libero_custom`（`repo_id="local/libero_spatial"`） | 沿用项目原有转换链路 |
| C（仅冒烟） | 现有 1/16 分片 / 2 个 episode | 30 MB | `pi05_libero_custom` | 只验证脚本能跑通，**不得用于结论** |

传输一律「Mac 下载 → rsync」，方法同 §7.1：

```bash
# ===== 本地 Mac =====
pip install -U huggingface_hub            # 或在有 hf CLI 的环境里
export HF_HUB_ENABLE_HF_TRANSFER=0

# 方案 A：LeRobot 数据集（注意：这是 dataset 仓库，用 hf download --repo-type dataset）
hf download physical-intelligence/libero --repo-type dataset --local-dir ~/hf/libero

# 方案 B：RLDS 原始数据（只取 libero_spatial 一个套件即可）
hf download openvla/modified_libero_rlds --repo-type dataset \
   --include "libero_spatial_no_noops/*" --local-dir ~/hf/modified_libero_rlds

rsync -avP --partial -e "ssh -p 20920" ~/hf/libero/ \
  vipuser@223.109.239.30:/mnt/data/vla-policy/assets/datasets/lerobot/physical-intelligence/libero/
# 或
rsync -avP --partial -e "ssh -p 20920" ~/hf/modified_libero_rlds/ \
  vipuser@223.109.239.30:/mnt/data/vla-policy/assets/datasets/rlds/modified_libero_rlds/
```

> 21.31 GiB 走 3.4 MB/s ≈ 1.8 小时；9.53 GiB ≈ 50 分钟。**方案 B 更省时间**，且与项目原有 `pi05_libero_custom` 链路一致，建议先做 B、需要复现官方数字时再做 A。

### 9.2 方案 B：RLDS → LeRobot 转换

```bash
cd /mnt/data/vla-policy/openpi
source /mnt/data/vla-policy/venvs/openpi-py311/bin/activate
export PATH=$HOME/.local/bin:$PATH UV_PROJECT_ENVIRONMENT=/mnt/data/vla-policy/venvs/openpi-py311
export HF_LEROBOT_HOME=/mnt/data/vla-policy/assets/datasets/lerobot
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

# 转换脚本需要 tensorflow / tensorflow_datasets（uv 的 rlds 依赖组，仅 cp311 有轮子）
GIT_LFS_SKIP_SMUDGE=1 uv sync --group rlds --no-dev

# 先 dry-run 看结构
uv run --no-sync scripts/convert_tfrecord_to_lerobot.py \
  --data-dir /mnt/data/vla-policy/assets/datasets/rlds/modified_libero_rlds --dry-run

# 正式转换（注意：脚本会在目标 repo 目录已存在时先 rmtree 再重建）
uv run --no-sync scripts/convert_tfrecord_to_lerobot.py \
  --data-dir /mnt/data/vla-policy/assets/datasets/rlds/modified_libero_rlds \
  --repo-id local/libero_spatial \
  --output_dir /mnt/data/vla-policy/assets/datasets/lerobot \
  2>&1 | tee /mnt/data/vla-policy/openpi/agent-work-space/logs/M4_convert.log
```

**校验**（必须与数据源一致）：

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path("/mnt/data/vla-policy/assets/datasets/lerobot/local/libero_spatial/meta/info.json")
d = json.loads(p.read_text())
print({k: d[k] for k in ("total_episodes", "total_frames", "total_tasks", "fps")})
n = len(list((p.parent.parent / "data").rglob("*.parquet")))
print("parquet files:", n)     # 应与 total_episodes 一致
PY
```
期望 `total_episodes = 432`【推断：TFRecord 16 分片 × 27 episode】、`fps = 10`。**若 parquet 数 ≠ total_episodes，先停下排查，不要进入训练。**

同时清理仓库内那份不一致的旧副本（`libreo_raw_data/local/libero_spatial`，只有 2 个 episode 却声明 432），避免被误加载。

### 9.3 归一化统计量

```bash
cd /mnt/data/vla-policy/openpi
source /mnt/data/vla-policy/venvs/openpi-py311/bin/activate
export PATH=$HOME/.local/bin:$PATH UV_PROJECT_ENVIRONMENT=/mnt/data/vla-policy/venvs/openpi-py311
export HF_LEROBOT_HOME=/mnt/data/vla-policy/assets/datasets/lerobot

# 方案 B / 配置 pi05_libero_custom
uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_libero_custom

# 方案 A / 配置 pi05_libero（数据放在 $HF_LEROBOT_HOME/physical-intelligence/libero）
# uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_libero
```

输出位置：`/mnt/data/vla-policy/openpi/assets/<config_name>/<repo_id>/norm_stats.json`。

**判据**：`state` 的统计量维度必须是 **8**，`actions` 是 **7**；分位数（q01/q99）不得出现常数或 NaN。若维度不是 8/7，说明数据转换阶段的字段定义与 `LeRobotLiberoDataConfig`（`src/openpi/training/config.py:282-330`）不匹配。

**注意**：`pi05_libero_custom` 使用的是 `local/libero_spatial` 数据（Panda 8 维状态 / 7 维动作），它与 ECO65 的观测语义并不相同；微调后的权重仍然是在"Panda 动作空间 + 图像/语言条件"上训练的，接到 ECO65 仿真里属于**跨本体迁移**，不是 ECO65 数据微调。报告里必须写清这一点。

---

## 10. 阶段 M5：LoRA 微调（24 GB 单卡唯一可行路线）

### 10.1 显存判断

| 方案 | 官方标注 | 24 GB 3090 |
| --- | --- | --- |
| 全参微调 | >70 GB（A100 80G / H100） | ❌ 不可行 |
| LoRA 微调 | >22.5 GB（RTX 4090 24G） | ⚠️ 勉强，需 batch 4～8 起步 |
| 推理 | >8 GB | ✅ |

### 10.2 新增 LoRA 配置（追加到 `src/openpi/training/config.py` 的 `_CONFIGS` 列表末尾）

```python
    # ── ECO65/知行 复现用：24 GB 单卡 LoRA 微调 ──
    TrainConfig(
        name="pi05_libero_custom_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="local/libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        # 建议先镜像到本地再改成绝对路径，避免慢速下载：
        # weight_loader=weight_loaders.CheckpointWeightLoader(
        #     "/mnt/data/vla-policy/assets/openpi-assets/checkpoints/pi05_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=5e-5, decay_steps=5_000, decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        num_train_steps=5_000,
        log_interval=50,
        save_interval=500,
        keep_period=1_000,
    ),
```

（这是模板，**必须先跑 10 步确认不 OOM**；`get_freeze_filter()` 对 `pi05=True` 的行为属【推断】，若报错则参照 `pi0_libero_low_mem_finetune`（`config.py:678-698`）的写法调整。）

### 10.3 训练与监控

```bash
cd /mnt/data/vla-policy/openpi
source /mnt/data/vla-policy/venvs/openpi-py311/bin/activate
export PATH=$HOME/.local/bin:$PATH UV_PROJECT_ENVIRONMENT=/mnt/data/vla-policy/venvs/openpi-py311
export HF_LEROBOT_HOME=/mnt/data/vla-policy/assets/datasets/lerobot
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 先短跑验证（10 步）
uv run --no-sync scripts/train.py pi05_libero_custom_lora \
  --exp-name eco65-lora-smoke --overwrite --no-wandb-enabled --num-train-steps 10

# 正式训练
uv run --no-sync scripts/train.py pi05_libero_custom_lora \
  --exp-name eco65-lora-r1 --overwrite --no-wandb-enabled \
  2>&1 | tee /mnt/data/vla-policy/openpi/agent-work-space/logs/M5_train_lora_r1.log
```

监控（另开终端）：

```bash
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 10 \
  | tee /mnt/data/vla-policy/openpi/agent-work-space/logs/M5_gpu.csv
```

OOM 处理顺序：`batch_size` 8 → 4 → 2；`num_workers` 降到 1；确认没有别的进程占显存。注意 `--fsdp-devices` 在单卡上无收益。

### 10.4 微调后评测

```bash
# 服务端换成自己的 checkpoint
uv run --no-sync scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_libero_custom_lora \
  --policy.dir /mnt/data/vla-policy/openpi/checkpoints/pi05_libero_custom_lora/eco65-lora-r1/<step> \
  --port 8000
```

客户端命令与 §8.4 完全相同，只改 `--args.video-out-path` 与日志名，保证「同任务、同 seed、同初始化方式、同 replan_steps」的**配对对照**。成功率的比较必须给出配对统计（不要只看单点差）。

---

## 11. 阶段 M6：ECO 离线推理（可选：量化准备）

```bash
cd /mnt/data/vla-policy/openpi
source /mnt/data/vla-policy/venvs/libero-py38/bin/activate   # 需要 numpy/PIL/openpi-client
python ECO_data/remote_inference_client.py \
  --host 127.0.0.1 --port 8000 \
  --chunk ECO_data/vla_dataset/chunk-001 \
  --max-frames 3 \
  --task "pick up the object" \
  2>&1 | tee /mnt/data/vla-policy/openpi/agent-work-space/logs/M6_eco_offline.log
```
产物：`inferred_actions.npy`（每个动作块的第一步）。

**这条链路的能力边界**（务必写进报告）：
- 数据是 12 维 state / 13 维 action、单路图像，脚本只取 `state[:8]` 并把同一张图同时当外部视角和腕部视角；
- 不读取 CSV 里的 `action` 作为对照，也不驱动机械臂 → **它只能验证"数据读取 + 网络推理 + 保存"**，既不是离线精度评估，也不是真机执行。

**量化**：`docs/VLA量化部署技术方案_v1.md` 是**尚未实施的设计稿**（其自述"实施设计，尚未执行量化"），其中拟新增的 `src/openpi/quantization/`、`scripts/quantization/`、`configs/quantization/`、`reports/quantization/` 在仓库中**均不存在**。建议在 M1–M3 打通、P0/P95 时延测出来之后再启动，且严格按该文档的 P0→P3 顺序做。当前 3090 的驱动 535 / CUDA 12.2 组合满足其"单张 NVIDIA GPU"前提，但 INT4 kernel 支持情况需实测。

---

## 12. 附录 A：命令速查

```bash
# 环境变量（建议写进 agent-work-space/env.sh 后 source）
export PATH=$HOME/.local/bin:$PATH
export UV_PROJECT_ENVIRONMENT=/mnt/data/vla-policy/venvs/openpi-py311
export UV_CACHE_DIR=/mnt/data/vla-policy/caches/uv
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_LEROBOT_HOME=/mnt/data/vla-policy/assets/datasets/lerobot
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONPATH=/mnt/data/vla-policy/openpi:/mnt/data/vla-policy/openpi/packages/openpi-client/src:/mnt/data/vla-policy/openpi/third_party/libero
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export LIBERO_CONFIG_PATH=/mnt/data/vla-policy/libero-config
```

| 目的 | 命令 |
| --- | --- |
| 模型环境 | `source /mnt/data/vla-policy/venvs/openpi-py311/bin/activate` |
| 仿真环境 | `source /mnt/data/vla-policy/venvs/libero-py38/bin/activate` |
| 起服务 | `uv run --no-sync scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero --policy.dir /mnt/data/vla-policy/assets/openpi-assets/checkpoints/pi05_libero --port 8000` |
| 评测 | `python examples/libero/main.py --args.host 127.0.0.1 --args.port 8000 --args.task-suite-name libero_spatial --args.num-trials-per-task 10` |
| 统计量 | `uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_libero_custom` |
| 训练 | `uv run --no-sync scripts/train.py pi05_libero_custom_lora --exp-name <run> --overwrite --no-wandb-enabled` |
| 离线推理 | `python ECO_data/remote_inference_client.py --chunk ECO_data/vla_dataset/chunk-001 --max-frames 3` |

---

## 13. 附录 B：待核验清单

| # | 项目 | 核验方式 | 归属阶段 |
| --- | --- | --- | --- |
| 1 | 服务端 8000 端口是否被占用 | `ss -ltnp \| grep 8000` | M1 |
| 2 | JAX 是否识别 GPU、显存占用 | `jax.devices()`、`nvidia-smi` | M1 |
| 3 | `import robosuite` 是否命中本地副本 | 打印 `robosuite.__file__` | M2 |
| 4 | EGL 无头渲染是否可用 | §5.4 渲染自检 | M2 |
| 5 | 知行夹爪 reset 后是否抖动（单值 init_qpos 广播） | 记录前 20 步关节角 | M2 |
| 6 | 单 episode 实际耗时与显存峰值 | 计时 + `nvidia-smi -l 5` | M2 |
| 7 | `robosuite` 本地版是否真的被 LIBERO 使用（是否被 pip 版遮蔽） | 在 `main.py` 里打印 `robosuite.__file__` | M2 |
| 8 | `done` 未初始化导致的 NameError 是否出现 | 检查日志 traceback | M3 |
| 9 | 官方权重在 ECO65+随机初始化下的成功率量级 | M3 评测 | M3 |
| 10 | 转换后 episode/frame 数是否与数据源一致 | §9.2 校验脚本 | M4 |
| 11 | norm stats 维度是否 8/7 | 读 `norm_stats.json` | M4 |
| 12 | LoRA 微调是否 OOM、单步耗时 | 10 步短跑 | M5 |
| 13 | `get_freeze_filter()` 在 `pi05=True` + LoRA 下是否可用 | 短跑日志 | M5 |
| 14 | JAX 与 PyTorch 后端是否需要对齐 | 同输入对比动作 | M5（量化前） |

---

## 14. 附录 C：阶段记录模板

```markdown
### M<n> <阶段名> — <日期>
- 代码状态：git 无法使用（未上传 .git），以文件名+mtime 记录；关键文件 md5：
- 环境：python=..., robosuite.__file__=..., mujoco=..., CUDA/driver=...
- 命令：（原样粘贴）
- 参数：checkpoint=..., seed=..., num_trials=..., replan_steps=...
- 结果：成功 x/y = z%；失败原因分布
- 资源：峰值显存 ... MiB，单次推理 ... ms，总耗时 ...
- 产物：logs/...、eval/...、reports/...
- 异常与偏差：（视频覆盖、NameError、随机初始化、PD 项……）
- 结论与下一步：
```

---

## 15. 执行顺序速览（TL;DR）

```text
1) 修 F1（libero_policy.py 转 UTF-8）                 ← 5 分钟，不修则一切免谈
2) Mac 下载 uv + CPython3.11 → 上传 → 建 openpi-py311  ← 30–60 分钟（TUNA 镜像）
3) 建 libero-py38（uv sync + 两个 -e 安装）            ← 10–20 分钟
4) Mac 镜像 pi05_libero → rsync 上传（11.58 GiB）      ← ≈1.5 小时
5) 起服务 + ping_policy.py 验证                        ← 10 分钟
6) LIBERO 配置 + M2 单任务冒烟 + EGL 自检              ← 30 分钟
7) M3 评测（先 1 次/任务，再 10 次/任务）               ← 数小时（后台跑）
8) 拉数据（方案 B，9.53 GiB）→ 转换 → 统计量           ← ≈1.5 小时
9) LoRA 10 步短跑 → 正式训练 → 配对评测                ← 小时～天级
10) 可选：ECO 离线推理验证 / 按量化方案文档推进         ← 视需要
```

**最小可交付复现**：步骤 1–7。它产出的东西是「官方 π0.5 LIBERO 权重在 ECO65 + 知行夹爪 MuJoCo 环境中的闭环评测」——这正是本项目适配工作的核心可验证部分；数据与微调（8–9）属于在此基础上继续推进的内容。
