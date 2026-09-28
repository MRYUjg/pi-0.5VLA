# π0.5 VLA 单卡量化部署技术方案 v1

版本日期：2026-09-24  
适用项目：OpenPI π0.5 + ECO65 六轴机械臂 + ZhiXing CRT-CTAG2F90 夹爪  
部署目标：单张 NVIDIA GPU 服务器，GPU 型号与显存待补充  
文档状态：实施设计，尚未执行量化、安装依赖或修改推理代码。

## 1. 推荐方案与预期交付

建议采用 **PyTorch 混合浮点基线 → 语言主干 MLP INT8 → 语言主干 MLP INT4 → 按瓶颈扩大量化范围 → ECO65/ZhiXing 闭环验证 → 独立量化服务部署** 的路线。

第一版使用训练后量化（PTQ），优先选择 TorchAO。保留现有 WebSocket 接口、图像处理、动作归一化和 OSC 控制协议；第一轮保留视觉编码器、动作专家、归一化层、时间条件模块和动作输入输出投影的原有浮点精度。

本项目的“BF16 基线”指现有 BF16/FP32 混合精度推理：部分归一化、视觉输入和动作相关模块仍为 FP32。本文不将全模型强制转换成 BF16 作为前提。

第一版交付物应包括：

1. 一个可重复运行的浮点基线及数据回放集。
2. 至少 INT8、INT4 两类量化候选及实际量化层清单。
3. 模型加载、量化、保存、独立进程重载和 WebSocket 服务链路。
4. 延迟、显存、动作偏差与闭环任务成功率对照报告。
5. 可复建的部署环境、模型清单和回退到浮点版本的方法。

首轮目标是回答“哪些模块可以压缩、节省多少显存、是否保持任务能力，以及是否真的降低延迟”。具体收益由服务器实测确定，不预先承诺模型总显存下降 75% 或推理加速若干倍。

## 2. 范围、假设与启动前信息

用户已明确选择单张 NVIDIA GPU 服务器。当前工作目录是远程项目的本地副本，远程服务器为实际实施环境。

| 项目 | 当前状态 | 实施要求 |
| --- | --- | --- |
| GPU 型号、显存、计算能力 | 未提供 | P0 阶段记录后，验证所选低精度 kernel |
| Linux / CUDA / 驱动 | 副本包含 Linux CUDA 容器方案 | 记录实际环境，不能只依据 Dockerfile |
| 基线权重 | 项目有默认 pi05_libero 与自定义配置 | 优先选择已经能在远程跑通、来源清楚的权重 |
| 当前模型后端 | 项目支持 JAX 和 PyTorch | 若远程仅跑通 JAX，先建立 PyTorch 对齐基线 |
| 评测环境 | ECO65 + ZhiXing + LIBERO | 固定模型、控制器、初始状态和相机设置 |
| 实践范围 | 模型量化和服务器部署 | 第一阶段不做真实机械臂自动闭环 |

默认固定 batch size=1、两路 224×224 图像、`action_horizon=10`、flow matching 采样步数=10、`replan_steps=5`。修改这些参数属于独立性能实验，不能和量化同时变更后把全部收益归因于量化。

本方案与 [项目详细说明](/Users/yuyu/Documents/openpi/docs/ECO65_ZhiXing_项目说明.md) 配套使用。前者说明拟实施内容，后者说明当前代码已经具备的内容。

## 3. 项目现状对量化的约束

### 3.1 模型不是普通自回归文本生成模型

[PI0Pytorch.sample_actions()](/Users/yuyu/Documents/openpi/src/openpi/models_pytorch/pi0_pytorch.py:377) 先计算图像和语言前缀及其 KV cache，再反复运行动作专家完成动作采样。当前默认一次输出 10×32 的内部动作，经 [LiberoOutputs](/Users/yuyu/Documents/openpi/src/openpi/policies/libero_policy.py:86) 返回 10×7 的环境动作。

一次请求可分解为：

```text
图像/语言预处理
  → 视觉编码与多模态投影
  → 语言主干前缀计算与 KV cache
  → 动作专家 denoise_step × 10
  → 动作反归一化、取前 7 维
  → WebSocket 返回动作块
```

量化语言主干会改变供动作专家读取的前缀缓存；即使动作专家保持浮点，最终动作仍可能受到量化误差影响。因此只测语言层输出误差不足以证明部署质量。

### 3.2 现有状态输入配置保持不变

当前 `pi05_libero`、`pi05_libero_custom` 都设置 `discrete_state_input=False`；结合 π0.5 模型分支，状态数值没有作为离散或连续状态 token 输入网络，但接口仍依赖状态字段。

量化实验继续使用现有 8 维状态协议，不顺带改变状态输入开关或 ZhiXing 状态映射。否则无法区分模型条件输入变化和量化变化。

### 3.3 需要处理的代码细节

| 位置 | 当前行为 | 量化实施时的处理 |
| --- | --- | --- |
| [PI0Pytorch 构造器](/Users/yuyu/Documents/openpi/src/openpi/models_pytorch/pi0_pytorch.py:114) | 默认在构造时包装 `sample_actions` 为编译函数 | 构造时设 `pytorch_compile_mode=None`；量化完成后再显式编译 |
| [policy_config.py](/Users/yuyu/Documents/openpi/src/openpi/policies/policy_config.py:48) | 按 model.safetensors 存在与否选择后端，并进行 BF16 转换 | 新增显式量化 artifact 类型；统一精度处理必须在量化之前完成 |
| [Policy 初始化](/Users/yuyu/Documents/openpi/src/openpi/policies/policy.py:55) | 再次 `.to(device)` 并绑定采样函数 | 让量化加载器掌握设备迁移；完成量化与编译后才构造 Policy |
| [Gemma 补丁](/Users/yuyu/Documents/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py:506) | 根据 q_proj.weight.dtype 决定 hidden_states 是否转 BF16 | 量化 Q/K/V 前，改用明确的计算 dtype 契约并回归验证 |
| [gemma_pytorch.py](/Users/yuyu/Documents/openpi/src/openpi/models_pytorch/gemma_pytorch.py) | 存在自定义 adaRMS、前缀/后缀模型调用和缓存传递 | 使用实际 sample_actions 路径验证，不能仅用标准 HF generate 测试 |
| [JAX 转换脚本](/Users/yuyu/Documents/openpi/examples/convert_jax_model_to_pytorch.py:518) | `strict=False` 加载；assets 从 checkpoint 父目录复制 | 检查缺失/多余键，并核对统计量的真实位置和哈希 |

本地已有模型和 Transformers 补丁是兼容性基线。不能直接套用 `AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)`，也不能直接覆盖当前环境升级整个 Transformers 栈。

## 4. 技术选型

### 4.1 主方案：TorchAO 权重量化

TorchAO 提供对 PyTorch Linear 的权重量化及动态激活量化接口，适合在当前自定义模型中按模块选择量化对象。第一版选择 W8A16 和 W4A16：W 表示权重位宽，A16 在本文指所选 Linear 的 BF16 计算输入；其余模块沿用原浮点精度。接口能力依据 [TorchAO 官方量化 API](https://docs.pytorch.org/ao/stable/api_reference/api_ref_quantization.html)。

**版本策略：优先试验 PyTorch 2.7.1 + TorchAO 0.12.0 + Transformers 4.53.2，并保留本项目补丁。** 官方兼容矩阵将 TorchAO 0.12.0 与 PyTorch 2.7.1 列为兼容组合；它是本项目的候选组合，不代表已经在目标 GPU 验证。[官方版本兼容表](https://github.com/pytorch/ao/issues/2919)

为量化创建独立环境或镜像，锁定精确版本、CUDA wheel 来源和补丁哈希。若 0.12.0 在目标平台无法工作，再独立测试新的 PyTorch/TorchAO 组合，同时重新验证原模型。不要把当前稳定版 API 示例直接复制到旧版环境。

### 4.2 备选与后续路线

| 路线 | 适合解决的问题 | 本项目安排 |
| --- | --- | --- |
| TorchAO INT8 weight-only | 建立低风险压缩对照 | 第一轮 |
| TorchAO INT4 weight-only | 进一步压缩语言主干权重 | 主力实验 |
| bitsandbytes NF4 | TorchAO 兼容性受阻时建立另一种 4-bit 权重基线 | 备选，单独适配 Linear，不混用现有 artifact 格式 |
| 动态 INT8 激活或 FP8 | 权重量化收益不足且硬件/kernel 合适 | 第二阶段，单独评估 |
| ModelOpt + TensorRT | 已识别稳定子图和性能瓶颈，需要进一步编译部署 | 后续研究路线 |
| QAT / 蒸馏恢复 | 混合精度 PTQ 无法达到任务质量要求 | 最后再考虑，需要额外数据和训练预算 |

bitsandbytes 的 `Linear4bit` 支持 NF4 等表示，并有明确的“先加载浮点权重、再移至 CUDA 量化”流程。NF4 是一种 4-bit 表示，不能与均匀 INT4 或原生硬件 FP4 混为一谈；具体 kernel 和速度仍需实测。[官方 Linear4bit 文档](https://huggingface.co/docs/bitsandbytes/main/en/reference/nn/linear4bit)

TensorRT 的低精度部署需要有实际支持的算子与量化图；ModelOpt 可以帮助构建 Q/DQ，但生成带量化节点的图并不等于本项目完整策略已获得加速。第一阶段不把导出整个 Python 动作采样循环作为前置条件。[TensorRT 官方量化工作流](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)

### 4.3 相关 VLA 量化研究的用法

QuantVLA 研究讨论了选择性量化、注意力温度匹配和输出校准，可以作为动作专家量化的后续参考。核查时其公开仓库说明主要围绕 GR00T 展开，不能当作当前 OpenPI π0.5 的即插即用实现。[论文](https://arxiv.org/abs/2602.20309)、[作者代码仓库](https://github.com/AIoT-MLSys-Lab/QuantVLA)

本项目首先建立可解释的逐模块 PTQ 基线，再考虑迁移论文中的校准方法。论文报告的压缩率或加速率不写入本项目的预期实测结果。

## 5. 量化范围与实验矩阵

### 5.1 模块分组

运行时先导出 `named_modules()`，并按实际执行情况分组，不依赖未经核验的字符串模糊匹配。

| 模块 | 第一版处理 | 原因 |
| --- | --- | --- |
| 语言主干 MLP：gate/up/down 投影 | INT8 → INT4 | 较容易独立替换，优先建立压缩基线 |
| 语言主干 attention Q/K/V/O | 初始保留，后续逐组加入 | 影响前缀缓存，且存在 dtype 判断耦合 |
| 视觉编码器 | 保持基线浮点 | 避免早期同时引入视觉表征变化 |
| 多模态 projector | 保持基线浮点 | 视觉与语言接口较敏感 |
| Gemma 动作专家 MLP | 主方案先保留；必要时 INT8 | 多次采样累计误差，需要覆盖完整时间步 |
| 动作专家 attention | 后续单独试验 | 涉及共享前缀 KV 和动作动态分布 |
| action_in_proj / action_out_proj | 保持基线浮点 | 直接关联连续控制动作 |
| time_mlp、adaRMS 条件投影 | 保持基线浮点 | 控制时间条件和归一化调制 |
| Embedding、LayerNorm/RMSNorm、位置编码 | 保持基线浮点 | 不属于首轮 Linear 量化目标 |
| KV cache、softmax、Euler 累加 | 保持基线精度 | 与权重量化分离，避免同时改变数值路径 |
| lm_head 等未执行模块 | 先检查调用与权重共享 | 不把未使用层的压缩当作有效推理计算优化 |

尤其需要按存储去重统计 tied weights；模型中存在的层不一定在 `sample_actions` 中执行。第一版不删除未执行模块，裁剪另作独立实验。

### 5.2 最小实验矩阵

| 编号 | 配置 | 用途 |
| --- | --- | --- |
| B0 | 当前远程已验证后端与权重 | 现有能力参考；若为 JAX，不直接作为量化加速分母 |
| B1 | PyTorch 混合浮点、eager | 权重转换与数值基线 |
| B2 | B1 + 编译 | 单独衡量编译收益 |
| Q1 | 语言 MLP W8A16 | 验证量化集成和较温和压缩 |
| Q2 | 语言 MLP W4A16，group_size=128 | 核心候选 |
| Q3 | Q2 改 group_size=64 | 质量/元数据开销对照 |
| Q4 | 最佳 MLP 候选 + 语言 attention 量化 | 仅在 dtype 和缓存路径验证后推进 |
| Q5 | 最佳语言候选 + 动作专家 MLP INT8 | 仅在动作专家被证明是瓶颈时推进 |

Q1～Q5 均先做 eager 数值检查，再测试编译版本。计算“量化独立收益”时，比较相同编译设置的浮点和量化版本；计算“最终部署收益”时，再与原运行版本比较，并分开列出。

TorchAO 新版 INT4 接口与 0.12.0 的 layout 参数存在区别；实施按锁定版本使用接口，记录实际 packing/layout。分组选择及旧版 API 可核对 [0.12.0 固定版本源码](https://github.com/pytorch/ao/blob/v0.12.0/torchao/quantization/quant_api.py)；新版说明见 [Int4WeightOnlyConfig](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int4WeightOnlyConfig.html)。

## 6. 显存与时延的测算方法

### 6.1 显存不能按“4-bit 模型=原显存四分之一”计算

量化后峰值显存包括：量化权重、scale/zero 元数据、未量化权重、激活与 KV、计算临时空间、编译/CUDA Graph 缓冲区、框架上下文，以及同卡仿真渲染占用。

设原本 16-bit 权重字节中有比例 f 被转换为 4-bit，暂忽略额外元数据：

```text
量化后的这部分权重总字节 / 原权重字节 ≈ (1-f) + f/4
```

例如 f=0.6 时，权重总字节约为原来的 55%，而不是 25%。这只是解释性计算；当前项目 f 值应由实际模块与存储统计得出，FP32 保留部分另外计入。

必须同时记录：模型驻留显存、推理峰值 allocated、峰值 reserved、NVML 进程显存、量化转换峰值、冷启动峰值，以及量化 artifact 字节数。

### 6.2 首轮量化可能主要节省内存

近似时延分解为：

```text
T_request = T_preprocess + T_vision + T_prefix
            + N_denoise × T_expert + T_postprocess + T_transport
```

若语言主干仅占总时间的 35%，即使其时间减半，整体加速也只有约 `1/(0.65+0.35/2)=1.21`。此处比例是示例，不是本项目测量结果。

因此先 profile 再决定是否量化动作专家；减少采样步数、切换 attention kernel、裁剪未执行头和更改图像输入都应独立做消融实验。

## 7. 阶段 P0：冻结基线与环境检查

### 7.1 记录环境

在远程服务器记录以下内容，形成 `environment.json` 与依赖锁文件：

- GPU 名称、显存、compute capability、驱动与 CUDA runtime。
- PyTorch、TorchAO、Transformers、JAX、MuJoCo、robosuite 版本及导入路径。
- 项目版本、补丁文件哈希、模型权重与 norm_stats 哈希。
- 模型配置：pi05、action_dim、action_horizon、采样步数、状态开关。
- 图像尺寸、相机、任务集、控制频率、replan_steps、是否同卡渲染。

不在主环境直接执行无版本限制的 `pip install -U`。先在独立环境验证 Linear 微基准，再加载完整策略。使用实际层形状和实际 BF16 输入检查量化、kernel 调用、保存重载及峰值显存。

如果目标 GPU 不支持所需 BF16 或量化 kernel，先停止该组合试验；需要 FP16 替代路线时，应单独建立 FP16 数值与任务基线，再选择兼容的量化实现，不能直接对当前混合精度模型整体调用 `.half()` 后继续沿用原基线结论。

### 7.2 JAX → PyTorch 对齐

若基线已经是可信的 PyTorch 权重，可直接核验并跳过转换。否则使用项目的 [转换脚本](/Users/yuyu/Documents/openpi/examples/convert_jax_model_to_pytorch.py)，但先解决以下事项：

1. 明确 `--config-name`，与原模型配置保持一致。
2. 检查 `load_state_dict(..., strict=False)` 返回的 missing/unexpected keys；仅允许明确记录的共享权重或非执行模块差异。
3. 检查 `assets` 来源。脚本当前取 checkpoint 父目录 assets，而策略加载器读取 checkpoint 自身 assets，二者未必一致。
4. 优先保留 FP32 转换源，再由策略加载器施加原有混合精度规则，避免转换脚本全模型 BF16 cast 提前降低动作投影等模块精度。
5. 保存完整配置，包括 `pi05`、状态开关、token 长度和补丁版本；转换脚本自带的 config.json 字段不够完整。

对同一预处理输入，给 JAX 与 PyTorch 提供同一份显式噪声数组，而不只是相同随机种子。分别比较前缀、中间采样结果和最终 7 维动作。

若后端转换误差尚未解释，暂停量化实验。不能把转换、补丁或精度错误归因于量化。

## 8. 阶段 P1：数据准备与可重复回放

### 8.1 数据分成四种用途

| 数据 | 建议首轮规模 | 用途 |
| --- | --- | --- |
| smoke 样本 | 16～32 个观测 | 形状、dtype、有限值和服务连通性 |
| 校准集 | 512～1,024 个观测 | 激活统计、敏感度分析、后续 AWQ/激活量化 |
| 开发回放集 | 200～500 个观测 | 选择位宽、分组、回退层 |
| 锁定测试集 | 独立 episode 的 200～500 个观测，另配闭环任务 | 最终报告，禁止反复据此调参 |

这些是实践起点，不是理论保证。按 episode、任务实例或采集轨迹划分，避免相邻视频帧分别进入校准集和测试集。覆盖接近、对齐、闭合、提起、移动、释放阶段，兼顾成功和失败状态。

纯 weight-only 的基础量化可直接由权重决定尺度，不要求运行激活校准。此时数据主要用于选择量化范围和质量验证；后续涉及激活、AWQ 或误差恢复时再启用校准过程。

### 8.2 数据来源优先级

优先录制当前 ECO65/ZhiXing 浮点策略的真实仿真观测，覆盖实际相机和任务。补充 LIBERO 数据用于检查是否过拟合少数场景。

现有 [ECO 离线客户端](/Users/yuyu/Documents/openpi/ECO_data/remote_inference_client.py) 将同一张图同时填入两个视角，并截取 12 维状态前 8 维，不宜直接当作最终部署分布的校准基准。已有 CSV/图像可以用于读取和服务 smoke test。

本地 LIBERO 转换数据目录仅有元信息，不能假定完整训练/校准数据已随副本存在；应从远程数据集获取实际观测。

### 8.3 回放记录格式

拟保存每个样本的两路图像、状态、prompt、task_id、episode_id、step_id、预处理版本和显式初始噪声。记录输出动作时同时保存网络输出的前 7 维归一化值、反归一化动作及控制器实际接收值。

噪声固定为 `(1, 10, 32)` 或按接口明确保存的无 batch 形式。通过本地 Python `Policy.infer(obs, noise=...)` 做配对比较；现有 WebSocket 协议并没有显式传递这个 keyword 参数，不能把噪声字段随意塞进 obs 后假定它生效。

若需要服务端确定性评测，新增仅供评测使用的 seed/noise 处理入口，或在测试 wrapper 中控制采样噪声，并在输出中记录 noise_id。

## 9. 阶段 P2：按模块实施量化

### 9.1 加载、量化和编译的顺序

```text
解析并验证模型/量化配置
  → 创建未编译的 PyTorch 模型
  → 加载浮点参数或显式量化 artifact
  → 应用基线混合精度规则（仅浮点加载路径）
  → 校验层清单、dtype、共享权重和设备
  → 按后端要求分配到 GPU 并量化指定层
  → smoke test / 质量检查
  → 保存并在新进程重载校验
  → 可选 torch.compile
  → 构造 Policy 并绑定最终 sample_actions
  → warmup 完成后对外提供服务
```

首轮在线量化需要 GPU 能容纳浮点基线及转换临时内存。若量化目标设备无法容纳浮点模型，使用离线预量化 artifact 或逐层加载方案；不能要求一个只能装下量化模型的设备先把整个浮点模型搬到 GPU。

量化后不再对整个模型调用 `.half()`、`.bfloat16()` 或通用 dtype 转换。Packed tensor 的移动和恢复由量化后端处理；`Policy` 不重复执行不受控的设备或精度变换。

### 9.2 首轮白名单与参考代码

下面是实施设计片段，不是仓库中已存在的脚本。API 按候选 TorchAO 0.12.0 编写，完整模块名必须以运行时导出结果为准。

```python
import torch
from torchao.quantization import (
    quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig,
)

# model 已加载权重、完成基线 dtype 设置，且尚未编译。
# 先记录清单，再由人工/规则审核后固定到 artifact manifest。
prefix = "paligemma_with_expert.paligemma.model.language_model.layers."
allowed_names = {
    name for name, module in model.named_modules()
    if name.startswith(prefix)
    and ".mlp." in name
    and name.rsplit(".", 1)[-1] in {"gate_proj", "up_proj", "down_proj"}
    and isinstance(module, torch.nn.Linear)
}
assert allowed_names, "量化白名单为空，先检查真实模块名称"

def select(module, fqn):
    return fqn in allowed_names and isinstance(module, torch.nn.Linear)

# 每个实验从同一份浮点模型独立加载，只选择下面一种配置。
recipe = Int8WeightOnlyConfig()
# recipe = Int4WeightOnlyConfig(group_size=128)

quantize_(model, recipe, filter_fn=select)
```

实施时还要检查每层输入维度与 group_size、layout 和 kernel 的兼容性；量化后逐层确认实际权重表示，不能只检查 API 没有抛异常。有些不满足条件的层可能被跳过，必须写入报告。

不要在已经量化的对象上再次执行另一种量化；每个候选从干净的浮点权重或指定 artifact 创建。

### 9.3 误差过大时的处理顺序

1. 先确认预处理、统计量、权重、噪声与采样步数一致。
2. 对误差较大的 MLP 层做逐层/逐组回退，检查前缀缓存与最终动作。
3. 将敏感层恢复为 INT8 或原浮点精度。
4. 尝试更小 group_size，记录元数据与 kernel 代价。
5. 若仍不满足闭环要求，再考虑激活感知校准或 QAT。

不能通过额外截断、平滑动作或改变夹爪阈值来“掩盖”量化退化；这些控制修改应另立实验。

## 10. 阶段 P3：动作专家与编译优化

### 10.1 为什么动作专家放到后面

动作专家被重复调用，少量局部误差可能在积分和后续闭环中放大。它虽然可能是延迟瓶颈，但不宜与语言主干在首轮同时量化，否则难以定位质量变化。

扩展时先尝试动作专家 MLP INT8，保留 attention、adaRMS 条件投影、时间 MLP 和动作输出投影。只有该候选稳定后，才考虑更低位宽或更大范围。

### 10.2 校准必须覆盖采样轨迹

对于需要激活校准的方法，使用真实 `sample_actions` 或等价的 `denoise_step` 路径覆盖全部 10 个时间步，记录每步 `x_t`、时间条件与关键激活分布。

可先在浮点轨迹上采集统计，再用量化模型的轨迹检查分布漂移。仅用文本、单次前向、t=0，或者只运行训练用联合 forward，不足以代表当前推理时的前缀缓存和重复后缀调用。

### 10.3 编译策略

从 eager 通过的模型开始，量化后再编译。首轮固定图像与 token 的 padding 形状、batch size 和动作块长度，分别测试整段 sample_actions 与较小子图。

当前采样中存在 tensor 控制的 while 条件、KV cache 对象和 attention 配置赋值，可能产生 graph break 或重复编译。优先检查编译日志和 profiler，再决定拆分视觉、前缀、单步动作专家。

如果后续把循环改成固定次数的 for 循环，必须先证明时间点、积分规则和输出与原实现对齐，并将循环改写的收益独立列出。第一版不要求无 graph break，也不把成功调用 torch.compile 当作加速证据。

## 11. 正确性与任务质量评测

### 11.1 三级验证

| 层级 | 内容 | 作用 |
| --- | --- | --- |
| 模块级 | 已选择 Linear 的有限值、dtype、层输出误差和量化覆盖率 | 排查接口与 kernel 错误 |
| 完整动作级 | 同观测、同噪声下的 10×7 输出 | 排查多步采样误差 |
| 闭环任务级 | ECO65/ZhiXing 多任务配对 rollout | 判断是否保留实际任务能力 |

后两级必须执行。浮点输出是压缩一致性的参考，不是动作正确性的真值；最终仍需看任务成功与轨迹行为。

### 11.2 离线指标

分别报告位置、旋转和夹爪，不把三类量纲直接混成一个 RMSE：

- 前 6 维每维 MAE、RMSE、P95/P99 绝对误差；同时报告前 5 个执行步和全部 10 步。
- 按基线/量化共同使用的 norm_stats 计算归一化动作误差，不纳入 25 维填充动作。
- 将前 6 维动作分别代入实际 OSC 缩放，报告目标平移增量差和目标旋转增量差；这不是实际末端位置误差。
- 夹爪开/合方向一致率、零输入保持状态的变化、反转次数，以及基线接近零值的样本比例。
- 动作越界、NaN/Inf、控制器饱和比例。

ZhiXing 使用 `sign(action)` 驱动累加，因此夹爪第七维的小误差也可能翻转开合方向。统计全部符号变化，另报告 `|a_baseline|>0.1` 的稳定方向样本；阈值仅用于分析，不改变控制器。

### 11.3 闭环协议

第一轮选择当前浮点模型确实能完成的 3～5 个任务，每任务 10 次作为开发筛选；最终候选至少覆盖 `libero_spatial` 全部 10 个任务、每任务 50 次，共 500 次/模型。若统计不确定性仍过大，再增加样本，不能把 500 次当作精度保证。

所有版本使用同一 ECO65 模型、ZhiXing 模型、额外关节 PD、OSC 配置、相机、采样步数与重规划间隔。为每个 episode 保存并复用 ECO65 环境对应的初始状态；不能恢复 Panda 的整机状态。

初始状态复现不能只保存 qpos：应复现或重建 qvel、物体位置、随机状态、控制器目标、夹爪 current_action 和等待动作历史，并检查首次决策前观测一致。可采用一致 reset 种子与初始化流程，辅以状态快照和观测哈希确认。

成功/失败以任务条件和统一终止逻辑记录。保存包含 experiment_id、task_id、episode_id、seed 的独立视频和 JSONL 记录，避免当前主脚本同名视频覆盖。

量化模型与基线应配对运行并交替实验顺序，减少温度、时钟或负载变化。测纯策略性能时关闭录像和逐帧 cv2.imwrite；测完整系统时两者使用同一记录策略。

### 11.4 低成功率基线的处理

如果 ECO65 浮点基线本身接近零成功率，量化模型同样为零不能说明量化“无损”。先恢复可用基线，或者用可成功的任务子集开展开发验证，并明确适用范围；必要时另建原生 Panda 参考环境区分机器人迁移问题与模型量化问题，不能悄悄改动当前 ECO65 配置来改善对照结果。

## 12. 性能测量与控制时序

### 12.1 测量口径

现有 [Policy.infer()](/Users/yuyu/Documents/openpi/src/openpi/policies/policy.py:94) 用 CPU 时钟包围模型调用，而 GPU 可能异步执行，`policy_timing.infer_ms` 不能直接视为准确 GPU 计算时间。现有 `server_timing` 也不是网络端到端时间。

拟新增三类计时：

| 指标 | 测量方式 |
| --- | --- |
| GPU 前缀/视觉/各 denoise 阶段时间 | CUDA Events 或 profiler；读取前完成同步 |
| 服务端请求时间 | 包含预处理、传输到设备、采样、传回 CPU、后处理的同步 wall clock |
| 客户端端到端时间 | 从发送前到完整接收并解码后；包含网络和序列化 |

正式性能样本建议预热 30～50 次，之后记录至少 500 个请求，并重复 3 轮；记录 P50/P95/P99、均值、标准差和温度/时钟。编译时间、首次推理、权重量化时间单独列出。

B1/B2/Q 各实验尽量使用独立进程，以免 allocator 缓存影响峰值。单 GPU 时依次测浮点与量化，不要求两个完整模型同时驻留。模型独占测量与同卡 MuJoCo 渲染测量分别报告。

### 12.2 仿真步频与实时控制不同

当前环境默认控制频率 20 Hz，执行 5 步后重新推理，因此动作块覆盖 250 ms 仿真时间。但客户端是同步推理，队列空后会等待模型返回；低于 250 ms 的推理延迟本身不等于已经实现无停顿的 20 Hz 真机控制。

若执行本身耗时为 `5/20=0.25 s`，额外同步推理耗时 T，忽略其他开销时有效频率约为：

```text
f_effective ≈ 5 / (0.25 + T)
```

例如 T=0.2 s 时约为 11.1 Hz。这是时序示例，实际以系统日志为准。

第一版保持同步架构便于公平比较。异步预取、时间戳对齐和动作队列管理作为后续系统优化；不能仅通过调大 replan_steps 获得更高表面执行频率而忽略反馈变慢。

## 13. 验收标准与决策规则

以下是本项目 v1 的建议工程门槛，属于预先约定的实验目标，不是工具或论文保证。

| 维度 | 建议标准 | 决策 |
| --- | --- | --- |
| 功能正确性 | 所有测试输入形状正确、无 NaN/Inf；加载/重载后输出在固定噪声下符合约定容差 | 必须通过 |
| 转换一致性 | JAX/PyTorch 差异解释清楚，浮点 PyTorch 有可用闭环基线 | 未通过不得开始量化结论 |
| 离线动作筛选 | 每维归一化 MAE ≤0.02、P99 绝对误差 ≤0.10；分别报告前 5 步/全 10 步 | 初始筛选值，结合任务敏感度在开发集冻结 |
| 夹爪方向 | 稳定方向样本一致率 ≥99%，并检查所有过零事件 | 不满足则优先回退敏感层 |
| 闭环质量 | 整体成功率下降目标 ≤3 个百分点；关注逐任务明显退化 | 配对统计检验后决定 |
| 显存收益 | 目标推理峰值 allocated 至少下降 25%，并报告 NVML 总量 | 未达到则报告实际结果，检查覆盖率 |
| 延迟收益 | 争取服务端 P95 加速 ≥1.2 倍；至少不明显劣于对应浮点基线 | 只省显存则标记为压缩方案，不称为加速方案 |
| 长时间运行 | ≥1,000 次连续请求，零服务崩溃/输出异常，无持续显存增长 | 部署前完成 |
| 重建与回退 | 新进程加载 artifact、版本/哈希验证、恢复浮点服务均通过 | 部署必需 |

成功率以百分点差报告，例如 80% 降至 77% 是下降 3 个百分点。使用按任务分层、episode 配对的 bootstrap 或等价方法估计 `p_baseline - p_quantized` 的置信区间；若其单侧 95% 上界仍高于 0.03，不能声称已证明满足非劣效目标，应增加样本或标记证据不足。逐任务少量 episode 的不确定性也需列出。

当两个候选都通过质量门槛时，根据实际目标选择：显存受限优先峰值更低者；时延受限优先 P95 更低者。保留质量、显存、时延的折中表，不强行选择最低位宽。

## 14. 工程改造清单

下表中的新增路径均是拟实现内容，本次仅创建方案文档。

| 文件或模块 | 拟实现职责 |
| --- | --- |
| `src/openpi/quantization/config.py` | 量化后端、位宽、分组、白名单、计算精度、编译配置 |
| `src/openpi/quantization/torchao_backend.py` | 层发现、替换、实际覆盖验证、量化统计 |
| `src/openpi/quantization/artifact.py` | manifest、后端专用保存/加载、哈希校验 |
| `scripts/quantization/inspect_model.py` | 参数/存储/执行层统计及 dtype 清单 |
| `scripts/quantization/collect_replay.py` | 观测、任务 ID、噪声、数据划分清单 |
| `scripts/quantization/quantize_policy.py` | 从浮点 checkpoint 构建独立量化 artifact |
| `scripts/quantization/benchmark_policy.py` | 阶段耗时、显存、固定回放与输出指标 |
| `scripts/quantization/eval_rollouts.py` | 配对初始化、统一成功率和轨迹记录 |
| `scripts/quantization/serve_quantized_policy.py` | 量化 artifact 服务入口，复用 WebSocket 协议 |
| `configs/quantization/` | B1/Q1/Q2 等实验配置与版本锁定信息 |
| `reports/quantization/` | 每轮 metrics.json、rollouts.jsonl、图表与结论 |

现有 [policy_config.py](/Users/yuyu/Documents/openpi/src/openpi/policies/policy_config.py)、[policy.py](/Users/yuyu/Documents/openpi/src/openpi/policies/policy.py) 只做必要的加载/绑定扩展；量化实现集中在独立模块，不侵入 robosuite 控制器。

拟提供的量化配置字段：

```yaml
# 设计样例：尚未接入当前 CLI。
experiment_id: pi05_libero_q2_v1
train_config: pi05_libero
source_checkpoint: /path/to/verified/pytorch_checkpoint
backend: torchao
backend_version: 0.12.0
recipe: int4_weight_only
scope: language_mlp
group_size: 128
compute_dtype: bfloat16
keep_baseline_float_modules: true
action_horizon: 10
num_denoise_steps: 10
replan_steps: 5
compile_mode: null
device: cuda:0
module_manifest: /path/to/approved_module_manifest.json
output_dir: /path/to/artifacts/pi05_libero_q2_v1
```

配置名必须对应真实 checkpoint；如果权重来自 `pi05_libero_custom`，需要同步修改数据资产与配置，不能沿用官方 LIBERO 统计量冒充一致环境。

## 15. Artifact、服务与回退设计

### 15.1 Artifact 内容

```text
<量化产物目录>/
  manifest.json
  model_config.json
  quantization_config.json
  module_manifest.json
  environment.json
  weights/                 # 后端约定格式，不预设为普通 safetensors
  assets/                  # 与源 checkpoint 一致的统计量
  validation/metrics.json
```

manifest 至少记录源权重哈希、配置名、量化算法与版本、实际层列表及排除原因、packing/layout、计算 dtype、硬件与 CUDA、补丁哈希、数据划分哈希、采样参数及评测结论。

TorchAO 使用量化 tensor 表示，保存和恢复需要遵循相应版本的序列化方式；不能默认现有 `safetensors.torch.load_model()` 会恢复所有量化元数据。应实现独立加载器，并在新进程验证状态、压缩存储和真实 kernel。[TorchAO 官方序列化说明](https://docs.pytorch.org/ao/stable/eager_tutorials/serialization.html)

v1 原型允许从浮点权重在启动时量化以便调试，但必须单独记录量化时间和峰值；最终部署交付应验证无需重新校准的 artifact 恢复。编译缓存与权重分开管理，硬件或依赖改变后重新验证。

### 15.2 服务行为

保留客户端输入的图像、状态、prompt 和输出 `actions` 协议。服务 metadata 增加 experiment_id、source_hash、quantization_summary、action_horizon、num_denoise_steps，便于确认正在运行哪个版本。

加载、校验和 warmup 完成后才宣告 ready。现有 `/healthz` 只返回通用 OK，量化服务需要确保它或新增 readiness 检查代表模型真正可推理。

先以单客户端顺序请求部署，限制并发队列，避免同一卡上多请求堆积引入不受控延迟。不会把提高吞吐量当作降低机器人单请求时延。

### 15.3 回退

浮点与量化权重分别保存，不覆盖原 checkpoint。版本切换在 episode 边界进行；客户端清空旧动作队列并重建 episode。单卡显存不足以双驻留时，使用停止量化服务、重启浮点服务的回退方式，不承诺无停机切换。

服务异常、artifact 不匹配或输出非有限值时，应中止当前评测 episode 并记录失败原因，不能静默退回不同模型后继续计入同一实验。

## 16. 实施顺序、时间与里程碑

以下为单人熟悉代码、已有可用远程环境时的工作量估计，不包含下载大权重、长队列等待或大量闭环仿真时间。

| 阶段 | 预计工作量 | 交付与退出条件 |
| --- | --- | --- |
| P0 环境与浮点基线 | 1～3 天 | 环境/权重清单、转换一致性、可用闭环基线 |
| P1 数据与测量工具 | 1～2 天 | 固定回放集、显式噪声、准确计时和显存报告 |
| P2 Q1/Q2/Q3 | 2～3 天 | 量化层清单、数值与压缩对照、候选排序 |
| P3 编译与可选范围扩展 | 1～3 天 | kernel/profile 证据，明确瓶颈及最终候选 |
| P4 闭环与独立重载部署 | 2～3 天 | 配对成功率、artifact、服务与回退记录 |

最小可行版本预计约 7～14 个工作日，兼容性问题和评测时长可能延长。阶段门槛未通过时优先修复或回退，不为了完成所有低位宽实验而继续叠加变量。

建议首先只实施 P0～P2，形成一张可以回答“有没有省显存、有没有退化、瓶颈在哪里”的结果表，再决定是否投入动作专家量化和 TensorRT。

## 17. 命令与操作边界

下面前两类是现有接口的命令模板；路径须替换为远程实际路径。本次没有执行。

```bash
# 现有接口：硬件信息
nvidia-smi

# 现有接口：JAX 转 PyTorch。
# 执行前按 P0 核对 missing keys、assets 来源与精度转换。
cd /path/to/server/openpi
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /path/to/jax/checkpoint \
  --config-name pi05_libero \
  --output-path /path/to/pytorch/checkpoint \
  --precision float32

# 现有接口：启动已经完成核验的浮点 PyTorch checkpoint。
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /path/to/pytorch/checkpoint
```

原服务会按当前默认配置编译，以上启动命令不等于 B1 eager 实验；B1/B2 的明确开关由拟新增的基准工具或加载选项提供。

以下为**拟新增工具的 CLI 设计**，目前尚不能直接运行：

```bash
python scripts/quantization/inspect_model.py --config configs/quantization/b1.yaml
python scripts/quantization/collect_replay.py --config configs/quantization/replay.yaml
python scripts/quantization/quantize_policy.py --config configs/quantization/q2.yaml
python scripts/quantization/benchmark_policy.py --config configs/quantization/q2.yaml
python scripts/quantization/eval_rollouts.py --config configs/quantization/q2.yaml
python scripts/quantization/serve_quantized_policy.py --artifact /path/to/artifacts/q2
```

在这些工具实现前，不能把命令样例当作现有功能，也不能仅安装 TorchAO 就宣称完成量化部署。

## 18. 结果报告模板

每个候选必须有完整配置和结果；无法测量的字段写“未测”，不填估计值。

| 版本 | 后端/精度 | 实际量化权重比例 | artifact GiB | 峰值显存 GiB | 服务端 P50/P95 ms | 客户端 P95 ms | 夹爪方向一致率 | 成功率及区间 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | PyTorch 混合浮点 eager | 0 | 未测 | 未测 | 未测 | 未测 | 参考 | 未测 |
| B2 | PyTorch 混合浮点 compiled | 0 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| Q1 | MLP INT8 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| Q2 | MLP INT4 g128 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |
| Q3 | MLP INT4 g64 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |

报告另外提供：模块级耗时图、各时间步动作误差曲线、逐任务成功率、量化层排除清单、失败视频索引和阶段退出原因。最终结论可以是“量化节省显存但未加速”，也可以是“INT8 比 INT4 更适合该 GPU 与任务”；以实际部署目标和证据选择配置。

## 19. 启动实施时最先要做的五件事

1. 获取服务器 GPU 型号/显存及当前已跑通的 checkpoint、配置名、后端。
2. 在独立环境锁定 PyTorch/TorchAO/Transformers 和本项目补丁，运行真实层形状微基准。
3. 建立未量化 PyTorch 基线，核验转换、assets 和数值路径。
4. 录制 ECO65/ZhiXing 的可重复双视角回放，固定采样噪声，完成延迟和显存 profile。
5. 仅对白名单中的语言 MLP 做 INT8、INT4 对照，依据离线和闭环结果决定下一步。

本方案基于本地副本阅读和 2026-09-24 查阅的官方工具文档设计；工具页面会持续更新，实际实施应以锁定版本源码、目标 GPU 的 kernel 验证和项目实验记录为准。
