+++
date = '2024-10-01'
draft = false
title = 'SDXL 推理极限优化实战（二）：全面优化实践指南'
tags = ['Stable Diffusion', '推理优化', 'TensorRT']
categories = ['Diffusion', 'inference']

+++

> **系列导读**：本系列共三篇文章，渐进式地探讨 Stable Diffusion XL（SDXL）模型的推理优化。[第一篇](./SDXL推理优化实战(一)：模型剖析与性能Profiling.md)聚焦于问题背景与性能剖析，本篇展开全面的单项优化实践，第三篇进行混合组合优化与吞吐工程部署。

---

在上一篇中，我们通过 Profiling 明确了 SDXL 推理的性能瓶颈：UNet 去噪循环占据 90%+ 的推理时间，Attention 层受限于访存带宽，大量小 Kernel 引入启动开销，CFG 使计算量翻倍。本篇将基于这些发现，**从三个维度逐一展开优化实践**：

```
优化维度总览
══════════════════════════════════════════════════════
                                        精度
维度 1  数值精度优化                     无损 ──────▶ 有损
        FP32 → FP16 / BF16 / TF32      ████░░░░░░

维度 2  编译与算子层优化                 无损
        torch.compile / StableFast      ██████████
        OneDiff / TensorRT

维度 3  模型组件级优化                   无损 ──────▶ 有损
        VAE Fix / TinyVAE / CFG /       ████████░░
        DeepCache / 蒸馏 / 显存优化
══════════════════════════════════════════════════════
```

> **Baseline 约定**：所有实验以 **FP16、30 步、1024×1024** 为基准（延时 5.5s，显存 11.24GB），在 NVIDIA L20 上执行。

---

## 1. 数值精度优化

数值精度是推理优化中 **投入最小、收益最大** 的一环。它的本质是用更少的 bit 来表示数字，从而同时减少计算量和访存带宽需求。

### 1.1 浮点格式速览

在深入实验之前，先理解四种常用浮点格式的区别：

```
        符号  指数   尾数     总位数   数值范围         精度
FP32:   1    8      23       32      ±3.4×10³⁸       ~7 位有效数字
TF32:   1    8      10       19*     ±3.4×10³⁸       ~3 位有效数字
FP16:   1    5      10       16      ±65504           ~3 位有效数字
BF16:   1    8       7       16      ±3.4×10³⁸       ~2 位有效数字

* TF32 仅在 Tensor Core 内部使用，存储仍为 32 bit
```

**关键差异**：
- **FP16** vs **BF16**：FP16 精度更高（10 bit 尾数），但数值范围小（易溢出）；BF16 范围与 FP32 一致（8 bit 指数），但精度较低
- **TF32**：NVIDIA Ampere+ 架构的 Tensor Core 特有格式，存储为 FP32（不省显存），计算用 TF32（略微加速）

### 1.2 FP32 → FP16：最基本的优化

FP16 优化的核心原理：

1. **计算加速**：FP16 Tensor Core 吞吐量是 FP32 的 2~4 倍（取决于架构）
2. **带宽减半**：每个数据从 4 Byte 变为 2 Byte，对访存密集型算子（如 Attention）收益显著
3. **显存减半**：模型权重和中间激活值的显存占用大幅降低

```python
from diffusers import StableDiffusionXLPipeline
import torch

# FP32（默认）
pipe_fp32 = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32
).to("cuda")

# FP16
pipe_fp16 = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
```

**实测结果**：

| 精度   | 步长  | 延时（s） | 速度（step/s） | 显存（GB） | 加速比       |
| ---- | --- | ----- | ---------- | ------ | --------- |
| FP32 | 30  | 16.3  | 1.84       | 18.08  | 1×        |
| FP16 | 30  | 5.5   | 5.45       | 11.24  | <span class="metric-speedup">**2.96×**</span> |
| FP32 | 50  | 26.9  | 1.85       | 18.07  | 1×        |
| FP16 | 50  | 8.7   | 5.74       | 11.24  | <span class="metric-speedup">**3.09×**</span> |

**图片质量对比**：

| FP32                                                         | FP16                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317012810956.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317012845502.png) |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317012852883.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317012856052.png) |

FP16 与 FP32 的生成质量几乎无差异，肉眼不可区分。FP16 精度对于扩散模型的去噪过程完全足够。

### 1.3 TF32：看似折中，实则鸡肋

TF32（TensorFloat-32）是 NVIDIA 在 Ampere 架构中引入的计算格式，设计初衷是"无需改代码即可获得加速"。它使用与 FP32 相同的指数位（保持数值范围），但将尾数截断到 10 bit（与 FP16 相同）。

```python
# 启用 TF32（PyTorch 2.x 中默认已启用）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**实测结果**：

| 精度 | 步长 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 |
|------|------|----------|---------------|-----------|--------|
| FP32 | 30 | 16.3 | 1.84 | 18.08 | 1× |
| TF32 | 30 | 12.4 | 2.41 | 18.08 | <span class="metric-speedup">1.31×</span> |

**为什么 TF32 加速不明显？**

TF32 的局限性在于：
- **不省显存**：数据存储仍为 32 bit，显存带宽没有减少
- **仅加速 Tensor Core 运算**：只有 GEMM 和卷积运算受益，Element-wise 操作不受影响
- **在 L20 上收益有限**：L20 的 FP32 性能已经很强，TF32 的加速幅度有限

> **结论**：TF32 加速仅 1.31×，远不如 FP16 的 2.96×，且不节省显存。**不建议在 SDXL 推理中使用**。

### 1.4 BF16：FP16 的强力替代

BF16（Brain Floating Point 16）与 FP16 的关键区别在于用更多的指数位换取更大的数值范围，牺牲了部分精度。

```python
pipe_bf16 = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16
).to("cuda")
```

**实测结果**：

| 精度 | 步长 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 |
|------|------|----------|---------------|-----------|--------|
| FP32 | 30 | 16.3 | 1.84 | 18.08 | 1× |
| BF16 | 30 | 5.4 | 5.55 | 9.62 | <span class="metric-speedup">**3.01×**</span> |

BF16 相比 FP16 的优势：
- 速度基本持平（5.4s vs 5.5s）
- **显存更低**（9.62 GB vs 11.24 GB，降低 14%）
- 数值范围更大，训练稳定性更好（但推理场景差异不大）

### 1.5 精度优化总结

| 精度 | 延时（s） | 显存（GB） | 加速比 | 质量 | 推荐 |
|------|----------|-----------|--------|------|------|
| FP32 | 16.3 | 18.08 | 1× | 基准 | 仅调试用 |
| TF32 | 12.4 | 18.08 | <span class="metric-speedup">1.31×</span> | 无损 | **不推荐** |
| FP16 | 5.5 | 11.24 | <span class="metric-speedup">2.96×</span> | 无损 | **默认首选** |
| BF16 | 5.4 | 9.62 | <span class="metric-speedup">3.01×</span> | 无损 | 显存紧张时选用 |

> **实践建议**：FP16 是 SDXL 推理的默认精度选择，兼顾通用性和性能。BF16 在显存受限时是更优解。后续所有优化均基于 FP16 展开。

---

## 2. 编译与算子层优化

精度优化解决了"用更少的 bit 做同样的计算"，编译优化则解决"用更少的 Kernel 做同样的计算"。这类优化工作在底层，**对模型精度完全无损**，但通常需要一次性的编译开销。

### 2.1 为什么需要编译优化？

在上一篇的 Profiling 中，我们观察到 SDXL 在 PyTorch eager 模式下存在两个关键问题：

1. **Kernel 数量爆炸**：UNet 单步前向传播触发 ~847 个 CUDA Kernel，每个 Kernel 启动有 5~10μs 的固定开销
2. **算子碎片化**：Conv → BN → ReLU 等模式被拆成独立 Kernel，中间结果要反复 store/load 到显存

编译优化的核心手段：

```
优化前（Eager Mode）:
  Conv Kernel → 写回显存 → 读取 → BN Kernel → 写回 → 读取 → ReLU Kernel
  (3 次 Kernel Launch, 4 次显存 Read/Write)

优化后（算子融合）:
  Fused_Conv_BN_ReLU Kernel
  (1 次 Kernel Launch, 1 次显存 Read/Write)
```

### 2.2 torch.compile：PyTorch 原生方案

`torch.compile` 是 PyTorch 2.0 引入的编译 API，通过 TorchDynamo 捕获计算图，再交由后端（TorchInductor）进行优化，包括算子融合和 CUDA Graphs。

#### 基础模式

```python
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# 编译 UNet 和 VAE
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=True)
```

编译耗时：约 **1 分钟**。

| 方法 | 步长 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 |
|------|------|----------|---------------|-----------|--------|
| Base (FP16) | 30 | 5.5 | 5.45 | 11.24 | 1× |
| torch.compile (reduce-overhead) | 30 | 5.2 | 5.77 | 11.24 | <span class="metric-speedup">1.05×</span> |

#### max-autotune 模式

`max-autotune` 模式会启用 CUDA Graphs 并尝试更多的 Kernel 配置来寻找最优方案：

```python
pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)
```

编译耗时：**较长且不稳定**（5~16 分钟）。

| 方法 | 步长 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 |
|------|------|----------|---------------|-----------|--------|
| Base (FP16) | 30 | 5.5 | 5.45 | 11.24 | 1× |
| torch.compile (max-autotune) | 30 | 4.9 | 5.77 | 11.24 | <span class="metric-speedup">**1.12×**</span> |

#### mode 参数说明

| Mode | 说明 | 编译耗时 | 推理加速 |
|------|------|---------|---------|
| `default` | 基础优化，平衡编译与运行时 | 短 | 小 |
| `reduce-overhead` | 减少 Python 开销和 Kernel Launch 开销 | 中 | 中 |
| `max-autotune` | 启用 CUDA Graphs + 自动调优，针对 latency 优化 | **长** | **最大** |
| `max-autotune-no-cudagraphs` | 自动调优但不使用 CUDA Graphs | 中 | 中 |

> **CUDA Graphs 原理**：将一系列 CUDA 操作录制为一个 Graph，后续执行时通过 **单次 CPU 调用** 触发整个 Graph，消除逐 Kernel 启动的开销。特别适合 SDXL 这种 "同一计算重复执行多步" 的场景。

### 2.3 Stable Fast：超轻量编译框架

[stable-fast](https://github.com/chengzeyi/stable-fast) 是专为 HuggingFace Diffusers 优化的超轻量推理框架，采用了多项底层优化技术：

- **cuDNN 卷积融合**：完整支持 Conv + Bias + Add + Act 计算模式的融合
- **低精度融合 GEMM**：使用 FP16 精度进行计算，比 PyTorch 默认（读写 FP16，计算 FP32）更快
- **NHWC 融合 GroupNorm**：通过 OpenAI Triton 实现高效的 GroupNorm + GELU 融合，消除 channels-last 下不必要的内存排列
- **全模型 Trace**：改进 `torch.jit.trace` 适配复杂模型，支持 ControlNet 和 LoRA
- **CUDA Graph**：将 UNet 捕获为 CUDA Graph，减少小 batch 下的 CPU 开销
- **融合多头自注意力**：集成 xformers 并使其与 TorchScript 兼容

```python
from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

config = CompilationConfig.Default()
config.enable_xformers = True
config.enable_triton = True
config.enable_cuda_graph = True

pipe = compile(pipe, config)
```

首次编译耗时：仅 **18 秒**（这是 Stable Fast 最大的优势）。

| 方法 | 步长 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 |
|------|------|----------|---------------|-----------|--------|
| Base (FP16) | 30 | 5.5 | 5.45 | 11.24 | 1× |
| Stable Fast | 30 | 5.0 | 6.0 | 12.12 | <span class="metric-speedup">**1.10×**</span> |
| Base (FP16) | 50 | 8.7 | 5.74 | 11.24 | 1× |
| Stable Fast | 50 | 7.9 | 6.32 | 12.12 | <span class="metric-speedup">**1.10×**</span> |

**分析**：速度提升 10%，显存略微增加（CUDA Graph 预分配的额外开销）。质量完全无损。

> **Stable Fast 的核心优势不在于绝对加速比，而在于 10 秒级的编译速度**。相比 TensorRT 需要 20 分钟的引擎构建，Stable Fast 在需要频繁切换模型（如 LoRA 热更新）的场景中有不可替代的优势。

### 2.4 OneDiff：一行代码加速

[OneDiff](https://github.com/siliconflow/onediff) 由硅基流动（SiliconFlow）开源，通过改进的注意力机制和深度图编译优化来加速扩散模型。

```python
from onediff.infer_compiler import oneflow_compile

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.unet = oneflow_compile(pipe.unet)
```

首次编译耗时：约 **55 秒**。

| 方法 | 步长 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 |
|------|------|----------|---------------|-----------|--------|
| Base (FP16) | 30 | 5.5 | 5.45 | 11.24 | 1× |
| OneDiff | 30 | 4.8 | 6.25 | **7.24** | <span class="metric-speedup">**1.14×**</span> |
| Base (FP16) | 50 | 8.7 | 5.74 | 11.24 | 1× |
| OneDiff | 50 | 7.5 | 6.66 | **7.24** | <span class="metric-speedup">**1.16×**</span> |

**分析**：OneDiff 不仅加速了 14~16%，还显著降低了显存占用（11.24 → 7.24 GB，降低 **36%**）。这得益于 OneFlow 后端更高效的内存管理和计算图优化。

### 2.5 TensorRT：NVIDIA 官方高性能推理引擎

[TensorRT](https://developer.nvidia.com/tensorrt) 是 NVIDIA 打造的专用推理优化器，拥有最深度的硬件级优化能力：

**优化方法**：
1. **算子融合（Layer & Tensor Fusion）**：将多个计算 op 融合，减少数据搬运和显存访问
2. **Kernel 自动调优（Auto-Tuning）**：根据具体 GPU 架构（SM 数量、频率等），在候选 Kernel 中搜索最优实现
3. **多流执行（Multi-Stream）**：利用 CUDA Stream 最大化并行执行
4. **CUDA Graphs**：减少 Kernel Launch 开销
5. **量化支持**：支持 INT8 / FP16 等低精度推理

```python
# TensorRT 集成方式（通过 diffusers 的 TensorRT 后端）
# 需要先将模型转为 ONNX，再构建 TensorRT Engine
# 详细步骤参见 NVIDIA TensorRT 文档

# 编译流程：Model → ONNX → TensorRT Engine
# 编译耗时：约 20 分钟（包括 ONNX 导出和 Engine 构建）
```

| 方法 | 步长 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 |
|------|------|----------|---------------|-----------|--------|
| Base (FP16) | 30 | 5.5 | 5.45 | 11.24 | 1× |
| TensorRT | 30 | 4.4 | 6.12 | 11.24 | <span class="metric-speedup">**1.25×**</span> |
| Base (FP16) | 50 | 8.7 | 5.74 | 11.24 | 1× |
| TensorRT | 50 | 7.2 | 6.58 | 11.24 | <span class="metric-speedup">**1.20×**</span> |

TensorRT 取得了编译优化中最高的加速比（1.25×），但也存在明显限制：

| 优势               | 限制                   |
| ---------------- | -------------------- |
| 加速比最高            | 编译耗时长（~20min）        |
| NVIDIA 官方维护，稳定可靠 | 不支持动态 shape（需预设输入尺寸） |
| 持续更新针对新架构的优化     | 使用不灵活，需提前构建 Engine   |
| 支持 INT8 量化进一步加速  | LoRA 切换需重新编译         |

### 2.6 编译优化对比总结

| 方案 | 加速比 | 显存 | 编译耗时 | 灵活性 | 适用场景 |
|------|--------|------|---------|--------|---------|
| torch.compile | <span class="metric-speedup">1.12×</span> | 11.24 GB | 5~16 min | 高 | 通用场景 |
| Stable Fast | <span class="metric-speedup">1.10×</span> | 12.12 GB | **18s** | 高 | 频繁切换模型 / LoRA |
| OneDiff | <span class="metric-speedup">1.16×</span> | **7.24 GB** | 55s | 中 | 显存敏感场景 |
| TensorRT | <span class="metric-speedup">**1.25×**</span> | 11.24 GB | ~20 min | **低** | 固定模型长期部署 |

> **选型建议**：
> - 追求最大加速且模型固定 → **TensorRT**
> - 需要频繁切换 LoRA / 模型 → **Stable Fast**（编译最快）
> - 显存紧张 → **OneDiff**（显存占用最低）
> - 不想引入额外依赖 → **torch.compile**（PyTorch 原生）

> **重要发现**：在 L20 上，编译优化的加速比（1.10~1.25×）普遍不及官方宣称的数据。这可能是因为 **L20 本身的 FP16 计算能力已经很强**，Eager 模式下 GPU 利用率已达 ~98%，留给编译优化的空间有限。在算力更弱的消费级 GPU 上，编译优化的收益可能更显著。

---

## 3. 模型组件级优化

与编译优化作用于底层 Kernel 不同，组件级优化直接修改 SDXL Pipeline 的各个组成部分。这些优化有的完全无损，有的会轻微影响图片质量，但它们可以 **叠加使用**，累积效果可观。

### 3.1 VAE FP16 Fix（无损）

#### 问题背景

SDXL 默认的 VAE 在 FP16 下会产生 NaN（数值溢出），因此官方代码在解码前强制将 VAE upcast 到 FP32：

```python
# diffusers 内部逻辑
pipe.upcast_vae()  # 强制 VAE 以 FP32 运行
```

这意味着即使整体 Pipeline 使用 FP16，VAE 解码阶段仍然在浪费 FP32 的计算和显存。

#### 解决方案

社区开发者 [madebyollin](https://huggingface.co/madebyollin) 创建了修补版 VAE，使其能够在 FP16 下稳定运行：

```python
from diffusers import AutoencoderKL

# 加载 FP16 兼容的 VAE
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16
).to("cuda")
```

| 方法 | 步长 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 |
|------|------|----------|---------------|-----------|--------|
| Base (FP16) | 30 | 5.5 | 5.45 | 11.24 | 1× |
| VAE FP16 Fix | 30 | 5.4 | 5.55 | **9.62** | <span class="metric-speedup">1.01×</span> |

**图片质量对比**：

| Base                            | VAE FP16                        |
| ------------------------------- | ------------------------------- |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013522687.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013526614.png) |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013531721.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013536984.png) |

**分析**：延时几乎不变（VAE 本身占比小），但显存从 11.24 GB 降至 **9.62 GB**（节省 14%）。图片质量完全无损。

> **结论**：VAE FP16 Fix 是零成本优化，**建议始终启用**。

### 3.2 Tiny VAE（基本无损）

#### 原理

SDXL 默认 VAE 拥有约 **5000 万** 参数。[TAESD](https://huggingface.co/madebyollin/taesdxl)（Tiny AutoEncoder for Stable Diffusion）是原始 VAE 的精简版本，仅 **100 万** 参数（缩减 50 倍），且原生支持 FP16。

```python
from diffusers import AutoencoderTiny

# 使用 Tiny VAE
vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesdxl",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16
).to("cuda")
```

| 方法          | 步长  | 延时（s） | 速度（step/s） | 显存（GB）   | 加速比       |
| ----------- | --- | ----- | ---------- | -------- | --------- |
| Base (FP16) | 30  | 5.5   | 5.45       | 11.24    | 1×        |
| Tiny VAE    | 30  | 5.1   | 5.88       | **7.57** | <span class="metric-speedup">**1.08×**</span> |

**图片质量对比**：

| Base                            | Tiny VAE                        |
| ------------------------------- | ------------------------------- |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013522687.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013617863.png) |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013531721.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013626462.png) |

**分析**：

- 延时减少 0.4s（VAE 解码加速显著）
- 显存降低 33%（11.24 → 7.57 GB）
- 图片质量基本无损，极少数情况可能出现细微的纹理差异

> **结论**：Tiny VAE 以微小的质量代价换取了可观的显存节省和轻微的速度提升，**推荐在显存敏感场景使用**。

### 3.3 禁用 CFG（轻微有损）

#### 原理

Classifier-Free Guidance（CFG）是扩散模型中控制生成质量的核心技术。它通过同时预测 **有条件** 和 **无条件** 两个噪声，然后按比例混合：

```
noise_pred = noise_uncond + guidance_scale × (noise_cond - noise_uncond)
```

这意味着每个去噪步骤中，UNet 需要执行 **两次** 前向传播（通过将 conditional 和 unconditional 拼接为 batch=2 实现），计算量直接翻倍。

**核心思考**：CFG 在早期步骤中至关重要（引导模型走向正确的语义方向），但在后期步骤中，模型已经"走上正轨"，是否还需要 CFG？

#### 实验：在后续步骤中关闭 CFG

```python
# 在 75% 的步骤后关闭 CFG
image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    # 自定义回调：在第 22 步后将 guidance_scale 设为 1（即关闭 CFG）
    callback_on_step_end=lambda pipe, step, ts, kwargs:
        {**kwargs, "guidance_scale": 1.0} if step > 22 else kwargs
)
```

| 方法                | 步长  | 延时（s） | 速度（step/s） | 显存（GB） | 加速比       |
| ----------------- | --- | ----- | ---------- | ------ | --------- |
| Base (FP16)       | 30  | 5.5   | 5.45       | 11.24  | 1×        |
| CFG 75%（后 25% 关闭） | 30  | 5.1   | 5.88       | 11.24  | <span class="metric-speedup">**1.08×**</span> |
| CFG 50%（后 50% 关闭） | 30  | 4.6   | 6.52       | 11.24  | <span class="metric-speedup">**1.19×**</span> |

**图片质量对比**：

| Base                            | CFG 75%                         | CFG 50%                         |
| ------------------------------- | ------------------------------- | ------------------------------- |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013801172.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013805293.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013809090.png) |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013815492.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013819058.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317013824655.png) |

**分析**：
- CFG 75%：图片与 Baseline 差异极小，大多数场景可接受
- CFG 50%：开始出现可感知的细节差异，但整体构图和语义不变

> **结论**：CFG 75% 是一个不错的折中点（加速 8%，质量几乎无损）。CFG 50% 加速更大（19%）但需要接受一定的质量波动。**注意：该优化对不同 prompt 的稳定性不一致**，部分场景可能导致质量显著下降，建议配合业务验证使用。

### 3.4 DeepCache（有损，高加速比）

#### 原理

[DeepCache](https://github.com/horseee/DeepCache)（CVPR 2024）的核心思想是：在 UNet 的去噪循环中，**相邻步骤的高层特征变化很小**。因此可以每隔若干步才真正执行一次完整的 UNet 前向传播，中间步骤复用（缓存）之前的高层特征。

```
正常推理（30步）:  UNet UNet UNet UNet UNet ... (30 次完整计算)
DeepCache(间隔2): UNet Cache UNet Cache UNet ... (15 次完整 + 15 次缓存)
DeepCache(间隔3): UNet Cache Cache UNet Cache Cache ... (10 次完整 + 20 次缓存)
```

两个关键参数：
- `cache_interval`：缓存更新频率，值越大加速越多但质量越低
- `cache_branch_id`：指定缓存 UNet 的哪个层级（0 = 最深层，值越大 = 越浅层）

```python
from DeepCache import DeepCacheSDHelper

helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(cache_interval=3, cache_branch_id=0)
helper.enable()

image = pipe(prompt, num_inference_steps=30).images[0]
```

| 方法 | cache_interval | cache_branch_id | 延时（s） | 速度（step/s） | 加速比 |
|------|---------------|----------------|----------|---------------|--------|
| Base (FP16) | - | - | 5.5 | 5.45 | 1× |
| DeepCache | 2 | 1 | 3.3 | 9.09 | <span class="metric-speedup">**1.66×**</span> |
| DeepCache | 2 | 0 | 3.2 | 9.37 | <span class="metric-speedup">**1.72×**</span> |
| DeepCache | 3 | 1 | 2.5 | 12.0 | <span class="metric-speedup">**2.20×**</span> |
| DeepCache | 3 | 0 | **2.4** | **12.5** | <span class="metric-speedup">**2.29×**</span> |

**图片质量对比**：

| Base                               | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014046425.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014049833.png) |
| ---------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| cache_interval=2 cache_branch_id=1 | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014053989.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014057125.png) |
| cache_interval=2 cache_branch_id=0 | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014101759.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014105091.png) |
| cache_interval=3 cache_branch_id=1 | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014109982.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014113254.png) |
| cache_interval=3 cache_branch_id=0 | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014118706.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014124744.png) |

**分析**：
- `cache_interval=2` 时加速约 1.7×，图片整体布局不变，细节有轻微差异
- `cache_interval=3` 时加速高达 **2.29×**，但图片细节变化更明显
- 图片的整体构图和语义一致性保持良好，主要差异在纹理细节

> **结论**：DeepCache 提供了所有组件优化中 **最高的加速比**。虽然有质量损失，但布局不变，非常适合 **Prompt 调试和快速预览** 场景。对于最终出图建议关闭或使用保守参数（`cache_interval=2`）。

### 3.5 蒸馏模型：SDXL-Lightning（有损，极高加速比）

#### 原理

[SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) 是字节跳动推出的蒸馏模型，采用 **渐进式对抗蒸馏（Progressive Adversarial Distillation）** 技术，使模型能在极少步数（2~8 步）内生成高质量图像。

这不是对原始模型的加速，而是用一个全新的蒸馏模型 **替代** 原始模型。

```python
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download

# 加载 SDXL-Lightning 的 LoRA 权重
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# 使用 4-step 版本
pipe.load_lora_weights(hf_hub_download(
    "ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors"
))
pipe.fuse_lora()

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)

image = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
```

| 方法 | 步长 | 延时（s） | 显存（GB） | 加速比 |
|------|------|----------|-----------|--------|
| Base (FP16) | 30 | 5.5 | 11.24 | 1× |
| SDXL-Lightning | 8 | 1.5 | 11.25 | <span class="metric-speedup">**3.67×**</span> |
| SDXL-Lightning | 4 | 1.1 | 11.25 | <span class="metric-speedup">**5×**</span> |
| SDXL-Lightning | 2 | 1.0 | 11.25 | <span class="metric-speedup">**5.5×**</span> |

**图片质量对比**：

| Base   | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014244886.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014247709.png) |
| ------ | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Step 2 | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014251093.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014254165.png) |
| Step 4 | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014257906.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014300949.png) |
| Step 8 | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014304439.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014307106.png) |


**分析**：
- 加速比极高（4 步版本可达 5×）
- 但生成风格与原始 SDXL **存在明显差异**，倾向于更"插画风"
- 2 步版本质量明显下降，4 步和 8 步版本质量较为可接受

> **结论**：蒸馏模型提供了极致的速度，但以 **风格一致性** 为代价。适用于对风格要求宽松、追求极速的场景（如实时预览、批量草图生成）。

### 3.6 显存优化（牺牲速度换显存）

以下两种优化专注于降低显存占用，适用于显存紧张的场景。它们 **会增加推理延时**，仅在硬件条件受限时使用。

#### 模型 CPU 卸载

在推理过程中，按需将当前阶段的模型移入 GPU，其余模型暂存于 CPU 内存：

```python
# 加载顺序：text_encoder → text_encoder_2 → image_encoder → unet → vae
pipe.enable_model_cpu_offload()
```

| 方法 | 步长 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 |
|------|------|----------|---------------|-----------|--------|
| Base (FP16) | 30 | 5.5 | 5.45 | 11.24 | 1× |
| CPU Offload | 30 | 6.7 | 4.48 | **7.05** | <span class="metric-slowdown">0.82×</span> |

延时增加 22%，显存降低 37%。模型搬运（CPU ↔ GPU）引入了额外的数据传输开销。

#### 批处理加载

对一个 batch 的数据，**顺序加载** Pipeline 中的各个组件并执行推理，每完成一个组件就将其卸载。本质上是 CPU Offload 的 batch 化变体，用批次处理来掩盖模型加载延时。

| 方法 | 步长 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 |
|------|------|----------|---------------|-----------|--------|
| Base (FP16) | 30 | 5.5 | 5.45 | 11.24 | 1× |
| Batch Process (B=4) | 30 | 6.0 | 5.0 | **5.76** | <span class="metric-slowdown">0.91×</span> |

> **注**：此处延时为 Batch=4 时的平均延时。

> **结论**：显存优化方案以牺牲速度为代价。在 L20（48GB）上通常不需要使用。仅当部署在消费级 GPU（如 RTX 3060 12GB）时才考虑。

### 3.7 质量优化（不加速，提升生成效果）

以下两种方法专注于提升图片质量，不影响推理速度，作为可选项供参考。

#### FreeU：平衡 U-Net 特征

[FreeU](https://github.com/ChenyangSi/FreeU) 通过调整 UNet 跳跃连接（Skip Connection）和主干特征图的贡献权重，抑制不自然的高频细节，使生成结果更真实。

```python
# 启用 FreeU
pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)
```

| 方法 | 步长 | 延时（s） | 显存（GB） |
|------|------|----------|-----------|
| Base (FP16) | 30 | 5.5 | 11.24 |
| FreeU | 30 | 5.5 | 11.24 |

速度和显存完全不变，仅影响生成风格。


| Base                            | FreeU                           |
| ------------------------------- | ------------------------------- |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014631784.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014634958.png) |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014640207.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014642847.png) |

#### Refiner：专用细化模型

SDXL 提供了一个专用的 Refiner 模型，通过 **Ensemble of Expert Denoisers** 方式使用：Base 模型执行前面若干步，Refiner 模型执行后面若干步。

```python
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Base 20步 + Refiner 10步
image = base(prompt, num_inference_steps=20, denoising_end=0.8, output_type="latent").images
image = refiner(prompt, image=image, num_inference_steps=10, denoising_start=0.8).images[0]
```

| 方法 | 步长 | 延时（s） | 显存（GB） |
|------|------|----------|-----------|
| Base (FP16) | 30 | 5.5 | 11.24 |
| Base + Refiner | 20+10 | 5.6 | **18.3** |

| Base                            | Refiner                         |
| ------------------------------- | ------------------------------- |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014631784.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014714123.png) |
| ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014640207.png) | ![](/images/SDXL推理优化实战(二)：全面优化实践指南/file-20260317014721004.png) |

**分析**：延时基本不变，但显存大幅增加（需要同时加载两个大模型）。图片在细节层面有提升。

---

## 4. 组件优化全景对比

将所有单项优化汇总为一张对比表：

| 类别     | 方法              | 加速比       | 显存（GB）   | 图片质量     | 使用建议     | 备注             |
| ------ | --------------- | --------- | -------- | -------- | -------- | -------------- |
| **精度** | FP16            | —         | 11.24    | 无损       | **始终使用** | 作为 Baseline    |
| **编译** | torch.compile   | <span class="metric-speedup">1.12×</span>     | 11.24    | 无损       | 可以使用     | 编译耗时不稳定        |
|        | Stable Fast     | <span class="metric-speedup">1.10×</span>     | 12.12    | 无损       | 可以使用     | **编译最快（18s）**  |
|        | OneDiff         | <span class="metric-speedup">1.16×</span>     | **7.24** | 无损       | 可以使用     | **显存最低**       |
|        | TensorRT        | <span class="metric-speedup">**1.25×**</span> | 11.24    | 无损       | 考虑使用     | 不灵活，需预构建引擎     |
| **组件** | VAE FP16 Fix    | <span class="metric-speedup">1.01×</span>     | **9.62** | 无损       | **始终使用** | 零成本优化          |
|        | Tiny VAE        | <span class="metric-speedup">1.08×</span>     | **7.57** | 基本无损     | 推荐使用     |                |
|        | 禁用 CFG 75%      | <span class="metric-speedup">1.08×</span>     | 11.24    | 细微差别     | 考虑使用     | 稳定性因 prompt 而异 |
|        | DeepCache (i=3) | <span class="metric-speedup">**2.29×**</span> | 11.59    | **有损**   | 可选       | 适合 Prompt 调试   |
|        | 蒸馏 (Lightning)  | <span class="metric-speedup">**5×**</span>    | 11.25    | **风格变化** | 可选       | 适合快速预览         |
| **显存** | CPU Offload     | <span class="metric-slowdown">0.82×</span>     | **7.05** | 无损       | 不推荐      | 显存不够时考虑        |
|        | Batch Process   | <span class="metric-slowdown">0.91×</span>     | **5.76** | 无损       | 不推荐      | 显存不够时考虑        |
| **质量** | FreeU           | 1×        | 11.24    | 风格变化     | 可选       | 不影响速度          |
|        | Refiner         | <span class="metric-slowdown">0.98×</span>     | 18.3     | 细节增强     | 可选       | 需额外显存          |

---

## 5. 小结与展望

### 本篇核心收获

1. **精度优化是收益最大的单项优化**：FP32 → FP16 带来 ~3× 加速，是后续所有优化的基础
2. **编译优化无损但有限**：在 L20 上加速 1.10~1.25×，不同框架各有侧重（速度 vs 灵活性 vs 显存）
3. **组件优化可叠加**：VAE FP16 Fix + Tiny VAE + 部分禁用 CFG 的组合可以在几乎不损失质量的前提下获得可观加速
4. **DeepCache 和蒸馏是"空间换质量"的极端方案**：加速比极高但有明显的质量代价
5. **没有银弹**：每种优化都有其 trade-off，需要根据业务场景（质量要求、延时预算、显存限制）做取舍

### 关键 Trade-off 决策树

```
你的场景是什么？
│
├── 追求最大质量，延时可接受
│   └── FP16 + VAE FP16 Fix + (可选) Refiner
│
├── 追求低延时，质量基本无损
│   └── FP16 + TensorRT/OneDiff + Tiny VAE + VAE FP16 Fix
│
├── 追求极致延时，接受轻微质量损失
│   └── FP16 + OneDiff + Tiny VAE + CFG 75%
│
├── Prompt 调试 / 快速预览
│   └── FP16 + DeepCache (interval=3)
│
└── 实时交互 / 极速生成
    └── SDXL-Lightning (4-step)
```

### 下篇预告

在第三篇（终篇）中，我们将：

1. **混合组合优化**：将编译优化与组件优化叠加，测试 TensorRT + Tiny VAE、OneDiff + Tiny VAE + CFG 等组合的实际效果
2. **吞吐优化实战**：深入对比 Batch 推理、多实例部署、CUDA MPS 三种吞吐提升策略
3. **最佳工程部署实践**：给出不同并发场景下的最优部署方案和配置建议

敬请期待。

---

**参考资料**：
1. [Ultimate Guide to Optimizing Stable Diffusion XL - Felix Sanz](https://www.felixsanz.dev/articles/ultimate-guide-to-optimizing-stable-diffusion-xl)
2. [stable-fast - GitHub](https://github.com/chengzeyi/stable-fast)
3. [OneDiff - GitHub](https://github.com/siliconflow/onediff)
4. [DeepCache: Accelerating Diffusion Models for Free (CVPR 2024)](https://github.com/horseee/DeepCache)
5. [SDXL-Lightning - ByteDance](https://huggingface.co/ByteDance/SDXL-Lightning)
6. [NVIDIA TensorRT Documentation](https://developer.nvidia.com/tensorrt)
7. [HuggingFace Diffusers - Optimization Guide](https://huggingface.co/docs/diffusers/optimization/opt_overview)
