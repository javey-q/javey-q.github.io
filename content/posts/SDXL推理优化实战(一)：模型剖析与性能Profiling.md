+++
date = '2024-09-01'
draft = false
title = 'SDXL 推理极限优化实战（一）：模型剖析与性能 Profiling'
tags = ['Stable Diffusion', '推理优化', 'TensorRT']
categories = ['Diffusion', 'inference']

+++

> **系列导读**：本系列共三篇文章，渐进式地探讨 Stable Diffusion XL（SDXL）模型的推理优化。第一篇聚焦于问题背景与性能剖析，第二篇展开全面的优化实践，第三篇进行混合组合优化与吞吐工程部署。

---

## 1. 为什么要优化 SDXL 推理？

### 1.1 SDXL 的工程地位

Stable Diffusion XL（SDXL）是 Stability AI 于 2023 年发布的旗舰级文生图模型，相较于 SD 1.5，它在图像质量、分辨率（原生支持 1024×1024）和语义理解能力上有了质的飞跃。SDXL 已成为开源社区中应用最广泛的高分辨率生成模型之一，广泛用于创意设计、电商素材生成、游戏概念图制作等场景。

然而，更强的模型能力伴随着更高的计算开销。SDXL 的参数量达到约 **3.5B**（Base UNet 2.6B + Text Encoders 等），相较于 SD 1.5 的 ~0.9B，推理成本大幅增加：

| 指标 | SD 1.5 | SDXL |
|------|--------|------|
| UNet 参数量 | ~860M | ~2.6B |
| 原生分辨率 | 512×512 | 1024×1024 |
| Text Encoder | CLIP ViT-L/14 | CLIP ViT-L/14 + OpenCLIP ViT-bigG |
| FP32 显存占用 | ~4 GB | ~18 GB |
| FP16 单次推理延时（30步） | ~1.5s | ~5.5s |

### 1.2 优化的业务价值

在生产环境中，推理性能直接决定了：

- **用户体验**：单张图片从 5.5s 降至 2.8s，交互延迟感知有本质差异
- **服务成本**：GPU 是最昂贵的资源，推理加速 = 同等算力下服务更多用户
- **吞吐能力**：高并发场景下，从 0.25 image/s 提升到 0.37+ image/s 意味着单卡产能提升近 50%

### 1.3 实验环境

本系列所有实验基于以下硬件与软件环境：

**硬件：NVIDIA L20**

| 规格 | 参数 |
|------|------|
| GPU 架构 | NVIDIA Ada Lovelace |
| CUDA 核心 | 10240 |
| 频率 | 高达 2.2 GHz |
| 显存 | 48GB GDDR6 |
| 显存位宽 | 384 位 |
| 显存带宽 | 864 GB/s |
| FP32 算力 | 90 TFLOPS |
| FP16 Tensor Core 算力 | ~181 TFLOPS |

**软件环境**：
- CUDA 12.x
- PyTorch 2.x
- Diffusers (HuggingFace)
- NVIDIA Nsight Systems / Nsight Compute

### 1.4 优化评估三要素

在评估推理优化效果时，我们关注三个核心指标：

1. **速度（Latency）**：单次推理端到端延时，直接影响用户体验
2. **资源占用（Memory）**：主要关注显存占用，决定了并行度上限和可部署的 Batch Size
3. **图片质量（Quality）**：优化不能以牺牲可感知的图片质量为代价，需要在速度和质量间取得平衡

---

## 2. SDXL 推理 Pipeline 深度剖析

### 2.1 端到端推理流程

SDXL 的推理是一个多模型协作的 Pipeline，而非单一模型的前向传播。理解这一点是性能优化的前提。

```
┌─────────────┐    ┌─────────────────────────────────────┐    ┌───────────┐
│ Text Encode │───▶│         UNet Denoising Loop          │───▶│ VAE Decode│
│ (CLIP×2)    │    │  (Iterative: 20~50 steps)           │    │           │
└─────────────┘    └─────────────────────────────────────┘    └───────────┘
     ~0.1s                    ~5.0s (主要瓶颈)                    ~0.3s
```

完整的推理流程如下：

**Step 1：文本编码（Text Encoding）**

SDXL 使用双 Text Encoder 架构：
- **CLIP ViT-L/14**：输出 768 维文本嵌入
- **OpenCLIP ViT-bigG/14**：输出 1280 维文本嵌入

两个编码器的输出被拼接为 2048 维向量，作为 UNet 的条件输入。此阶段通常只执行一次，耗时占比极小（<2%）。

**Step 2：UNet 去噪循环（Denoising Loop）—— 计算核心**

UNet 是整个 Pipeline 的计算瓶颈，占据了 **90%+ 的推理时间**。在每个去噪步骤中，UNet 接收当前的 noisy latent 和文本条件，预测噪声残差。

关键参数：
- 输入 latent shape：`(B, 4, 128, 128)` — 对应 1024×1024 图像
- 模型参数量：~2.6B
- 每步包含大量 Self-Attention、Cross-Attention 和 Conv 运算

如果启用了 **CFG（Classifier-Free Guidance）**，每个去噪步骤实际上需要执行 **两次** UNet 前向传播（一次有条件、一次无条件），计算量翻倍。

**Step 3：VAE 解码（Latent → Pixel）**

最后，VAE Decoder 将去噪后的 latent（4 通道，128×128）解码为 RGB 图像（3 通道，1024×1024）。

需要注意的是，SDXL 默认的 VAE **不支持 FP16 推理**，会自动 upcast 到 FP32，这是一个可优化的点。

### 2.2 UNet 架构细节

SDXL 的 UNet 采用经典的 Encoder-Decoder + Skip Connection 结构，但在内部组件上做了重要改进：

```
UNet 结构概览（简化）:

Input (4, 128, 128)
    │
    ▼
┌───────────────────────────┐
│  DownBlock 0 (320ch)      │ ──────────────────────────────┐
│  [ResNet + Transformer]   │                                │
├───────────────────────────┤                                │
│  DownBlock 1 (640ch)      │ ─────────────────────┐        │
│  [ResNet + Transformer]   │                       │        │
├───────────────────────────┤                       │        │
│  DownBlock 2 (1280ch)     │ ────────────┐        │        │
│  [ResNet + Transformer]   │              │        │        │
├───────────────────────────┤              │        │        │
│  MidBlock (1280ch)        │              │        │        │
│  [ResNet + Transformer]   │              │        │        │
├───────────────────────────┤              │        │        │
│  UpBlock 0 (1280ch)       │ ◄────────────┘        │        │
│  [ResNet + Transformer]   │  (skip connection)    │        │
├───────────────────────────┤                       │        │
│  UpBlock 1 (640ch)        │ ◄─────────────────────┘        │
│  [ResNet + Transformer]   │  (skip connection)             │
├───────────────────────────┤                                │
│  UpBlock 2 (320ch)        │ ◄──────────────────────────────┘
│  [ResNet + Transformer]   │  (skip connection)
└───────────────────────────┘
    │
    ▼
Output (4, 128, 128)
```

每个 Transformer Block 内部包含：
- **Self-Attention**：latent 特征间的空间注意力，复杂度 O(n²)
- **Cross-Attention**：latent 特征与文本嵌入间的交叉注意力
- **FFN（Feed-Forward Network）**：通道维度上的非线性变换

### 2.3 计算量分布分析

通过分析各组件的计算量占比，可以明确优化的优先级：

| 组件 | 每步执行次数 | 占比（估算） | 备注 |
|------|-------------|-------------|------|
| UNet Attention (Self + Cross) | 1×（或 2× w/ CFG） | ~40% | 内存带宽密集 |
| UNet Conv + ResNet | 1×（或 2× w/ CFG） | ~35% | 计算密集 |
| UNet FFN | 1×（或 2× w/ CFG） | ~15% | 计算密集 |
| Text Encoder | 仅 1 次 | <2% | 可忽略 |
| VAE Decode | 仅 1 次 | ~5-8% | FP32 时较慢 |
| Scheduler / 其他 CPU 开销 | 每步 | <1% | CPU-GPU 同步开销 |

从上表可以看出：**UNet 去噪循环是绝对瓶颈，占总推理时间的 90% 以上**。优化 UNet 的执行效率是提升整体性能的关键。

---

## 3. 性能 Profiling 方法论

在开始优化之前，必须先通过系统化的 Profiling 找到真正的性能瓶颈。盲目优化不仅效率低下，还可能适得其反。本节介绍如何使用 NVIDIA 提供的两款核心分析工具。

### 3.1 Nsight Systems：系统级时间线分析

**Nsight Systems** 是系统级的性能分析工具，适用于观察 **宏观层面** 的执行行为，包括：

- CPU 与 GPU 的执行时间线
- CUDA Kernel 的启动和执行时序
- 内存拷贝操作（HtoD / DtoH）
- CPU-GPU 同步开销
- 各阶段（Text Encode / UNet / VAE）的耗时分布

#### 使用方法

```bash
# 基本用法：对推理脚本进行 Profiling
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=sdxl_profile \
    --force-overwrite=true \
    python sdxl_inference.py
```

为了获得更有意义的 Profiling 结果，建议在代码中添加 **NVTX 标记** 来标注各阶段：

```python
import torch
import nvtx

# 在推理代码中标注关键阶段
with nvtx.annotate("Text Encoding", color="blue"):
    prompt_embeds = pipe.encode_prompt(prompt)

with nvtx.annotate("UNet Denoising Loop", color="red"):
    for i, t in enumerate(timesteps):
        with nvtx.annotate(f"Step {i}", color="orange"):
            noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

with nvtx.annotate("VAE Decode", color="green"):
    image = vae.decode(latents)
```

#### 关键观察点

通过 Nsight Systems 生成的时间线视图，我们重点关注以下几点：

**1. GPU 利用率与空闲时间（GPU Idle Gaps）**

在 SDXL 推理的时间线中，典型的观察是：
- 每个 UNet step 之间存在微小的 **CPU-GPU 同步间隙**
- Scheduler 的步骤计算在 CPU 上执行，会引入少量空闲
- 这些间隙单独看很小（~μs 级），但累积 20~50 步后变得可观

**2. Kernel 启动开销（Kernel Launch Overhead）**

SDXL 的 UNet 在 PyTorch eager 模式下，一次前向传播会触发 **数百到上千个小 CUDA Kernel**。每个 Kernel 的启动有固定开销（~5-10μs），当 Kernel 数量极多时，启动开销不可忽略。

因此后续可考虑采用 `torch.compile` 和 TensorRT 等编译优化方式，来减少 Kernel 数量。

**3. 内存拷贝操作**

如果在时间线中发现频繁的 HtoD（Host to Device）或 DtoH 拷贝，说明存在不必要的 CPU-GPU 数据搬运。在正确的实现中，推理过程应基本保持数据在 GPU 上。

**4. CFG 导致的计算翻倍**

启用 CFG 时，Nsight Systems 时间线会清晰地展示每个 step 内 UNet 被调用了两次（batch 维度加倍），这为"部分禁用 CFG"的优化方案提供了直观的依据。

#### Nsight Systems 分析示例输出

```
SDXL FP16 Inference Timeline (20 steps, 1024×1024)
=====================================================
Phase            | Duration | GPU Active | Kernel Count
-----------------+----------+------------+-------------
Text Encoding    |   82 ms  |    78 ms   |     126
UNet Step (avg)  |  185 ms  |   181 ms   |     847
 ├─ Self-Attn    |   72 ms  |    71 ms   |     198
 ├─ Cross-Attn   |   43 ms  |    42 ms   |     132
 ├─ Conv/ResNet  |   58 ms  |    57 ms   |     389
 └─ Other        |   12 ms  |    11 ms   |     128
UNet Total (×20) | 3700 ms  |  3620 ms   |   16940
VAE Decode       |  280 ms  |   275 ms   |     312
-----------------+----------+------------+-------------
Total            | 4062 ms  |  3973 ms   |   17378
GPU Utilization  |          |   97.8%    |
```

> **Insight**：GPU 利用率已经很高（97.8%），进一步优化的方向不在于"让 GPU 更忙"，而在于"让每个 Kernel 跑得更快"或"减少 Kernel 数量"。

### 3.2 Nsight Compute：Kernel 级深度分析

**Nsight Compute** 是 Kernel 级别的性能分析工具，用于深入分析 **单个 CUDA Kernel 的执行效率**，帮助我们理解瓶颈的微观本质。

#### 使用方法

```bash
# 分析特定范围的 Kernel（避免全量分析耗时过长）
ncu --set full \
    --kernel-name "regex:.*attention.*" \
    --launch-skip 100 --launch-count 20 \
    --output sdxl_kernel_profile \
    python sdxl_inference.py
```

> **注意**：Nsight Compute 的 Profiling 开销极大（单个 Kernel 可能需要重放数十次），建议只分析关键的 Kernel，而非全量扫描。

#### 关键分析维度

**1. Roofline 分析：计算密集 vs 访存密集**

Roofline 模型是判断 Kernel 性能瓶颈最直观的工具：

```
         Peak FLOPS
            │
            │         ╱ Roofline
            │       ╱
Performance │     ╱
(FLOPS)     │   ╱  ● Attention Kernel（访存密集区）
            │ ╱
            │╱         ● Conv Kernel（计算密集区）
            └──────────────────────────
              Arithmetic Intensity (FLOP/Byte)
```

对于 SDXL 的关键 Kernel：

| Kernel 类型 | 算术强度 | 瓶颈类型 | 优化方向 |
|-------------|---------|---------|---------|
| Self-Attention | 低 | **访存密集** | Flash Attention、减少显存访问 |
| Cross-Attention | 低 | **访存密集** | 融合 Kernel、xformers |
| Conv 3×3 | 高 | **计算密集** | cuDNN 融合、低精度计算 |
| GroupNorm | 极低 | **访存密集** | 算子融合（与后续激活融合） |
| GELU/SiLU | 极低 | **访存密集** | 算子融合 |

**2. Occupancy 分析：SM 利用率**

Occupancy 衡量每个 SM（Streaming Multiprocessor）上活跃 warp 数与理论最大值的比值。低 Occupancy 意味着 SM 资源未被充分利用。

SDXL UNet 中常见的 Occupancy 限制因素：

- **寄存器压力**：Attention Kernel 使用大量寄存器，限制了每个 SM 可同时调度的 warp 数
- **共享内存占用**：部分 Kernel 使用较多共享内存，同样限制并发度

```
Kernel: fused_attention_forward_fp16
  Theoretical Occupancy:  50.0%
  Achieved Occupancy:     46.2%
  Limiting Factor:        Registers (128 per thread)
  Shared Memory:          48 KB / 100 KB available
```

> **Insight**：SDXL 推理的瓶颈在于 **寄存器和共享内存的限制导致并发度不够**。这一发现对后续的多实例部署策略（MPS）具有重要指导意义。

**3. Memory Throughput 分析**

```
Kernel: attention_softmax_fp16
  Memory Throughput:
    L1 Hit Rate:     82.3%
    L2 Hit Rate:     64.7%
    DRAM Bandwidth:  78.2% of peak (675 GB/s)

  Warp Stall Reasons:
    Memory Dependency:  43.2%  ← 主要瓶颈
    Execution:          28.7%
    Other:              28.1%
```

对于访存密集型 Kernel，**Warp Stall** 的主要原因是内存依赖（等待数据从显存加载）。优化方向包括：
- 使用 Flash Attention 减少 HBM 访问
- 算子融合减少中间结果的 store/load
- 使用低精度（FP16/BF16）减少每次访问的数据量

### 3.3 Profiling 小结与优化路线图

综合 Nsight Systems 和 Nsight Compute 的分析结果，我们可以得出以下结论：

| 发现 | 影响 | 对应优化方向 |
|------|------|-------------|
| UNet 占 90%+ 推理时间 | 高 | 优先优化 UNet |
| GPU 利用率已较高（~98%） | 中 | 重点不在"填满GPU"，而在"每个op更快" |
| Attention 为访存密集型 | 高 | Flash Attention / xformers |
| 大量小 Kernel 启动开销 | 中 | torch.compile / CUDA Graphs / TensorRT |
| CFG 导致计算翻倍 | 高 | 部分禁用 CFG |
| VAE 强制 FP32 | 低 | VAE FP16 Fix |
| 寄存器/共享内存限制 Occupancy | 中 | 影响多实例部署策略 |

基于以上分析，我们规划了如下优化路线图（将在后续两篇文章中展开）：

```
优化路线图
═══════════════════════════════════════════════════════

Layer 1: 基础优化（无损）
├── 数值精度：FP32 → FP16/BF16
├── VAE FP16 Fix
└── 预期收益：~3x 加速

Layer 2: 编译与框架加速（无损）
├── torch.compile (max-autotune)
├── Stable Fast / OneDiff
├── TensorRT
└── 预期收益：额外 1.1x~1.25x

Layer 3: 组件级优化（可能轻微有损）
├── Tiny VAE
├── 部分禁用 CFG
├── DeepCache
└── 预期收益：额外 1.1x~2.3x

Layer 4: 吞吐工程优化
├── Batch 策略
├── 多实例部署
├── CUDA MPS
└── 目标：最大化 image/s
```

---

## 4. Baseline 建立

在开始优化之前，我们首先建立一个合理的 Baseline 并记录各项指标。

### 4.1 FP32 Baseline

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
```

| 配置 | 步长 | 分辨率 | 延时（s） | 速度（step/s） | 显存（GB） |
|------|------|--------|----------|---------------|-----------|
| FP32 | 30 | 1024 | 16.3 | 1.84 | 18.08 |
| FP32 | 50 | 1024 | 26.9 | 1.85 | 18.07 |

### 4.2 FP16 Baseline（后续对比基准）

```python
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
```

| 配置 | 步长 | 分辨率 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 |
|------|------|--------|----------|---------------|-----------|--------|
| FP32 | 30 | 1024 | 16.3 | 1.84 | 18.08 | 1× |
| FP16 | 30 | 1024 | 5.5 | 5.45 | 11.24 | <span class="metric-speedup">**2.96×**</span> |
| FP32 | 50 | 1024 | 26.9 | 1.85 | 18.07 | 1× |
| FP16 | 50 | 1024 | 8.7 | 5.74 | 11.24 | <span class="metric-speedup">**3.09×**</span> |

FP16 带来了接近 **3 倍** 的加速和 **38%** 的显存节省，且图片质量几乎无损。这是最基本也是收益最高的优化。

> **后续所有实验均以 FP16 作为 Baseline**（30 步，1024×1024，延时 5.5s，显存 11.24GB）。

### 4.3 其他精度格式对比

| 精度 | 延时（s） | 速度（step/s） | 显存（GB） | 加速比 | 建议 |
|------|----------|---------------|-----------|--------|------|
| FP32 | 16.3 | 1.84 | 18.08 | 1× | 仅用于调试 |
| TF32 | 12.4 | 2.41 | 18.08 | <span class="metric-speedup">1.31×</span> | 不推荐（加速有限） |
| FP16 | 5.5 | 5.45 | 11.24 | <span class="metric-speedup">2.96×</span> | **推荐作为默认** |
| BF16 | 5.4 | 5.55 | 9.62 | <span class="metric-speedup">3.01×</span> | 可选（显存更低） |

> **TF32** 的加速效果远不如 FP16/BF16，因为它仅改变了 Tensor Core 的计算精度，没有减少内存带宽消耗。在 L20 上不建议使用。
>
> **BF16** 与 FP16 速度接近，但显存更低（9.62 vs 11.24 GB），适合显存紧张的场景。

---

## 5. 小结与展望

### 本篇核心收获

1. **SDXL 推理是多阶段 Pipeline**：Text Encoding → UNet 去噪循环 → VAE 解码，其中 UNet 占 90%+ 的计算时间
2. **UNet 内部以 Attention 和 Conv 为主**：Attention 是访存密集型，Conv 是计算密集型，优化策略不同
3. **Profiling 先于优化**：通过 Nsight Systems 看宏观时间线，通过 Nsight Compute 看微观 Kernel 效率
4. **FP16 是最基本的优化**：简单切换精度即可获得 ~3× 加速，是所有后续优化的基础
5. **Occupancy 受限于寄存器和共享内存**：这一发现将影响第三篇中的多实例部署策略

### 下篇预告

在第二篇文章中，我们将在 FP16 Baseline 上逐一实践以下优化：

- **编译与框架加速**：torch.compile、Stable Fast、OneDiff、TensorRT
- **组件级优化**：VAE FP16 Fix、Tiny VAE、禁用 CFG、DeepCache
- **蒸馏方案**：SDXL-Lightning
- **显存优化**：模型 CPU 卸载、批处理加载

每种方案都将给出完整的代码示例、实测数据和适用场景分析，敬请期待。

---

**参考资料**：
1. [Stability AI - SDXL Technical Report](https://arxiv.org/abs/2307.01952)
2. [Ultimate Guide to Optimizing Stable Diffusion XL](https://www.felixsanz.dev/articles/ultimate-guide-to-optimizing-stable-diffusion-xl)
3. [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
4. [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
5. [HuggingFace Diffusers - Optimization Guide](https://huggingface.co/docs/diffusers/optimization/opt_overview)
