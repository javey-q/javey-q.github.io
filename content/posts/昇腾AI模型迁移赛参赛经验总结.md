+++
date = '2025-01-01'
draft = false
title = '昇腾 AI 模型迁移赛参赛经验总结'
tags = ['MindSpore', '昇腾', '模型迁移', '推理优化', 'FlashAttention']
categories = ['inference', '竞赛']

+++

## 一、背景与结果概述

本次模型迁移赛的任务，是将 BERT、CLIP 等 Transformer 类模型迁移到昇腾平台，并在保证精度的前提下尽可能降低单次推理耗时。经过完整的优化迭代，BERT 单次推理从约 48ms 降至 5-6ms，CLIP 从 80-90ms 降至约 10ms，量级上均为 8 倍左右的提升。

这一提升并非来自某个单点技巧，而是三个层次优化叠加的结果：

| 层次 | 优化内容 | 作用机理 |
| --- | --- | --- |
| 代码写法层 | Tensor 索引、切片、permute、broadcast 的等价重写 | 减少 Host 侧算子下发次数与冗余小算子 |
| 算子与结构层 | QKV 融合、FlashAttention、mask 预构造、局部 fp16 | 提高单次下发的计算密度，降低访存量 |
| 执行模式层 | 静态图与整图静态化 | 消除 Python/Host 侧逐算子调度，转为整图连续执行 |

理解这三层的边界，是整个比赛中最关键的认知。初看 BERT、CLIP 这类模型时，注意力很容易集中在 attention、GELU、LayerNorm 等局部算子上，认为优化就是"把慢算子换成快算子"。但实测数据表明：在动态图推理下，真正的瓶颈大量分布在 Host 侧调度、单算子下发、动态 shape 处理和 Python 分支上，Device 侧计算反而不是第一约束。因此，优化的主线可以概括为一句话——**先稳定测量，再定位热点，然后把动态图、小算子、动态 shape 和低效写法，系统性地改造成静态 shape、大算子、图模式与硬件友好的执行路径。**

## 二、性能台阶图示

为避免 BERT 与 CLIP 的阶段标注相互挤压，下面将二者分开展示。图中区间值采用代表值绘制，重点表达优化路线中的性能下降台阶，而非将不同实验分支做严格线性叠加。

![BERT 模型迁移优化性能台阶](/images/昇腾AI模型迁移赛参赛经验总结/BERT性能台阶图.svg)

![CLIP 模型迁移优化性能台阶](/images/昇腾AI模型迁移赛参赛经验总结/CLIP性能台阶图.svg)

## 三、先建立可靠基线，而不是马上改代码

优化初期最容易犯的错误，是看到某个慢算子就直接动手改。但如果测量环境本身不稳定，优化方向很容易被噪声误导。

首先要处理的是运行稳定性。动态图推理中，Python 侧与 Host 侧调度占比很高，一旦 CPU 发生跨核调度，缓存失效与上下文切换会让耗时明显波动。绑核在这里价值显著：BERT 原始耗时约 48ms，绑核后可降到 41ms 甚至 38ms；CLIP 也从 80-90ms 降到约 76ms。这部分收益本身不是最终核心，但它让后续 profile 与版本对比变得可信。

在此基础上，任何性能优化开始前都应先固定以下条件：

- 固定输入 shape、batch size、seq length 与图片尺寸；
- 固定模型为 eval 模式；
- 固定绑核策略；
- 做足 warmup，避免把图编译或首次运行开销计入推理时间；
- 记录 median、min、p90，而不是只看单次结果；
- 每次只打开一个优化开关，保证收益可归因。

基线不稳，后续优化就会退化为"碰运气"。

## 四、BERT 的优化路线

BERT 是 Encoder-only 结构，层次规整、输入长度固定，非常适合静态图与 attention 大算子优化。

### 1. QKV 融合：把三个小 GEMM 合成一个大 GEMM

self-attention 中的 Query、Key、Value 原本各走一次 Linear 投影。三者输入完全相同（同一份 hidden state），只是权重不同，因此可以把三组权重与 bias 沿输出维拼接，做一次大的 QKV Linear，再 split 出 Q、K、V。

其收益来自两个互相独立的机理：

- **Host 侧**：算子下发次数从 3 次降为 1 次。在动态图下，每次下发都伴随 Python 调用、参数打包与流同步开销，这部分开销与矩阵规模无关，是纯粹的固定成本。
- **Device 侧**：NPU 的 Cube 单元对大矩阵才能跑满算力。三个 `[B*S, H] × [H, H]` 的小 GEMM 合成一个 `[B*S, H] × [H, 3H]` 的大 GEMM 后，权重只需一次搬入，计算访存比（arithmetic intensity）提升，更容易从访存受限转向计算受限。

BERT 中该项把耗时从绑核后的约 38-41ms 降到约 37.8ms。单看收益不大，但它统一了 attention 的数据布局，是后续 FlashAttention 与整图静态化的结构基础。

需要注意的是，QKV 融合有其前提：**三者的输入必须是同一个张量**。Encoder self-attention 满足这一条件；而 Decoder 的 cross-attention（Q 来自 Decoder，K/V 来自 Encoder 输出）或带 KV Cache 的路径（K/V 需要跨步复用），强行融合会破坏缓存复用与生成逻辑。

### 2. FlashAttention：消除 O(N²) 中间矩阵的访存

传统 attention 会展开成 QK^T、scale、mask、softmax、dropout、乘 V 等多个步骤。问题的核心不在于计算量——而在于中间会实体化一个 `[B, H, S, S]` 的 attention score/probability 矩阵，并在 HBM 上完整地写一次、读一次。当 S 较大时，这部分访存量随 S² 增长，attention 因此是典型的**访存受限**算子，而非计算受限。

FlashAttention 的本质是**分块计算 + online softmax**：将 Q、K、V 按块载入片上缓存，在片上完成 QK^T、softmax 与乘 V 的全部运算，通过在线更新最大值与归一化因子来保证 softmax 数学等价，从而使巨大的中间矩阵**从不落回 HBM**。访存复杂度由 O(S²) 降至 O(S)，同时把六七个小算子的下发合并为一次大算子调用，Host 侧开销一并消除。

接入时有三个关键点：

- **布局对齐**：需按大算子要求组织为 BNSD（batch、head、seq、dim）；
- **mask 规格**：mask 的 shape 与 dtype 必须符合接口约定；
- **dtype 策略**：通常让 Q/K/V 在 FlashAttention 内部走 fp16 以利用 Cube 单元，输出再转回 fp32。

这一步不是简单替换函数名。QKV 的 reshape、permute、mask、scale 与 dtype 若没有一并处理，性能和精度都会出问题。

### 3. `mint` 替换低效写法：削减动态图的下发开销

MindSpore 迁移中一个很实用的经验是：许多看似普通的 Tensor 写法，在动态图下会触发不理想的执行路径。以 `x[:, :seq_len]` 这类切片为例，它需要在 Host 侧解析 Python slice 对象、推导输出 shape、再合成对应的 Device 算子；高级索引（fancy indexing）更是可能被拆解为多个 gather/scatter 小算子。这些开销**不随张量大小变化**，但在 Transformer 中会在每一层、每次推理中反复出现，累计效应显著。

`mindspore.mint` 提供了与之语义等价、但下发路径更短、更贴近底层原生算子的接口：

- 用 `mint.narrow` 替代 `x[:, :seq_len]` 这类切片；
- 用 `mint.permute` 替代 Tensor 自带 permute；
- 用 `mint.split` 替代部分 split；
- 尽量用 `gather`、`index_select`、`Embedding` 替代高级 Tensor 索引。

单个改动看起来微不足道，但 BERT 完成 `mint` 替换后，耗时进一步降到约 21ms——这个量级本身就说明，此前有近一半的时间消耗在 Host 侧的"胶水代码"上。

### 4. 静态图与整图静态化：改变执行模式本身

BERT 最终从 20ms 左右降到 5-6ms，关键不是又替换了哪个算子，而是整图静态化。这个数字反过来印证了前面的判断：即使做完所有算子级优化，动态图模式下 Host 侧调度与单算子下发**仍然占据了近四分之三的耗时**。

静态图的收益机理有三层：一是执行模式从"Python 逐算子解释下发"变为"整图一次下发、Device 端连续执行"，Host 与 Device 之间不再逐算子同步，流水线得以打满；二是编译器拿到完整计算图后，可以做跨算子融合、常量折叠与内存复用；三是 shape 固定后，所有 tiling 策略与内存分配可在编译期确定，运行期没有任何 shape 推导开销。

要让静态图稳定工作，必须做出相应取舍：

- 固定输入 shape；
- 固定 `output_attentions=False`、`output_hidden_states=False` 等选项；
- 避免 `head_mask`、`past_key_values` 这类 JIT 不友好的分支进入主路径；
- 把可提前创建的 mask、position ids、token type ids 变为 buffer 或常量；
- 尽量减少 Python 动态逻辑与复杂返回结构。

这也引出了比赛中一个重要思路：**评测场景通常远比通用库场景固定**。通用模型代码为了兼容训练、推理、多任务、多种输出，保留了大量分支；参赛优化时完全可以围绕评测路径做专门化裁剪。

## 五、CLIP 的优化路线

CLIP 是双 Encoder 结构，包含 Text Encoder 与 Vision Encoder。整体仍然规整，但优化时要同时兼顾文本路径、图像路径与最后的特征对齐。

### 1. Text Encoder

attention 部分的优化思路与 BERT 一致（QKV 融合、FlashAttention、`mint` 替换），但多出一个重点：**causal mask**。

若 causal mask 每次推理都动态构造，就等于在主路径上引入了一段与序列长度相关的 Host 侧计算。正确做法是预先创建最大长度的 mask 并常驻，推理时按实际长度用 `mint.narrow` 截取——把运行期的构造成本转移到初始化期。同时需确认 mask 的维度与语义（加性 mask 还是布尔 mask）与 FlashAttention 接口一致。

文本侧还有一个典型索引优化点：取 EOT token 对应的 hidden state。原始写法通常使用高级索引，优化后可用 `argmax + gather` 组合表达同一语义，避免动态图下高级索引被展开为多个小算子。

### 2. Vision Encoder

Vision 侧的 patch embedding 本质是一次 Conv2d，将图像切成 patch 后送入 Transformer。优化重点包括：

- patch embedding 后的 flatten + transpose 改为语义更明确的 `mint.permute`；
- class embedding 的 broadcast 改为 `mint.broadcast_to`；
- CLS token pooling 避免 `last_hidden_state[:, 0, :]` 这类索引，改用 `index_select`；
- Vision Encoder 内部 attention 同样应用 QKV 融合与 FlashAttention。

### 3. 整图静态化：收益来源与 BERT 一致，但更能说明问题

CLIP 的整图静态化思路与 BERT 相同——固定输入 shape、裁剪评测路径以外的分支、把 mask 与 position ids 变为常量，让文本塔与图像塔各自编译为一张连续执行的静态图。性能演进同样由执行模式层主导：原始 80-90ms，转静态图后约 43ms，整图静态化后 12ms，最终约 10ms。

真正值得记录的，是 CLIP 实验中出现的一个反直觉现象：**某些单项优化版本并不比整图静态化版本快**。单独的 `mint` 版本约 27ms，而整图静态化版本约 12ms。

这说明优化收益并非可加的独立项，而取决于它与图编译、大算子、shape 固定之间的组合关系。其内在逻辑是：动态图下 `mint` 只能压缩**单次下发的固定成本**，但只要执行模式仍是"逐算子下发"，下发次数这个乘数就无法消除，收益必然存在上限；而整图静态化直接把这个乘数消掉，同时给编译器留出了融合与内存复用的空间，因此收益更彻底。

**优化要看的是瓶颈所在的层次，而不是技巧本身的先进程度。**

## 六、精度边界与踩过的坑

性能优化如果破坏精度，等于无效优化。本次比赛中，精度约束是一条硬边界。

### 1. fp16 不是无脑加速

最初很容易认为 fp16 一定更快，直接把模型或大部分计算转为 fp16，但实测结论明确：**全局 fp16 精度不够**。原因在于 Transformer 的残差累加、LayerNorm 的方差统计与最终 logits 都对数值范围与有效位数敏感，误差会沿层数累积放大。

稳妥做法是**局部 fp16**：在 FlashAttention 内部让 Q/K/V 走 fp16 以利用 Cube 单元算力，输出后立即转回 fp32；LayerNorm、残差、最终 logits 或 feature 输出等关键路径保持 fp32。

### 2. GELU 快算子要谨慎

BERT 的 MLP 中 GELU 看似诱人的优化点，`fast_gelu` 或自定义近似实现都尝试过，但结论同样是**精度不够**。近似激活函数的局部误差可能很小，但经过多层 Transformer 累积后，最终输出会超出容忍范围。

GELU、LayerNorm、Softmax 这类数值敏感算子，不能只看性能，必须逐层对比误差。

### 3. FlashAttention 的 mask 最容易出错

FlashAttention 接口对 layout、mask 维度与 mask 类型都有严格要求。BERT 是双向 attention，CLIP Text 是 causal attention，两者不能混用同一套 mask 逻辑。

验证时不能只跑一个正常样例，必须覆盖：短序列、最大长度序列、含 padding 的样例、不同 batch size、以及 CLIP 文本中 EOT token 位置不同的样例。否则很可能在某些输入下结果悄悄出错。

### 4. 通用模型代码不等于比赛最优代码

MindNLP 或 Transformers 风格的模型代码为了通用性保留了大量分支：训练/推理、是否返回 attention、是否返回 hidden states、是否使用 cache、是否传入 inputs_embeds 等。评测通常只走其中一条路径。

优化时要敢于围绕评测路径做专门化，但也要清楚自己删掉或绕开的是什么能力——例如 BERT Encoder 推理不需要 KV Cache，就不应让相关分支干扰静态图编译。

## 七、方法论沉淀

### 1. 优化流程：分层优化、分层验证

综合两个模型的实践，一套稳妥的优化流程可以固化为：

1. **固定环境与基线**——绑核、warmup、固定输入、固定测量方式；
2. **Profile 定位热点**——用 runtime profiler 与 Chrome tracing 判断瓶颈在 Host 还是 Device，这决定了后续应从哪一层入手；
3. **先改高频低风险写法**——替换 Tensor 索引、切片、permute、split、broadcast；
4. **再做结构融合**——QKV 融合、mask 预构造、layout 整理；
5. **接入大算子**——FlashAttention，必要时尝试 LayerNorm/AddNorm、MLP 融合；
6. **最后做静态图/整图静态化**——固定 shape，裁剪分支，把动态逻辑移出主推理图；
7. **每一步都做精度回归**——先对齐局部输出，再看端到端输出。

其核心原则是"分层优化、分层验证"：一次只改一层，否则性能变快了不知道原因，精度坏了也不知道是谁导致的。这个顺序本身也有讲究——低风险写法优先，是因为它们收益确定且不改变数值语义；静态图放在最后，是因为它要求前面各层的结构已经稳定。

### 2. 最终收获

如果用一句话总结这次参赛经验：**不要只盯着单个慢算子，而要把模型结构、框架图模式、输入 shape、算子布局、dtype 精度与硬件执行特点放在一起看。**

回到开篇提出的三个层次——代码写法层削减固定开销，算子结构层提高计算密度，执行模式层消除调度乘数。三者中收益最大的是执行模式层，因为它从根本上改变了模型与硬件的交互方式，让计算过程更接近 NPU 所擅长的连续大图执行。真正有效的优化，往往来自这些因素的组合，而非任何单一技巧。
