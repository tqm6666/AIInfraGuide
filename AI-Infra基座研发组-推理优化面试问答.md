# AI Infra 基座研发组：推理优化面试问答

> 候选人：田乔木  
> 目标方向：AI Infra / LLM 推理引擎 / GPU 性能优化  
> 依据材料：`田乔木ai infra简历.pdf`  
> 使用说明：答案中的 `【面试前补齐】` 必须替换成真实数据；禁止为了“答案完整”而编造项目细节。
>
> **全文披露规则：** 自我介绍、项目题、行为题和反问中的内部模型名、绝对吞吐、硬件数量、并行配置、函数名、Trace、请求数据及 MR 细节，只有在信息已公开或公司明确批准时才能口述。否则改用抽象机制、控制变量和相对趋势；简历中已经出现某个数字，也不自动代表可以补充其内部配置。

---

## 目录

1. [面试前必须修正的简历风险](#1-面试前必须修正的简历风险)
2. [自我介绍与求职动机](#2-自我介绍与求职动机)
3. [SGLang 与 AMD 推理优化项目深挖](#3-sglang-与-amd-推理优化项目深挖)
4. [性能分析与 Benchmark](#4-性能分析与-benchmark)
5. [自研 C++/CUDA 推理原型](#5-自研-ccuda-推理原型)
6. [Lua VM 与 AOT 经历](#6-lua-vm-与-aot-经历)
7. [LLM 推理基础](#7-llm-推理基础)
8. [KV Cache、PagedAttention 与调度](#8-kv-cachepagedattention-与调度)
9. [量化与投机解码](#9-量化与投机解码)
10. [TP、DP、CP、EP 与 MoE](#10-tpdp-cpep-与-moe)
11. [GPU、CUDA、ROCm 与 Kernel](#11-gpucudarocm-与-kernel)
12. [生产推理服务系统设计](#12-生产推理服务系统设计)
13. [行为面试](#13-行为面试)
14. [反问面试官](#14-反问面试官)
15. [面试前材料清单](#15-面试前材料清单)

---



# 1. 面试前必须修正的简历风险

这部分优先级高于背题。如果简历事实存在明显矛盾，面试官会沿着矛盾持续追问。

## 1.1 标准 GPT-2 不使用 RMSNorm、RoPE 或 SwiGLU

简历目前同时出现：

- GPT-2 Small；
- RMS Norm；
- RoPE；
- SwiGLU FFN。

其中 RMSNorm、RoPE 可以是阅读 llama.cpp 时学习的其他模型算子，但标准 GPT-2 的结构是：

- 12 层；
- hidden size 768；
- 12 个 Attention Head；
- learned absolute positional embedding；
- LayerNorm；
- GELU MLP；
- MLP 中间维度 3072；
- Token Embedding 与 LM Head 权重共享。

标准 GPT-2 Small 参数量约为：

```text
124,439,808 ≈ 124M
```

早期经常出现的 117M 是旧统计口径，不建议继续使用。

### 如果代码实现的是标准 GPT-2

建议改为：

```text
用 C++ 从零实现 GPT-2 Small（12 层、hidden=768、约 124M 参数）的完整前向与自回归解码流程，覆盖 Multi-Head Attention、KV Cache、GELU FFN、LayerNorm 与采样。
```



### 如果代码确实使用了 SwiGLU

不要称为标准 GPT-2，建议改为：

```text
用 C++ 从零实现 GPT-2 规模的自定义 decoder-only Transformer，覆盖 Multi-Head Attention、KV Cache 与 SwiGLU FFN。
```



### 面试回答边界

> 我阅读 llama.cpp 时重点研究了 RMSNorm、RoPE 和量化 GEMV；我自己实现的模型是否严格遵循 GPT-2，需要与当前代码再次核对。如果使用了 SwiGLU，它应被称为 GPT-2 规模的自定义 Transformer，而不是标准 GPT-2。

不要强行解释成“GPT-2 本来就有 SwiGLU”。

## 1.2 `7395 → 9100` 对应约 `+23.1%`

计算如下：

```text
(9100 - 7395) / 7395 ≈ 23.06%
```

因此简历中的 `+22%` 与端点数字不一致。

处理方式二选一：

1. 如果原始多次实验均值确实是 22%，展示真实均值，不同时展示 7395 和 9100；
2. 如果 7395 和 9100 是同口径结果，将提升改为约 23%。

面试前无法确认时，直接回答：

> 在固定场景下，吞吐由 7395 提升到 9100 tokens/s/GPU。

不要再口述 22%。

## 1.3 Lua 元方法应为 `__index`

Lua 元方法是双下划线：

```lua
__index
```

不是 `_index`。

## 1.4 TBO、DP、CP 的术语需要消歧

首次出现应说明：

```text
TBO = Two-Batch Overlap
```

同时区分：

- 普通推理 DP：复制完整模型，不同副本处理不同请求；
- DP Attention：Attention 与 MoE 采用不同并行组织；
- CP：沿序列维度切分长上下文；
- EP：不同专家分布在不同设备。

“DP-TBO”需要说清楚是：

- DP Attention 场景下的 TBO；
- 还是普通副本 DP 中的流水优化。

不要把两者混为一谈。

## 1.5 AITER 与 RCCL 是不同库，但不能绝对划成“计算 vs 通信”

- AITER：面向 AMD GPU 的高性能算子库，除计算算子外，也可能提供定制 All-Gather、Reduce-Scatter 等通信路径；
- RCCL：AMD GPU Collective Communication Library，提供标准 Collective；
- 实际执行可能选择 AITER 定制通信、RCCL，或在能力不满足时进入 Fallback；
- 内部算子库：若 TACOps 属于内部组件，公开面试中应泛化为“内部算子库”。

推荐表述：

> 我负责将 SGLang 的相关执行路径接入 AITER、内部算子库与 RCCL，并核对实际调用的是 AITER 定制通信、RCCL 还是 Fallback，完成不同并行模式和请求 Shape 的正确性与性能验证。



## 1.6 “完整推理引擎”可能被认为范围过大

如果项目没有实现以下能力：

- Continuous Batching；
- Paged KV Cache；
- 多请求调度；
- 分布式执行；
- OpenAI API Server；
- 可观测性与容错；

更准确的名称是：

```text
教学型 C++/CUDA 推理原型
```

可以主动限定：

> 这个项目的目标是理解模型前向、自回归解码、KV Cache 和 CUDA Kernel 优化，不是生产级 Serving Engine。



## 1.7 所有性能数字都要补测试条件

以下数字均需准备完整口径：

- TBO `+17%`；
- CP-TBO `7395 → 9100 tokens/s/GPU`；
- Lua VM `+10%`；
- Lua 访问 C++ 对象 `+60%`；
- CPU `15 tok/s` 到 Tensor Core `1400 tok/s`。

至少准备：

```text
硬件
模型与精度
软件版本/Commit
并行配置
Batch/并发
输入与输出长度
Prefix Ratio
是否启用 Prefix Cache
Warmup 次数
正式重复次数
均值/P50/P95/波动
计时范围
正确性标准
唯一变化变量
```

---



# 2. 自我介绍与求职动机



## Q1：请做一个 60 秒自我介绍



### 建议回答

> 面试官您好，我叫田乔木，目前在中国科学技术大学攻读软件工程硕士，主要方向是大模型推理、GPU 性能优化和系统运行时。
>
> 目前我在腾讯计算加速中心从事 SGLang 在 AMD MI355X、MI308X 上的推理适配与优化，工作覆盖 AITER 算子、RCCL 通信、TP/DP/CP、长上下文 Prefill 和多请求 Shape。我使用 Perfetto 定位过 TBO 中的 Stream 与资源争用问题，并独立完成 CP-TBO 的执行流水优化，在固定 32K Prefill 场景中将吞吐从 7395 提升到 9100 tokens/s/GPU。
>
> 此前我做过 Lua 虚拟机热路径和 AOT 优化，也实现过一个教学型 C++/CUDA 推理原型，逐步将 GEMM 从 CPU 优化到 Tiled CUDA 和 Tensor Core。我的优势是能把 Kernel、通信、调度、Profiler 和 Benchmark 串起来解决端到端问题，希望继续在 AI Infra 基座研发组深入做生产级推理优化。



### 注意

- 如果模型或内部算子名称涉及保密，用“大规模 MoE 模型”“内部算子库”替代；
- 不要在自我介绍中同时堆十几个缩写；
- 60 秒只讲一条主线：生产推理优化 → 系统底层经验 → 求职目标。



## Q2：请做一个 3 分钟自我介绍



### 建议回答

> 面试官您好，我叫田乔木，目前在中国科学技术大学攻读软件工程硕士。我希望长期从事 AI Infra 和大模型推理引擎研发，我的经历主要分成生产推理优化、运行时优化和自研推理原型三个部分。
>
> 第一部分是目前在腾讯计算加速中心的工作。我负责 SGLang 在 AMD MI355X、MI308X 上的大模型推理适配，接入 AITER、内部算子库和 RCCL，覆盖 TP、DP、CP、长上下文 Prefill 及多请求 Shape。除了让功能跑通，我还会核对实际执行路径、不同 Shape 的后端选择，以及 Stream、通信和 Buffer 生命周期。【面试前核实：只有实际处理过 Stride、Graph Capture 或 Fallback 问题时，才在面试中展开。】
>
> 在 TBO 优化中，我先固定模型、请求 Shape 和并行配置，用 Perfetto 和 torch profiler 对比 CPU Launch、GPU Kernel、RCCL Collective 与不同 Stream 的时间线，定位到额外 Stream 和资源争用影响关键路径。修复后，吞吐相对初版提高 17%。之后我独立开发 CP-TBO，通过拆分 Attention Prepare/Finish、调整 All-Gather 发起与消费时序，并限制 AITER All-Gather 的 Block 数，在 MI355X 32K Prefill 场景下将吞吐从 7395 提升到 9100 tokens/s/GPU。
>
> 我也建立了覆盖吞吐、正确性和稳定性的 Benchmark，使用经过公司批准的数据或等价 Shape 做 Dump/Replay，覆盖不同 AMD GPU、DP/CP、Prefix Ratio 和长上下文 Shape，并结合 Kernel Timing、MFU 和 Trace 分解 Attention、【MHC 全称面试前补齐】以及 MoE Stage1/Stage2 的瓶颈。【面试前核实：如果不能确认脱敏与外披授权，不要声称使用了可对外描述的真实请求。】
>
> 第二部分是游戏前沿技术经历。我做过 Lua `__index` 热路径和 AOT 优化，这些经历让我积累了 C/C++、虚拟机、ABI、对象布局以及性能分析能力。
>
> 第三部分是教学型 C++/CUDA 推理原型。我从 llama.cpp 的 Tensor Layout 和 Kernel 入手，学习 RMSNorm、RoPE 和量化 GEMV，并实现模型前向、KV Cache 与自回归解码，再将 GEMM 从 CPU 逐步替换为 Naive CUDA、Shared-Memory Tiled Kernel 和 WMMA。
>
> 这些经历让我形成了一套比较稳定的工作方式：先建立可复现 Baseline，再用 Trace 和性能模型找关键路径，做最小改动，通过正确性、性能和稳定性三类测试验证，最后推动合入与交付。我希望在 AI Infra 基座研发组继续把这套能力用于更大规模的推理系统。



## Q3：为什么从游戏运行时转到 AI Infra？



### 建议回答

> 两段经历在底层能力上是连续的。Lua VM 和 AOT 关注解释器热路径、对象布局、编译期信息和运行时开销；推理引擎同样需要处理算子执行、内存布局、调度和性能归因。我在做运行时优化时发现自己更喜欢“通过系统和硬件理解解释性能”的问题，因此主动从 llama.cpp 和 CUDA 推理原型开始学习，之后转到计算加速中心做 AMD 生产推理优化。这个转变不是完全换方向，而是把运行时与性能工程能力迁移到了更大规模的 AI 系统。



## Q4：为什么想加入 AI Infra 基座研发组？



### 建议回答

> 我希望做的不只是某个孤立 Kernel，而是端到端推理基座。基座组通常需要同时理解模型执行、算子、通信、调度、资源管理、Benchmark 和生产交付，这与我目前的工作方式匹配。我已有 AMD GPU、SGLang、CP/TBO 和性能分析经验，但还希望继续补齐更通用的 Scheduler、KV Cache、在线 Serving 和多机容错能力。



## Q5：你的核心竞争力是什么？



### 建议回答

> 第一，我能从 Trace 和数据出发做性能归因，不会看到长 Kernel 就直接优化。第二，我同时接触过 Kernel、Collective、Stream 依赖和框架执行路径，能够处理跨层问题。第三，我不只看单点吞吐，还建立了正确性、稳定性和多 Shape Benchmark，并参与 MR 和镜像交付。我的不足是生产 Serving 的在线流量治理和大规模集群调度经验还不够完整，这也是我下一步希望补强的方向。

---



# 3. SGLang 与 AMD 推理优化项目深挖

回答项目题时统一遵循：

```text
测试条件
→ Baseline
→ Trace 证据
→ 根因假设
→ 受控实验
→ 最小改动
→ 正确性/性能/稳定性验证
→ 局限与适用边界
```

> **披露门禁：** 本章中的绝对吞吐、硬件数量、并行配置、函数名和执行细节只是面试准备清单，不代表已经获得对外披露授权。只有公开信息或公司明确批准的内容才能口述；否则使用抽象 DAG、方法和相对趋势。简历中已经出现某个数字，也不自动等于可以继续补充其内部配置。



## Q6：用两分钟介绍你当前的推理优化项目



### 建议回答

> 项目目标是在 AMD MI355X、MI308X 上完成 SGLang 对目标大模型的推理适配和性能交付。我负责的范围主要有三部分。
>
> 第一是执行路径适配，包括 AITER、内部算子库、RCCL，以及 TP、DP、CP、长上下文 Prefill 和多请求 Shape。这里的难点不是单一 API 接入，而是不同 GPU 架构、数据类型和 Shape 的组合。【面试前核实：Stride、Graph Capture 和 Fallback 只保留你确实处理过的部分。】
>
> 第二是 TBO 与 CP-TBO 的流水优化。我使用 Perfetto 分析多 Stream 和 Collective 的关键路径，定位资源争用并完成修复；之后独立实现 CP-TBO，通过 Attention 拆分、All-Gather 时序和 Block Cap 控制通信计算重叠。
>
> 第三是 Benchmark 和交付。我建立请求 Dump/Replay、正确性、吞吐和稳定性测试矩阵，覆盖不同硬件、并行模式和 Prefix Ratio，并推动跨版本 MR 与镜像交付。对 Dump 只说明获批的脱敏方式或等价 Shape，不披露请求内容。



## Q7：哪些是你独立完成的，哪些是团队已有能力？



### 建议回答模板

> 我独立负责的是【面试前补齐：CP-TBO 的具体模块、函数和测试】；团队 MoE Kernel 由其他同学提供，我负责接入、定位它与框架及通信路径的交互问题，并完成不同 Shape 的正确性和性能验证。AITER 和 RCCL 是外部库能力，我的贡献是 Dispatch、参数与执行时序适配，不把库本身的实现算作个人工作。



### 追问

- 你实际修改了哪些模块？
- MR 中你负责多少代码？
- 设计方案是谁提出的？
- Benchmark 和交付是否也由你负责？

回答时把“设计、编码、调试、测试、合入”分别说明，避免用“负责整个项目”覆盖贡献边界。

## Q8：TBO 是什么？



### 30 秒答案

> TBO 是 Two-Batch Overlap。它把一个 Batch 切成两个 Micro-Batch，在执行图的 Yield Point 交替推进，让一个 Micro-Batch 的计算覆盖另一个 Micro-Batch 的通信，例如 MoE Dispatch/Combine 或某类 Collective。理想情况下，串行的 `C + M` 可以接近 `max(C, M)`，但拆分后 GEMM 变小、Launch 和同步增多，因此不是所有 Shape 都有收益。



### 深挖

- 两个 Micro-Batch 如何划分？
- 重叠的具体阶段是什么？
- 哪个 Stream 执行通信？
- 如何管理双缓冲？
- 流水启动和排空开销是多少？
- 什么场景下关闭 TBO？



## Q8.1：DP Attention / DP-TBO 在你的项目中具体是什么？

### 回答原则

“DP-TBO”没有脱离实现的通用标准答案。必须先根据真实代码画出以下拓扑，不能把普通副本 DP、DP Attention 和 EP 混为一谈：

```text
1. 请求或 Token 如何分配到 DP Rank；
2. Attention 权重和 KV 在哪些 Rank 驻留或分片；
3. MoE Expert 是否按 EP 切分；
4. Attention 前后发生哪些 Collective；
5. 两个 Micro-Batch 分别在哪个 Stream 推进；
6. TBO 重叠的是 Attention、Dispatch/Combine、All-Gather 还是其他阶段。
```

### 建议回答模板

> 我项目中的 DP-TBO 指【面试前补齐：普通 DP 还是 DP Attention】场景下的 Two-Batch Overlap。并行组由【补齐 DP/TP/EP/CP Group】组成；每个 Rank 持有【补齐 Attention 权重、KV 和 Expert 的实际布局】。Micro-Batch A 在执行【阶段 A】时，Micro-Batch B 发起【Collective B】，两者通过【HIP Event/框架依赖】同步。真正的收益来自隐藏【通信】，代价是【Micro-Batch GEMM 变小、Launch 增加或 Buffer 加倍】。

### 如果实现属于常见的 DP Attention + EP MoE 形态

可以用下面的抽象图帮助解释，但必须以实际代码为准：

```text
Micro-Batch A: Attention → Token Dispatch → Expert Compute → Combine
Micro-Batch B:             Attention → Token Dispatch → Expert Compute
                         <------- overlap window ------->
```

### 高频追问

- DP Rank 之间是否共享或交换 KV？
- Attention 与 MoE 使用同一 Process Group 吗？
- Dispatch/Combine 是 All-to-All 还是定制通信？
- 两个 Micro-Batch 如何保持 Collective 顺序一致？
- 为什么该方案在小 Token 数下可能回退？
- +17% 相对的是无 TBO，还是 TBO 初版？

## Q9：为什么增加 Stream 反而可能让吞吐下降？



### 30 秒答案

> Stream 只是调度和依赖抽象，不代表独占硬件。不同 Stream 上的 RCCL、Attention 和 MoE Kernel 仍会竞争 CU、HBM、Cache、互联或硬件队列。额外 Stream 还可能因为 Event 放置错误、默认 Stream 语义或 Buffer 生命周期导致隐式串行。判断是否有效不能只看时间线上颜色重叠，而要看端到端关键路径是否缩短。



### 深挖

- 高优先级 Stream 能否抢占正在执行的 Kernel？
- 如何区分资源争用和依赖错误？
- 两个 Kernel 并发后都变慢怎么办？
- 如何测量真正的 Overlap？



### 推荐回答

> 我会对同一请求做串行、原并发和限资源并发三组实验。如果输出始终一致，但并发时两个 Kernel 都变慢，而限制通信 Block 后 Wall Time 降低，就更支持资源争用假设；如果去掉或调整 Event 后结果错误，则说明还有依赖问题。



## Q10：你如何定位 TBO 初版的问题？



### 建议回答模板

> 我首先固定【模型、硬件、并行配置、请求 Shape、精度】，排除请求波动。然后使用 torch profiler 的 `record_function` 标记 Attention、MoE 和通信阶段，通过 Perfetto 查看 CPU Launch 与 GPU Stream 时间线。我观察到【面试前补齐：具体 Kernel/Collective】运行在额外 Stream，并与【具体计算】发生破坏性并发。
>
> 为验证因果，我进行了三组 A/B：【原始实现】、【强制串行或关闭该 Stream】、【调整时序/资源后的实现】。结果显示【面试前补齐：关键路径、Kernel Slowdown 与吞吐变化】，因此确定问题不是简单的单 Kernel 性能，而是并发资源争用。最终修改【面试前补齐】，吞吐较初版提升 17%。



### 必须补齐

- 额外 Stream 的来源；
- 争用的具体资源；
- 关键路径缩短多少；
- 17% 的起止吞吐；
- 哪些 Shape 有提升，哪些没有；
- 正确性与稳定性如何验证。



## Q11：CP-TBO 的核心设计是什么？



### 30 秒答案

> 核心是把 CP Attention 中“通信前必须完成的工作”和“真正消费通信结果的工作”拆开。Prepare 阶段产出通信输入并尽早发起 All-Gather；另一 Micro-Batch 在通信期间执行可独立计算；Finish 只在首次消费远端结果前等待 Event。再通过 All-Gather Block Cap 控制通信对 CU 的占用，使重叠后的端到端关键路径最短。若实际 API 仅将数据收集到 Root，才应称 Gather；若分片不等长，还需确认是否属于 All-Gatherv。



### 追问

- Prepare 和 Finish 各自处理什么 Tensor？
- All-Gather 的输入/输出 Tensor Shape 是什么？
- 为什么可以提前发起？
- 为什么不能无限提前？
- Event 记录和等待在哪里？
- Buffer 是否双缓冲？
- 多 Rank Collective 顺序如何保证？

凡涉及内部实现，只能依据你实际做过且获准披露的代码回答；否则画抽象依赖 DAG，并明确受 NDA 限制，不要用通用 Attention 流程猜测。

## Q12：Attention Prepare/Finish 拆分为什么有用？



### 建议回答

> 拆分本身不会自动产生收益，它只是暴露调度边界。Prepare 必须完成通信输入的生产，并记录 Producer Event；All-Gather 在通信 Stream 发起。Finish 是最早需要完整 All-Gather 结果的 Consumer，只在这里等待。这样将通信等待从整个 Attention 路径推迟到真实依赖点，中间窗口可以执行另一批计算。



### 常见错误

- 把函数拆成两个名字就声称实现了 Overlap；
- 没有说明 Producer/Consumer；
- CPU `synchronize()` 阻塞整个线程；
- 两个 Micro-Batch 复用同一 Buffer 产生 Race；
- 不同 Rank 的 Collective 顺序不一致。



## Q13：All-Gather 时序如何选择？



### 30 秒答案

> 理想位置是“最晚 Producer 之后、最早 Consumer 之前”。过早时输入还没有完成或 Buffer 仍在复用；过晚会缩短可重叠窗口。需要结合执行 DAG、Event 和 Trace 选择，而不是单纯把 All-Gather 移到函数开头。面试时还应说明输入是每 Rank 的本地分片、输出是否在所有 Rank 形成完整张量。



## Q14：为什么限制 All-Gather Block 数可能提升端到端性能？



### 30 秒答案

> Block 多通常会降低通信本身延迟，但也可能占用更多 CU、带宽和 Cache，导致并发计算变慢。限制 Block 数是在“通信自身速度”和“给计算让资源”之间取舍。最佳点不是 All-Gather 单测最快，而是 `通信 + 计算 + 流水首尾` 的 Wall Time 最短。



### 追问

- Cap 太小会怎样？
- Cap 太大会怎样？
- 是否所有消息大小使用同一 Cap？
- 是否影响多机通信？
- 如何做参数 Sweep？



### 建议回答

> 我会固定消息大小和计算 Shape，扫描 Cap，记录通信单独时延、并发后通信时延、计算 Slowdown 和最终 Wall Time。最终选择端到端最优且在多轮测试中稳定的区间，而不是只看 RCCL/AITER 单测峰值。



## Q15：如何解释 `7395 → 9100 tokens/s/GPU`？



### 建议回答

> 在已获准披露的范围内，这个结果来自固定模型、硬件、并行配置和 32K Prefill Workload。Baseline 是相同 CP 配置下关闭 TBO，优化组只打开 CP-TBO 和对应时序参数。经过 Warmup 和多轮重复，吞吐由简历中的 7395 提升到 9100 tokens/s/GPU，按端点是约 23.1%。如果卡数、节点拓扑、绝对容量或内部配置未获准外披，只说明控制变量、相对趋势和验证方法，不继续补充。



### 不能漏掉

- 32K 是单请求输入长度、总 Batch Token 还是 Chunk Size；
- `tokens/s/GPU` 是系统总有效 Token 除以 GPU 数，还是单卡直接统计；
- 是否包含前一项 TBO +17%；
- 两个提升不能相加。



## Q16：如何证明提升不是 Benchmark 噪声？



### 建议回答

> 我固定模型、镜像、Commit、并行配置、请求 Dump、ISL/OSL、Prefix Ratio、并发和 GPU 状态；先 Warmup，再交替运行 Baseline 与优化版，避免温度和缓存随时间偏移。除均值外还记录 P50/P95、波动、显存、错误率和输出一致性。如果收益明显大于噪声区间，且多轮顺序切换后仍然稳定，才认为优化成立。



## Q17：如何验证多 Stream 优化的正确性？



### 建议回答

> 我会分四层验证：
>
> 1. 单算子与参考实现比较；
> 2. 单请求多步 Logits/Token 对齐；
> 3. 多请求、多 Shape、不同 CP/DP 配置；
> 4. 长时间压力测试检查 Race、Hang、显存增长和 Collective 顺序。
>
> 多 Stream 问题不能只跑一次 `allclose`，需要覆盖 Buffer 复用、请求交错和 Rank 间顺序。



## Q18：MI355X 与 MI308X 适配有什么差异？



### 建议回答

> 我不会先背型号参数判断瓶颈，而是先用 `rocminfo`、`amd-smi` 和软件环境确认实际 GFX Target、可用数据类型、内存与矩阵指令，再分别做 Shape Sweep。不同架构的最优 Tile、LDS/VGPR 压力、量化路径和 Kernel Binary 可能不同，因此需要按架构 Dispatch 并保留正确性 Fallback。



### 注意

- 不要编造 MI308X 未公开或无法确认的硬件规格；
- MI355X 与 MI308X 不能共用一个未经验证的最优配置；
- WMMA 是 NVIDIA CUDA API，不要说用于 AMD 交付。



## Q19：跨版本迁移最容易出什么问题？



### 建议回答

> 常见问题包括框架内部 Metadata 变化、Tensor Layout/Stride 变化、算子 Dispatch 条件变化、Graph Capture 兼容性、ROCm/PyTorch/AITER ABI 组合以及 Collective Group 初始化顺序。我会建立版本兼容矩阵，先跑 Reference Path，再逐项打开优化后端；每个 Feature 都有 Capability Check 和显式 Fallback。



### 面试前补齐

准备一个真实案例：

```text
旧版本行为
→ 新版本变化
→ 失败表现
→ 定位证据
→ 兼容方案
→ 回归测试
→ 如何避免再次发生
```



## Q20：如何处理保密内容追问？



### 建议回答

> 这部分涉及内部模型、请求和部署配置。我可以基于公开的 SGLang、AITER 和 RCCL 路径解释依赖关系、定位方法和相对性能趋势，但不能提供内部源码、真实请求、集群拓扑、镜像标签或未公开参数。

可以讲：

- 抽象后的执行时间线；
- 通用 Stream/Event 机制；
- 公开框架源码；
- Benchmark 方法；
- 相对趋势；
- 个人贡献边界。

不能讲：

- 真实 Token Dump；
- 未公开模型结构；
- 内部 MR 内容；
- 客户数据；
- 内部镜像与集群地址；
- 未公开绝对容量。

---



# 4. 性能分析与 Benchmark



## Q21：你通常如何定位推理性能瓶颈？



### 30 秒答案

> 我先明确业务指标和阶段，再建立可复现 Baseline。然后从服务级指标定位是排队、Prefill、Decode 还是通信问题；用 Trace 找关键路径和空洞；对关键路径上的 Kernel 或 Collective 做 Microbenchmark；最后通过受控 A/B 和端到端 Replay 验证。不会直接把所有 Kernel 时长相加，也不会只优化最长 Kernel。



### 完整流程

```text
SLO/吞吐异常
→ 固定 Workload
→ 服务阶段拆分
→ Perfetto 时间线
→ 关键路径
→ Kernel/Collective Microbenchmark
→ 性能模型
→ 最小优化
→ 端到端回归
```



## Q22：Perfetto 和 torch profiler 分别做什么？



### 建议回答

> torch profiler 负责采集 CPU Operator、Runtime API、GPU Kernel、Shape 和用户标记，Kineto 可以导出 Trace；Perfetto 更适合查看跨线程、跨 Stream 的时间线。我的做法是用 `record_function` 标记 Attention、通信和 MoE Stage，通过 Correlation ID 从 Python Op 追到 HIP API 和 GPU Kernel。



### 追问

- 为什么不 Profile 第一轮？
- Profiler 自身会产生多大开销？
- `self_gpu_time` 与总时间有什么区别？
- 如何从 CPU Launch 定位到 GPU Kernel？
- Hardware Counter 从哪里获取？



### 回答要点

- 第一轮包含 JIT、Module Load、缓存初始化和首次分配；
- 使用 wait/warmup/active schedule；
- 最终性能数字要关闭 Profiler；
- Perfetto 负责时间线，硬件 Counter 需要 rocprofiler-compute 等工具。



## Q23：Trace 上有重叠，为什么性能仍可能下降？



### 建议回答

> 时间重叠不等于关键路径缩短。并发后通信和计算可能因为 CU、HBM 或 Cache 争用同时变慢；Micro-Batch 变小也可能降低 GEMM 效率；还可能增加 Launch、Event 和流水首尾开销。因此要比较串行 Wall Time、并发 Wall Time以及双方的 Slowdown，而不是追求最大重叠面积。



## Q24：如何正确测 Kernel 时间？



### 建议回答

> GPU 执行是异步的，CPU Timer 只测到 Launch 时间。单 Kernel Microbenchmark 应 Warmup，使用 HIP/CUDA Event 在同一 Stream 上记录开始和结束，结束后同步 Event，再重复多轮统计。端到端测试则测 Wall Time，并明确是否包含 H2D、Tokenizer、权重加载和首次编译。



### 常见错误

- 不同步就读取 CPU 时间；
- 把 Profiler 时间直接当正式 Benchmark；
- 只测一次；
- 包含首次 JIT；
- 优化组与 Baseline 精度不同；
- 只测最佳 Shape。



## Q25：什么是 Dump/Replay？如何保证可信？



### 建议回答

> Dump/Replay 的目的，是将真实 Workload 的 Shape、Dtype、Stride、Sequence Metadata、路由分布和并行配置固定下来，便于离线复现。Manifest 还应记录代码 Commit、模型版本、随机种子和 Rank。Replay 先与参考路径比较正确性，再恢复多 Rank 和性能测试。



### 风险

- 只保存一个 Tensor，丢失语义 Metadata；
- MoE 路由分布失真；
- Collective 没有保存各 Rank 输入和顺序；
- Dump 含敏感 Token；
- Replay 只能复现 Kernel Shape，不能复现线上到达与排队。



## Q26：你如何定义正确性、性能和稳定性？



### 建议回答模板

> 正确性方面，我使用【面试前补齐：max abs/max relative/cosine/logit top-k/token/任务集】；性能方面记录【吞吐、TTFT、TPOT/ITL、显存、Kernel/Collective 时间】；稳定性方面运行【时长和并发】，检查 OOM、Hang、RCCL Timeout、错误率、显存增长和输出漂移。

不要只说：

```text
结果差不多，跑起来也没问题。
```



## Q27：MFU 是什么？推理中如何计算？



### 30 秒答案

> MFU 是有效模型 FLOPs/s 除以对应精度下的硬件峰值。若 `tokens/s` 使用全局吞吐，分母需要乘 GPU 数：
>
>
> MFU = \frac{T_{global}\times\text{模型 FLOPs/token}}
> {\text{GPU 数}\times\text{峰值 FLOPS/GPU}}
>
>
> 如果使用 `tokens/s/GPU`，分母就不能再次乘 GPU 数。Dense Transformer 前向常粗略按每 Token `2P`，但长 Prefill 需要加入 Attention 项；MoE 应按激活专家计算；量化模型要按实际执行指令与计算精度选择峰值，而不是只看权重存储格式。Prefix Cache 命中的逻辑 Token 没有重新执行模型 FLOPs，不能直接计入硬件 MFU。Decode 通常受带宽限制，MFU 低不一定代表实现差。



### 常见错误

- 使用训练的 `6P` 估算推理；
- MoE 使用所有驻留参数计算 FLOPs；
- 用 FP4 峰值评价 BF16 Kernel；
- 把 Padding 和无效计算当作有效模型 FLOPs；
- 只看 MFU，不看带宽。



## Q28：TTFT、ITL、TPOT、Throughput 和 Goodput 分别是什么？



### 建议回答

- TTFT：请求到达至首 Token 返回，包含排队和 Prefill；
- ITL：相邻输出事件/Token 的时间间隔；
- TPOT：首 Token 后平均每输出 Token 时间；
- Throughput：Request/s 或 Input/Output Token/s；
- Goodput：满足 TTFT、ITL、E2E 等 SLO 的有效请求吞吐。

需要说明：

- Speculative Decode 一次可能返回多个 Token，ITL 与 Token 级 TPOT 不能机械等同；
- 所有指标都要报告 P50/P95/P99；
- `tokens/s/GPU` 必须写清 Input、Output 还是总 Token。



## Q29：如何设计可信的性能 Benchmark？



### 建议回答

> 固定代码、镜像、模型、精度、硬件、并行配置和缓存状态；充分 Warmup；覆盖真实输入长度、输出长度、并发、Prefix Ratio 和多请求 Shape；同时记录吞吐、尾延迟、显存、正确性、错误率和稳定性。最终数字关闭 Profiler，并进行多轮交替 A/B。



### 推荐结果表字段

```text
commit
image
hardware
world_size
TP/DP/CP/EP
dtype
ISL/OSL
concurrency
prefix_ratio
cache_state
warmup
repeat
throughput
TTFT p50/p99
ITL p50/p99
memory
correctness
error_rate
```

---



# 5. 自研 C++/CUDA 推理原型



## Q30：你为什么自己实现一个推理原型？



### 建议回答

> 目标不是重复生产框架，而是通过最小闭环理解模型执行。我希望把 Transformer 数学、Tensor Layout、KV Cache、自回归解码和 CUDA Kernel 联系起来。因此先实现正确的 CPU/朴素版本，再逐层替换 GEMM 和其他 Kernel，并使用相同输入做正确性和性能对比。



## Q31：一条 Token 的生成流程是什么？



### 建议回答

```text
Token ID
→ Token Embedding + Position Encoding
→ N 个 Transformer Block
  → Norm
  → Q/K/V Projection
  → RoPE 或模型对应位置编码
  → 写入 KV Cache
  → Attention
  → Output Projection + Residual
  → Norm
  → FFN + Residual
→ Final Norm
→ LM Head
→ Logits
→ Sampling
→ Next Token
```

标准 GPT-2 使用 learned absolute position embedding，不使用 RoPE。回答时必须根据实际实现区分。

## Q32：如何手算标准 GPT-2 Small 参数量？



### 回答要点

Embedding：

```text
Token Embedding: 50,257 × 768 = 38,597,376
Position Embedding: 1,024 × 768 = 786,432
```

每层主要参数：

```text
QKV: 768 × (3 × 768)
Attention Output: 768 × 768
MLP Up: 768 × 3072
MLP Down: 3072 × 768
加上 Bias 与两个 LayerNorm
```

12 层、Final LayerNorm、Embedding 与 LM Head 权重共享后：

```text
总参数约 124,439,808
```



## Q33：ggml 的 `ne/nb` 是什么？



### 建议回答

> `ne` 表示各维度元素数量，`nb` 表示沿各维前进一步的字节跨度。不能只看 Shape 判断 Tensor 是否连续；View、Transpose 和量化 Tensor 都可能有特殊 Stride。Kernel 接入时如果默认连续布局，可能产生错误结果或隐藏的 Layout Copy。



### 追问

- 连续 Tensor 的 `nb` 如何计算？
- Transpose 后 `ne/nb` 如何变化？
- 量化 Block 的 `nb[0]` 是否仍等于单元素字节数？
- 如何检测 Non-Contiguous Input？



## Q34：为什么 Prefill 通常是 GEMM，Decode 通常是 GEMV？



### 30 秒答案

> Prefill 同时处理多个 Token，线性层形状接近 `[tokens, K] × [K, N]`，权重可被多个 Token 复用，因此算术强度较高；低 Batch Decode 每步只有一个或少量 Token，退化为 GEMV 或 Small-M GEMM，每步都要重新读取大部分权重，所以常受 HBM 带宽限制。高并发 Decode 也可能重新形成较大的 GEMM，不能绝对化。



## Q35：KV Cache 如何工作？



### 建议回答

> 自回归 Decode 时，历史 Token 的 K/V 不会变化，因此每层只计算新 Token 的 Q/K/V，将新 K/V 追加到缓存，再让新 Q 与全部历史 K 做 Attention。这样每步避免重算整个历史，但 KV Cache 显存随序列长度线性增长。

每 Token KV 字节：


2\times L\times n_{kv}\times d_h\times bytes


其中 2 表示 K 和 V。

## Q36：Naive GEMM 的瓶颈是什么？



### 建议回答

> Naive 实现通常让每个线程计算一个输出元素，并在 K 维循环中反复从 Global Memory 读取 A 和 B。同一元素被相邻线程重复读取，数据复用差，Global Memory 流量大，无法充分利用计算单元。



## Q37：Tiled GEMM 为什么更快？



### 建议回答

> Thread Block 协作将 A、B 的 Tile 搬到 Shared Memory，每个元素被多个线程复用；累加器保存在寄存器中；Global Load 合并；每轮 Tile 之间同步。性能取决于 Tile Size、向量化、Bank Conflict、寄存器/LDS 占用、双缓冲和边界处理。



### 常见追问

- 为什么 Tile 不能越大越好？
- `__syncthreads()` 应放在哪里？
- 非整除 Shape 如何处理？
- Shared Memory Bank Conflict 如何产生？
- 如何估算 Global Memory 流量减少比例？



## Q38：WMMA 是什么？



### 建议回答

> WMMA 是 NVIDIA CUDA 的 Warp-Level Matrix Multiply-Accumulate API。一个 Warp 协同执行 `load_matrix_sync`、`mma_sync` 和 `store_matrix_sync`，需要满足支持的 Dtype、Tile Shape、Layout、Alignment 和 Leading Dimension。Tensor Core 只加速矩阵乘累加，数据搬运、Padding、Layout 转换和 Epilogue 仍可能成为瓶颈。



### 追问

- 使用的 M/N/K Tile 是什么？
- 输入与累加精度是什么？
- 如何证明真正执行了 Tensor Core 指令？
- 非 16 倍数 Shape 怎么处理？
- Decode 中 M=1 时为什么 WMMA 可能没有优势？



### 关键提醒

> 如果 `1400 tok/s` 主要来自 Prefill 或较大 Batch，必须明确。不要使用 GEMM Microbenchmark 代替单请求 Decode 结论。



## Q39：如何证明 `15 → 1400 tok/s` 公平？



### 建议回答模板

> 四档使用相同模型、输入、输出、Batch 和正确性标准；分别记录 CPU、Naive CUDA、Tiled GEMM 与 WMMA。计时【是否包含 Tokenizer、H2D、权重加载】必须明确；GPU 测试经过 Warmup，并使用 Event/Wall Time 正确同步。CPU 是否使用 BLAS 和多线程、各版本精度是否相同也必须说明。



### 面试前补齐

- CPU 和 GPU 型号；
- CPU 线程数；
- CUDA 版本；
- Dtype；
- Batch；
- Prompt/Output Length；
- 是否启用 KV Cache；
- 1400 是 Prefill、Decode 还是端到端；
- Logit/Token 正确性。



## Q40：RMSNorm 如何实现？



### 建议回答

RMSNorm：


y_i=x_i\cdot\frac{1}{\sqrt{\frac{1}{d}\sum_j x_j^2+\epsilon}}\cdot w_i


与 LayerNorm 不同，它不减均值。

Kernel 通常：

1. 每个 Block/Warp 处理一行；
2. FP32 累加平方和；
3. Warp Shuffle 或 Shared Memory 归约；
4. 计算 `rsqrt`；
5. 融合 Scale 和 Weight；
6. 向量化读写。

AMD Wave64 与 NVIDIA Warp32 不能硬编码相同归约步长。

## Q41：RoPE 的数学原理是什么？



### 建议回答

对每对维度做二维旋转：


y_0=x_0\cos\theta-x_1\sin\theta



y_1=x_0\sin\theta+x_1\cos\theta


常见错误包括：

- Interleaved 与 Split-Half 配对混淆；
- Position Offset 错误；
- Partial Rotary Dimension 错误；
- Scaling 方案不一致；
- Q/K Layout 错误。

标准 GPT-2 不使用 RoPE，回答时说明这是 llama.cpp 源码学习内容或自定义模型能力。

## Q42：`block_q8_0` 如何工作？



### 建议回答

> Q8_0 通常按固定 Block 保存若干 int8 权重和一个 Scale，近似恢复为 `w_i ≈ q_i × d`。Decode 中更好的实现是在寄存器中反量化后立即与 Activation 做 Dot Product，而不是先将完整 FP16 权重写回 HBM，从而降低权重流量和中间 Tensor。



### 注意

- 不同 llama.cpp Commit 的 Scale 类型可能不同；
- 面试前准备精读 Commit；
- 不要从“权重是 int8”直接推断使用了 int8 Tensor Core；
- 对称量化通常没有 Zero Point，但以实际源码为准。

---



# 6. Lua VM 与 AOT 经历



## Q43：Lua 经历与 AI Infra 有什么关系？



### 建议回答

> 两者都属于性能敏感的系统软件。Lua VM 优化让我理解解释器 Dispatch、对象布局、缓存失效和热路径；AOT 项目让我接触编译期信息与运行时 ABI；这些能力可以迁移到推理引擎的 Kernel Dispatch、Tensor Layout、Graph Capture 和运行时调度。



## Q44：`__index` 的语义是什么？



### 建议回答

当 Table 中不存在 Key 时：

- 如果 Metatable 的 `__index` 是 Table，继续在该 Table 查找；
- 如果 `__index` 是 Function，调用函数；
- 可能形成多层 Metatable Chain；
- 修改 Metatable 后，任何缓存都必须正确失效；
- 需要处理循环引用和递归保护。



### 项目回答模板

> 我通过 Profile 确定 `__index` Dispatch 是热路径，再分析【面试前补齐：重复查找链】并增加【具体 Fast Path/缓存】。优化必须保持 Table/Function 两类语义，并处理 Metatable 变化后的失效。+10% 是【微基准/实际 Workload】下的虚拟机执行效率，不等于游戏整体帧率。



## Q45：静态反射为什么能加速 Lua 访问 C++ 对象？



### 建议回答

> 运行时反射通常需要通过名称查找属性或函数，再做类型和偏移解析。如果编译期已知类型，可以把属性偏移、函数入口或符号索引固化，运行时从多次哈希/反射查找变为直接偏移访问。



### 风险

- C++ ABI 与类布局变化；
- 继承和虚函数；
- Hot Reload；
- 不同编译器或编译选项；
- 对象生命周期；
- 生成代码与类型版本不一致。

回答时必须说明如何校验版本和失效。

---



# 7. LLM 推理基础



## Q46：Prefill 和 Decode 有什么区别？



### 30 秒答案

> Prefill 一次处理 Prompt 中多个 Token，生成每层初始 KV，线性层通常是大 GEMM，关注 TTFT；Decode 每次生成一个或少量新 Token，读取历史 KV 并追加新 KV，线性层常是 GEMV/Small GEMM，关注 ITL/TPOT。Prefill 往往计算受限，低 Batch Decode 往往带宽受限，但应通过 Roofline 和 Trace 判断，不能绝对化。



## Q47：为什么 Decode 不能一次并行生成所有 Token？



### 建议回答

> 自回归模型满足：
>
>
> P(x_{1:n})=\prod_{t=1}^{n}P(x_t|x_{<t})
>
>
> 第 `t` 个 Token 的输入依赖前面已经采样出的 Token，因此时间维是串行的。Speculative Decode 只能让小模型先提出候选，再由 Target 并行验证，不能直接消除分布依赖。



## Q48：推理性能瓶颈如何用 Roofline 判断？



### 建议回答


P\le\min(P_{peak},BW_{peak}\times AI)


其中：

```text
AI = FLOPs / Bytes
```

- 落在斜线附近：带宽受限；
- 落在水平线附近：计算受限；
- 远低于两条 Roof：可能是并行度、Occupancy、依赖、访存合并或 Launch 问题。

需要使用实际访存字节，而不只是算法理想流量。

## Q49：如何计算 KV Cache 容量？



### 标准公式


KV_{bytes/token}=2\times L\times n_{kv}\times d_h\times bytes

该式首先给出标准、未分片模型的一份逻辑 KV 总量。单 GPU 物理驻留量必须将 `n_kv` 替换为实际本地 KV Head 数，并乘上复制因子；某些 CP Prefill 路径虽然切分计算，却会在 All-Gather 后为 Decode 复制持久 KV，因此不能默认再除以 CP Size。


总量：


KV_{total}=KV_{bytes/token}\times tokens


若：

```text
L = 48
n_kv = 8
d_h = 128
BF16 = 2 bytes
```

则：

```text
每 Token KV = 2 × 48 × 8 × 128 × 2
            = 196,608 bytes
            = 192 KiB
```

32K Token 约为：

```text
6 GiB
```

还要考虑：

- TP 是否切分 KV Heads；
- Block 向上取整；
- 尾块碎片；
- Scale 元数据；
- Prefix Sharing；容量规划按物理驻留 Block 计算，不能把共享前的逻辑 Token 简单相加；
- Graph Pool 与 Workspace；
- MLA 等不同缓存结构。



## Q50：为什么 GQA/MQA 能降低 KV Cache？



### 建议回答

> MHA 为每个 Query Head 保存独立 K/V；GQA 让多个 Query Head 共享一个 KV Head；MQA 进一步让所有 Query Head 共享一组 K/V。KV Cache 与 `n_kv_heads` 成正比，因此能显著减少显存和 Attention 读带宽，但模型质量和 Kernel Layout 需要配套设计。

---



# 8. KV Cache、PagedAttention 与调度



## Q51：PagedAttention 解决什么问题？



### 30 秒答案

> 它将逻辑 KV Block 映射到不连续的物理 Block，按需分配，避免为最大长度预留连续 KV 空间，降低外部碎片和过度预留，并支持 Block 级共享与 Copy-on-Write。它不改变 Attention 的数学复杂度，也不会自动压缩有效 KV。



### 代价

- Block Table；
- 地址间接访问；
- 尾块内部碎片；
- Allocator 与 Kernel 接口复杂；
- Block Size 需要权衡。



## Q52：KV Block Size 如何选择？



### 建议回答

> 小 Block 减少尾块浪费、提高回收和前缀共享粒度，但 Metadata 更多、分配更频繁、访存更离散；大 Block 有利于连续访问和 Kernel 向量化，但平均浪费约半个尾块，部分前缀也更难复用。应根据长度分布和 Kernel 支持实测。



## Q53：Continuous Batching 为什么提高吞吐？



### 建议回答

> 静态 Batch 要等整批请求完成，短请求结束后产生空槽；Continuous Batching 在每个迭代边界移除完成请求、释放 KV，并加入新请求，使 Active Batch 持续饱和。它依赖 Ragged Attention、动态 Slot Mapping 和 KV 管理。



### 注意

Continuous Batching 是 Iteration-Level Scheduling，不是在任意 Kernel 中途插入请求。

## Q54：为什么需要 Chunked Prefill？



### 建议回答

> 32K Prefill 如果一次执行，可能长时间阻塞 Decode，造成 ITL 尾延迟。Chunked Prefill 将 Prompt 切块，与 Decode 交错调度，限制单轮 Token Budget 和 Workspace。块太小会让 GEMM 变小、Launch 和 Collective 增多；块太大又无法隔离，因此需要按 TTFT/ITL SLO 调整。



## Q55：`max_num_batched_tokens` 如何调？



### 建议回答

> 它控制单轮可调度的 Decode Token 与未缓存 Prefill Token 总量。增大有利于形成大 GEMM，但单轮时间、ITL 和临时显存可能增加；减小改善公平性，却增加小 Kernel 和调度开销。我会使用真实长度分布做 Load Sweep，以满足 p99 TTFT/ITL 的最大 Goodput 为目标。



## Q56：Prefix Cache 需要满足什么条件？



### 建议回答

必须具有相同 Token 前缀，并保证影响隐藏状态的条件一致：

- 模型版本；
- Tokenizer；
- LoRA Adapter；
- Position ID；
- RoPE 配置；
- 多模态输入；
- Prompt Template。

采样 Temperature、Top-p 通常不影响 Prompt KV。

## Q57：SGLang RadixAttention 与 vLLM Prefix Cache 有什么区别？



### 建议回答

> PagedAttention 主要是 KV 的物理存储方式；Prefix Cache 是复用策略。SGLang 使用 Radix Tree 管理可变长共享前缀，并结合前缀感知调度；vLLM 常使用 Block Hash 做自动前缀缓存。两者都可建立在 Paged KV 上，不能脱离版本、模型和 Workload 宣称谁一定更快。



## Q58：KV 即将 OOM 时，调度器怎么办？



### 建议回答

- Admission Control；
- 根据可用 KV Block 和 Token Budget 接纳请求；
- 回收已完成请求；
- Evict 低价值 Prefix Cache；
- 对低优先级请求做 Preemption/Recompute；
- 背压或拒绝超长请求；
- 保留未知输出长度的安全余量；
- 必要时将长请求隔离到独立队列。

CPU Swap 延迟通常很高，不能默认使用。

---



# 9. 量化与投机解码



## Q59：Weight-Only INT8 为什么更适合 Decode？



### 建议回答

> 低 Batch Decode 每步都要读取大部分权重，Weight-Only INT8 相比 BF16 近似减少一半权重流量，因此能缓解带宽瓶颈；前提是反量化与 Dot/GEMM 融合。Prefill 是大 GEMM，收益依赖硬件是否有高效低精度矩阵路径和转换开销。



## Q60：W8A8 与 Weight-Only INT8 有何区别？



### 建议回答

- Weight-Only：权重量化，Activation 保持 FP16/BF16；
- W8A8：权重和 Activation 都量化；
- Weight-Only 更容易部署，主要节省权重带宽；
- W8A8 有机会使用整数矩阵指令，但需要 Activation Scale、校准与高效 Kernel；
- 端到端收益取决于 Cast、Scale 和 Layout 是否融合。



## Q61：FP8/FP4 的关键设计点是什么？



### 建议回答

- 数值格式：如 E4M3、E5M2；
- Weight、Activation、KV 哪些量化；
- Per-Tensor、Per-Channel、Per-Block Scale；
- 静态还是动态量化；
- Accumulator 精度；
- Outlier；
- Packing 与 Alignment；
- 硬件原生支持；
- Norm、Softmax、Router 是否保留高精度；
- 校准集是否覆盖真实长尾 Shape。

不能将模型文件缩小比例直接等同于端到端加速比例。

## Q62：KV Cache 量化和权重量化有什么不同？



### 建议回答

> KV 量化同时降低常驻显存和长上下文 Attention 读带宽，还能提高并发；但每步 Attention 都要读取并反量化，Scale 元数据和 Kernel 支持非常关键。Key 误差影响 Attention Logits，Value 误差影响聚合结果，长上下文检索往往比短文本困惑度更敏感。



## Q63：Speculative Decode 如何工作？



### 建议回答

> Draft 模型先自回归提出 `k` 个候选 Token，Target 一次前向并行验证这些位置。标准 Speculative Sampling 按 `min(1, p/q)` 接受候选，首次拒绝时从修正分布采样，从而保持 Target 分布；Greedy 场景则接受到首个 Argmax 不一致的位置。



### 追问

- 全部接受后的 Bonus Token；
- Draft 与 Target Tokenizer；
- Provisional KV；
- 拒绝后的 KV 回滚；
- 不同请求接受长度形成 Ragged Batch；
- 高 Batch 下为什么收益下降。



## Q64：什么时候 Speculative Decode 会变慢？



### 建议回答

> Draft 不够便宜、接受率低、`k` 太大、Target 验证开销高、Batch 已经很大、Draft 占用过多显存或 Scheduler/KV 回滚开销过高时都可能变慢。不能只看接受率，要看每轮实际产出 Token 与 Draft、Verify、调度总成本。

---



# 10. TP、DP、CP、EP 与 MoE



## Q65：TP、DP、CP、EP 分别解决什么问题？



### 建议回答

- TP：切分层内矩阵，使一个请求跨卡执行，主要解决权重容量和单请求计算；
- 普通 DP：复制模型并处理不同请求，扩展总吞吐和故障隔离；
- CP：切分单条长序列，分摊长上下文 Attention 计算与部分临时 Activation；持久 KV 是否分片取决于实现，某些 Prefill CP 路径会在 All-Gather 后为 Decode 复制 KV；
- EP：不同专家放到不同设备，通过 All-to-All 路由 Token。

一般：

```text
TP/EP 先解决模型放不下
CP 解决超长上下文
DP 扩展服务吞吐
```



## Q66：Column Parallel 与 Row Parallel 如何配合？



### 建议回答

> Column Parallel 按输出维切权重，各 Rank 得到输出 Shard；后续 Row Parallel 按输入维消费这些 Shard，各 Rank 产生部分和，再通过 All-Reduce 或 Reduce-Scatter 合并。成对设计可以避免中间张量反复复制。

TP 过大时：

- Local GEMM 变小；
- Collective 延迟增加；
- 单请求性能不再线性扩展。



## Q67：CP 的 All-Gather 与 Ring Attention 有什么取舍？



### 建议回答

> All-Gather 汇集 K/V 后，每个 Rank 对自己的 Q Shard 做 Attention，简单且能利用高效 Collective，但峰值内存较高。Ring Attention 分步传递 K/V Block，并使用 Online Softmax 合并局部结果，峰值内存更低且可流水，但实现更复杂，有多轮启动和因果负载不均。

Online Softmax 跨块合并必须维护：

- 当前最大值；
- 指数和；
- 加权输出；
- 最大值变化后的重缩放。

不能直接相加各 Rank Softmax 输出。

## Q68：常见 Collective 的语义与通信量是什么？

以下是大消息、Ring 类算法的近似带宽项。设完整逻辑张量为 `N` 字节、Rank 数为 `P`：

- All-Reduce：每 Rank 输入和输出都是 `N`；
- Reduce-Scatter：每 Rank 输入 `N`，输出 `N/P`；
- All-Gather：每 Rank 输入 `N/P`，输出 `N`。

后续公式表示每 Rank 的近似单向发送量；接收量近似相同，不包含协议头、启动延迟和算法差异：

- Ring All-Reduce 每 Rank 发送量约：


2\frac{P-1}{P}N


- Reduce-Scatter：


\frac{P-1}{P}N


- All-Gather：


\frac{P-1}{P}N


- All-to-All：每个 Rank 向不同目标发送不同数据，实际性能强依赖发送矩阵和负载均衡。

术语必须区分：

- Gather：仅 Root 获得完整结果；
- All-Gather：所有 Rank 都获得完整结果；
- All-Gatherv：各 Rank 分片长度可以不同。

除了总字节，还要考虑：

- 消息步数；
- 启动延迟；
- 拓扑；
- 最慢链路；
- Rank/NUMA/NIC 绑定；
- 消息大小；
- 算法选择。



## Q69：MoE 一层的数据路径是什么？



### 建议回答

```text
Router Logits
→ Top-k Expert
→ Token Count / Prefix Sum
→ Token Permute & Pack
→ EP All-to-All Dispatch
→ Local Grouped Expert GEMM
→ All-to-All Combine
→ Unpermute
→ 按 Router Weight 聚合
```

端到端瓶颈不只是 Expert GEMM，还包括：

- Router；
- Token 重排；
- Padding；
- Dispatch/Combine；
- 热专家；
- 最慢 Rank。



## Q70：MoE 负载不均如何分析？



### 建议回答

不能只看平均 Token/Expert，需要看：

- 最大 Token/Expert；
- 方差；
- 每 Rank 总 Token；
- All-to-All Send Matrix；
- 最热专家；
- 最慢 Expert GEMM；
- Padding 比例；
- Straggler Rank。

处理方式：

- Expert Placement；
- 热专家副本；
- 动态负载均衡；
- 跨请求聚合；
- 减少 Padding；
- 调整路由策略，但必须重新验证模型质量。



## Q71：如何分析 RCCL 慢或 Overlap 无收益？



### 建议回答

> 先使用 rccl-tests 按真实消息大小确认通信基线，再检查 XGMI/PCIe/IB 拓扑、Rank—NUMA—NIC 绑定；之后在 Trace 中看 RCCL Kernel、Event 和空洞。Overlap 还需要异步提交、独立计算、资源余量和双缓冲。如果通信与计算都占用 CU/HBM，强行并发可能让两者同时变慢。



## Q72：为什么要做 Prefill/Decode 解耦？



### 建议回答

> Prefill 偏大 GEMM，关注 TTFT；Decode 偏带宽和低抖动，关注 ITL。混部时长 Prefill 会干扰 Decode。PD 解耦允许独立扩缩容和 SLO 隔离，但引入 KV 传输、路由、故障恢复和 Cache Locality 成本。只有减少的排队和干扰超过这些成本时才值得。



## Q73：如何判断 KV 传输是否抵消 PD 收益？



### 建议回答


T_{KV}\approx\frac{KV\ Bytes}{Effective\ Bandwidth}+T_{protocol}


需要比较：

```text
KV 传输时间
vs
解耦后减少的排队与干扰时间
```

还要考虑：

- Advertised Bandwidth 与实际带宽；
- 最慢 Rank；
- Rank-to-Rank 映射；
- 协议和注册开销；
- 是否只传未命中 Suffix；
- 是否可以 Layerwise Streaming；
- 故障后 KV 如何恢复。

---



# 11. GPU、CUDA、ROCm 与 Kernel



## Q74：CUDA Warp 与 AMD Wavefront 有什么区别？



### 建议回答

> CUDA Warp 通常是 32 Lane，MI 系列 CDNA 常见 Wavefront 是 64 Work-Item。相同 256 Thread/Work-Item Block，在 NVIDIA 上是 8 Warp，在 Wave64 上是 4 Wave。归约、Shuffle、Lane Mask、分支发散和每 Wave 的寄存器成本都不同，可移植代码不能硬编码 32。



## Q75：GPU 内存层级如何影响 Kernel？

常见层级：

```text
Register
→ Shared Memory / LDS
→ L1
→ L2 / Last-Level Cache
→ HBM
```

优化顺序通常是：

1. 合并 Global Memory Access；
2. 提高 Tile 数据复用；
3. 避免 Shared/LDS Bank Conflict；
4. 减少 Register Spill；
5. 改善 Cache Locality；
6. 平衡资源占用与 Occupancy。

CUDA 的 Local Memory 往往位于设备内存，不应误认为片上私有缓存。

## Q76：什么是 Coalesced Access？



### 建议回答

> 相邻 Lane 访问连续且对齐的地址，使硬件能够用尽量少的内存事务完成整个 Warp/Wave 的访问。Stride 过大、错位或不规则 Gather 会增加事务数，降低有效带宽。



## Q77：什么是 Bank Conflict？



### 建议回答

> Shared Memory/LDS 被划分为多个 Bank，同一 Warp/Wave 中多个线程访问同一 Bank 的不同地址时，请求可能被串行处理。可以通过 Padding、改变 Layout、转置或向量化减少冲突。广播同一地址是否冲突取决于硬件语义，不能一概而论。



## Q78：Occupancy 越高越好吗？



### 建议回答

> Occupancy 是每个 SM/CU 驻留 Warp/Wave 数相对上限的比例，受线程数、VGPR/SGPR、Shared/LDS 和硬件限制。高 Occupancy 有助于隐藏延迟，但不是最终目标；为了提高 Occupancy 强行减少寄存器可能导致 Spill，计算密集 Kernel 也可能在较低 Occupancy 下达到更高性能。

需要区分：

- Theoretical Occupancy；
- Achieved Occupancy；
- GPU Utilization；
- Pipeline Utilization。



## Q79：Stream 与 Event 的语义是什么？



### 建议回答

- 同一 Stream 内按入队顺序执行；
- 不同 Stream 只具备并发可能，不保证并发；
- Event 表示某个 Stream 时间点；
- 另一个 Stream `wait_event` 建立设备侧依赖；
- `event.synchronize()`/Device Synchronize 会阻塞 Host；
- 高优先级 Stream 不保证任意粒度抢占正在运行的 Kernel；
- PyTorch 跨 Stream Tensor 还要考虑 Allocator 生命周期和 `record_stream`。



## Q80：如何判断 Kernel 是 Compute-Bound 还是 Memory-Bound？



### 建议回答

结合：

- Arithmetic Intensity；
- Achieved FLOPS；
- HBM/L2 带宽；
- Roofline；
- Compute Pipeline Utilization；
- Cache Hit；
- Stall Reason；
- Occupancy；
- Shape Sweep。

只看 GPU Utilization 不足以判断。

## Q81：量化 Kernel 变快，为什么端到端不变？



### 建议回答

可能新增：

- Quant/Dequant；
- Cast；
- Scale 读取；
- Layout Conversion；
- Padding；
- Fallback；
- 更多 Kernel Launch；
- 质量回退导致无法启用。

需要比较：

```text
目标 Kernel
相关转换 Kernel
总 GPU 时间
Wall Time
任务质量
```



## Q82：如何迁移 Warp32 优化到 Wave64？



### 建议回答

- 使用实际 `warpSize`；
- 调整 Shuffle 归约步长；
- 检查 Lane Mask；
- 重新评估 Workgroup Size；
- 重新评估 VGPR/LDS；
- 检查分支发散；
- 用 Known-Answer Test 验证；
- 不直接复制 NVIDIA 的经验参数。

---



# 12. 生产推理服务系统设计



## Q83：设计一个支持长上下文、多租户和流式输出的推理服务



### 第一步：先澄清需求

必须询问：

- 模型大小与结构，Dense 还是 MoE；
- 精度；
- 输入/输出长度分布；
- Prefix Ratio；
- 峰值 QPS 和并发；
- TTFT、ITL、E2E SLO；
- 可用性；
- 成本预算；
- 单机还是多机；
- 是否需要 LoRA、VLM、Tool Calling。

没有数据时使用变量，不编造容量数字。

### 第二步：总体架构

```text
Client
→ Gateway/Auth/Rate Limit
→ Admission Control
→ Model Router
→ Request Queue
→ Scheduler
→ Model Replica
   → TP/CP/EP Workers
   → KV Cache
   → Prefix Cache
→ Streaming Response
```

旁路系统：

- Model Registry；
- 镜像和配置管理；
- GPU Scheduler；
- Metrics/Logs/Trace；
- Benchmark/Replay；
- 灰度发布和回滚。



### 第三步：调度

- Continuous Batching；
- Token Budget；
- Chunked Prefill；
- KV-Aware Admission；
- 长短请求隔离；
- 优先级和 Aging；
- Prefix-Aware Routing；多租户默认使用由 `tenant_id + model_version + tokenizer + adapter` 组成的 Cache Namespace，并对 Hash 加 Salt；
- 超长请求限额；
- 取消请求及时回收 KV。

多租户 Prefix Cache 默认隔离。只有租户明确授权、数据分类允许且审计规则完备时，才可在指定共享域复用；同时需要租户级 KV 配额、缓存计费、公平调度和防止时间侧信道的策略。

### 流式输出与背压

- Gateway 与客户端之间使用有界发送缓冲，不能让慢客户端无限占用内存；
- 客户端消费变慢时实施背压、限速或取消请求；
- 客户端断连要向 Scheduler 传播取消，及时停止生成并回收 KV；
- 明确 Gateway、Server 和单 Token 的超时；
- 首 Token 后发生故障时返回部分响应或错误终止，不能透明重试并声称结果幂等；
- 记录断连率、发送缓冲水位和取消到 KV 释放的延迟。



### 第四步：并行

- TP/EP：模型容量和单请求执行；
- CP：超长上下文；
- DP：扩吞吐；
- PD：只有在干扰与排队收益大于 KV 传输成本时采用。



### 第五步：可靠性

- TP/CP Replica 任一 Rank 失败时隔离整个 Replica；
- Collective Timeout；
- 首 Token 前可安全重试；
- 流式输出后不能承诺无缝重试；
- 模型、框架、算子、通信库使用版本兼容矩阵；
- 镜像不可变；
- 灰度和快速回滚；
- 长时间内存泄漏检查。



### 第六步：指标

- TTFT/ITL/E2E；
- Request/s；
- Input/Output Token/s/GPU；
- Queue Time；
- Active Sequence；
- Batch Token；
- KV 使用率；
- Prefix Cache Hit；
- GPU/HBM；
- Kernel/Collective；
- OOM/Timeout/Restart；
- Goodput。



## Q84：如何估算服务容量？



### 建议回答

只有 Prefill/Decode 使用独立资源池时，才可以分别粗估卡数。并且应使用未被 Prefix Cache 命中的输入 Token：

\[
N_P\ge\frac{\lambda\times E[uncached\ input\ tokens]}
{prefill\ tokens/s/GPU}\times margin
\]

Decode 需要使用满足 ITL SLO 时的实测输出能力：

\[
N_D\ge\frac{\lambda\times E[output\ tokens]}
{decode\ tokens/s/GPU\ under\ SLO}\times margin
\]

其中 `margin > 1` 表示突发、故障和波动余量；也可等价地除以目标利用率 `\rho < 1`，但不要两种安全系数重复计算。Prefill/Decode 混部时共享 GPU，不能把 `N_P` 与 `N_D` 当成两个独立资源池直接求卡数，应使用联合服务曲线或离散事件/压力测试。最终都必须使用真实长度分布和 Open-Loop Load Sweep。

## Q85：70% 请求共享 8K System Prompt，10% 请求为 32K，如何优化？



### 建议回答

> 使用 Paged KV 和 Prefix Cache，并做 Cache-Aware Routing，提高共享 8K 前缀的 Token Hit Rate；多租户场景默认按租户隔离 Cache Namespace，只有授权共享域可以复用。使用 Continuous Batching 与 Chunked Prefill，避免 32K 请求阻塞 Decode；长请求可进入独立队列或 CP Tier；模型容量由 TP/EP 解决，总吞吐由 DP 扩展。如果混部仍违反 ITL，再评估 PD 和 KV 传输，不默认拆分。



### 追问

- 高 Cache Hit 是否会造成节点热点？
- Prefix Cache 如何 Evict？
- 模型升级后如何失效？
- 多租户是否允许共享 KV？
- 32K 请求如何防止饿死短请求？



## Q86：如何做模型和 Kernel 灰度发布？



### 建议回答

```text
离线正确性矩阵
→ 固定请求 Replay
→ 性能回归
→ 长稳测试
→ Shadow
→ Canary 小流量
→ 分阶段放量
→ 全量
```

需要：

- 新旧版本可并行；
- 路由权重可快速切换；
- Prefix Cache 按版本隔离；
- 正确性、延迟和错误率门禁；
- 自动或人工回滚；
- 镜像、模型、配置可追溯。



## Q87：如何处理大请求导致 OOM？



### 建议回答

- 请求长度预校验；
- KV Token Budget；
- Admission Control；
- Chunked Prefill；
- 限制输出上限；
- 长请求队列；
- Prefix Cache 回收；
- 拒绝或降级；
- 预留安全显存；
- 不依赖 OOM 后再恢复。



## Q88：如何避免长请求饿死短请求？



### 建议回答

- Token-Level Fairness；
- 长短队列；
- 每租户配额；
- Chunked Prefill；
- Aging；
- Priority；
- 每轮 Token Budget；
- Preemption；
- 防止长请求永久饥饿。

---



# 13. 行为面试

行为题建议控制在 60～90 秒：

```text
20% 背景
60% 个人行动
20% 结果和反思
```



## Q89：讲一次最有挑战的项目



### 建议回答

> 我认为最有挑战的是 CP-TBO，因为它不是单 Kernel 优化，而是 Attention、Collective、Stream、Buffer 和多 Rank 顺序共同决定性能与正确性。
>
> 我先建立关闭 TBO 的同配置 Baseline，再用 Trace 画出 Producer、All-Gather 和 Consumer 的依赖。之后将 Attention 拆成 Prepare/Finish，调整 All-Gather 时序，并通过 Block Cap 控制通信资源。每项改动都做正确性、性能和多 Shape 回归。最终在固定 32K Prefill 场景中由 7395 提升到 9100 tokens/s/GPU。
>
> 这个项目让我认识到，Overlap 的目标不是让时间线看起来更满，而是缩短端到端关键路径。



## Q90：讲一次失败

不要把“项目很难”当作失败。必须说明自己的错误。

### 真实答案骨架

> 初版 TBO 中，我最初假设【面试前补齐：真实错误假设】。结果【真实负面结果】。我冻结后续功能改动，重新建立 Baseline，并用 Perfetto 对比 Stream 和 Kernel 时间线，发现【证据】。随后通过【受控实验】推翻原假设并完成修复，最终相对初版提升 17%。
>
> 反思是：增加并发不等于增加有效重叠。以后我会在设计阶段先画依赖 DAG，并提前验证资源余量，而不是只验证功能正确。

如果没有真实的个人错误，不要虚构；换一个真实失败案例。

## Q91：讲一次与他人的技术分歧



### 答案骨架

> 在【真实项目】中，我与【角色】对【性能、兼容性、交付周期或维护成本】存在不同判断。我先把争论拆成事实、假设和决策标准，并构造统一的复现 Case 和 Benchmark。数据表明【结果】，最终选择【方案】，同时将短期交付和长期上游合入拆开处理。结果是【真实结果】。

必须补齐真实人物关系和冲突点，不能把普通讨论包装成激烈冲突。

## Q92：讲一次 Ownership



### 建议回答

> CP-TBO 中我不仅完成代码，还负责 Baseline、执行时间线、正确性验证、性能测试和后续回归。对我来说 Ownership 不是提交最多代码，而是能明确问题边界、推动依赖方协同，并对最终结果的可信度负责。

只有确实由你负责的环节才可以说。

## Q93：如何快速学习陌生技术？



### 建议回答

> 我通常使用“源码主线—最小实现—数据验证”三步。学习推理时，我先从 llama.cpp 的 Tensor Layout、RMSNorm、RoPE 和量化 GEMV 建立源码主线，再自己实现模型前向与 KV Cache，最后用 Tiled GEMM 和 WMMA 做性能实验。进入 AMD 推理优化后，我继续用同样方法：先走通 Reference Path，再隔离 AITER/RCCL 能力，最后通过 Dump/Replay 和 Trace 验证。对我来说，能运行和能测量的实现比只阅读资料更能检验理解。



## Q94：你的缺点是什么？



### 建议回答

> 我的实践目前更集中在 Kernel、通信和离线/半在线性能验证，对大规模在线 Serving 的流量治理、跨机容错和资源调度还不够系统。为补齐这一点，我正在系统学习 Paged KV、Scheduler、PD 解耦、Kubernetes 和容量规划，并尝试把性能指标与在线 SLO 联系起来。

不要回答：

- 过于追求完美；
- 工作太认真；
- 没有缺点。



## Q95：为什么应该录用你？



### 建议回答

> 我的匹配点不是只会使用推理框架，而是已经在真实 AMD 环境中处理过框架、算子、通信、Stream 和多 Shape 的交叉问题，也有 C++/CUDA 与运行时优化基础。我能从 Profiler 证据出发形成 Baseline—优化—验证—交付闭环。作为校招生，我仍需要补充更大规模线上系统经验，但可以较快承担推理性能定位、后端适配和 Benchmark 建设类工作。

---



# 14. 反问面试官

最后选择 2～3 个，不要全部询问。

1. 组内当前最核心的瓶颈更偏 Kernel、通信、Scheduler，还是跨硬件交付？
2. 新人前三个月通常负责什么问题，什么结果算达到预期？
3. 团队如何定义推理优化的上线门禁？
4. 正确性、性能和稳定性分别如何验证？
5. 团队如何平衡上游开源版本与内部定制分支？
6. 一个性能问题通常由个人端到端负责，还是由 Kernel、框架和交付团队协同？
7. 当前长上下文和 MoE 场景最难解决的问题是什么？
8. 团队如何维护不同 GPU 或加速卡的兼容矩阵？
9. 对校招生而言，团队更看重系统能力、Kernel 能力还是工程交付能力？
10. 推理与训练基座之间是否共用通信、调度或算子基础设施？
11. 团队怎样评价一个优化是否值得长期维护？
12. 面试官认为我目前最需要补强的能力是什么？

---



# 15. 面试前材料清单



## 15.1 简历口径

- [ ] 确认标准 GPT-2 还是 GPT-2-like；
- [ ] 修正 GELU/SwiGLU、RMSNorm/LayerNorm、RoPE/绝对位置编码；
- [ ] 参数量改为约 124M，或解释自定义模型；
- [ ] `7395 → 9100` 统一为约 23.1%；
- [ ] `_index` 改为 `__index`；
- [ ] TBO 首次出现给出全称；
- [ ] DP、DP Attention、CP、EP 术语统一；
- [ ] AITER、RCCL、内部算子库分类正确；
- [ ] “完整推理引擎”根据实际能力改为“推理原型”；
- [ ] 两段腾讯经历的内部转组时间说明清楚。



## 15.2 TBO/CP-TBO

- [ ] TBO 的完整执行时间线；
- [ ] 两个 Micro-Batch 的划分方式；
- [ ] 重叠的具体通信与计算；
- [ ] 额外 Stream 来源；
- [ ] 争用的具体资源证据；
- [ ] Prepare/Finish 的输入和输出；
- [ ] All-Gather 输入/输出 Tensor Shape；
- [ ] Event Producer/Consumer；
- [ ] Double Buffer 与生命周期；
- [ ] Block Cap 的值和选择依据；
- [ ] 三项改动的 Ablation；
- [ ] +17% 的起止数据；
- [ ] 7395 与 9100 的完整测试条件；
- [ ] 正确性和稳定性阈值；
- [ ] 无收益或回退的 Shape。



## 15.3 Benchmark

- [ ] 模型、精度、Commit、镜像；
- [ ] GPU 数量与拓扑；
- [ ] ROCm、PyTorch、SGLang、AITER、RCCL 版本；
- [ ] TP/DP/CP/EP；
- [ ] ISL/OSL；
- [ ] Batch/Concurrency；
- [ ] Prefix Ratio；
- [ ] Cache State；
- [ ] Warmup 和重复次数；
- [ ] P50/P95/P99 和波动；
- [ ] Token/s/GPU 口径；
- [ ] TTFT/ITL；
- [ ] 正确性指标；
- [ ] 稳定性时长；
- [ ] Dump 脱敏方式。



## 15.4 自研推理原型

- [ ] CPU/GPU 型号；
- [ ] CUDA 与编译器版本；
- [ ] Batch、Prompt、Output Length；
- [ ] 每个版本的精度；
- [ ] 是否包含 Tokenizer、H2D、权重加载；
- [ ] 是否启用 KV Cache；
- [ ] 1400 tok/s 属于 Prefill、Decode 还是端到端；
- [ ] CPU 是否使用 BLAS/多线程；
- [ ] Tiled GEMM 的 Tile；
- [ ] WMMA 的 M/N/K、Layout 和累加精度；
- [ ] Tensor Core 指令证据；
- [ ] 与参考实现的 Logit/Token 误差；
- [ ] llama.cpp 精读 Commit；
- [ ] `block_q8_0` 的实际结构。



## 15.5 行为面

- [ ] 一个真正包含个人错误的失败案例；
- [ ] 一个真实技术分歧；
- [ ] 一个 Ownership 案例；
- [ ] 一个跨团队协作案例；
- [ ] 一个快速学习案例；
- [ ] 一个时间紧张下做取舍的案例；
- [ ] 每个案例能在 90 秒内讲完；
- [ ] 每个结果都能说明个人贡献边界。

---



# 附录 A：十个必须能够白板推导的公式



## A.1 KV Cache


`KV_bytes/token = 2 × L × n_kv × d_h × b`

这是未分片的一份逻辑 KV；单卡物理量需使用本地 KV Head 数和实际复制因子。


## A.2 Dense 模型前向 FLOPs 粗估


`FLOPs/token ≈ 2P`


## A.3 Roofline


`P ≤ min(P_peak, BW_peak × AI)`


## A.4 Arithmetic Intensity


`AI = FLOPs / Bytes`


## A.5 MFU


`MFU = T_global × FLOPs/token ÷ (GPU count × Peak FLOPS/GPU)`

若使用 `tokens/s/GPU`，不要再次除以 GPU 数；只统计真正执行了模型计算的 Token。


## A.6 Ring All-Reduce 每 Rank 通信量


`2(P - 1)N / P`


## A.7 All-Gather/Reduce-Scatter 每 Rank 通信量


`(P - 1)N / P`

这里 `N` 为最终完整逻辑张量字节数，公式是假设 Ring 类算法时每 Rank 的近似单向发送量。


## A.8 KV 传输时间


`T_KV ≈ KV Bytes / Effective Bandwidth + T_protocol`


## A.9 Little 定律


`N = λW`


## A.10 理想两阶段 Overlap


`T_serial = T_C + T_M`



`T_ideal ≈ max(T_C, T_M)`


实际还要加 Pipeline Fill/Drain、同步与资源争用。

---



# 附录 B：回答性能项目的固定模板

```text
1. 场景：
   模型、硬件、精度、并行方式、请求 Shape。

2. 指标：
   关注吞吐、TTFT、ITL、显存还是稳定性。

3. Baseline：
   原始数值、版本和测试方法。

4. 证据：
   Trace、Counter、Kernel Timing 或错误日志。

5. 根因：
   必须说明因果验证，不只描述相关性。

6. 改动：
   具体改了哪个执行边界、Kernel、通信或配置。

7. 验证：
   新增测试、正确性、性能、稳定性和多 Shape 回归。

8. 结果：
   原始数字、波动和适用范围。

9. 局限：
   哪些 Shape 没收益，哪些配置不支持。

10. 贡献边界：
    哪部分由自己完成，哪部分来自团队或开源库。
```

