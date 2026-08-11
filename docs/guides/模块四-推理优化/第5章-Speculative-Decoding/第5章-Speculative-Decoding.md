---
title: "第5章：Speculative Decoding"
description: "理解投机解码的 Draft + Verify 与 Rejection Sampling，比较 N-gram / 独立 Draft / Self-Draft，并划清接受率边界与 vLLM 实战验收"
pubDate: 2026-04-16
category: "inference-optimization"
order: 34
tags: ["Speculative Decoding", "投机解码", "Medusa", "EAGLE", "N-gram", "Rejection Sampling"]
---

## 本章简介

前四章分别解决了装请求、喂饱 GPU、读懂 vLLM 调度，以及用量化砍比特。Decode 仍常卡在"一步只推进约 1 个 Token"。本章换一条杠杆：**用便宜草稿提议多 Token，再让 Target 并行验证**——在规则正确时保持与 Target 采样同分布。重点放在机制、草稿来源、收益边界与 vLLM 配置验收；调度/KV/量化细节只在叠加处挂接前文，不重复深讲。

**核心原理**钉死 Draft + Verify 与 Rejection Sampling 的保真直觉，并用带毫秒单位的粗算说明：接受长度不够或 Draft 不便宜时，投机可以净亏损。

**Draft 模型与 N-gram**比较独立小模型与无模型匹配两条路：前者付显存换通用接受率，后者在复述/代码/模板上常是 ROI 冠军；$\gamma$ 需要按场景扫参。

**Self-Draft**覆盖 Medusa 多头与 EAGLE-2/3 动态 Draft Tree：把"猜"内化进 Target 周边，用树降低线性草稿早断的风险，并承认训练/引擎门槛。

**收益边界与限制**分高/低接受率场景，解释单请求加速推不到高并发的原因，以及与量化、Continuous Batching 叠加时的缩水与关闭条件。

**vLLM 实战**给出 ngram / draft_model / EAGLE 的配置骨架（以当前文档为准），并用代码 vs 对话对照实验做路由化取舍，而不是承诺统一加速比。

## 本章章节

- **5.1 核心原理**：Draft + Verify、Rejection Sampling、加速粗算与不变量
- **5.2 Draft 模型与 N-gram**：独立小模型选型、无模型匹配、$\gamma$ 与显存账
- **5.3 Self-Draft 方案**：Medusa、EAGLE-2/3、Draft Tree 与递进选型
- **5.4 收益边界与限制**：场景表、并发缩水、量化/批处理张力、关闭决策
- **5.5 vLLM 投机解码实战**：三类配置骨架、分桶对照、灰度与监控
