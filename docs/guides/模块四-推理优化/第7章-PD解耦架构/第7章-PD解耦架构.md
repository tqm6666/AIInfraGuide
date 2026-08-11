---
title: "第7章：Prefill/Decode 解耦架构"
description: "从混合 Batching 互扰到 P/D 解耦：KV 传输、Goodput、池配比与 vLLM Disaggregated Prefill 实战"
pubDate: 2026-04-16
category: "inference-optimization"
order: 36
tags: ["P/D解耦", "DistServe", "Splitwise", "KV Cache传输", "Goodput", "SLO"]
---

## 本章简介

前六章把单实例与分布式扩展主线钉死了：分页 KV、连续批处理、vLLM 调度、量化、投机，以及 TP/PP/DP 装载与吞吐。再往前走，交互式服务常撞上另一类矛盾——**Prefill 与 Decode 同池混跑时，TTFT 与 TPOT 互相挤压**。Chunked Prefill（[2.4](../第2章-推理引擎核心技术/2.4-Chunked%20Prefill%20与统一调度.md)）能削峰，却消不掉阶段特性差异；当双 SLO 同时很紧、负载长短混杂时，才会认真评估把两阶段拆到不同资源池。本章不承诺「解耦必涨吞吐」，而是讲清互扰账、传输税、Goodput 目标与配比校准，并落到 vLLM Disaggregated Prefill。

**混合 Batching 的问题**用带单位的示意账说明：同一步里 Prefill 拉长步墙钟，Decode 的瞬时 TPOT 如何被抬高，以及保护 Decode 时 TTFT 如何反噬。

**解耦架构设计**抽出 DistServe / Splitwise / TaiChi / Microserving 的共通骨架：P/D 角色、状态转移、KV 交接；明确解耦换来什么、运维多了什么。

**KV Cache 传输与 Connector**做字节粗算与带宽门槛，介绍 Connector 抽象与后端选型——传输税吃掉隔离收益时，应先优化通路而非盲目加卡。

**Goodput 与 SLO 感知调度**把 Raw QPS 与「满足双 SLO 的有效吞吐」分开，并讨论接纳控制等调度取舍。

**解耦挑战与配比**给出 $N_p:N_d$ 手算直觉与上线前挑战清单；配比以 Goodput 压测校准，而非经验比例定终身。

**vLLM Disaggregated Prefill 实战**给最小可跑路径与验收指标：互扰因子、分段 TTFT、配比扫描表。

## 本章章节

- **7.1 混合 Batching 的问题**：TTFT/TPOT 互挤的定量直觉与压测钉死法
- **7.2 解耦架构设计**：P/D 角色、状态转移、论文路线与运维代价
- **7.3 KV Cache 传输与 Connector**：字节粗算、带宽门槛、Connector 抽象
- **7.4 Goodput 与 SLO 感知调度**：有效吞吐目标与调度取舍
- **7.5 解耦挑战与配比**：复杂度清单、$N_p:N_d$ 手算与决策
- **7.6 vLLM Disaggregated Prefill 实战**：最小可跑路径与验收指标
