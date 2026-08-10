---
title: "第3章：深入 vLLM 架构与源码"
description: "深入 vLLM 的整体架构、V1 引擎设计、调度器源码与关键配置调优，并横向对比 SGLang、TensorRT-LLM"
pubDate: 2026-04-16
category: "inference-optimization"
order: 32
tags: ["vLLM", "V1引擎", "Scheduler", "源码导读", "SGLang", "TensorRT-LLM"]
---

## 本章简介

vLLM 是本模块的实例主线。前两章讲过的 PagedAttention、Continuous Batching、Prefix Cache、Chunked Prefill，本章不再重复深讲机制，而是落到 vLLM 的真实组件、调度与调参里看它们如何被组织；最后用可执行的 POC 对照 SGLang 与 TensorRT-LLM，帮助你做框架选型。

**快速入门**用最短路径跑通安装、离线批量推理与 OpenAI 兼容服务，并给出 OOM / Chat 乱码等排错顺序；原理细节挂回第 2 章。

**整体架构**拆解 V1 分层：`LLM` / `AsyncLLM` 入口、`EngineCore` 三拍循环、`Scheduler`、`KVCacheManager`、`Worker` / `ModelRunner`，用"请求的一生"串数据流，并给出症状→层级的排障定位表。

**V1 引擎**讲清多进程隔离、统一 Token 预算、近零开销 Prefix Cache、Persistent Batch 各自换来什么、付出什么；官方约 1.7x 吞吐附带基线与负载前提，避免当成无条件承诺。

**调度器源码导读**抓住 Waiting/Running、先 Running 再 Waiting 的非显然设计、Token Budget 手算，以及抢占后 `num_computed_tokens` 归零一类可检验点。

**关键配置调优**围绕显存利用率、并发序列数、每步 Token 预算与 `block_size`，强调参数耦合、症状→旋钮速查与单变量压测顺序。

**框架横向对比**用统一维度与可操作决策树对照 vLLM / SGLang / TensorRT-LLM，并用同一流量做最小 POC——匹配场景，而不是评出永远的第一名。

## 本章章节

- **3.1 vLLM 快速入门**：安装、离线推理、在线服务与排错
- **3.2 vLLM 整体架构**：分层地图、请求一生、排障定位
- **3.3 V1 引擎深度解析**：多进程、统一调度、Prefix Cache、Persistent Batch 与 1.7x 前提
- **3.4 调度器源码导读**：Token Budget、双队列顺序、抢占与 Recompute
- **3.5 关键配置调优**：容量/并发/步预算旋钮、耦合与单变量实验
- **3.6 框架横向对比**：vLLM / SGLang / TensorRT-LLM 决策树与 POC
