<!-- translated-from: README.jp.md blob:80d32423befc341b6fe6042218539fa246d5aca6 date:2026-09-19 progress:done -->
# ANET Documentation

> Primary perspective: overall structure (guide to documents organized by function and workflow)

## 1. Introduction

### 1.1 Purpose

This document is the entry point for documentation on using, analyzing, and developing ANET. It briefly introduces the framework and lists documents that cover the details.

### 1.2 Audience

- Users running training or evaluation with ANET
- Users analyzing Run results
- Developers modifying ANET itself, Agents, Envs, or applications

### 1.3 Scope

This document provides an overview of ANET and an index of the documents under `docs/design/`. Follow the links for details on configuration, operation, and implementation design.

## 2. ANET Overview

ANET is a C++20 reinforcement learning experiment framework built on libtorch. It constructs an Env and Agent from configuration and runs training and evaluation through the wxWidgets-based `AnetRLRunner`. The GUI shows runtime state, and metrics recorded for each Run can be analyzed with the Java/Spring-based Metrics Viewer and supporting tools.

The currently verified runtime environment is Windows 11 x64 with NVIDIA CUDA. Other operating systems and CPU-only configurations have not been verified for the framework as a whole, even where code paths exist. See the [ANET Framework Overview](010_framework_overview.en.md) for the overall structure and environment requirements.

## 3. Document Index

Document numbers normally increase in steps of 10 to leave room for later additions. `0xx` covers the overview, usage, and development environment; `1xx` covers shared framework design; and `2xx` covers concrete implementation specifications. The `.jp.md` suffix denotes the Japanese edition. English editions are added as `.en.md` files with the same number and base name.

### 3.1 Overview

| Number | Document | Main topics | Audience |
|---:|---|---|---|
| - | This document | Introduction, system overview, document index | Everyone |
| 010 | [ANET Framework Overview](010_framework_overview.en.md) | Basic concepts, overall structure, feature list, main processing flows | Everyone |

### 3.2 User Guides

The user guides follow the sequence of tasks performed by users.

| Number | Document | Main topics | Audience |
|---:|---|---|---|
| 020 | [Run Execution Guide](020_user_guide_run.en.md) | Writing configuration files, startup, screens, basic operations, Run artifacts | Users running experiments |
| 030 | [Run Analysis Guide](030_user_guide_analysis.en.md) | Metrics, Metrics Viewer, Run comparison, external analysis tools | Users analyzing results |

### 3.3 Development Guide

| Number | Document | Main topics | Audience |
|---:|---|---|---|
| 040 | [Development Environment Setup Guide](040_development_environment.en.md) | Windows/MSVC, dependencies, builds, tests | Developers |

### 3.4 Design Guides by Functional Category

The design guides are divided by functional category. Within each document, the relevant processing stages are explained chronologically.

| Number | Document | Main topics | Audience |
|---:|---|---|---|
| 100 | [Runtime Infrastructure and Configuration](100_runtime_and_configuration.en.md) | Startup, configuration resolution, Run construction, lifecycle | Framework developers |
| 110 | [Agents and Learning](110_agents_and_learning.en.md) | Shared contracts and ownership for Agent, Actor, and Learner | Agent and framework developers |
| 120 | [Environments](120_environments.en.md) | Env, BatchEnv, Reset, Step, Env implementations | Env and framework developers |
| 130 | [Neural Networks](130_neural_networks.en.md) | NetworkModel, modules, forward, optimizer | NN and Agent developers |
| 140 | [Observability](140_observability.en.md) | Event, Observer, metrics, visualization, profiling | Framework and analysis feature developers |
| 150 | [ReplayBuffer](150_replay_buffer.en.md) | Experience, N-step, PER, transfer, prefetch | Agent and performance developers |
| 160 | [Applications and Tools](160_applications_and_tools.en.md) | Runner GUI, Metrics Viewer, supporting tools | Application developers |

### 3.5 Concrete Implementation Specifications

These specifications describe component structure, configuration, data structures, and internal contracts for individual implementations built on the shared design. They cover both Agent families and applications with independent execution units.

| Number | Document | Main topics | Audience |
|---:|---|---|---|
| 200 | [DQN Agents](200_dqn_agents.en.md) | DefaultDQN, Rainbow, shared DQN components, learning, synchronization, PER | DQN Agent developers |
| 210 | [Metrics Viewer](210_metrics_viewer.en.md) | Ingestion, cache DB, range queries, rendering, configuration, dependencies | Metrics Viewer developers |
| 220 | [Atari Env (ALE Integration)](220_atari_env.en.md) | AtariEnv, ALE configuration key contracts, preprocessing, presets, AtariView | Env developers and Atari experiment users |

## 4. Related Documents

- [Project README](../../README.md)
- [Domain Glossary](../../CONTEXT.md)
- [Agent Implementation Ownership Guidelines](../ownership_guideline.md)
- [Architecture Decision Records](../adr/)
- [Implementation Plans and Investigation Notes](../memo/)
- [ANET Overview PDF](../anet_overview_ja.pdf)
