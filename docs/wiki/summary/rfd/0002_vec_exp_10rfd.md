---
title: Ape-X型 空間探索（Vectorized Exploration）機能の追加
page_kind: サマリ
material: "[[docs/memo/done/0002_vec_exp_10rfd.md|0002_vec_exp_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/framework/321_agent_core.md|321 Agent 共通基盤]]"
updated: 2026-11-26
---
# Ape-X型 空間探索（Vectorized Exploration）機能の追加

## 元の材料

[Ape-X型 空間探索（Vectorized Exploration）機能の追加](../../../memo/done/0002_vec_exp_10rfd.md) blob:b8f4131a362d24b002cc1e600740c7081e2b241f

## 要点

- DefaultDQNAgent の並列環境ごとに、異なる探索の強さを割り当てる。
- 空間探索を有効にすると、時間経過による ε または τ の減衰を止める。
- 最小探索の値は先頭の環境に置き、TrainView の既定表示でその行動を追えるようにする。
- UQE と ε-greedy は環境ごとのパラメータで行動を選ぶ。
- 既定は無効とし、従来の時間方向の探索を使用できる。

## 内容

### 概要

時間経過による探索パラメータ（$\epsilon$ や $\tau$）のアニーリングを無効化し、並列環境（`num_envs`）ごとに異なる固定のパラメータを空間的に割り当てる「空間的アニーリング」を導入する。これにより、Replay Bufferの多様性と学習の安定性を向上させる。

### 外部仕様（Config拡張）

`*_end`: インデックス `0`（最小探索環境）のパラメータ値。TrainViewが既定表示するlaneを最小探索側にする。

### 内部仕様

空間探索有効時は、時間経過によるパラメータの減衰（アニーリング）が存在しないため、無駄な計算をスキップする。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [Replay ratioとlane窓](../../concepts/methods/replay_ratio_lane_window.md)
- [321 Agent 共通基盤](../../concepts/framework/321_agent_core.md)
