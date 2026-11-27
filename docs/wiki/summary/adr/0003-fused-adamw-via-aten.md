---
title: optimizer step を ATen `_fused_adamw_` 直接呼び出しの FusedAdamW に置き換える
page_kind: サマリ
material: "[[docs/adr/0003-fused-adamw-via-aten.md|0003-fused-adamw-via-aten]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/313_profiling.md|313 プロファイリング]]"
updated: 2026-11-26
---
# optimizer step を ATen `_fused_adamw_` 直接呼び出しの FusedAdamW に置き換える

## 元の材料

[ADR 0003](../../../adr/0003-fused-adamw-via-aten.md) blob:119ebbc434761fb0b2fa04b4eb3d5637ac32443e

## 要点

- 小型 MLP の DQN 学習では AdamW のテンソルごとの演算が多数の CUDA カーネルを起動していた。
- FusedAdamW は LibTorch の AdamW を継承し、ATen の fused 演算を直接呼ぶ。
- 既存の options と optimizer state を引き続き使える。
- foreach で演算を組む案は起動回数と実装量が増える。
- 厳密な比較には従来経路を使う。演算順の差があり、内部 API は LibTorch 更新時に確認する。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [313 プロファイリング](../../concepts/framework/313_profiling.md)
