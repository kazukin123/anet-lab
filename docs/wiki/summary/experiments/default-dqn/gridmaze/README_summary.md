---
title: DefaultDQNAgent / GridMaze 探索記録
page_kind: サマリ
material: "[[docs/experiments/default-dqn/gridmaze/README.md|default-dqn/gridmaze/README]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/framework/322_env_individual.md|322 Env個別実装]]"
updated: 2026-11-28
---
# DefaultDQNAgent / GridMaze 探索記録

## 元の材料

[DefaultDQNAgent / GridMaze 探索記録](../../../../../experiments/default-dqn/gridmaze/README.md) blob:747096376ae75f55cc9c403645794b5aa56993c5

## 要点

- DefaultDQNAgent を GridMaze で評価した資料の索引である。
- TempoRL のスキップ方策は 6×10 の迷路で比較した。
- 比較したのは UQE による長さ選択と ε-greedy による選択で、各 3 seed である。
- この比較の記録は 2026-10-28 の campaign にある。
- RFD0098 の初期検証と RFD0183 の追加検証の正本は各 RFD の検証節にある。

## 主要な概念

- [行動の時間尺度](../../../../concepts/methods/temporal_abstraction.md)
- [322 Env個別実装](../../../../concepts/framework/322_env_individual.md)
