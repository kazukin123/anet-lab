---
title: NN Forward の設定解釈を事前準備へ移し、用途別実行オブジェクトが State と Network の寿命を持つ
page_kind: サマリ
material: "[[docs/adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md|0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
updated: 2026-11-26
---
# NN Forward の設定解釈を事前準備へ移し、用途別実行オブジェクトが State と Network の寿命を持つ

## 元の材料

[ADR 0051](../../../adr/0051-nn-forward-prepares-concrete-calls-and-purpose-owned-execution.md) blob:5c009e4612c4d0774c9045d7ee093127e9fbca6b

## 要点

- Forward の設定は毎回解釈せず、選択された設定から初期化時に必要な処理を準備する。
- 具象 Module・Head が自身に関係する設定を解釈する。
- 用途別の実行オブジェクトが Network の寿命、準備済み処理、専用状態と時計を持つ。
- 通常の層にまで一律のラッパーを導入しない。
- 後の改訂では BindNetwork による状態共有をやめ、Network ごとに独立した実行器を同じ factory で作る。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [NoisyNetの観測](../../concepts/methods/noisynet.md)
