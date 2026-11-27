---
title: 可塑性 weight norm（feature/readout 分割）PRD
page_kind: サマリ
material: "[[docs/memo/done/0063_plasticity_weight_norm_10rfd.md|0063_plasticity_weight_norm_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
updated: 2026-11-26
---
# 可塑性 weight norm（feature/readout 分割）PRD

## 元の材料

[可塑性 weight norm（feature/readout 分割）PRD](../../../memo/done/0063_plasticity_weight_norm_10rfd.md) blob:2c260764400cf4e418c22d5b2f902ebf64adad84

## 要点

- Breakout の RR8 では、評価低下と同時に srank の低下や dormant unit の増加は見られなかった。
- 3.5M〜5M の窓では q_max が約4%増にとどまり、feature_norm は約26%増えた。
- 特徴を作る側と、その特徴を値へ読み出す側に学習パラメータを分け、それぞれの L2 ノルムを測る。
- 単純な body と head の境界で分けず、特徴への依存関係に基づいて所属を決める。
- bias を含む全学習パラメータを数え、活性のノルムだけでは区別できない変化を追う。

## 内容

### Context（背景・目的）

Q(s,a) ≈ w・φ(s) の観点で「Q 横ばい + ‖φ‖ 成長」は、w が縮んでいるのか φ が Q に寄与しない方向へ成長しているのかの二択だが、現メトリクスはパラメータ側を測っておらず分離できない。パラメータノルムを φ を作る側（feature）と読み出す側（readout）の 2 群で出し、この帰属を閉じる。

## 主要な概念

- [可塑性指標と成績](../../concepts/phenomena/plasticity.md)
- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [Breakout](../../concepts/subjects/breakout.md)
