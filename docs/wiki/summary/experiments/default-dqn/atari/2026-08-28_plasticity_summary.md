---
title: Atari / 可塑性メトリクスの較正と保護機構の初回評価
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-08-28_plasticity.md|default-dqn/atari/2026-08-28_plasticity]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
updated: 2026-11-28
---
# Atari / 可塑性メトリクスの較正と保護機構の初回評価

## 元の材料

[Atari / 可塑性メトリクスの較正と保護機構の初回評価](../../../../../experiments/default-dqn/atari/2026-08-28_plasticity.md) blob:c630b33f8d5a3e435afe6e5222e54378104b3d26

## 要点

- 摩耗は勾配 step の関数で、`replay_ratio` にほぼ依存しない。
- 膨らむのは feature 側で readout ではない。
- `dead_ratio` の谷が転換点を予告する。
- srank は今回の損傷モードを見ない。
- RR8 / 5M / 旧バンドルは 54 分で回る崩壊アッセイになる。
- `probe.batch_size` は隠れた測定パラメータで、`dead_ratio` を 5 倍変える。
- GroupNorm 単独は悪化させ、weight_decay は `weight_norm_feature` を用量順に制御するが崩壊は止めない。

## 内容

### 探索ブロック 01: RR ラダーでの器の初回検証

反復のブレ幅（r1 / r2、非決定 backend）: srank ±2%、`q_max` ±5%、`dead` / `feature_norm` ±10〜19%。RR8 の崩壊は eval ピークが同一窓・同値（65.0 / 64.2）で再現した。

### 探索ブロック 02: コスト A/B

需要駆動化（購読された統計だけ計算する変更）の後、probe batch 4096 で srank の interval だけを 10 分の 1 にした比較では、ドリフト補正後で coarse が +1.09% 速い（control 3 本と coarse 3 本が完全分離、p≈0.05）。SVD は batch 4096 のペナルティ 6.81% のうち約 1/6 で、支配的なのは probe forward 側である。

### 探索ブロック 04: 正規化 × weight_decay

GroupNorm 単独は明確に悪化した。 2 本ともピーク・終端の両方で対照より下、`dead` は 2 倍。ピーク位置も早い。

## 主要な概念

- [可塑性指標と成績](../../../../concepts/phenomena/plasticity.md)
- [計測の負荷と比較条件](../../../../concepts/measurement/diagnostic_cost.md)
- [Breakout](../../../../concepts/subjects/breakout.md)
- [Replay ratioとlane窓](../../../../concepts/methods/replay_ratio_lane_window.md)
