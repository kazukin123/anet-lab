---
title: Atari / 可塑性の保護機構 screening（新バンドル・RR8 アッセイ）
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-08-30_protection-screening.md|default-dqn/atari/2026-08-30_protection-screening]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# Atari / 可塑性の保護機構 screening（新バンドル・RR8 アッセイ）

## 元の材料

[Atari / 可塑性の保護機構 screening（新バンドル・RR8 アッセイ）](../../../../../experiments/default-dqn/atari/2026-08-30_protection-screening.md) blob:1f4bb8aedb82c4118abb1ff65665ad05483192ad

## 要点

- 5 群試して、成績で対照を有意に上回った群はゼロ。
- 機序指標は大きく動く。 `01_dormant_ratio` は SiLU で −10.9σ、LN512 で −9.0σ。`45_probe_srank_ratio` は LN512 で +14.2σ。表現の健全性は明確に改善できる（探索ブロック 03）。
- 機序側だけなら LN512 が総合 1 位。
- Spectral Norm には「①が止まる範囲」と「発散しない範囲」の両立点が無い。
- 判定の物差しは「落差」ではなく「終端の絶対値」。
- zero-init residual（`init2.mode`）は未検証のまま。

## 内容

### 探索ブロック 01: Spectral Norm の適用範囲

ただし SN 自体は設計どおり動いていた。 発散前（0〜28k 勾配）で PRD が予測した三点証拠が揃う。

### 探索ブロック 02: screening 4 群の成績

機序の各段を叩いた群が、崩壊アッセイの成績を動かすか。判定は対照 4 本の平均 ± 標準偏差に対する z。

`51_eval1/10_game_score_mean` の 1M 窓平均。

C2 は「落差は同等だが到達点が低い」という別の負け方。 1-2M で 62.0（対照 168.9 ± 25.7 に対し −4.2σ）と立ち上がりが遅い。`spectral` は常時除算なので初期に全 residual conv の実効重みが σ≈1.89 で割られて 5 割強に縮み、かつ C2 は `init2=he` で残差枝が最初からフルに効く。D2（cap + zero-init）が序盤で対照と同じなのと対照的である。

### 探索ブロック 03: 機序指標と成績の乖離

表現の健全性は明確に改善できる。 SiLU は③を完全に消し（dead が厳密ゼロ、dormant −10.9σ）、LN512 は実効ランクを +14.2σ 押し上げた。複製間の一致も良い（LN512 の dormant −8.8 / −9.0、srank +14.2 / +13.7）。再現性のある実効果である。

## 主要な概念

- [可塑性指標と成績](../../../../concepts/phenomena/plasticity.md)
- [Breakout](../../../../concepts/subjects/breakout.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
