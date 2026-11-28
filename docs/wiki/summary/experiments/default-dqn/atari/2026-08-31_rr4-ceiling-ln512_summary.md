---
title: "Atari Breakout: RR4 天井と LayerNorm — 2026-08-31"
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-08-31_rr4-ceiling-ln512.md|default-dqn/atari/2026-08-31_rr4-ceiling-ln512]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
updated: 2026-11-28
---
# Atari Breakout: RR4 天井と LayerNorm — 2026-08-31

## 元の材料

[Atari Breakout: RR4 天井と LayerNorm — 2026-08-31](../../../../../experiments/default-dqn/atari/2026-08-31_rr4-ceiling-ln512.md) blob:fdf8fc13ff53fc97a8cd39c226ee3cffe481492f

## 要点

- LayerNorm(Linear512 直後)は RR4 の天井を大きく押し上げた。
- RR1 では eval 終端平均が対照と同等だった。
- しかし eval1 は壁突破を検出できない。
- train 側で見ると RR1+LN512 は対照の 2.8 倍の突破率だった。
- スコア分布は 432 点に壁がある。 400-449 に集中し、475-624 がほぼ空で、その上に高得点が散在する。Breakout の 1 画面は 18 列 ×(1+1+4+4+7+7) = 432 点である（探索ブロック 04）。
- 勾配ステップ軸を揃えると RR1 が RR4 を大きく上回る。

## 内容

### 探索ブロック 01: RR4 + LN512 の 50M

先行 20M Run との一致は良い（5-10M 328.3 対 331.1、10-15M 350.5 対 352.0、15-20M 355.4 対 358.2）。

### 探索ブロック 02: RR1 + LN512 の 50M

到達速度は 2.5M 窓で明確に差がある。RR1+LN512 は 15.0-17.5M で 437.9、対照が同水準に届くのは 37.5-40.0M（424.8）である。

### 探索ブロック 03: RR1 対照の相転移

37.5M で一段上がり、以降維持されている。`≥700` の出現は 37.5-50M プールで 4/49、それ以前は 0。

### 探索ブロック 04: スコア分布の壁

400-449 に集中し、475-624 がほぼ空で、その上に散在する。旧バンドル RR1 100M の 50M 以降（n=78）も同型で、400-449 が 54 本、475-624 が 3 本、625-874 が 11 本である。

### 探索ブロック 08: 勾配ステップ軸をそろえた比較

同一 grad で RR1 が 110 点上回る。60-80k 帯では差が無く、分岐は 80k-150k で起きている。

## 主要な概念

- [可塑性指標と成績](../../../../concepts/phenomena/plasticity.md)
- [Breakout](../../../../concepts/subjects/breakout.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
- [Replay ratioとlane窓](../../../../concepts/methods/replay_ratio_lane_window.md)
