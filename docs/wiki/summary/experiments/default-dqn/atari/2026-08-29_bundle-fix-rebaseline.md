---
title: Atari / 設定修正後の新バンドル再ベースラインと RR の勾配軸決着
page_kind: サマリ
material: "[[docs/experiments/default-dqn/atari/2026-08-29_bundle-fix-rebaseline.md|default-dqn/atari/2026-08-29_bundle-fix-rebaseline]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
  - "[[docs/wiki/concepts/phenomena/plasticity.md|可塑性指標と成績]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
updated: 2026-11-26
---
# Atari / 設定修正後の新バンドル再ベースラインと RR の勾配軸決着

## 元の材料

[Atari / 設定修正後の新バンドル再ベースラインと RR の勾配軸決着](../../../../../experiments/default-dqn/atari/2026-08-29_bundle-fix-rebaseline.md) blob:eb24abb7c41428c7e1ae48f608bd1361e2258b06

## 要点

- 設定修正 2 件は実効になったが、RR1 Breakout の軌道は動かさなかった。
- 崩壊アッセイは新バンドルでも成立する。
- 新バンドルはピークが約 2 倍。 2-3M 窓で 323.7 / 239.7 に対し旧バンドルは 165.8。BTR 寄せと今回の修正のどちらが効いたかは本記録では分離していない（探索ブロック 01）。
- RR1 は 50M でまだ登り坂。 終盤 2 窓で 454.4 / 426.6、単発最高 797。40-50M で初めて 700 超が 3 本出た（探索ブロック 02）。
- RR4 は 5-10M で天井を打ち、以後 40M 走っても動かない。
- RR4 の天井位置は RR8 の崩壊点と同じ勾配座標にある。

## 内容

### 探索ブロック 01: 崩壊アッセイのバンドル移行

2/2 で崩壊し、ピーク窓が旧バンドルと一致した。 ピーク高の 323.7 / 239.7 は −26% 差で、既知の変動幅（±26%）と整合する。

### 探索ブロック 02: RR1 の新アンカー（50M）

可塑性メトリクス（learn_step 軸。`exp = grad × 256 + 0.2M`）。

srank が初めて意味のある大きさで動いた。 RR8 アッセイでは δ 3 種 × probe batch 2 種で −3〜6% にとどまり「この損傷モードを見ない」と結論していたが、RR1 の長期では単調 −20% である。ユニット死とは別のランク浸食が長時間スケールで進んでいる可能性があり、RR8 アッセイはこのモードを再現しない。保護機構の screening をアッセイのみで判定してはならない。

### 探索ブロック 03: RR4 の天井（50M）

RR4 は 4 倍の勾配 step と 2 倍の実時間を使って、RR1 の 2/3 で頭打ちになる。 摩耗はすべての指標で約 2 倍。BTR の RR=4 を保護機構なしで写すことに利がないことが、新バンドルでも確認された。

## 主要な概念

- [Breakout](../../../../concepts/subjects/breakout.md)
- [可塑性指標と成績](../../../../concepts/phenomena/plasticity.md)
- [Replay ratioとlane窓](../../../../concepts/methods/replay_ratio_lane_window.md)
