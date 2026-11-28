---
title: Q 値統計による多環境の設定異常の検出案
page_kind: サマリ
material: "[[docs/memo/0153_q_statistics_config_screening_10rfd.md|0153_q_statistics_config_screening_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/target_update.md|Target更新の比較]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
  - "[[docs/wiki/concepts/subjects/breakout.md|Breakout]]"
updated: 2026-11-28
---
# Q 値統計による多環境の設定異常の検出案

## 元の材料

[RFD0153　Q 値統計による多環境の設定異常の検出案](../../../memo/0153_q_statistics_config_screening_10rfd.md) blob:45fcb9c936b374a97f3f5ddf247187e4a6c04dd9

## 要点

- 複数環境での Q 統計には実測があるが、その数値を設定異常の一般的な判定に使う案は未確定である。
- Breakout のターゲット更新の比較では、最適な係数の選択には実際の成績が必要だった。
- 最適値を予測する問いと、明らかに不健全な設定を早期に検出する問いを区別する。
- Q 値の大きさや推移を候補指標にして、崩れた条件と健全な条件を比較する。
- 一つの環境で成立した閾値をそのまま使わず、異なる報酬尺度や環境でも判別できるか確かめる。

## 内容

### Context / Problem Statement

ハイパラの最適値は環境の関数だが、多環境へ展開すると人手探索が `O(env × 群)` で破綻する。Atari-5 なら 5 ゲーム × 4 点 = 20 Run、フル 57 ゲームなら論外である。

### 実測

成績は逆 U 字（頂点 0.008）。`11_q_max_mean` は τ=0.001 だけが高原（11.0〜11.7、幅 ±3%）から 17% 低く、0.008 以降は飽和する。`03_q_delta_abs` は τ とともに単調増加する。遅すぎ側と速すぎ側で別の指標が反応し、最適点は「`q_max` が飽和し、`q_delta` がまだ立ち上がっていない膝」にある。

### 案

開始点でアンカーして符号と env スケールから独立させる。
全 env をスイープせず、事前分布で 1 本走らせ、ガードが鳴った env だけスイープする。

### 限界

予算不足と交絡する。 「まだ伸びている」は「τ が遅すぎ」でも「予算が短い」でも起きる。走り切ってからの事後監査であり、早期検出ではない。
判定ではなくフラグである。 スイープが要る env を絞るのが役割で、τ の値そのものは決められない。

### 着手条件

Munchausenが TD target に `α·τ·log π` を加算するため、Q のスケールが変わる。 本 RFD の閾値は全て Munchausen 前の Q 統計で校正しており、実装後に校正し直しになる。

## 主要な概念

- [Target更新の比較](../../concepts/methods/target_update.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
- [評価と比較の条件](../../concepts/measurement/evaluation.md)
- [Breakout](../../concepts/subjects/breakout.md)
