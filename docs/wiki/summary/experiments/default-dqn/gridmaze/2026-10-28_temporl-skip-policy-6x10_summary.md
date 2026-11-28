---
title: TempoRL のスキップ長の方策（UQE 対 ε-greedy）、GridMaze 6×10 探索履歴
page_kind: サマリ
material: "[[docs/experiments/default-dqn/gridmaze/2026-10-28_temporl-skip-policy-6x10.md|default-dqn/gridmaze/2026-10-28_temporl-skip-policy-6x10]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/subjects/gridmaze.md|GridMaze]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-28
---
# TempoRL のスキップ長の方策（UQE 対 ε-greedy）、GridMaze 6×10 探索履歴

## 元の材料

[TempoRL のスキップ長の方策（UQE 対 ε-greedy）、GridMaze 6×10 探索履歴](../../../../../experiments/default-dqn/gridmaze/2026-10-28_temporl-skip-policy-6x10.md) blob:6783506a15be112a9d6b6e61e5448e132806f7a2

## 要点

- ε-greedy は UQE と同じ速さで届き、同じく最後まで保った。
- ε-greedy では、6×10 の方策はほぼスキップしない形に落ち着いた。
- ビルドの差は無い。 2026-10-28 のビルドの UQE 3 本は、10/26 の 3 本（PRD098 §7 の段 1）と届く速さ・終盤・判断率が完全に一致した（探索ブロック 01）。
- 評価 policy の EMA（`21_eval/04_policy_reward_ema`）が初めて 0.5 以上になった step は、UQE 237k／238k／273k、ε-greedy 270k／247k／252k。
- 試した両方式・各3 seed の最終 EMA はいずれも +0.93 だった。

## 内容

### 探索ブロック 01: スキップ長の方策を UQE と ε-greedy で比べる（6×10、3 seed）

10/26 の UQE 3 本（`t98_gm_sep_s1〜3`）は、届く速さ 269,056／322,816／358,656、最後の EMA 0.93、判断率 0.77／0.64／0.80 で、2026-10-28 の UQE 3 本と同じ値だった。

## 主要な概念

- [行動の時間尺度](../../../../concepts/methods/temporal_abstraction.md)
- [GridMaze](../../../../concepts/subjects/gridmaze.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
