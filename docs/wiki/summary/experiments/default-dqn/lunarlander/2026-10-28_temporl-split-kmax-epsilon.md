---
title: TempoRL 第二弾の段 2（分割・k_max の重み・スキップ長の ε-greedy）、LunarLander 探索履歴
page_kind: サマリ
material: "[[docs/experiments/default-dqn/lunarlander/2026-10-28_temporl-split-kmax-epsilon.md|default-dqn/lunarlander/2026-10-28_temporl-split-kmax-epsilon]]"
material_kind: 実験記録
concepts:
  - "[[docs/wiki/concepts/methods/temporal_abstraction.md|行動の時間尺度]]"
  - "[[docs/wiki/concepts/subjects/lunarlander.md|LunarLander]]"
  - "[[docs/wiki/concepts/measurement/evaluation.md|評価と比較の条件]]"
updated: 2026-11-26
---
# TempoRL 第二弾の段 2（分割・k_max の重み・スキップ長の ε-greedy）、LunarLander 探索履歴

## 元の材料

[TempoRL 第二弾の段 2（分割・k_max の重み・スキップ長の ε-greedy）、LunarLander 探索履歴](../../../../../experiments/default-dqn/lunarlander/2026-10-28_temporl-split-kmax-epsilon.md) blob:0219473ae6ff9cb540fdda7e25146601e4f4f869

## 要点

- 3 群とも基準以上で、基準を上回る群は無い。
- `kmax` は長い j を増やさなかった。
- `[greedy]`（行動もスキップ長も greedy）の報酬は `[eval]`（行動は UQE τ 0.25）より低い Run がある。
- 分割で primary に入る skip の loss の勾配は、skip Network の勾配より大きく、クリップにはほぼ掛からない。
- 基準の 3 本は 10/26 の同 seed の Run と最初から違う。
- 3 seed のブレ幅は分散を小さく見積もる。

## 内容

### 探索ブロック 01: 合算の基準に対する分割・`kmax`・ε-greedy の 3 seed 比較と `[greedy]` session の読み

`17`・`18` は分割のときだけ値を持つ（`stop`・`merge` では NaN で出力されない）。

PRD103 §9 の 1: LunarLander の既定は据え置き（上回る群が無い）。

PRD103 §7 の段 3: `kmax` が基準以上だったので、段 3 の群に分割・`kmax` を足す。

### 探索ブロック 02: 基準の 3 本と 10/26 の同 seed の Run の照合

GridMaze 6×10 では、10/28 の朝のビルド（`[greedy]` session を足す前）の UQE 3 本が 10/26 の 3 本と完全に一致した（2026-10-28_temporl-skip-policy-6x10）ので、違いは PRD103 の実装のあとに入った設定（`[greedy]` session）による。

### pending

`[greedy]` session を足したときに同じ seed の軌跡が変わる経路の特定（同じ設定の 2 本が一致するか、`[greedy]` を外すと 10/26 と一致するか）。PRD103 §5.7 に事実だけ記した。

## 主要な概念

- [行動の時間尺度](../../../../concepts/methods/temporal_abstraction.md)
- [LunarLander](../../../../concepts/subjects/lunarlander.md)
- [評価と比較の条件](../../../../concepts/measurement/evaluation.md)
