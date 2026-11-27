---
title: Actor Q ヒントを常時 5 列にし、価値の分解の Head ごとの値を運ぶ。列の数はモードで変えない
page_kind: サマリ
material: "[[docs/adr/0075-actor-q-hint-five-columns-carry-per-head-values.md|0075-actor-q-hint-five-columns-carry-per-head-values]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
updated: 2026-11-26
---
# Actor Q ヒントを常時 5 列にし、価値の分解の Head ごとの値を運ぶ。列の数はモードで変えない

## 元の材料

[ADR 0075](../../../adr/0075-actor-q-hint-five-columns-carry-per-head-values.md) blob:13e14b4bc24e8db0a07fb16b2dcd22057c2bdedd

## 要点

- 価値分解時の PER は Head ごとの TD 誤差から優先度を求める。
- Actor の近似初期優先度も同じ構造にできるようヒントを常に5列にする。
- 先頭3列は行動価値、状態価値、Munchausen 項、末尾2列は内的 Head の行動価値と状態価値である。
- 分解を使わないときは後ろを0とし、従来の初期優先度の値を保つ。
- モード別に列数を変えず、不正な列数は互換の読み替えをせずエラーにする。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
