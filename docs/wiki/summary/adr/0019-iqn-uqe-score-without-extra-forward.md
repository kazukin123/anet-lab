---
title: IQN+UQE の Q 系値は同一 forward の risk-biased action score を使う
page_kind: サマリ
material: "[[docs/adr/0019-iqn-uqe-score-without-extra-forward.md|0019-iqn-uqe-score-without-extra-forward]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
updated: 2026-11-26
---
# IQN+UQE の Q 系値は同一 forward の risk-biased action score を使う

## 元の材料

[ADR 0019](../../../adr/0019-iqn-uqe-score-without-extra-forward.md) blob:3d9fc733570b9cae51d9e5a8b2cefc35c40db2d4

## 要点

- IQN＋UQE は行動選択用の1回の forward にリスクを反映した分位を渡す。
- 設定に応じて上側区間の平均または指定分位1点を行動スコアにする。
- Q 系メトリクスと Actor Q ヒントもこのスコアを使う。
- 全分布の診断を有効にした場合は同じ forward に別の分位を連結し、出力を分ける。
- 全分布の出力を行動スコアに混ぜず、期待値だけを求める追加推論も行わない。
- 全分布の照会は既定で無効。両方の分位を処理するときは、1回の推論でも分位方向の出力幅が増える。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
