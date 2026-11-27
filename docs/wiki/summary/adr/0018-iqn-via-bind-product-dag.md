---
title: IQN を bind `*` DAG 汎用化と最小 Head で表現する
page_kind: サマリ
material: "[[docs/adr/0018-iqn-via-bind-product-dag.md|0018-iqn-via-bind-product-dag]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
updated: 2026-11-26
---
# IQN を bind `*` DAG 汎用化と最小 Head で表現する

## 元の材料

[ADR 0018](../../../adr/0018-iqn-via-bind-product-dag.md) blob:fa13041fb5aece46670d8a90db1167648ad94b3a

## 要点

- IQN の分位埋め込みと特徴量の融合は既存の branch DAG の設定で表す。
- taus を観測辞書の固定キーとして渡し、CosineEmbedding と Linear/ReLU を組み合わせる。
- bind の要素積で特徴と分位埋め込みを融合し、Head は最終射影を担う。
- 専用 IQNNetwork に融合を閉じ込める案と比べ、既存 block を差し替えやすい。
- 未使用入力の診断は局所的な参照を調べる。最終出力への到達性までは保証しない。

## 内容

### 配線と検証

要素積は feature を末尾にそろえ、rank が異なる場合は batch の直後に長さ1の次元を補う。taus は IQN の入力 spec へ明示して、分位ブランチと特徴ブランチの結合を設定に示す。

### 責務

Head に埋め込みや乗算まで固定すると、既存の構造ブロックを再利用しにくい。ネットワークの未使用入力検査とアルゴリズム側の検査は各々の局所的な契約で行う。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
