---
title: NN Head の最終射影と出力変換の分担
page_kind: サマリ
material: "[[docs/memo/0165_nn_head_projection_separation_10rfd.md|0165_nn_head_projection_separation_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/315_nn.md|315 ニューラルネットワーク]]"
  - "[[docs/wiki/concepts/methods/noisynet.md|NoisyNetの観測]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# NN Head の最終射影と出力変換の分担

## 元の材料

[RFD0165　NN Head の最終射影と出力変換の分担](../../../memo/0165_nn_head_projection_separation_10rfd.md) blob:269c028cc0d6b9d553be57f668fd43b3b55550d8

## 要点

- NoisyNet の検討から分けた暫定案である。最終射影の所有を変えるか、演算を共用するだけで足りるか、対象 Agent と公開契約を含めて未裁定である。
- Body の Linear と Head 内の最終射影で、構築・設定・検証の経路が分かれている。
- NoisyNet のように両方へ適用する演算を増やす際、同じ変更が複数の場所に必要になる。
- 最終射影と、射影後の Q や分位点への変換をどう分担するかを比較する。
- 既存 Head が重みを持つ構成も意図された設計として扱い、変更による重複の削減と移行費用を比べる。
- NoisyNet は既存の射影所有のまま先行し、この裁定を前提にしない。

## 内容

### 問題と目的

新しい射影機能を追加するとき、Body の Linear と Head 内の Linear で、構築・設定・検証の経路が分かれている。NoisyNet の検討では両方へ同じ演算を適用する必要があり、演算の共用に加えて、射影の所有や設定経路まで揃える価値があるかが論点となった。

### User Stories

射影機能を追加する実装者として、同じ演算を Body と Head に組み込むための重複実装や検証を減らしたい。
Agent を利用する実験者として、構造整理によって Q・分位・logits の shape、出力キー、精度、診断の意味が意図せず変わらないことを確認したい。

### 正式化への昇格条件

次を裁定し、選択しなかった案の理由も残してから、正式な RFD・実装計画へ進む。
最終射影を含む精度境界と、通常 forward / 診断経路での実現方法。

## 主要な概念

- [315 ニューラルネットワーク](../../concepts/framework/315_nn.md)
- [NoisyNetの観測](../../concepts/methods/noisynet.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
