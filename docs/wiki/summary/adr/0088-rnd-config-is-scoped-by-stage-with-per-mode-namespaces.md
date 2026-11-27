---
title: RND の設定は、段階ごとのスコープとモード別の名前空間で持ち、被るキーは割り切る
page_kind: サマリ
material: "[[docs/adr/0088-rnd-config-is-scoped-by-stage-with-per-mode-namespaces.md|0088-rnd-config-is-scoped-by-stage-with-per-mode-namespaces]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/316_config.md|316 設定管理]]"
  - "[[docs/wiki/concepts/methods/intrinsic_novelty.md|内的報酬による探索の比較]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-26
---
# RND の設定は、段階ごとのスコープとモード別の名前空間で持ち、被るキーは割り切る

## 元の材料

[ADR 0088](../../../adr/0088-rnd-config-is-scoped-by-stage-with-per-mode-namespaces.md) blob:8b6a8da68873b2ab8429018143b00bbcd650ebdd

## 要点

- RND と補助学習の設定を統計、誤差、predictor、補助学習の処理ごとに分ける。
- 選べる処理には mode とモード別の名前空間を設ける。
- 待つ件数や clip など、単位や適した値が異なるものはモード別に設定する。
- 非選択モードのキーも読み取って値域を検査するが、休止だけを理由に通知しない。
- 各モード内の有効・無効は列挙値を増やさず真偽で表す。

## 主要な概念

- [316 設定管理](../../concepts/framework/316_config.md)
- [内的報酬による探索の比較](../../concepts/methods/intrinsic_novelty.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
