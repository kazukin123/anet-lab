---
title: Replay優先度更新にはgeneration付きreplay item keyを使用する
page_kind: サマリ
material: "[[docs/adr/0011-generation-aware-replay-item-key.md|0011-generation-aware-replay-item-key]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# Replay優先度更新にはgeneration付きreplay item keyを使用する

## 元の材料

[ADR 0011](../../../adr/0011-generation-aware-replay-item-key.md) blob:d9d7dfd9721fbff47d64c6ab2fc50273eebd9b55

## 要点

- 抽選後にリングの slot が別の経験で上書きされると、物理 index だけでは誤った経験に優先度を返しうる。
- Sample 時の経験をそのまま学習することと、別の経験を更新することを区別する。
- 世代と物理 slot を1本の int64 キーに符号化し、抽選と優先度の返却で往復させる。
- 現在の世代と一致する要素だけ反映し、過去の世代は破棄数を記録する。
- キーの寿命は生成元 ReplayBuffer に限る。更新順序の暗黙の FIFO 契約や旧 index の互換経路は持たない。

## 内容

### 上書きと識別

generation は real・dummy のいずれも slot を書くたびに増える。丸めた実容量を基数に、世代と位置を一つのキーとして符号化する。

### 更新時の扱い

同じバッチに古いキーが混ざっても有効な分は反映する。古いキーは木、優先度の種別、最大値を変更しない。キーは保存用の ID として直列化しない。

## 主要な概念

- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
