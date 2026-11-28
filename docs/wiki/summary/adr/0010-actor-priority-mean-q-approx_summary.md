---
title: Actor初期優先度は平均Q空間の近似とし、Learner target-network評価/UQE targetの再現を見送る
page_kind: サマリ
material: "[[docs/adr/0010-actor-priority-mean-q-approx.md|0010-actor-priority-mean-q-approx]]"
material_kind: ADR
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
updated: 2026-11-28
---
# Actor初期優先度は平均Q空間の近似とし、Learner target-network評価/UQE targetの再現を見送る

## 元の材料

[ADR 0010](../../../adr/0010-actor-priority-mean-q-approx.md) blob:1de9a00614ab0f636b96da04c473ee65685f8313

## 要点

- Actor による replay 初期優先度は行動選択の online 出力を使う近似である。
- Learner と同じ target network 評価のための追加 forward は行わない。
- 当初は平均 Q に基づく行動価値と状態価値を想定した。IQN＋UQE では行動選択と同じ risk-biased score を使う。
- 後にヒントは5列になり、強制行動を含む最終行動を Actor が確定した上で計算する。
- UQE の target を再現する案は、順位相関と初期優先度の効果に明確な不足が出たときに再検討する。

## 内容

### ヒントの意味

オンラインで計算済みの Q から実行行動の値と状態価値を取り出す。Learner の target network と異なるため、両者が同じ順位を付けることまで保証しない。

### 再訪の条件

UQE 構成で優先度どうしの順位相関が明確に弱く、近似の効果も不十分なら再検討する。行動マスク導入時には、無効行動を最大値に含めるかを改めて扱う。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
