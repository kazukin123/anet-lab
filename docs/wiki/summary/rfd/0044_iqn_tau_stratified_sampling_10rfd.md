---
title: IQN tau sampling modes(stratified / systematic / antithetic)
page_kind: サマリ
material: "[[docs/memo/done/0044_iqn_tau_stratified_sampling_10rfd.md|0044_iqn_tau_stratified_sampling_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
updated: 2026-11-26
---
# IQN tau sampling modes(stratified / systematic / antithetic)

## 元の材料

[IQN tau sampling modes(stratified / systematic / antithetic)](../../../memo/done/0044_iqn_tau_stratified_sampling_10rfd.md) blob:f02ad5d46c7c0b4c818651163cd76147c62720d1

## 要点

- IQN の τ 配置を変え、評価点の総数と forward の形状をそろえたまま被覆とランダム性を比較する。
- stratified は区間を分けて各区間から選び、τ の偏った集中を抑える。
- systematic は等間隔の点を共通のランダムな位置ずれで動かす。
- antithetic は対になる点を使い、既存の random と fixed も比較対象に残す。
- 学習結果の優劣を先に決めず、DropMerge などで計算量をそろえた条件を組めるようにする。

## 内容

### Problem Statement

`random` は各 forward で τ を独立に一様抽出するため、IQN の連続的な分位関数学習に必要なランダム性を持つ。一方で、有限個の τ が特定区間へ偏り、1回の minibatch 内で分位範囲の一部が未被覆になることがある。DropMerge の Double Suika 達成間際のような希少遷移では、その遷移が ReplayBuffer から選ばれる回数自体が少ないため、偏った τ 集合が割り当てられると学習機会を十分に活用できない懸念がある。
`fixed` は指定範囲を等幅区間へ分割し、その中点を使うため全域を安定して被覆するが、forward ごとの τ のランダム性を失う。これは QR 寄せの control として有用だが、IQN が持つ連続分位関数の確率的な学習という性質を弱める。

### Solution

IQN の `sample_mode` に `stratified`、`systematic`、`antithetic` を追加し、現行契約を `random | fixed | stratified | systematic | antithetic` とする。
per-env 下限 `τ_min[b]` を使う場合も同じ正規化位置を用い、各行の `[τ_min[b], τ_max]` へ写像する。antithetic の鏡映は正規化空間で行うため、対称中心は行ごとの範囲中点 `(τ_min[b] + τ_max) / 2` となる。

### Implementation Decisions

DefaultDQNの全tau ruleは3つの新modeを受理する。既定値は一切変更せず、既存configの解決結果を維持する。

### 検証方針

テストは内部の演算手順ではなく、生成Tensorと設定境界から観測できる契約を検証する。
既定configのmodeとtau数が変更されていないことを既存のdefault testで継続確認する。

### Out of Scope

loss駆動の τ 重点サンプリング(τ ビンごとのloss履歴から非一様サンプル+IS重み補正。FQF-lite)と、FQF型 fraction proposal network。いずれも hot path へのフィードバック経路や追加networkを要するため別PRDとする。

### Further Notes

`antithetic` の分散低減理論は単調な分位関数を条件として Q/target の期待値推定に適用できる。現行IQNはquantile crossingを許すため、実モデルでの効果とQR lossの勾配への効果はいずれも経験的検証事項である。この区別を評価時に混同しない。

## 主要な概念

- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
