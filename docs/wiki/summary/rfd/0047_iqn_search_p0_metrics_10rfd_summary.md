---
title: IQN探索P0診断メトリクス
page_kind: サマリ
material: "[[docs/memo/done/0047_iqn_search_p0_metrics_10rfd.md|0047_iqn_search_p0_metrics_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
updated: 2026-11-28
---
# IQN探索P0診断メトリクス

## 元の材料

[IQN探索P0診断メトリクス](../../../memo/done/0047_iqn_search_p0_metrics_10rfd.md) blob:f8b5013f92b7494fa3253b319873d7050ad5e471

## 要点

- IQN 探索の実験で、最終報酬だけでは判断できない要因を同じ Run に記録する。
- DropMerge では Double Suika の生成数と達成有無をエピソード終端で測り、終端外は非イベントとして扱う。
- 方策では既存の分位を使い、上位行動の score gap と推定のばらつきの関係を測る。
- Learner と PER の診断も既存の forward と損失計算の値を使い、学習結果や乱数消費を変えない。
- 探索に要る指標だけの設定グループを設け、全 scalar を有効にする負荷を避ける。

## 内容

### Problem Statement

`q_std` はreturn distributionの幅であり、τサンプリングによる推定誤差そのものではない。

### Solution

Train/Evalの双方で`exp_step`軸に記録する。生成数はraw値を残し、達成有無は達成率としてEMAを残す。episode終了外の`NaN`は非イベントを0件として扱わないための契約である。
標準偏差はfloat32・不偏分散で計算する。`random`ではMonte Carlo平均の安定度、`fixed` / `stratified`では積分解像度のproxyとして解釈する。IQN+UQE以外、または`K < 2`では`NaN`とする。

### Implementation Decisions

IQN loss計算内に既に存在する`B x N x M`の`delta`を再利用し、診断のためにpairwise Tensorを再生成しない。
`BatchUpdateResult`、`DQNActionInfo`、`DropMergeEnv::GetScalar()`のキーを新しい観測interfaceとする。既存キーの名称と意味は変更しない。

### 検証方針

IQN診断計算をCPU Tensorの手計算値と比較する。`N != M`、相殺率`0`、相殺率`1`、sample lossの`/N`正規化を含める。
first maskが0件の場合に`iqn_first_*`が`NaN`、`per_sample_initial_count`が`0`になることを検証する。

### Further Notes

`iqn_priority_mc_ratio`は現行priority式を変更せず、その平均TD信号を有限τのscaleと比較する。`iqn_first_pair_abs_td`と`iqn_first_cancellation_ratio`は、将来の「初回優先度のquantileペア考慮」を議論するための観測根拠である。
`fixed` / `stratified`ではforwardごとのMonte Carlo分散を直接測る指標ではないため、`*_mc_*`は積分解像度proxyとして解釈する。この制約をRun分析ガイドにも明記する。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
