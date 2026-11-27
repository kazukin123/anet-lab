---
title: 分位tail探索診断メトリクス
page_kind: サマリ
material: "[[docs/memo/done/0048_quantile_tail_metrics_10rfd.md|0048_quantile_tail_metrics_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/framework/317_metrics.md|317 可視化・メトリクス]]"
updated: 2026-11-26
---
# 分位tail探索診断メトリクス

## 元の材料

[分位tail探索診断メトリクス](../../../memo/done/0048_quantile_tail_metrics_10rfd.md) blob:5ffb555ebd1156e022e58e600545d9891315c930

## 要点

- QR と IQN の分位列で、中央値より上・下の裾の広がりを対称に測る。
- 分位の所属は予測値を並べ替えず、評価点 τ の順番で決める。
- 奇数個の分位では中央値自身を除き、上下に同数の点を使う。
- 中央値からの二乗偏差の平均の平方根を用い、Q 値と同じ単位で表す。
- Policy と Learner の分布で6つの scalar を取り、行動選択・loss・PER の抽選には用いない。

## 内容

### Problem Statement

下側tailを負のbonusとして扱った場合に、risk-neutralな行動選択がどの程度変わり得るか。
DLTVのupper-tail variabilityは、分布が非対称な場合に全分散よりも探索側の幅を直接捉えられる。ただし、upper-tail幅だけではdownside riskを捉えられず、return distributionの幅にはparametric uncertaintyとintrinsic uncertaintyが混在する。また、分布情報を直ちにPolicy scoreや内的報酬へ組み込むと、QR / IQNの基本評価と探索方式の効果が分離できなくなる。

### Solution

QR / IQNに共通するtau順のquantile列から、medianを基準とする上下対称のtail幅を定義する。これをPolicy側のfull distributionとLearner側のcurrent distributionへ適用し、6個のscalar keyとして公開する。

### Implementation Decisions

Policy側は、QRでは既存`q_quantiles`、IQNではfixed full queryの`full_q_quantiles`を診断用full distributionとして選択する。診断のためのforwardやtau生成を追加しない。
`BatchUpdateResult::GetScalar()`の新しい観測interfaceは`upper_tail_priority_spearman`とする。PER無効時は計算用readbackを増やさず`NaN`を返す。
metric登録はDropMergeへ6本だけ追加し、すべてEMA `0.01`、`interval:100`とする。raw、Train、`eval1`、LunarLanderへの重複登録は行わない。

### 検証方針

テストは内部payloadの配置ではなく、公開`GetScalar()`、公開Learner更新経路、設定解決結果から観測できる振る舞いを検証する。production codeへtest-only APIを追加しない。
選択actionにcrossingがない場合、および全quantileが同値でrangeが`0`の場合、`policy_selected_crossing_depth_p90_ratio`が`0`になることを検証する。`K < 2`は他のPolicy側quantile診断と同じく`NaN`とする。
minibatch sizeが`2`未満、upper-tail幅が定数、raw priorityが定数、PER無効の場合に`NaN`になることを検証する。

### Out of Scope

`mean + sigma_+`、`mean - sigma_-`、上下を混合したscoreによる実際のPolicy変更。
metricのTrain Policy版、`eval1`版、raw版、LunarLander既定tag。

### Further Notes

Policy側metricはfixed full queryを使うため、IQNのrisk queryに使ったtau集合そのものの幅ではない。行動scoreと独立した観測用full distributionの形を測る。
`upper_tail_priority_spearman`はPERにより既に偏って抽出されたminibatch内の相関である。ReplayBuffer全体の冗長性や、upper-tail priorityを採用した場合の性能を直接示さない。

## 主要な概念

- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [317 可視化・メトリクス](../../concepts/framework/317_metrics.md)
