---
title: IQN の初回 Learner 優先度を分位点対から求める案
page_kind: サマリ
material: "[[docs/memo/0147_iqn_first_priority_quantile_pair_10rfd.md|0147_iqn_first_priority_quantile_pair_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/methods/tau_sampling.md|τ配置とIQN比較]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
updated: 2026-11-26
---
# IQN の初回 Learner 優先度を分位点対から求める案

## 元の材料

[RFD0147　IQN の初回 Learner 優先度を分位点対から求める案](../../../memo/0147_iqn_first_priority_quantile_pair_10rfd.md) blob:eca94d16d41147cbbf027760c8b9724abffe42ae

## 要点

- 初回の優先度計算は探索候補としての暫定案である。式、適用範囲、設定名は裁定前である。
- 現行方式は予測と目標の分位点をそれぞれ平均し、その差の絶対値を優先度の基にする。
- 分位点対の誤差に正負が混在すると、個々のずれが大きくても平均後の値は小さくなる。
- 最初の Learner 更新で各対の絶対誤差を集約する候補を設け、希少な遷移がもう一度抽選されるかを調べる。
- 投入時の Actor 推定優先度と Learner が上書きする優先度を分け、既存の相殺指標も読む。

## 内容

### Problem Statement

現行IQN Learnerは、experienceのpriority更新にcurrent quantileの平均とtarget quantileの平均の差を使う。current quantileを`z[b,i]`、target quantileを`y[b,j]`、pairwise TDを`delta[b,i,j] = y[b,j] - z[b,i]`とすると、clip前のpriority信号は次に相当する。

### Solution

IQNの初回Learner priority更新行に限り、現行の平均TD priorityとquantile-pair由来priorityを切り替えられる実験設定を追加する。
初回更新以外は現行の平均TD priorityを維持する。IQN loss、target、optimizer、初期priority方式、Replay sampling、importance sampling weightは変更しない。

### Implementation Decisions

同一replay itemが同じminibatchへ重複して現れた場合は、各行のsample時点sourceに従う。ReplayBufferの重複更新順序とgeneration検査は変更しない。
stale updateは従来どおり棄却する。計算時に初回行だったことを理由に、上書き済みslotへpriorityを適用しない。
IQN loss計算で既に生成する`delta[B,N,M]`から、RFD0047で既に計算済みの`pair_abs_td[B]`を再利用する。

### Testing Decisions

手計算できる`N != M`のTensorで`mean_td`と`pair_abs_td`を検証する。
正負が完全相殺し、`mean_td=0`かつ`pair_abs_td>0`になるケースを検証する。
`per_eps=0`、clip無効、clip境界未満・等値・超過、`per_alpha=0`を含め、既存変換が一度だけ適用されることを検証する。

### Out of Scope

初回以外を含む全Learner更新へのpair方式適用。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [τ配置とIQN比較](../../concepts/methods/tau_sampling.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
