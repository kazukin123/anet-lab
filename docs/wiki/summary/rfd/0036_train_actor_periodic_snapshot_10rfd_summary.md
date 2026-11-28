---
title: DefaultDQN Train Actor定期network snapshot PRD
page_kind: サマリ
material: "[[docs/memo/done/0036_train_actor_periodic_snapshot_10rfd.md|0036_train_actor_periodic_snapshot_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/target_update.md|Target更新の比較]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
updated: 2026-11-28
---
# DefaultDQN Train Actor定期network snapshot PRD

## 元の材料

[DefaultDQN Train Actor定期network snapshot PRD](../../../memo/done/0036_train_actor_periodic_snapshot_10rfd.md) blob:9d0d0f88456c015f9735eb17eb7429418d2d3afe

## 要点

- DropMerge の連続行動中にネットワークの更新が与える影響を、探索設定を保ったまま調べる。
- 学習用 Actor に Learner の online network の parameter と buffer を定期的に複製する選択肢を加える。
- 同期間隔は経験 step に応じた ProfiledValue で表す。
- 共有ネットワークを既定にし、別の評価環境や追加の推論を必要としない。
- Actor の resource を複製するかどうかは、Runner からの指定がなければ各 Agent が判断する。

## 内容

### 問題

DropMergeのTBO/UQE spatial explorationでは、ENV laneごとの`epsilon`と`tau`を維持したまま、near-greedy laneで連続行動中のnetwork変化を切り分けたい。Eval専用Actor、別ENV batch、追加network forwardを導入すると、評価条件への特化とGPUコストが増える。既存Train Actorのnetworkだけを一定期間固定し、周期的にLearner online networkから更新できるopt-in機能が必要である。

### 目的

shared networkを既定として既存Runのaction選択forward回数、保存形式、Train動作を維持する。
snapshotの現在周期とageを、既存の`$action_info`メトリクス経路から観測できるようにする。

### 対象範囲

周期snapshot機能の対象は`DefaultDQNAgent`のTrain Actorだけとする。
ImageClsのTrain Actorはshared networkによるリアルタイム反映を維持する。episodeの概念がなく、同期を遅らせる理由がないため、設定と同期実装を変更しない。

### 解決方針

Run途中で初めてTrain Actorを生成するlifecycleはv1の対象外とする。将来その経路を追加する場合は、生成時点の`train_step`を初期age基準として渡す契約を別途設計する。
条件成立時はsource online networkのparameterとbufferをsnapshotへcopyし、`last_sync_train_step = step.train_step`とする。

### GPU copyとprofiling契約

CUDA Event、別stream、明示的な`cudaDeviceSynchronize`や`torch::cuda::synchronize`を追加しない。

### 受入条件

DefaultDQN Trainの明示`true` / `false`がAgent設定をoverrideし、Evalの明示指定が従来どおり機能する。ImageCls、Rainbow、MuZeroにもoptional signatureを反映するが、既存のclone解釈を変更せず、新しいfail-fastを追加しない。

### テスト方針

CUDA D2Dの非同期完了時刻を必須CIテストにはしない。CUDA挙動は既存`Network::CopyTo()`の契約とNsight Systemsで確認し、テストのためのproduction同期を追加しない。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [Target更新の比較](../../concepts/methods/target_update.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
