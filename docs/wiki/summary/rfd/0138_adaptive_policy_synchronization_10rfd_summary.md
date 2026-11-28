---
title: Train Actor の方策同期を適応させる案
page_kind: サマリ
material: "[[docs/memo/frozen/0138_adaptive_policy_synchronization_10rfd.md|0138_adaptive_policy_synchronization_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/measurement/diagnostic_cost.md|計測の負荷と比較条件]]"
updated: 2026-11-28
---
# Train Actor の方策同期を適応させる案

## 元の材料

[RFD0138　Train Actor の方策同期を適応させる案](../../../memo/frozen/0138_adaptive_policy_synchronization_10rfd.md) blob:ca94ae9563793ca873f957c3e701f224c255addf

## 要点

- 適応同期は凍結中である。固定の cosine スケジュールを長期 Run で評価し、それを上回る根拠が出たときに検討を再開する。
- 同じ同期間隔でも学習率や1環境 step あたりの更新回数によって Actor の古さは異なる。
- DropMerge の条件では200 train step が最大約500 Learner 更新に相当し、同期境界をまたぐエピソードも多かった。
- 方策やパラメータのずれを観測して同期時点を変える案を比較する。
- 最短・最長の間隔を設ける案を含むが、ずれの定義、設定名、採用方式は未決である。

## 内容

### 背景

PRD 036では、DefaultDQN Train ActorがLearner online networkを直接共有せず、privateなnetwork snapshotを一定期間使う機能を追加した。

### 起草時点の現時点の観測

snapshot intervalを長くすると、Train側でNoDropを経験する件数は増えた。
interval 200と400では、Eval側NoDropはshared networkより少なかった。

### 問題

現在のDropMerge設定では1 `train_step`あたり概算2.5 learner updateであるため、interval 200は最大約500 learner update、interval 400は最大約1000 learner updateのlagになる。

### 目的

固定時間ではなく、Actor snapshotとLearner online networkの差に基づいて同期を判断できるようにする。
model driftが大きいときは短周期、driftが小さいときは長周期になる制御を可能にする。

### 暫定解決方針

`min_interval`: behavior networkを最低限継続使用する期間。
`cosine_restart`はlagを周期的に急変させるため、採用根拠が得られるまで使用しない。

### 所有権と依存方向

source online network、optimizer、共有drift集計ResourceはAgentが所有する。
learner updateに伴うdrift Stateは、実際に更新するLearner側moduleが所有する。

### 設定候補

interval単位を`train_step`のまま維持するか、learner update ageも選べるようにするか。
不正な閾値、非finite値、`min_interval > max_interval`はfail-fastする。

### 候補を採る場合の検証方針

v1が追加action forwardを行わないことをmock networkのforward回数で確認する。
configの不正mode、非finite threshold、範囲逆転をfail-fastする。

### 一般事例との位置付け

Ape-XはActor networkをLearnerから周期的に更新するが、adaptive drift thresholdを中心機構にはしていない。
IMPALAはtrajectory開始時にActor方策を更新し、残るpolicy lagをV-traceで補正する。

### 採用判断

現行`ProfiledValue`による一方向cosineを長期Run候補として評価する。
snapshot interval、learner update age、drift、episode跨ぎをどう観測するか決める。

## 主要な概念

- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
- [Replay ratioとlane窓](../../concepts/methods/replay_ratio_lane_window.md)
- [計測の負荷と比較条件](../../concepts/measurement/diagnostic_cost.md)
