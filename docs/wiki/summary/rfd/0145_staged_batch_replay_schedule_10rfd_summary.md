---
title: DQN のバッチサイズとリプレイ比率を途中で切り替える案
page_kind: サマリ
material: "[[docs/memo/0145_staged_batch_replay_schedule_10rfd.md|0145_staged_batch_replay_schedule_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/replay_ratio_lane_window.md|Replay ratioとlane窓]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-28
---
# DQN のバッチサイズとリプレイ比率を途中で切り替える案

## 元の材料

[RFD0145　DQN のバッチサイズとリプレイ比率を途中で切り替える案](../../../memo/0145_staged_batch_replay_schedule_10rfd.md) blob:c34a2c520ed1300d253595bf553f7f2ef355acc7

## 要点

- 長期 Run の成績は観測記録であり、それを自動の切り替えに使う案は暫定である。正式な設定名、既定値、切り替え時点は未決である。
- DropMerge の長期 Run で、継続時に batch size と replay ratio を段階的に変えた系列が得られた。
- 学習の段階に応じて標本数と更新頻度を変える効果を、一つの Run の中で確かめる案である。
- 同じ経験数でも Learner の更新回数と一度に見る経験数が変わるため、両方を記録する。
- 比較では切り替え時点、バッチサイズ、リプレイ比率と元の設定を区別する。

## 内容

### Problem Statement

DropMerge の長期 Run では、結果的に batch size と replay ratio を段階的に変更した lineage が成立した。
loss 実装が静的 batch size を shape の正本にすると、動的 batch に追従できない。
Agent checkpoint を別 Run で load すると Run-local `exp_step` が再開せず、schedule が初期 phase へ戻る。

### Solution

DefaultDQN Learner に、batch size と replay ratio を一組として解決する複合 ReplayUpdateSchedule を導入する案を採用候補とする。
ReplayUpdateSchedule は `constant` と `phased` を持つ。
phase ごとの batch size の増減を許容し、利用目的に合った schedule の選択は利用側の責任とする。
phase が変わった場合は、ReplayBuffer prefetch と遅延 Push を quiesce する。

### Implementation Decisions

ReplayBuffer は phase の切替時期を判断せず、Learner が要求する quiesce / cold-rearm 境界だけを提供する。
current phase は lineage exp-step から再解決し、保存値との不一致がある場合は fail-fast または明示診断する。具体契約は正式実装 RFD で確定する。
phased mode で phase が空、一覧にある phase の定義がない、phase 名が重複する場合は fail-fast する。

### Testing Decisions

テストは内部変数の代入方法ではなく、外部から観測できる update 回数、sample サイズ、transition、例外、metric、save / load の結果を検証する。
constant B128/RR1.25 で、固定step列に対する optimizer update 回数が現行計算と一致する。
phase 境界直前は旧 pair、境界stepから新 pairが使われ、同一 `UpdateFromBatch` 内で混在しない。
ReplayBufferが要求数と異なるsampleを返した場合にfail-fastする。

### Out of Scope

reward、loss、Q、TD、NEET発生率を入力にしたadaptive / closed-loop制御。

## 主要な概念

- [Replay ratioとlane窓](../../concepts/methods/replay_ratio_lane_window.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
