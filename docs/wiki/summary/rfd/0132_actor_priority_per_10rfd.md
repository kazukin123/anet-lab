---
title: Actor が見積もる PER 初期優先度の旧草案
page_kind: サマリ
material: "[[docs/memo/dropped/0132_actor_priority_per_10rfd.md|0132_actor_priority_per_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/subjects/dropmerge.md|DropMerge]]"
  - "[[docs/wiki/concepts/framework/324_dqn.md|324 Agent個別実装 DQN系列]]"
updated: 2026-11-26
---
# Actor が見積もる PER 初期優先度の旧草案

## 元の材料

[RFD0132　Actor が見積もる PER 初期優先度の旧草案](../../../memo/dropped/0132_actor_priority_per_10rfd.md) blob:a2ff078f678205c2789293c59989110576765f6b

## 要点

- Actor による初期優先度の旧草案である。RFD0035 として正式採番・実装されたため、この文書での検討を終了した。
- 新規遷移に固定値または最大値を与える方式では、Learner の初回更新まで遷移ごとの差が付かない。
- Actor の推定 TD 誤差で未更新の遷移の中にも優先順を付ける案を検討した。
- DropMerge の記録では固定初期値1.0でも、初期優先度の遷移が優先度総量の7割強を占めた。
- 未抽選で追い出される割合を併せて見て、古い Actor や低い優先度が標本の範囲を狭めないか調べる。

## 内容

### Problem Statement

問題は、`per_initial_priority` を固定値として統計から調整する運用が、環境・報酬スケール・TBO・network 変化・学習段階に強く依存することである。固定値を下げると PER 再利用は増えるが coverage が悪化し、`max_prio` に寄せると新規優先がさらに強くなりやすい。

### Solution

重要なのは、「初期 priority という概念を完全に無くす」のではなく、固定または max による仮 priority を、actor 推定 priority に置き換えることである。learner がその transition を sample した後は、従来通り learner 側 TD error で priority を上書きする。
invalid / missing / unsupported の場合は従来の初期 priority に fallback する。

### Implementation Decisions

hint が存在し、finite で、正の値である場合だけ ReplayBuffer の初期 priority として使う。
hint が無い場合、または invalid の場合は従来の `per_initial_priority` / `max_prio` 初期化へ fallback する。
ReplayBuffer 側は priority hint の source を知らなくてもよいが、metrics のために priority source は最低限区別できるようにする。

### Testing Decisions

テストは private SumTree の内部配列ではなく、外部から見える sampling priority と metrics で確認する。
QR-DQN / TBO が未対応の場合は、明示的に fallback することをテストする。
PrefetchingReplayBuffer の `Push` / `Sample` / `UpdatePriorities` ordering は既存 regression を維持する。

### Out of Scope

`max_prio` 初期化の再設計全般。比較対象として残すが、本 PRD の主目的ではない。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [DropMerge](../../concepts/subjects/dropmerge.md)
- [324 Agent個別実装 DQN系列](../../concepts/framework/324_dqn.md)
