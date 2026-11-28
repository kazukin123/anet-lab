---
title: ReplayBuffer PER priority 転送境界整理 PRD
page_kind: サマリ
material: "[[docs/memo/done/0021_replay_per_priority_transfer_10rfd.md|0021_replay_per_priority_transfer_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/methods/prioritized_replay.md|優先度付き経験再生（PER）]]"
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
updated: 2026-11-28
---
# ReplayBuffer PER priority 転送境界整理 PRD

## 元の材料

[ReplayBuffer PER priority 転送境界整理 PRD](../../../memo/done/0021_replay_per_priority_transfer_10rfd.md) blob:bf21ba56f86a0ff98fffd4f40bc7cceb94964fbf

## 要点

- PER の sampled index は CPU 上の SumTree を更新するためのメタデータとして保持する。
- 学習に使わない index を GPU へ送って再び戻す往復を除く。
- TD error に由来する優先度は GPU で得られるため、CPU への転送が必要になる。
- 優先度の転送開始を前に寄せ、backward と optimizer の処理に重ねる。
- ReplayBuffer の公開インターフェースと利用者の設定項目は維持する。

## 内容

### Problem Statement

`learner.use_rb_prefetch=true` では `Sample+To` と armed 後 `Push` が background worker に移ったが、Tracy 上では実時間性能がほぼ改善しないケースが残っている。現在の有力な同期源は PER 優先度更新まわりの CPU/GPU 転送境界である。
PER priority の D2H を共通部品化し、TD error 確定直後に非同期開始して、SumTree 更新直前まで待ちを遅延する。

### Implementation Decisions

ReplayBuffer の `UpdatePriorities` 呼び出し位置は従来どおり `Optimize` 後に残す。priority の値計算と D2H 起動だけを前倒しする。
source-ready / done の 2 event はいずれも `EventRecycler` から `Acquire` する。`Prepare` は学習 step 毎に呼ばれるため、transient event の per-call `cudaEventDestroy` を避ける。

### 検証方針

テストは stream の内部実装ではなく、外から見える契約を確認する。
`UpdatePerPriorities` は CPU indices を使って ReplayBuffer の priority を更新し、更新後の sampling behavior / priority 値が従来と一致することを確認する。

## 主要な概念

- [優先度付き経験再生（PER）](../../concepts/methods/prioritized_replay.md)
- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
