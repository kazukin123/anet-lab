---
title: Device 転送共通部品 `anet/transfer.hpp` と PrefetchingReplayBuffer 適用 PRD
page_kind: サマリ
material: "[[docs/memo/done/0020_device_transfer_common_part_10rfd.md|0020_device_transfer_common_part_10rfd]]"
material_kind: RFD
concepts:
  - "[[docs/wiki/concepts/framework/314_tensor.md|314 テンソル処理]]"
  - "[[docs/wiki/concepts/framework/318_rl_core.md|318 強化学習 共通基盤]]"
  - "[[docs/wiki/concepts/framework/313_profiling.md|313 プロファイリング]]"
updated: 2026-11-26
---
# Device 転送共通部品 `anet/transfer.hpp` と PrefetchingReplayBuffer 適用 PRD

## 元の材料

[Device 転送共通部品 `anet/transfer.hpp` と PrefetchingReplayBuffer 適用 PRD](../../../memo/done/0020_device_transfer_common_part_10rfd.md) blob:58aa17f3553a8142c020ac85fddab1d1a4c03f5b

## 要点

- ReplayBuffer の先読みで転送完了イベントと転送元の寿命を管理する。
- 実験用の退避方式では、長い Run で pinned memory が増え続けた。
- 共通の EventRecycler が完了を調べ、保留していた payload を解放する。
- イベントのハンドルは再利用し、破棄のコストが各バッチに繰り返し入ることを避ける。
- CPU と CUDA の転送を共通の部品から扱い、終了時は残った処理を回収する。

## 内容

### Context / Problem Statement

`learner.use_rb_prefetch=true` の `PrefetchingReplayBuffer` で、Nsight 実測により `Sample()` hot path の主因が ReplayBuffer sample でも H2D でもなく `c10::cuda::CUDAEvent::~CUDAEvent -> cudaEventDestroy`（代表 +7.459ms） と判明している。原因は、`PrefetchedBatch` が `Sample()` 末尾で破棄され、保持していた `CUDAEvent` を 毎 batch hot path で destroy していたこと。event の record/wait 自体は H2D と learner stream を安全につなぐために必要で、消すべきではない。

### Solution（概要）

`anet/transfer.hpp` に 2 部品＋1 concept＋1 helper を定義し、`PrefetchingReplayBuffer` をこれを使う形へ置換する。

### 公開 API スケルトン（

`~EventRecycler`: `Drain()` 後 `free_` の event が破棄される。`Drain`: pending を `synchronize()` して `free_` へ。

### Further Notes

名前は仮確定: namespace `anet::transfer`、`EventRecycler` / `DeviceTransfer` / `RecordStreamOn` / `TransferableSamples`。「CUDA」を型名に含めない（CPU 互換のため）。最終はクラス名とファイル名を揃える。

## 主要な概念

- [314 テンソル処理](../../concepts/framework/314_tensor.md)
- [318 強化学習 共通基盤](../../concepts/framework/318_rl_core.md)
- [313 プロファイリング](../../concepts/framework/313_profiling.md)
