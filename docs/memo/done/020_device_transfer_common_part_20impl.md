# Device 転送共通部品 `anet/transfer.hpp` 実装メモ

## 概要
`PrefetchingReplayBuffer::Sample()` hot path から batch ごとの `CUDAEvent` 破棄を除き、`EventRecycler` で event と転送元 payload を完了後に回収・再利用する。

`ReplayBuffer` public interface、`learner.use_rb_prefetch`、1-deep prefetch、armed 後 Push write-behind、ADR 0005 の stale PER 契約は変更しない。`CONTEXT.md` と ADR は更新しない。

## 主な変更
- 新規 `core/anet-core/include/anet/transfer.hpp` に `namespace anet::transfer` を追加し、`TransferableSamples` concept、`EventRecycler<Payload>`、`DeviceTransfer<Samples>`、`RecordStreamOn(samples, stream)` を定義する。
- `EventRecycler` は mutex 内で `pending_` と `free_` を管理する。`Acquire()` は完了済み event を回収して再利用し、無ければ lazy に新規 event を返す。`Retire()` は payload と event を pending に積むだけで CUDA 呼び出しをしない。`Poll()` は非同期回収、`Drain()` は synchronize して全回収する。
- `DeviceTransfer` の CPU constructor は `device_samples = cpu_samples.To(device)` とし `ready_event = nullopt`。CUDA constructor は `retained_source` を pin_memory 済みにし、copy stream guard 下で `To(device, true)`、最後に再利用 event へ `record(copy_stream)` する。
- `RecordStreamOn` は CUDA tensor だけに `tensor.record_stream(consumer_stream)` を打つ。`PrefetchingReplayBuffer::Sample()` では `ready_event->block(current_stream)` の直後に呼び、allocator の早期再利用を防ぐ。
- `TensorDict::ForEachTensor(F&&)` と `ExperienceSamples::ForEachTensor(F&&)` を追加する。`ExperienceSamples` 側は template callable のため `rl.hpp` 内 inline 実装にし、undefined tensor は訪問しない。
- `PrefetchingReplayBuffer::State` に `EventRecycler<ExperienceSamples>` を in-place 所有させる。`ANET_EXPERIMENT_RETIRE_PREFETCH_READY_EVENT` と `retired_batches` は削除する。
- `PrefetchedBatch` は `DeviceTransfer<ExperienceSamples>` ベースの move-only 値に置換する。旧 `unique_ptr<CUDAEvent>` と `PrefetchPinCpu*` helper は削除し、pin/copy/event record は `DeviceTransfer` に集約する。
- `StopPrefetch()` は future と queued Push を待った後、CUDA context が生きている間に `state_->event_recycler.Drain()` を呼ぶ。

## テスト
- `[transfer]` を追加し、`ForEachTensor` が TensorDict 内を含む全 defined tensor を可変訪問し undefined をスキップすることを確認する。
- `[transfer]` で `DeviceTransfer` CPU constructor が CPU samples と `ready_event == nullopt` を返すことを確認する。
- `[transfer]` で `EventRecycler` の `Retire -> Poll/Acquire -> Drain` の `PendingCount` / `FreeCount` 推移を CUDA 不要で確認する。
- 既存 `[replay_buffer][prefetch]`、`[dqn][prefetch][determinism]`、`[replay_buffer]` を維持する。
- CUDA 利用可能環境では `DeviceTransfer` CUDA path の値一致、event 再利用、`RecordStreamOn` 後の使用を追加で確認する。

## 検証
```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[transfer]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][prefetch]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][prefetch][determinism]"
core\anet-core\bin\Debug\anet-core-test.exe
cmd /s /c "git diff --check"
```

## 前提
- 現在のワークツリーには `replay_buffer_impl.cpp`、`replay_buffer_test.cpp`、014 memo、ADR 0005 などの既存未コミット変更がある。これらを既存作業として扱い、巻き戻さない。
- 020 の範囲では `ExperienceSamples::To` の field 方針は変えない。`indices` CPU 固定と PER priority D2H の二段階化は 021 の後続作業に残す。
