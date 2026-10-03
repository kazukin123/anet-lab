# ReplayBuffer PER Priority 転送境界整理 実装メモ

## 概要
`ExperienceSamples::To(cuda)` では sampled `indices` を ReplayBuffer の CPU metadata として固定し、learner device へ転送しない。PER priority は TD error 確定直後に `HostReadback` で D2H を開始し、`Optimize` 後の `ReplayBuffer::UpdatePriorities` 直前まで wait を遅延する。

`ReplayBuffer` public interface、config key、`PrefetchingReplayBuffer` の 1-step stale ordering、armed 後 Push write-behind は変更しない。`CONTEXT.md` と ADR は更新しない。

## 主な変更
- `core/anet-core/include/anet/transfer.hpp` に単一 tensor 用 `HostReadback` を追加する。CPU path は即時 CPU tensor、CUDA path は producer stream の source-ready event と copy stream の done event を `EventRecycler<torch::Tensor>` から取得し、pinned CPU destination への non-blocking D2H copy を積む。
- `ExperienceSamples::To(cuda)` は `indices` だけ CPU のまま返す。`ValidateDeviceSamples` は learner 入力 tensor が `device_`、`indices` が CPU であることを検証する。
- `Learner` に PER priority readback 用 copy stream と `EventRecycler<torch::Tensor>` を持たせる。`PreparePerPriorityUpdate` は CPU indices vector 化と priority readback enqueue、`ApplyPerPriorityUpdate` は wait、CPU vector 化、clip 件数確定、ReplayBuffer priority tree 更新を行う。
- `TDLearner` と `QRLearner` は TD error 確定直後に `PreparePerPriorityUpdate`、`Optimize` 後に `ApplyPerPriorityUpdate` を呼ぶ。既存 `UpdatePerPriorities` は test/helper 用の同期 wrapper として残す。
- `BatchUpdateResult` の PER metrics は ReplayBuffer 更新に使った CPU materialized priority と CPU scalar `per_clipped_count` を再利用する。`per_is_weights` の observer sync は本作業では全面解消しない。
- profiling 名は `Learner::PerPriorityD2H.launch`、`Learner::PerPriorityD2H.wait`、`Learner::PerPriorityD2H.vector_copy`、`Learner::UpdatePerPriorities.indices_cpu`、`Learner::UpdatePerPriorities.update_tree` に揃える。

## テスト
- `[transfer]` に `HostReadback` CPU immediate path の値一致を追加する。
- `[transfer][cuda]` に CUDA 利用可能時だけ D2H enqueue、done synchronize 後の CPU 値一致、event recycler の pending/free 推移を追加する。
- `ExperienceSamples::To(cuda)` は CUDA 利用可能時に `indices` が CPU に残り、他の learner input tensor が CUDA へ移ることを確認する。CUDA 不可環境では CPU path の契約だけ確認する。
- `[dqn][per]` で `Prepare` / `Apply` 経由の priority 値、clip 件数、ReplayBuffer priority update の結果を確認する。既存 helper テストは新 API に合わせる。
- 既存 `[transfer]`、`[replay_buffer][prefetch]`、`[dqn][prefetch][determinism]`、`[replay_buffer]`、full test を回す。`true` vs `false` prefetch の bit 一致は要求しない。

## 検証
```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe "[transfer]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][prefetch]"
core\anet-core\bin\Debug\anet-core-test.exe "[dqn][prefetch][determinism]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer]"
core\anet-core\bin\Debug\anet-core-test.exe
cmd /s /c "git diff --check"
```

## 前提
- 現在のワークツリーには 020/PrefetchingReplayBuffer 系の未コミット変更と未追跡 `core/anet-core/include/anet/transfer.hpp` がある。既存差分を巻き戻さず、021 差分だけを分離して載せる。
- `per_clipped_count` は別 scalar D2H を増やさず、priority readback 後に CPU で exact に算出する。
- CUDA 実機で priority D2H が optimizer work と十分 overlap しない場合でも、Tracy 上で launch/wait/vector/tree update の同期源を分離して見える状態を受け入れ条件にする。
