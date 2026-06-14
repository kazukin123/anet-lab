# Sample+To Prefetch 実装メモ

## Summary

`docs/memo/013_sample_prefetch_10prd.md` に基づき、DQN 系 Learner の `ReplayBuffer::Sample` + `ExperienceSamples::To` を 1 バッチ先読みできるようにする。

既定は `learner.use_rb_prefetch = false` のままにし、DefaultDQNAgent で true のときだけ prefetch thread と CUDA copy stream を使う。RainbowAgent では `use_tbo` / `use_fused_optimizer` と同様に false 固定にする。serial 実装との bit 互換は前提にせず、同 seed / 同 flag での決定性を優先する。

## Key Changes

- `LearnerConfig` に `use_rb_prefetch` を追加し、`DefaultDQNAgent` の `learner.*` config から読む。RainbowAgent は false 固定にする。
- `ExperienceSamples::To` は CUDA 転送時に current CUDA stream を尊重する。
- `Learner` は pimpl の `SamplePrefetchState` を持ち、`PinnedThreadPool`、copy stream、future、次回 request を管理する。
- prefetch worker は CPU sample を pinned CPU tensor へコピーしてから `To(device, true)` を呼び、`CUDAEvent` を record する。
- `UpdateFromBatch` は cold start では同期 sample を使い、以後は future から GPU batch を受け取って `ready_event.block(current_stream)` 後に学習へ渡す。
- 次 batch の prefetch は `UpdateFromSamples` の直前に起動する。これにより Sample + H2D が現在 batch の forward/backward/optimizer と重なり、PER priority 反映前の stale sampling も維持する。
- `UpdatePerPriorities` 側の `MaybeLaunchSamplePrefetch()` は、別経路で事前 launch されなかった場合の保険として残す。

## ReplayBuffer Safety

- `DefaultReplayBuffer` に storage shared lock と metadata mutex を追加する。
- `Sample` は storage shared lock を保持したまま metadata snapshot を取り、metadata mutex を解放してから `ExtractSamples` する。
- `Push` は storage unique lock で storage write / queue processing を守り、`ValidIndexManager` と PER sum-tree の更新だけ metadata mutex で保護する。
- `UpdatePriorities`、`Size`、PER accessor、valid-index snapshot も metadata mutex で保護する。
- PRD の lock-free Extract 前提は採用しない。現実装は physical index だけを渡すため、ring 一周直後の最古 slot と Push 上書きの競合を shared lock で避ける。

## Tests

- `DefaultDQNAgentConfig` が `learner.use_rb_prefetch` を読み、`RainbowAgentConfig` は指定されても false のままにすること。
- CPU learner device で `use_rb_prefetch=true` が明示エラーになること。
- prioritized ReplayBuffer で `Sample`、`Push`、`UpdatePriorities` を並行実行して shape と index metadata が壊れないこと。

## Verification

実装後に以下で確認する。

```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe
```

CUDA overlap は別途 NSight / Tracy で、H2D が copy stream 側に出て compute と並行することを確認する。
