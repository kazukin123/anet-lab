# PrefetchingReplayBuffer 実装メモ

## 概要
`Learner` 内の prefetch 実装を `ReplayBuffer` decorator の `PrefetchingReplayBuffer` へ移す。`use_rb_prefetch=true` のときは `Learner::replay_buffer_` の実体を `PrefetchingReplayBuffer(inner DefaultReplayBuffer)` にし、`Learner` 本体は常に `Sample()` → `To(device_)` → validate → update の単一路線にする。現在の `use_rb_prefetch=true` は、armed 後の `Push` を同じ FIFO worker へ遅延投入する write-behind insert も含む。

CPU learner device でも `use_rb_prefetch=true` を許容する。CPU では CUDA stream/event/pin を使わず、background `Sample+To(CPU)` を 1-deep prefetch として動かす。

## 主な変更
- ANET の機能グループ単位の配置に合わせ、`PrefetchingReplayBuffer` の宣言は既存の `core/anet-core/include/anet/replay_buffer.hpp`、実装は `core/anet-core/src/replay_buffer_impl.cpp` に置く。`prefetching_replay_buffer.{hpp,cpp}` は作らない。
- `class PrefetchingReplayBuffer : public ReplayBuffer` を実装する。`Size` / accessor は inner へ委譲し、`Sample` と、storage/priority を変更する `Push` / `UpdatePriorities` で prefetch と順序保証を担う。
- `Sample()` は cold start で state mutex 内の同期 sample を返し、以後は前回 future を consume して次 future を起動する。worker thread は monolithic `Sample+To(device)` を行い、`SampleIndices` も background に残す。
- `Push()` は prefetch 未armed/cold/warmup では同期 inner forward のままにする。armed 後は runner 側で stable 化済みの `BatchExperience` を shallow copy し、同じ `PinnedThreadPool(1)` に `inner_->Push(snapshot)` を FIFO enqueue して即 return する。`action_info` は毎 step 新規生成・immutable 扱いを前提にする。
- `UpdatePriorities()` は in-flight future を `wait()` してから inner へ forward し、`Push -> next SampleIndices -> UpdatePriorities` の順序を固定する。`wait()` は例外伝播を目的にせず、prefetch 例外は次回 `Sample().get()` で表面化させる。queued Push 例外は次の同期境界で回収する。
- `dqn_based_agent` から `SamplePrefetchState` と関連メソッドを削除する。`SetupReplayBuffer()` で `config_.use_rb_prefetch` が true なら CPU/CUDA どちらでも `PrefetchingReplayBuffer` で wrap する。
- Rainbow は現状維持する。`RainbowAgentConfig` の `learner.use_rb_prefetch = false` 固定と既存テストは変更しない。

## コーディング方針
- `replay_buffer_impl.cpp` では実装全体を `namespace ... {}` で囲まず、既存の `using namespace anet::rl;` に合わせる。
- コメントは同期境界と順序理由を説明する粒度にし、単純な代入説明は増やさない。
- 可読性が落ちない aggregate 生成では指示付き初期化（Designated Initializers）を使う。

## テスト
- CPU device で `use_rb_prefetch=true` の `Learner` construction が throw しないよう既存 rejection test を更新する。
- CPU path の `PrefetchingReplayBuffer` 単体テストを追加し、sample 取得、同 seed 2 run の sampled index 一致、cold sample 中の `Push()` 待機、in-flight future に対する `Push()` 遅延投入、shallow snapshot の寿命保持、FIFO 上での next `Fetch()` 前実行、`UpdatePriorities()` の wait 順序保証を確認する。
- forwarding test で `Push` / `Size` / accessor が inner と同じ結果になることを確認する。
- `true` と `false` の sampled index 列一致は検証しない。1-step stale により異なるのが正常。

## 検証
```powershell
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
core\anet-core\bin\Debug\anet-core-test.exe
git diff --check
```

## 前提
- public `ReplayBuffer` virtual interface と config key は変更しない。
- 新規ソースファイルは追加しないため、`core/anet-core/CMakeLists.txt` の手編集は不要。
- ADR 0005 は、internal plan/fetch split ではなく monolithic background prefetch + write-behind Push + UpdatePriorities wait を採用した内容へ更新する。
