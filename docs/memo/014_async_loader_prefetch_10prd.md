# PrefetchingReplayBuffer リファクタ（Learner 埋め込み prefetch を decorator へ分離 + 再現性修正）仕様書

## Context（背景・目的）

`docs/memo/013_sample_prefetch_10prd.md` の Sample+To プリフェッチは実装済み（実測 8.8h→7.3h）。ただし現状は **`Learner` 内に prefetch ロジックが埋め込まれ**（`SamplePrefetchState`）、`Push()` / `UpdatePriorities()` と in-flight prefetch の順序境界も Learner 実装に漏れている。これを `ReplayBuffer` decorator に閉じ、同時に mutation 前 wait で replay ordering を固定する。

本リファクタは：
1. prefetch を **共通 decorator `PrefetchingReplayBuffer`（`ReplayBuffer` を wrap し、`ReplayBuffer` インターフェースを継承して提供する）に分離**。`Learner` を薄くする。
2. **replay ordering を固定**：`PrefetchingReplayBuffer::Push()` / `UpdatePriorities()` は in-flight future を `wait()` してから inner mutation へ進む。**公開 `ReplayBuffer` インターフェースは不変**。
3. overlap（`Sample+To` prefetch）と速度（8.8→7.3h 近傍）は維持。

責務分担：**replay ordering＝PrefetchingReplayBuffer の 1-deep prefetch + Push / UpdatePriorities 前 wait / データ整合性＝ReplayBuffer の storage・metadata lock / overlap＝Sample+To prefetch / public RB IF＝不変**。

## 確定した設計判断（グリリング + Codex レビュー結論）

- **A2'（monolithic background prefetch + mutation-wait）を採用**。A1'（internal plan/fetch split + mutation-wait）は `SampleIndices` を caller thread に停留させ、`DefaultReplayBuffer` の内部 seam も増やすため production では採用しない。
- **1ステップ stale な PER サンプリングは許容**（overlap の前提、決定的）。
- Codex レビューで整理した条件を invariant として守る（後述 §制約）。
- **N-deep 化（AsyncDataLoader）は対象外**。今回は 1-deep prefetch と mutation wait に限定する（`docs/memo/999_async_loader_ndeep.md` 参照）。
- **CPU learner device でも `use_rb_prefetch=true` を許容**。CUDA stream/event/pin は使わず、background `Sample()` の 1-deep prefetch として動かす。

## 1. 前提事実（調査済み・現状コード）

### 1.1 現状 prefetch（`core/anet-core/src/dqn_based_agent.cpp` / `.hpp`）
`Learner` に pimpl `SamplePrefetchState`（`PinnedThreadPool` 1本、copy stream `getStreamFromPool`、`std::future<PrefetchedBatch>`、`pending_request`）。helper `PinCpuSamples`/`PinCpuTensor`/`PinCpuTensorDict` と `PrefetchedBatch{ pinned_cpu_samples; dev_samples; CUDAEvent ready_event; }` も `Learner` 側に埋め込まれている。
- `UpdateFromBatch` ループ：`dev_samples = use_rb_prefetch ? ConsumePrefetchedSamplesOrSample(B,beta) : SampleAndTransferSynchronously(B,beta)` → `ValidateDeviceSamples` → `ArmSamplePrefetch` → `MaybeLaunchSamplePrefetch` → `UpdateFromSamples`。
- `UpdatePerPriorities`：`MaybeLaunchSamplePrefetch()`（保険）→ … → `replay_buffer_->UpdatePriorities(...)`。
- prefetch worker：`inner.Sample(cpu)` → `PinCpuSamples` → `CUDAStreamGuard(copy_stream)` + `cpu.To(device, non_blocking=true)` → `event.record(copy_stream)`。
- consume：`future.get()` → `ready_event.block(getCurrentCUDAStream())` → device batch。

### 1.2 ReplayBuffer（`core/anet-core/src/replay_buffer_impl.{hpp,cpp}`）
- 公開インターフェース（`core/anet-core/include/anet/rl.hpp:728`）：`Push` / `Sample`（monolithic, const）/ `Size` + `UpdatePriorities`（`ReplayPriorityController` 基底）+ Module 系（accessor / Save / Load）。**実装は `DefaultReplayBuffer` のみ**（MuZero は別系）。
- スレッド安全化済み：`mutable std::shared_mutex storage_mutex_`（Push=unique / Sample=shared）+ `mutable std::mutex metadata_mutex_`（sum-tree / index_manager の µs 操作）。`Sample` は metadata lock 内で SampleIndices、storage shared lock 下で Extract。
- **注**：013 ADR は「Extract はロックフリー（リング距離で安全）」としていたが**これは誤り**（最古 valid スロットは write cursor 直前で次 Push で上書きされ得る）。現実装は storage shared/exclusive lock で Push⇄Extract を排他しており正しい。本リファクタでもこの lock を維持する。

### 1.3 `ExperienceSamples::To`（`core/anet-core/src/rl.cpp:666`）
`getCurrentCUDAStream` 化済み（ambient stream 尊重、後方互換）。prefetch は copy stream guard 下で `non_blocking=true` を呼ぶ。

### 1.4 config
`learner.use_rb_prefetch`（`agent.hpp` LearnerConfig、default false）。`default_dqn_agent.hpp` で `ANET_READ_CONFIG` 済み。**本リファクタで変更なし。**
`RainbowAgentConfig` は現在 `learner.use_rb_prefetch = false` に固定しており、今回も現状維持する。

### 1.5 replay ordering の機構（核心）
`PrefetchingReplayBuffer` は in-flight prefetch を常に 1 本に限定する。`Push(caller)` / `UpdatePriorities(caller)` は in-flight future の完了を `wait()` してから inner へ進むため、background `Sample+To` と storage/priority mutation の順序が固定される。この契約は inner mutation が必ず wrapper 経由で行われることを前提にする。SDPA や CUDA kernel などの演算 bit determinism は別レイヤで扱う。

## 2. 設計方針

### A. `PrefetchingReplayBuffer` decorator（新規）
`class PrefetchingReplayBuffer : public ReplayBuffer`。`std::shared_ptr<ReplayBuffer> inner_` を wrap。コンストラクタで `inner` + target `device` を受け取り、prefetch pool と CUDA 時だけ copy stream + pinned helper を所有。デストラクタで prefetch 停止（future wait → pool stop）。
- ANET はクラス単位ではなく機能グループ単位でソースファイルを作るため、宣言は既存の `core/anet-core/include/anet/replay_buffer.hpp`、実装は `core/anet-core/src/replay_buffer_impl.cpp` に置く。`prefetching_replay_buffer.{hpp,cpp}` は作らない。
- `.cpp` 側は AGENTS.md の規約に従い、実装全体を `namespace ... {}` で囲まず、既存の `using namespace anet::rl;` の流儀に合わせる。局所 helper は状態を持たない `static` helper として `replay_buffer_impl.cpp` に置く。
- prefetch 状態（future, pending）は `Sample`(const) から触るため `mutable`（既存 mutable mutex と同様）。
- **自前パラメータは持たない**：Module 系（accessor / Save / Load / Size / Forward 等）は **inner へ素通し**（checkpoint は inner のもの、変更なし）。`Push` は prefetch 完了待ちで順序を固定してから inner へ委譲する。
- `PrefetchedBatch` などの aggregate 生成では、可読性が極端に落ちない範囲で指示付き初期化（Designated Initializers）を使う。

### B. `Sample(out, B, beta)`（prefetch 消費 + 次起動）
1. cold start（future 無効）：state mutex 内で同期 `inner.Sample(cpu)` + `cpu.To(device)`。future が無い瞬間でも `Push()` が `Sample()` 中に割り込まないようにする。
2. それ以外：`batch = future.get()` → `batch.ready_event.block(getCurrentCUDAStream())` → `out = batch.dev_samples`（**device 常駐**で返す）。
3. 次 prefetch を起動：worker で monolithic `inner.Sample(cpu)` → CUDA 時は `PinCpuSamples` → copy stream で `To(device, non_blocking=true)` → `event.record`、CPU 時は stream/event/pin なしで `To(CPU)`。
- PrefetchingReplayBuffer は **target device 向けサンプルを返す**ので、`Learner` 側の `samples.To(device_)` は no-op（同 device への To は同一 tensor を返す）。`Learner` 経路は分岐不要。

### C. `Push(batch)` / `UpdatePriorities(indices, priorities)`（再現性の要）
```
if (prefetch_future_.valid()) prefetch_future_.wait();   // ★ in-flight prefetch(=Sample+To) 完了を保証
inner_->Push(batch); または
inner_->UpdatePriorities(indices, priorities);
```
`wait()` は future を消費しない（次 `Sample()` の `get()` が成立）。

### D. その他は inner へ forward
Size / accessor / Save / Load は inner へ委譲。整合性は inner の storage lock と metadata lock が守る。

### E. `Learner` の簡素化
- `SamplePrefetchState` と prefetch 関連メソッド（`ConsumePrefetchedSamplesOrSample` / `ArmSamplePrefetch` / `MaybeLaunchSamplePrefetch` / `StopSamplePrefetch` / `SampleAndTransferSynchronously` / `EnsureSamplePrefetchState`）を**削除**。`PinCpuSamples` 等は `replay_buffer_impl.cpp` の PrefetchingReplayBuffer 実装近傍へ移動する。
- `SetupReplayBuffer()`：`replay_buffer_ = CreateReplayBuffer(...)` の直後に、`config_.use_rb_prefetch` なら `replay_buffer_ = std::make_shared<PrefetchingReplayBuffer>(replay_buffer_, device_)`。CPU device でも fail-fast しない。
- `UpdateFromBatch` ループ：分岐を消し、`replay_buffer_->Sample(samples, B, beta)` → `samples.To(device_)`（PrefetchingReplayBuffer 時は no-op）→ `ValidateDeviceSamples` → `UpdateFromSamples`。prefetch は PrefetchingReplayBuffer が透過的に行う。
- `UpdatePerPriorities`：`MaybeLaunchSamplePrefetch()` を削除し、`replay_buffer_->UpdatePriorities(...)`（PrefetchingReplayBuffer が wait を挟む）だけ残す。`UpdateFromBatch` 冒頭の `Push` も PrefetchingReplayBuffer 経由で wait される。

### 制約（Codex 条件・invariant として守る）
1. **Push / UpdatePriorities は必ず PrefetchingReplayBuffer 経由**。`Learner` は inner への生ポインタ storage/priority write 経路を持たない（`replay_buffer_` が PrefetchingReplayBuffer 実体）。
2. **in-flight prefetch は常に 1 本**（1-deep）。深い pipeline は本 PRD 対象外（999 参照）。
3. **`wait()` は順序保証であって例外伝播ではない**。prefetch 失敗は次 `Sample().get()` で表面化する（仕様として許容・明記）。
4. **SampleIndices は background `Sample()` 内で実行**。caller thread へ戻さず、`Push()` / `UpdatePriorities()` の wait 境界で mutation との順序を固定する。

## 3. 外部仕様（config）

`learner.use_rb_prefetch`（bool, default false）。**変更なし**。true で `Learner` が `replay_buffer_` を PrefetchingReplayBuffer で wrap。DefaultDQN/QR は CPU/CUDA とも有効化可能。Rainbow は現状どおり config 側で false 固定。

## 4. 修正対象ファイル

| ファイル | 変更 |
|---|---|
| `core/anet-core/include/anet/replay_buffer.hpp` | `PrefetchingReplayBuffer` の宣言を追加。既存 `ReplayBuffer` virtual interface と config key は変更しない |
| `core/anet-core/src/replay_buffer_impl.cpp` | `PrefetchingReplayBuffer` 実装、`Push` / `UpdatePriorities` 前 wait と `PinCpuSamples` 等の helper も同じ機能グループ内へ移設 |
| `core/anet-core/src/dqn_based_agent.cpp` / `.hpp` | `SamplePrefetchState` + prefetch メソッド削除、`SetupReplayBuffer()` で PrefetchingReplayBuffer wrap、`UpdateFromBatch`/`UpdatePerPriorities` 簡素化 |
| `core/anet-core/src/*_test.cpp` | テスト追加（§6）。既存の prefetch テストを PrefetchingReplayBuffer 用に移行 |

`rl.cpp`（To）と `DefaultReplayBuffer` の public `ReplayBuffer` virtual interface は不変。`DefaultReplayBuffer::Sample()` は monolithic のまま、storage shared lock と metadata lock で `SampleIndices` と `Extract` の整合性を守る。

## 5. 既存利用可能な部品（再利用先）

- `anet::PinnedThreadPool` + `EnqueueFuture`、`at::cuda::getStreamFromPool` / `CUDAStreamGuard` / `CUDAEvent`（現 `SamplePrefetchState` から移設）。
- `PinCpuSamples` / `PinCpuTensor`（`replay_buffer_impl.cpp` へ移設）。
- `DefaultReplayBuffer` の storage/metadata lock（整合性、そのまま）。
- `ExperienceSamples::To`（getCurrentCUDAStream 化済み）。

## 6. 検証方針

1. **再現性（核心・指摘#1）**：`use_rb_prefetch=true` で **同 seed 2 run の sampled index 列が bit 一致**（主判定）。loss 系列一致は副。**`true` と `false` は staleness で異なるのが正常** ―― ここを等価判定にしない。
2. **計測**：`ANET_PROFILE_SCOPE` 系で分割 ―― `PrefetchingReplayBuffer::Push.wait_prefetch` / `PrefetchingReplayBuffer::UpdatePriorities.wait_prefetch` / `PrefetchingReplayBuffer::Sample.consume_wait` / `PrefetchingReplayBuffer::Fetch.sample` / `PrefetchingReplayBuffer::Fetch.to`。`Sample+To` は worker 側で overlap する。
3. **スレッド安全性**：並行 Sample×Push×UpdatePriorities（既存テスト流用）でクラッシュ・破綻なし。
4. **overlap**：NSight で prefetch の `Sample`(CPU) + `To`(H2D copy engine) が compute と並行。8.8→7.3h 維持。
5. **整合性（credit loop 0回 / accessor）**：先読みが呼び出しを跨いで in-flight のまま次 Push が来ても storage lock で安全。metrics accessor 並行も lock 防御。
6. **ビルド**：VsDevCmd x64-Debug で `anet-core-test`。

## 7. Out of Scope

- **N-deep prefetch（AsyncDataLoader）**：`docs/memo/999_async_loader_ndeep.md` に分離。本 PRD は 1-deep + mutation wait のみ。
- **Push のスレッド化 / MuZero / `indices_cpu` D2H 同期除去**。
- public `ReplayBuffer` インターフェースの変更。
