# Sample+To プリフェッチ（ReplayBuffer サンプリング + H2D を GPU 学習と overlap）仕様書

## Context（背景・目的）

学習ループは serialized pipeline で、env と learner は `learn_future_` により既に 1ステップ overlap 済み、**learner（GPU）が per-step の long pole**。NSight/Tracy 実測で、その learner step の先頭に **GPU を遊ばせる CPU/転送バブルが2つ連続**することが判明：

- **`ReplayBuffer::Sample`（`ExtractSamples`）~6.5ms/step**：PER sum-tree 抽選 + 経験 gather（CPU）。この間 GPU idle。
- **`ExperienceSamples::To`（device 転送）~6.5ms/step**：CPU→GPU の H2D。**現状 `non_blocking=false`（同期）かつ default stream 固定**のため非同期化されておらず、Sample と同程度に遅い。

合わせて **~13ms/step の GPU バブル**。本仕様は **`Sample` + `To` をセットで prefetch スレッドに offload し、現バッチの GPU 更新と「次バッチのサンプリング + H2D」を overlap** して両方を隠す。learner long pole が縮み throughput が上がる。**再現性（同 seed → 同結果）は保つ。**

確定した設計判断（グリリング済み）：
1. **1ステップ stale な PER サンプリングを許容**。overlap の原理的前提（後述 §前提事実）。決定的（同 seed → 同結果）は保つが現行 serial とは bit 非互換。PER 優先度は緩やかに変化するため学習影響は軽微。
2. **スレッド化は ReplayBuffer 公開インターフェース単位（`Sample()` 丸ごと）+ `To()`** をセットで。Learner は内部に手を突っ込まない。スレッド安全性は ReplayBuffer の責務として内部カプセル化。
3. **Push は当面 main thread**（そこそこ遅いが今回は対象外）。
4. **prefetch は GPU 常駐バッチを生成**（Sample の CPU gather + To の H2D まで）。To の H2D を compute と overlap させるため copy stream + pinned + event を使う。
5. **スコープ**：`Learner` 基底 + `DefaultReplayBuffer` + `ExperienceSamples::To`。DefaultDQN/QR + Rainbow（同経路なら同梱可）。**config フラグで default OFF**。**MuZero 対象外**。

## 1. 前提事実（調査済み・再調査不要）

### 1.1 パイプライン（`core/anet-core/src/trainer.cpp`）
`PipelineTrainRunner::DoStep`（[:511](../../core/anet-core/src/trainer.cpp)）は冒頭 `learn_future_.get()`（前 learn join）→ `actor_->MakeAction` → `learn_future_ = learn_pool_->EnqueueFuture(0, [...]{ return learner->UpdateFromBatch(...); })`（:577）→ env Step。専用 `learn_pool_`（`anet::PinnedThreadPool` 1本）。**この future/pool パターンを prefetch にも踏襲する。**

### 1.2 learner ループ（`core/anet-core/src/dqn_based_agent.cpp`）
`Learner::UpdateFromBatch`（:1285）：`replay_buffer_->Push(experiences)` → `while (update_credit_ >= 1.0)` { `replay_buffer_->Sample(samples, B, beta)`（:1312）→ `auto dev_samples = samples.To(device_)`（:1326）→ `UpdateFromSamples(dev_samples)` }。`UpdateFromSamples`（QR は :1685）末尾で `UpdatePerPriorities`（sum-tree 書き込み）。**credit ループは可変回数**（replay_ratio・num_envs 依存）。prefetch はループ境界をまたいで「次に消費するバッチ」を1つ先読みする。

### 1.3 ReplayBuffer（`core/anet-core/src/replay_buffer_impl.cpp`）
- **スレッド安全でない**（mutex は metrics accessor cache 用のみ :1252）。sum-tree（`prio_controller_`）/`index_manager_`/`storage_` は無保護。
- `Sample`（:1227）= `index_manager_->GetValidIndices1D` + `sampler_->SampleIndices`（sum-tree 抽選, **µs**）+ `extractor_->ExtractSamples`（storage gather, **~6.5ms**）。
- `Push`（:1135）= storage write（新スロット）+ `index_manager_->MarkValid` + sum-tree 初期優先度。`UpdatePriorities`（:1242）= sum-tree 書き込み（サンプル済み index）。
- **storage は固定 pre-alloc リングバッファ**（`torch::empty({num_envs, capacity_per_env})` :257、resize なし、`write_cursor % capacity` :270）。**pinned**（`TensorOptions().pinned_memory(pin_memory && device.is_cpu())` :253、`pin_memory=true` 既定）。`PrioritizedSampler` は `RandomHolder`（RNG 状態あり :727）。

### 1.4 `ExperienceSamples::To`（`core/anet-core/src/rl.cpp:666`）
- `device.is_cuda()` 時に **`getDefaultCUDAStream()` + `CUDAStreamGuard` で default stream を強制**（:672）。各 tensor を `To(device, non_blocking)`。
- 呼び出しは `samples.To(device_)`＝**`non_blocking` 既定 false（同期 H2D）**。
- → 別スレッドに出しても default stream 固定なので compute と直列化し overlap しない。**default stream 強制を解除（ambient stream 尊重）しないと To は隠れない。**

### 1.5 overlap には 1ステップ stale が原理的に必須
extract(N+1) を update(N) の GPU 処理と重ねるには、その前に index（＝SampleIndices 結果）が要る。bit 互換を保つ＝update(N) の優先度更新を反映した index を使う、には update(N) 完了を待つことになり overlap 不可能。よって **batch N+1 を update(N) の優先度反映前から抽選＝1ステップ stale** が overlap の前提。

### 1.6 スレッド安全性の根拠
- **Extract（gather, 6.5ms）はロックフリー安全**：固定リング storage を index 読み。Push は write_cursor の新スロットのみ書込。サンプル可能な valid スロットは index_manager のシール解除済みで write_cursor から `capacity/num_envs`（数千ステップ）離れており、1回の Extract（6.5ms=1ステップ強）で write_cursor は数 num_envs しか進まず、サンプル済みスロットに到達しない。
- **sum-tree + index_manager は fine-grained ロックで保護**（Sample の SampleIndices 部 µs と Push/UpdatePriorities µs を相互排他）。
- **To はバッファ共有状態に触らない**（抽出済み tensor の純 CUDA 演算）→ スレッド安全性は Sample と同じ、追加の race なし。

## 2. 設計方針

### A. ReplayBuffer を並行安全化（内部 fine-grained ロック）
`DefaultReplayBuffer` に `std::mutex buffer_mutex_` を追加。
- `Sample` を「**ロック内**で GetValidIndices1D + SampleIndices（idx_result 確定）→ **ロック解放** → ExtractSamples（ロックフリー）」に再構成。重い Extract はロックを持たない。
- `Push` / `UpdatePriorities` は sum-tree / index_manager 更新部のみロック。
- storage の Extract 読み / Push 書きはロック外（§1.6 のリング距離で安全）。Push 内の index_manager 反映（MarkValid + 初期優先度）はロック内に置く。

### B. `ExperienceSamples::To` を overlap 可能にする
- rl.cpp:672 の **`getDefaultCUDAStream()` → `getCurrentCUDAStream()`**（ambient stream 尊重）。後方互換（既存呼び出しは current=default のまま挙動不変）。
- prefetch 側で copy stream を `at::cuda::CUDAStreamGuard` で設定して `To(device, /*non_blocking=*/true)` を呼ぶ → H2D が copy stream + 非同期に。

### C. Learner が「Sample + To」を prefetch future で1つ先読み（GPU 常駐 double-buffer）
- 専用 `anet::PinnedThreadPool`（1本、`learn_pool_` とは別、Learner 所有）+ 専用 **copy stream**（`at::cuda::getStreamFromPool()`）。
- メンバに二重スロット `struct PrefetchedBatch { ExperienceSamples dev_samples; at::cuda::CUDAEvent ready_event; }` の future を保持し、credit ループ／UpdateFromBatch 呼び出しをまたいで持続。
- 更新ループ（`use_rb_prefetch=true`）：
  1. **cold start**：先読みが無ければ Sample + To を同期実行。
  2. 先読み済み `{dev_samples(N), ready_event(N)}` を `future.get()`。**compute stream を `ready_event(N).block(at::cuda::getCurrentCUDAStream())` で H2D 完了待ち**（cross-stream 依存）。
  3. `UpdateFromSamples(dev_samples(N))`（GPU enqueue）。
  4. **GPU enqueue 後・`indices_cpu` 同期前**に次を起動：prefetch_pool で
     ```cpp
     at::cuda::CUDAStreamGuard g(copy_stream);
     ExperienceSamples cpu;
     replay_buffer_->Sample(cpu, B, beta);              // SampleIndices(µs)+Extract(6.5ms)
     ExperienceSamples dev = cpu.To(device_, /*non_blocking=*/true);  // copy stream, async H2D
     at::cuda::CUDAEvent ev; ev.record(copy_stream);
     return PrefetchedBatch{ std::move(dev), std::move(ev) };
     ```
     - SampleIndices は update(N) の UpdatePriorities 反映前ツリーを読む＝**決定的に1ステップ stale**（GPU 同期 ms ≫ SampleIndices µs で順序確定）。
     - To は copy stream + 非同期 → **H2D が main の compute（別 stream）と overlap**。
  5. `indices_cpu` 同期 → `UpdatePriorities`（prefetch の SampleIndices は既に完了）。
- `use_rb_prefetch=false` 時は現行の同期 Sample+To 経路（A/B・切り戻し用）。

### D. pinned source（H2D を真に非同期化する必須条件）
非同期 H2D（overlap）には **転送元 CPU tensor が pinned 必須**。storage は pinned だが Extract の gather 出力は新規確保で非 pinned の可能性。
- 対応（実装時に計測して選択）：(a) ExtractSamples の出力を pinned 確保、(b) To 直前に pinned double-buffer へコピー、(c) 非 pinned だと `non_blocking=true` が同期 fallback し overlap しない。**(a) を第一候補**（追加コピー無し）。
- 検証：NSight で copy engine（H2D）と compute が別 stream で並行になっているか確認。並行していなければ pinned が効いていない（§D-(c)）。

**決定性のフォールバック**：prefetch の SampleIndices が UpdatePriorities(N) より先にロック取得することが前提（GPU 遅延で実質確定）。万一非決定が観測されたら「**SampleIndices だけ main thread に残し、Extract + To のみ offload**」へ退避する（SampleIndices を main 同期実行すれば順序が単一スレッドで自明に決定的になる）。

## 3. 外部仕様（config 追加）

| キー | 型 | 既定 | 意味 |
|---|---|---|---|
| `DefaultDQNAgent.learner.use_rb_prefetch` | bool | **false** | true で `Sample`+`To` を prefetch スレッド（+copy stream）に offload し GPU 更新と overlap。1ステップ stale な PER サンプリング（決定的）。false で現行同期経路。|

- 既存 `learner.*` 読み込み（`ANET_READ_CONFIG`）に追加。`apps/runner/config/DropMerge.txt` の learner ブロックにキー + 用途コメント追記。未指定時は構造体既定値 false。

## 4. 修正対象ファイル

| ファイル | 変更内容 |
|---|---|
| `core/anet-core/src/rl.cpp` | `ExperienceSamples::To` の `getDefaultCUDAStream` → `getCurrentCUDAStream`（後方互換） |
| `core/anet-core/src/replay_buffer_impl.cpp` | `buffer_mutex_`、`Sample`/`Push`/`UpdatePriorities` の fine-grained ロック化（Extract はロック外）。必要なら ExtractSamples の pinned 出力 |
| `core/anet-core/src/dqn_based_agent.cpp` | `Learner` に prefetch pool + copy stream + `PrefetchedBatch` double-buffer、`UpdateFromBatch` 更新ループの prefetch 化（フラグ分岐）、`Config` に `use_rb_prefetch` 配線 |
| Learner/Config ヘッダ（`core/anet-core/include/anet/...`） | メンバ・config 宣言 |
| `apps/runner/config/DropMerge.txt` | `learner.use_rb_prefetch` 追記 |
| `core/anet-core/src/*_test.cpp`（replay/dqn のテスト同居先） | テスト追加（§6） |

Rainbow は `Learner` 基底経路を共有するなら自動で対象（フラグ default OFF）。MuZero は別経路のため触らない。

## 5. 既存利用可能な部品（再利用先）

- `anet::PinnedThreadPool` + `EnqueueFuture`（trainer.cpp の `learn_pool_`/`learn_future_` と同パターン）。
- `at::cuda::getStreamFromPool` / `at::cuda::CUDAStreamGuard` / `at::cuda::CUDAEvent`（libtorch、copy stream + event）。
- `DefaultReplayBuffer::Sample` / `Push` / `UpdatePriorities`、`ExperienceSamples::To`（既存 API、内部改修のみ）。
- storage の `pin_memory`（既に true）。
- `ANET_READ_CONFIG` / `ToConfigBool`。

## 6. 検証方針

テストは replay/dqn のテスト同居先（`*_test.cpp`）に追加（`anet-core-test` ターゲット）。
1. **再現性（核心）**：同一 seed で `use_rb_prefetch` true/false を比較。完全 bit 一致は期待しない（stale により異なる）が、**同フラグ・同 seed の2回実行は bit 一致**（決定性）。loss/q_max/grad_norm 曲線が prefetch 有無で**統計的に同等**（早期発散しない）。
2. **スレッド安全性**：ThreadSanitizer（or ASAN）または高頻度ロングランで Sample×Push×UpdatePriorities + 並行 Extract の競合・クラッシュが無いこと。
3. **overlap 実測（必須）**：NSight で learner step の `ExtractSamples`(CPU) と `To`(H2D) が GPU compute の裏に隠れること。**H2D が copy engine で compute と別 stream 並行**になっているか確認（§D）。GPU idle 縮小・steps/sec 向上を実測（ユーザー）。
4. **cross-stream 正当性**：`ready_event.block` を外すと compute が未転送データを読む race → 数値破綻することを確認（event 同期が効いている証明）。通常経路で NaN/shape チェック。
5. **cold start / credit ループ**：warmup 直後・credit が 0 や複数回のステップで先読みスロットが破綻しないこと（最初の同期 fallback、可変回数の消費）。
6. **ビルド/テスト**：VsDevCmd 経由で x64-Debug をビルドし `core\anet-core\bin\Debug\anet-core-test.exe` を実行（AGENTS.md 必須事項）。

## 7. Out of Scope

- **Push のスレッド化**（今回は Sample+To のみ。Push も遅いが別 PRD 候補）。
- **MuZero**（unroll_steps の extract が別系）。
- `indices_cpu` / `priorities_cpu` の D2H 同期除去（別件）。
- 自前の複数 actor / 完全非同期（Ape-X）化（本件は単一 learner の GPU バブル除去のみ）。
