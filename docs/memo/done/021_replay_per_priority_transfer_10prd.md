# ReplayBuffer PER priority 転送境界整理 PRD

> 旧 019。`docs/memo/020`（transfer.hpp 共通部品）との依存順に合わせて 021 に採番。019 は欠番。
> **020 に依存**: D2H staging は transfer.hpp の `EventRecycler` ＋ 可視 `CUDAEvent` を再利用する。実装は Codex 想定。

## Problem Statement

`learner.use_rb_prefetch=true` では `Sample+To` と armed 後 `Push` が background worker に移ったが、Tracy 上では実時間性能がほぼ改善しないケースが残っている。現在の有力な同期源は PER 優先度更新まわりの CPU/GPU 転送境界である。

現状の `ExperienceSamples::To(cuda)` は、ReplayBuffer の sampled index まで CUDA へ転送する。しかし sampled index は ReplayBuffer の CPU 側 SumTree / priority metadata を更新するためだけに使われ、学習計算では使わない。その結果、`UpdatePerPriorities` で indices を CPU へ戻す D2H が発生し、不要な H2D→D2H 往復と同期が入る。

一方、`per_priorities` は TD error 由来なので GPU 計算結果であり、CPU 側 SumTree 更新のための D2H 自体は避けられない。ただし現在は `Optimize` 後に `per_priorities.cpu()` を開始するため、priority D2H を backward / optimizer と overlap できない。indices の無駄な同期を消しても、次に `per_priorities` の D2H wait が表に出るだけでは prefetch の効果が伸びない。

目的は、ReplayBuffer の public interface と config key を増やさず、可読性を保ったまま以下を実現することである。

1. sampled index を CPU metadata として固定し、CUDA 転送対象から外す。
2. PER priority の D2H を共通部品化し、TD error 確定直後に非同期開始して、SumTree 更新直前まで待ちを遅延する。
3. `UpdatePriorities` の呼び出し順序と 1-step stale prefetch の replay ordering を維持する。
4. Tracy で「どこで待っているか」を追える粒度を残す。

## Solution

`ExperienceSamples` を「学習 device へ送る tensor」と「ReplayBuffer metadata として CPU に留める tensor」に分けて扱う。CUDA learner でも `indices` は CPU tensor のまま保持し、`ValidateDeviceSamples` などでその契約を明示する。`is_weights` は loss 計算に使うため従来どおり learner device に送る。

PER priority 更新は `Prepare` と `Apply` の二段階に分ける。`Prepare` は TD error が確定した直後、`Optimize` より前に呼び、priority tensor を生成して pinned CPU buffer への D2H copy を copy stream へ enqueue する。`Apply` は従来と同じ論理位置、つまり `Optimize` 後に呼び、D2H 完了を待って CPU vector 化し、ReplayBuffer の priority tree を更新する。

この二段階化により、priority D2H は backward / optimizer / PrefetchingReplayBuffer の `UpdatePriorities` wait 境界と重なり得る。ReplayBuffer の priority update 自体は CPU mutation なので従来どおり同期境界として残すが、CPU が GPU priority tensor の生成以降すべてを待つ形は避ける。

D2H 処理は PER 専用の inline 実装に埋め込まず、**`docs/memo/020` の transfer.hpp（`EventRecycler` ＋ 可視 `CUDAEvent`）を再利用**し、device→pinned host の単一 tensor 型 `HostReadback` だけを追加する。PER 側は indices vector 化、priority vector 化、ReplayBuffer 更新の順序だけを読む形にする。

## User Stories

1. As an RL experimenter, I want sampled indices to stay on CPU, so that PER priority updates do not pay a useless H2D→D2H round trip.
2. As an RL experimenter, I want PER priority D2H to start as soon as TD error is available, so that the unavoidable transfer can overlap with optimizer work.
3. As an RL experimenter, I want `learner.use_rb_prefetch=true` replay ordering to remain deterministic, so that same-seed prefetch runs remain comparable.
4. As an RL experimenter, I want `true` vs `false` prefetch results not to be forced bit-identical, so that the existing 1-step stale PER contract remains explicit.
5. As a performance investigator, I want Tracy zones around priority D2H launch, wait, vectorization, and tree update, so that hidden synchronization moves are visible.
6. As a performance investigator, I want `indices_cpu` to become cheap CPU vectorization, so that any remaining stall is attributed to the correct tensor.
7. As a maintainer, I want D2H staging encapsulated behind a small helper, so that CUDA event / pinned memory details do not spread through learner code.
8. As a maintainer, I want CPU learner behavior to keep working without CUDA-only branches leaking into the call sites, so that tests can still cover the non-CUDA path.
9. As a maintainer, I want `BatchUpdateResult` PER metrics to reuse the materialized CPU priority tensor where practical, so that observer metrics do not reintroduce a second priority D2H sync.
10. As a maintainer, I want no new ReplayBuffer public methods or config keys, so that this remains an internal performance correction rather than a user-facing API change.
11. As a test author, I want behavior-oriented tests for indices device placement and priority update values, so that tests do not depend on private stream implementation details.
12. As a future optimizer, I want the D2H helper to be reusable for other small GPU→CPU materialization points, so that similar sync issues can be fixed without rewriting CUDA plumbing.

## Implementation Decisions

### 1. `indices` は CPU-only replay metadata にする

- `ExperienceSamples::To(cuda)` は `obs` / `actions` / `target_returns` / `next_state` / `n_steps` / `is_weights` / `info` を learner device へ移すが、`indices` は CPU tensor のまま返す。
- CPU device への `To` では既存と同じく CPU tensor のまま扱う。CUDA branch と CPU branch の差分は「indices を device tensor にしない」点だけに抑える。
- `ValidateDeviceSamples` では、学習入力 tensor が learner device にあることに加えて、`indices` が CPU にあることを明示的に検証する。
- `UpdatePerPriorities` 側では `samples.indices.cpu()` を同期境界として扱わない。CPU tensor であることを前提に contiguous 化と vector 化だけ行う。
- sampled index は ReplayBuffer の CPU metadata であり、network forward / loss / optimizer の入力ではない、という契約を docs に明記する。

### 2. PER priority 更新を `Prepare` / `Apply` に分ける

- `MakePerPriorityUpdateInfo` 相当の priority tensor 生成は TD error が確定した時点で行う。
- `PreparePerPriorityUpdate` は以下をまとめた pending object（CUDA path は `HostReadback`）を返す。
  - PER 無効時は empty pending として即時完了扱いにする。
  - `indices` を CPU vector 化する。
  - `per_priorities` と `per_clipped_count` を計算する。
  - CUDA learner では `per_priorities` の D2H copy を `HostReadback`（transfer.hpp）で enqueue する。
  - CPU learner では同じ interface で即時 CPU tensor として扱う。
- `ApplyPerPriorityUpdate` は `Optimize` 後に呼び、pending priority D2H の完了を待って CPU vector を作り、ReplayBuffer の `UpdatePriorities` を呼ぶ。
- ReplayBuffer の `UpdatePriorities` 呼び出し位置は従来どおり `Optimize` 後に残す。priority の値計算と D2H 起動だけを前倒しする。
- `TDLearner` と `QRLearner` は同じ二段階 API を使う。片方だけの特殊実装にしない。

### 3. D2H staging は transfer.hpp（020）を再利用する

- 当初構想の bespoke pending object は作らない。020 で導入した **`EventRecycler<Payload>` をそのまま再利用**する（`CUDAEvent` をプール再利用し、紐づくリソースを完了まで延命。`cudaEventDestroy` は dtor/`Drain` のみ＝D2H でも step 毎の destroy を出さない）。
- 020 の **「event を隠さない」判断の見返りがここで効く**: 同じ可視 `CUDAEvent` に対し、H2D consumer は `block()`（stream 待ち）、**D2H consumer は `synchronize()`（host 待ち）** を選ぶだけでよい。Fence 抽象が無いので両方向を同じ部品で扱える。
- device→pinned host の **単一 tensor 用 `HostReadback`（仮称）を `transfer.hpp` に追加**する。`DeviceTransfer<Samples>`（H2D）の鏡像で、対象は tensor 単体に絞る（TensorDict / nested structure は本 PRD の対象にしない、という当初方針は維持）。
  - **CPU path**: `HostReadback(cpu_source)` は source をそのまま即時保持（event 無し）。呼び出し側の CUDA/CPU 分岐を最小化する。
  - **CUDA path**: `HostReadback(gpu_source, copy_stream, producer_stream, recycler)` が
    1. producer stream（`per_priorities` を生成した compute stream）の source-ready event を copy_stream で待たせ、
    2. pinned CPU destination へ `non_blocking` D2H copy を copy_stream に積み、
    3. copy_stream に done event を record する。
  - source-ready / done の **2 event はいずれも `EventRecycler` から `Acquire`** する。`Prepare` は学習 step 毎に呼ばれるため、transient event の per-call `cudaEventDestroy` を避ける。
- skeleton:

```cpp
// device→pinned host の単一 tensor readback（DeviceTransfer の鏡像）。D2H 用。
struct HostReadback {                                 // 仮称（PinnedReadback / HostMaterialize 等も可）
    torch::Tensor pinned_result;                      // synchronize 後に CPU 可読
    torch::Tensor retained_source;                    // GPU source。copy 完了まで保持
    std::optional<at::cuda::CUDAEvent> ready_event;   // done marker（可視）。CPU 経路は nullopt

    HostReadback() = default;
    explicit HostReadback(torch::Tensor cpu_source);  // CPU: 即時（そのまま）
    HostReadback(torch::Tensor gpu_source,            // CUDA: source-ready→D2H→done record
                 at::cuda::CUDAStream copy_stream,
                 at::cuda::CUDAStream producer_stream,
                 EventRecycler<torch::Tensor>& event_recycler);
    HostReadback(HostReadback&&) = default;
    HostReadback& operator=(HostReadback&&) = default;
    HostReadback(const HostReadback&) = delete;
    HostReadback& operator=(const HostReadback&) = delete;
};
```

- `Prepare` / `Apply` へのマップ:
  - `Prepare` = `per_priorities` から `HostReadback` を構築（D2H enqueue ＋ done record）。PER 無効 / CPU learner では event 無しの即時 readback。
  - `Apply` = `ready_event->synchronize()`（host 待ち）→ `pinned_result` を contiguous CPU vector 化 → ReplayBuffer の priority tree 更新 → `recycler.Retire(std::move(*ready_event), std::move(retained_source))`。

### 4. `BatchUpdateResult` と PER metrics は CPU materialized priority を使う

- `per_priorities` は ReplayBuffer 更新に使った CPU tensor を `BatchUpdateResult` に渡す。
- `per_clipped_count` も CPU scalar として返し、observer 側の `.item()` が GPU sync を起こさないようにする。
- `per_is_weights` は loss 用には learner device に残す。PER metric のための CPU copy を同時に持つかは、実装時に可読性と追加転送量を見て判断する。少なくとも priority 修正の効果測定を曇らせないよう、残る同期源として Tracy で区別できる名前を付ける。
- `td_error` や Q 値 metrics の lazy CPU 化は本 PRD では全面的に直さない。priority D2H と混同しない計測名にする。

### 5. PrefetchingReplayBuffer の replay ordering は維持する

- `PrefetchingReplayBuffer::UpdatePriorities` が in-flight prefetch を待つ境界は維持する。
- pending priority D2H の wait は learner 側の `ApplyPerPriorityUpdate` に置く。ReplayBuffer wrapper は CPU vector を受け取るだけにする。
- armed 後 `Push` の FIFO write-behind は変更しない。今回の目的は `Sample+To` 後の priority update 側の同期整理であり、Push ordering の再設計はしない。

### 6. Profiling 名を安定させる

- 既存の `Learner::UpdatePerPriorities.indices_cpu` は、CPU vector 化だけの短い範囲になる。
- priority D2H には少なくとも以下の安定した zone を入れる。
  - `Learner::PerPriorityD2H.launch`
  - `Learner::PerPriorityD2H.wait`
  - `Learner::PerPriorityD2H.vector_copy`
  - `Learner::UpdatePerPriorities.update_tree`
- 既存の `PrefetchingReplayBuffer::Fetch.to.*` 分割計測と並べて見られるよう、名前は実装後も変更しない。

### 7. record_stream の対称性（device source 側）

- H2D(020) と D2H(021) は CUDA caching allocator の早期再利用ハザードが鏡像になる:
  - H2D: 結果は copy_stream で alloc / compute stream で使用 → `RecordStreamOn(result, compute_stream)`（020 実装済）。
  - D2H: source（`per_priorities`）は compute stream で alloc / copy_stream で読み取り → **同じ早期再利用ハザード**。
- D2H 側の最小対処は **`HostReadback` が GPU source を `Apply` の `synchronize()` まで保持**すること。host 待ちがコピー完了を保証するので、その時点まで source が生きていれば allocator 再利用は起きない。
- source を早期に手放したい設計にする場合のみ `source.record_stream(copy_stream)` を保険として打つ。

## Testing Decisions

- テストは stream の内部実装ではなく、外から見える契約を確認する。
- `ExperienceSamples::To(cuda)` は CUDA 利用可能時に `indices` が CPU に残り、学習入力 tensor は CUDA へ移ることを確認する。CUDA 不可環境では CPU path の契約だけを確認する。
- `UpdatePerPriorities` は CPU indices を使って ReplayBuffer の priority を更新し、更新後の sampling behavior / priority 値が従来と一致することを確認する。
- `HostReadback` は CPU immediate path を常時テストする。CUDA 利用可能時だけ、非同期 D2H の値一致と `synchronize()` 後に CPU tensor が読めることを追加で確認する（020 の `[transfer][cuda]` と同じゲート方針）。
- `TDLearner` と `QRLearner` の両方で、PER 有効時の `BatchUpdateResult.per_priorities` と `per_clipped_count` が CPU tensor として metrics に使えることを確認する。
- prefetch determinism regression は、`use_rb_prefetch=true` 同士の same-seed sampled index 列一致を主ガードにする。`true` vs `false` の一致は要求しない。
- 既存 regression は少なくとも `[replay_buffer][prefetch]`、`[dqn][prefetch][determinism]`、`[replay_buffer]`、`[transfer]`、full test を対象にする。
- `git diff --check` を必ず実行する。

## Out of Scope

- ReplayBuffer public interface の変更。
- `learner.use_rb_prefetch` 以外の config key 追加。
- GPU resident SumTree / GPU priority tree。
- `SampleIndices` の caller thread 化。
- N-deep prefetch / AsyncDataLoader 化。
- Push write-behind の再設計。
- PER 以外の observer metric 同期をすべて解消すること。
- MuZero replay buffer の転送境界変更。
- `HostReadback` の TensorDict / nested structure 対応（単一 tensor のみ）。

## Further Notes

- この PRD は `docs/memo/014_async_loader_prefetch_10prd.md`、ADR 0005、**`docs/memo/020`（transfer.hpp 共通部品）** の次段の性能修正である。既存の 1-step stale PER と PrefetchingReplayBuffer decorator 方針は維持する。
- User Story #7（small helper で CUDA event / pinned 詳細を learner に漏らさない）と #12（D2H helper の再利用性）は、020 の `EventRecycler` ＋ 本 PRD の `HostReadback` で充足する。新規の bespoke plumbing は作らない。
- 020 は `RecordStreamOn` を **無条件で** `Sample()` に入れたため、本 PRD が loss `.item()` / priority `.cpu()` の付随同期を消しても **H2D 結果側の早期再利用ハザードは既に塞がれている**。安心して同期源を削れる。
- `indices` を CPU 固定にすると、`ExperienceSamples::To` の名前から期待される「全 field が device に移る」挙動とは少しずれる。実装時はコメントと validation で、`indices` が learner input ではなく replay metadata であることを明示する。
- priority D2H を早めても、GPU 側で TD error の生成自体が遅い場合や D2H が optimizer より長い場合は wait が残る。その場合でも、Tracy 上で「転送開始が遅い」のか「転送時間が長い」のかを分けて判断できることを受け入れ条件に含める。
- 実測 acceptance は DropMerge の git build A/B で見る。期待値は `indices_cpu` の同期消滅、`priorities_cpu` 相当の待ちの縮小または overlap、wall-clock steps/sec の改善である。改善しない場合は、priority D2H 以外の同期源、GPU stream dependency、observer metric sync、または learner compute 側が支配的である可能性を再調査する。
