# Device 転送共通部品 `anet/transfer.hpp` と PrefetchingReplayBuffer 適用 PRD

> 関連: `docs/memo/014`（prefetch decorator）, `docs/adr/0005`（sample prefetch stale PER）, `docs/memo/021`（PER priority async D2H）。

## Context / Problem Statement

`learner.use_rb_prefetch=true` の `PrefetchingReplayBuffer` で、Nsight 実測により `Sample()` hot path の主因が ReplayBuffer sample でも H2D でもなく **`c10::cuda::CUDAEvent::~CUDAEvent -> cudaEventDestroy`（代表 +7.459ms）** と判明している。原因は、`PrefetchedBatch` が `Sample()` 末尾で破棄され、保持していた `CUDAEvent` を **毎 batch hot path で destroy** していたこと。event の record/wait 自体は H2D と learner stream を安全につなぐために必要で、消すべきではない。

現状 `replay_buffer_impl.cpp` には短時間 profiling 用の実験コード（`ANET_EXPERIMENT_RETIRE_PREFETCH_READY_EVENT` ＝ `retired_batches` に退避して destroy を回避）が残るが、長時間 run で pinned memory が増え続ける袋小路。

恒久対策として、PyTorch の caching allocator と同型の **「event ハンドルは再利用プール／pending な (event,payload) はポーリングで随時回収」** を、ReplayBuffer ローカルではなく **CPU/CUDA 両対応の共通部品**として `core/anet-core/include/anet/transfer.hpp` に切り出す。`docs/memo/021`（PER priority の async D2H materialization）でも同じ部品を再利用できるよう、最初から一般化して作る（「後で一般化」は来ない、という判断）。

## Solution（概要）

`anet/transfer.hpp` に 2 部品＋1 concept＋1 helper を定義し、`PrefetchingReplayBuffer` をこれを使う形へ置換する。

- **`EventRecycler<Payload>`**: 完了待ち資源の回収器。`CUDAEvent` を再利用プールし、転送元 `payload` を event 完了まで生かして遅延解放。`cudaEventDestroy` は dtor/`Drain` のみ。
- **`DeviceTransfer<Samples>`**: 1 回の転送結果（全公開メンバの値オブジェクト＝struct）。`pin→H2D→record` を行う **constructor** を持ち、`CUDAEvent` を可視フィールドで返す（隠さない）。CPU 用の同期 constructor も持つ。
- **`TransferableSamples` concept**: `Samples` が `To(device, non_blocking)`（既存）＋ `ForEachTensor(fn)`（新規）を満たすこと。
- **`RecordStreamOn(samples, stream)`**: device 結果の全 CUDA tensor に `record_stream` を打つ generic helper。

### 設計判断（why。self-contained のため明記）

1. **event を隠さない（可視 `CUDAEvent`）**: `block(stream)`（stream 待ち・CPU 止めない／H2D consumer）と `synchronize()`（host 待ち／将来 D2H consumer）と `query()`（回収判定）を**呼び側が選べる**。Fence 抽象を作らないことで H2D/D2H を同じ結果型で扱える。
2. **再利用の安全性 = query ゲート**: event が `free_` に戻るのは `query()==true` 確認後のみ。未完了 event の再 record で前の待ち手を壊すことが**構造的に起きない**。
3. **`cudaEventDestroy` は dtor/`Drain` のみ**。`Acquire`/`Retire`/`Poll` では絶対に走らない（元バグの根絶）。
4. **event 再利用と payload 解放を 1 部品に束ねる**: pinned 解放可能時刻 = event 完了時刻 = event 再利用可能時刻 で同一。完了判定ループを 1 つにできるため、別部品にするより簡潔（`1関心1機構` ＝「完了待ち資源の回収」という単一関心）。
5. **cap は持たない**: in-flight は呼び側の 1-deep prefetch（`LaunchPrefetchLocked` の `if (future.valid()) return;`）で back-pressure され、event 総数は ≈depth+1 で有界。blocking cap は「worker Acquire ↔ learner future.get」のデッドロックを生むため**入れない**。異常検知が要るなら非ブロッキング `ANET_ASSERT`（任意）。
6. **`record_stream` 契約（部品1の核心）**: device 結果は `copy_stream` 上で確保されるため、consumer の compute stream で使う間に allocator が早期再利用してデータ破壊するハザードがある。`block` 後に `RecordStreamOn(device_samples, compute_stream)` を打って防ぐ。現状コードは付随同期（loss `.item()` / priority `.cpu()`）で**たまたま masking** されている可能性が高く、**021 がその同期を消すと顕在化**しうるため、ここで寿命契約を明示する。
7. **転送は型の `To` に委譲、pin/record_stream は `ForEachTensor` で generic**: `indices` を CPU 据え置きにする等の per-field 方針（021）は `ExperienceSamples::To` が握るので転送はそれを使う。pin（CPU tensor を pin）と record_stream（CUDA tensor のみ）は方針非依存の均一操作なので `ForEachTensor` 一点で部品側が汎用実装。
8. **一般化は template（compile-time concept）**: virtual interface は使わない（AGENTS.md「production で型分岐しない」「性能に妥協しない」、オーバーヘッドゼロ）。
9. **スレッド**: `Acquire`=worker（`Fetch`）/ `Retire`=consumer（`Sample`）の別スレッドを内部 mutex で保護。`Retire` は CUDA 呼び出しなし（consumer hot path 最小）。回収 query は `Acquire`（worker 側）に寄せ、consumer 側の critical path から CUDA 回収処理を排除。
10. **`EventRecycler` は mutex 保持で move 不可**。`PrefetchingReplayBuffer::State` が `copy_stream` と並べて **in-place 所有**。

## 公開 API スケルトン（`core/anet-core/include/anet/transfer.hpp`）

```cpp
#pragma once
#include <torch/torch.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAContext.h>   // CUDAStream / getCurrentCUDAStream / CUDAStreamGuard
#include <concepts>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace anet::transfer {

// Samples 契約: To(既存) + ForEachTensor(新規。全 leaf tensor を可変訪問)
template <class S>
concept TransferableSamples =
    requires(S s, torch::Device device, bool non_blocking) {
        { s.To(device, non_blocking) } -> std::same_as<S>;
        s.ForEachTensor([](torch::Tensor&) {});
    };

// 部品2: 完了待ち資源の回収器（Payload は movable なら何でも）
template <class Payload>
    requires std::movable<Payload>
class EventRecycler {
public:
    EventRecycler() = default;
    ~EventRecycler();                          // free_ の event をここで破棄（唯一の cudaEventDestroy）
    EventRecycler(const EventRecycler&) = delete;
    EventRecycler& operator=(const EventRecycler&) = delete;
    // mutex 保持のため move 不可。State が in-place 所有。

    at::cuda::CUDAEvent Acquire();                                  // worker: 完了分回収後に再利用 or 新規(lazy)
    void Retire(at::cuda::CUDAEvent event, Payload payload);        // consumer: 完了まで保持。CUDA 呼び出し無し
    void Poll();                                                   // 任意: 完了分回収。CPU 待ちなし
    void Drain();                                                  // shutdown: synchronize して全回収
    std::size_t PendingCount() const;
    std::size_t FreeCount() const;

private:
    struct Pending { at::cuda::CUDAEvent event; Payload payload; };
    void ReclaimCompletedLocked();             // 先頭から query、完了で回収、未完了で停止
    mutable std::mutex mutex_;
    std::vector<at::cuda::CUDAEvent> free_;     // query==true 確認済み
    std::deque<Pending> pending_;               // record 順
};

// 部品1: 1 回の転送結果（struct。生成は転送する constructor）
template <TransferableSamples Samples>
struct DeviceTransfer {
    Samples device_samples;                          // 使用本体（consumer 保持）
    Samples retained_source;                         // 転送元 pinned（完了まで生存→Retire）
    std::optional<at::cuda::CUDAEvent> ready_event;  // 完了マーカ（可視）。CPU 経路は nullopt

    DeviceTransfer() = default;
    DeviceTransfer(Samples cpu_samples, torch::Device device);  // CPU: 同期、event/pin 無し
    DeviceTransfer(Samples cpu_samples, torch::Device device,   // CUDA: pin→H2D→record
                   at::cuda::CUDAStream copy_stream, EventRecycler<Samples>& event_recycler);
    DeviceTransfer(DeviceTransfer&&) = default;
    DeviceTransfer& operator=(DeviceTransfer&&) = default;
    DeviceTransfer(const DeviceTransfer&) = delete;
    DeviceTransfer& operator=(const DeviceTransfer&) = delete;
};

// consumer helper: device 結果の CUDA tensor に record_stream
template <TransferableSamples Samples>
void RecordStreamOn(Samples& samples, at::cuda::CUDAStream consumer_stream);

}  // namespace anet::transfer
```

実装の肝（定義はヘッダ内、template のため）:
- `Acquire`: lock → `ReclaimCompletedLocked()` → `free_` 末尾 pop or `at::cuda::CUDAEvent{}`（lazy 新規）。
- `Retire`: lock → `pending_.push_back({move,move})` のみ。
- `ReclaimCompletedLocked`: `while (!pending_.empty() && pending_.front().event.query()) { free_へ move; pop_front; }`。
- CUDA ctor: ①`retained_source = move(cpu_samples)`、`ForEachTensor([](t){ if(t.is_cpu()&&!t.is_pinned()) t=t.pin_memory(); })` ②`ready_event.emplace(event_recycler.Acquire())` ③`{ CUDAStreamGuard g(copy_stream); device_samples = retained_source.To(device, true); }` ④`ready_event->record(copy_stream)`。
- CPU ctor: `device_samples = cpu_samples.To(device);`（`ready_event` は nullopt）。
- `RecordStreamOn`: `samples.ForEachTensor([&](t){ if (t.is_cuda()) t.record_stream(consumer_stream); });`
- `~EventRecycler`: `Drain()` 後 `free_` の event が破棄される。`Drain`: pending を `synchronize()` して `free_` へ。

## Samples 契約と既存型の変更（最小）

- `TensorDict::ForEachTensor(F&& fn)` を `tensor_util.hpp` に追加: `for (auto& kv : dict_) fn(kv.second);`（in-place 可変。`To`/`Cpu` と同じ `dict_` 反復）。
- `ExperienceSamples::ForEachTensor(F&& fn)` を `rl.hpp` 宣言 / `rl.cpp` 実装で追加: `obs.ForEachTensor(fn); fn(actions); fn(target_returns); next_state.next_obs.ForEachTensor(fn); fn(next_state.terminals); fn(n_steps); fn(indices); fn(is_weights); info.ForEachTensor(fn);`（undefined tensor はスキップ）。
- 既存 `ExperienceSamples::To`（rl.hpp:715）は温存（転送方針の置き場）。

## PrefetchingReplayBuffer の変更（`replay_buffer_impl.cpp` 中心）

CUDA ヘッダを `replay_buffer.hpp` に漏らさないため、`State`/`PrefetchedBatch` は前方宣言のまま（replay_buffer.hpp:95-99 はそのまま）、`transfer.hpp` の include は **.cpp のみ**。

- `replay_buffer_impl.cpp` 冒頭で `#include "anet/transfer.hpp"`。
- `State`（.cpp:1567）に **`transfer::EventRecycler<ExperienceSamples> event_recycler;` を in-place 追加**（`copy_stream` の隣）。`ANET_EXPERIMENT_RETIRE_PREFETCH_READY_EVENT`・`retired_batches` を**削除**。
- `PrefetchedBatch`（.cpp:1557）を `transfer::DeviceTransfer<ExperienceSamples>` で表現（前方宣言名 `PrefetchedBatch` を保つなら `struct PrefetchingReplayBuffer::PrefetchedBatch : transfer::DeviceTransfer<ExperienceSamples> {};` 等の薄い定義、または内部 alias）。`ready_event` は `optional<CUDAEvent>` 値（旧 `unique_ptr` 廃止、heap alloc 1 個減）。
- `TransferSamples`/`Fetch`（.cpp:1700-1746）: pin/copy/record の手書きを **`DeviceTransfer` ctor 呼び出しに置換**。CPU/CUDA は ctor のアリティで分岐（呼び側 `is_cuda()` 判定）。`PrefetchPinCpuSamples`/`PrefetchPinCpuTensor`/`PrefetchPinCpuTensorDict`（.cpp:1500-1555）は**削除**（pin は ctor の `ForEachTensor` へ）。
- `Sample`（.cpp:1621-1661）: `event_wait` 区間を `if (t.ready_event){ t.ready_event->block(getCurrentCUDAStream()); transfer::RecordStreamOn(t.device_samples, getCurrentCUDAStream()); }` に。`out_samples = std::move(t.device_samples);` 後、`if (t.ready_event) state_->event_recycler.Retire(std::move(*t.ready_event), std::move(t.retained_source));`。実験用 `retire_ready_event` ブロックは撤去。
- `StopPrefetch`（.cpp:1790）: `WaitForPrefetchLocked()/WaitForQueuedPushesLocked()` の後に **`state_->event_recycler.Drain();`**（CUDA context 健在のうちに）。`retired_batches.swap` 撤去。
- `<ATen/cuda/CUDAEvent.h>`/`<ATen/cuda/CUDAContext.h>` は `transfer.hpp` が供給するので、.cpp 側の直接 include は整理可（任意）。`rl.cpp:4` の未使用 `<ATen/cuda/CUDAEvent.h>` 撤去は任意の cleanup。

## Testing Decisions

- **CPU（CUDA 不要・常時）**:
  - `ExperienceSamples::ForEachTensor` が全 tensor（TensorDict 内含む）を訪問し、undefined をスキップすること。
  - `DeviceTransfer` CPU ctor が `device_samples` を CPU に揃え `ready_event==nullopt`。
  - 既存 `[replay_buffer][prefetch]`（CPU、deterministic / accessor / priority-wait / push-FIFO / shallow-alive、test:861-1052）が**そのまま pass**。determinism（`use_rb_prefetch=true` 同士の sampled index 列一致＝ADR 0005 主契約）維持。
  - `EventRecycler` の bookkeeping（未 record event は `query()==true` 扱いで `Retire→Acquire` 後に再利用、`PendingCount`/`FreeCount` 推移、`Drain` で空）。新タグ `[transfer]`。
- **CUDA（device 利用可時のみ）**:
  - event 再利用で steady-state の event 総数が depth+α に有界（無限増加しない）。
  - `Sample()` hot path で `cudaEventDestroy` が呼ばれないこと（カウンタ/Tracy zone）。
  - 値の正しさ（H2D 結果が同期版と一致、`RecordStreamOn` 後の使用で破壊なし）。可能なら小容量 allocator で H2D→即解放→再確保のストレス。
- `[replay_buffer][prefetch]`・`[dqn][prefetch][determinism]`・`[replay_buffer]`・`[transfer]`・full を対象。`git diff --check` 必須。

## Out of Scope

- D2H（021 の PER priority materialization）本体実装。本部品が `synchronize()`/`Retire` で再利用可能であることの確認まで。
- N-deep prefetch / AsyncDataLoader 化。
- ReplayBuffer public interface・config key の変更。
- `true` vs `false` prefetch の bit 一致要求（ADR 0005: `true` 同士の deterministic が主契約）。
- 未コミットの他作業の revert。

## 変更ファイル

- 新規: `core/anet-core/include/anet/transfer.hpp`（公開ヘッダ、template 定義含む）。
- 編集: `core/anet-core/include/anet/tensor_util.hpp`（`TensorDict::ForEachTensor`）。
- 編集: `core/anet-core/include/anet/rl.hpp` + `core/anet-core/src/rl.cpp`（`ExperienceSamples::ForEachTensor`、任意で dead include 撤去）。
- 編集: `core/anet-core/src/replay_buffer_impl.cpp`（State/PrefetchedBatch/Transfer/Sample/StopPrefetch、実験コード・PrefetchPinCpu* 撤去）。`replay_buffer.hpp` は原則無改修。
- 編集: `core/anet-core/src/replay_buffer_test.cpp`（`[transfer]` 追加、既存 prefetch 維持）。

## Verification

ビルド（MSVC 初期化必須）:
```
cmd /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build --preset x64-Debug --target anet-core-test'
```
テスト:
```
core\anet-core\bin\Debug\anet-core-test.exe "[transfer]"
core\anet-core\bin\Debug\anet-core-test.exe "[replay_buffer][prefetch]"
core\anet-core\bin\Debug\anet-core-test.exe
cmd /s /c "git diff --check"
```
CUDA 実機受け入れ（RelWithDebInfo / Tracy or Nsight）:
- `PrefetchingReplayBuffer::Sample` から `cudaEventDestroy` が消える。
- steady-state の event 総数が有界（pinned 無限増加が無い）。
- DropMerge の git build A/B で wall-clock steps/sec が劣化しない（できれば改善）。determinism regression 無し。

## Further Notes

- 名前は仮確定: namespace `anet::transfer`、`EventRecycler` / `DeviceTransfer` / `RecordStreamOn` / `TransferableSamples`。「CUDA」を型名に含めない（CPU 互換のため）。最終はクラス名とファイル名を揃える。
- 021 連携: D2H は同じ `EventRecycler` を再利用し、consumer が `synchronize()`（host 待ち）を選ぶ。copy 前に producer-ready を待つ precondition は D2H 側の追加引数で対応（本 PRD では作らない）。
- 017/da86c7c の Push micro-opt は性能劣化で revert 済み（50de61c）。本件は別系統（Sample 側の event lifetime）。
