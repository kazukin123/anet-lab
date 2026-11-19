#pragma once

#include <concepts>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>

namespace anet::transfer {

    /**
     * @brief copy stream へ非同期転送できる sample 型の要件。
     *
     * `To(device, non_blocking)` は転送後の同型オブジェクトを返す必要がある。
     * `ForEachTensor(fn)` は保持している defined tensor を列挙し、pinning や
     * `record_stream` のような leaf tensor 単位の処理を呼び出せるようにする。
     */
    template <class S>
    concept TransferableSamples =
        requires(S samples, torch::Device device, bool non_blocking) {
            { samples.To(device, non_blocking) } -> std::same_as<S>;
            samples.ForEachTensor([](torch::Tensor&) {});
        };

    /**
     * @brief CUDAEvent と payload lifetime をまとめて回収・再利用する補助クラス。
     *
     * copy stream へ積んだ非同期転送は、event 完了まで転送元や転送先の tensor storage を
     * 生かしておく必要がある。このクラスは完了 event と一緒に payload を保持し、
     * `Poll()` / `Acquire()` / `Drain()` のタイミングで完了済み event を再利用可能にする。
     *
     * `Retire()` 自体は CUDA 同期を行わない。shutdown や所有者破棄の前に全転送完了を保証
     * したい場合は `Drain()` が同期境界になる。
     */
    template <class Payload>
        requires std::movable<Payload>
    class EventRecycler {
    public:
        /// @brief 空の recycler を作成する。event は必要になるまで lazy に作られる。
        EventRecycler() = default;

        /// @brief 未完了 event をすべて同期回収してから破棄する。
        ~EventRecycler()
        {
            Drain();
        }

        /// @brief mutex と未完了 payload を持つためコピー不可。
        EventRecycler(const EventRecycler&) = delete;
        /// @brief mutex と未完了 payload を持つためコピー代入不可。
        EventRecycler& operator=(const EventRecycler&) = delete;
        /// @brief 所有者側で in-place に保持する前提のため move 不可。
        EventRecycler(EventRecycler&&) = delete;
        /// @brief 所有者側で in-place に保持する前提のため move 代入不可。
        EventRecycler& operator=(EventRecycler&&) = delete;

        /**
         * @brief 再利用可能な CUDAEvent を取得する。
         *
         * 呼び出し時に完了済み pending event を非同期に回収する。free event が無い場合は
         * default constructed event を返し、LibTorch 側で必要時に作られる。
         */
        at::cuda::CUDAEvent Acquire()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ReclaimCompletedLocked();
            if (free_.empty()) {
                return at::cuda::CUDAEvent{};
            }

            auto event = std::move(free_.back());
            free_.pop_back();
            return event;
        }

        /**
         * @brief 完了待ち event と、その完了まで生かす payload を pending に移す。
         *
         * この関数は `event.synchronize()` を呼ばない。payload は event が完了して
         * `Poll()` / `Acquire()` / `Drain()` で回収されるまで保持される。
         */
        void Retire(at::cuda::CUDAEvent event, Payload payload)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_.push_back(Pending{
                .event = std::move(event),
                .payload = std::move(payload)
            });
        }

        /// @brief 完了済み pending event を同期なしで free list に戻す。
        void Poll()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ReclaimCompletedLocked();
        }

        /**
         * @brief すべての pending event を同期して回収する。
         *
         * copy stream の非同期転送が残ったまま CUDA context や tensor storage が破棄される
         * ことを避けるため、所有者の停止処理やデストラクタで呼ぶ同期境界として使う。
         */
        void Drain()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            while (!pending_.empty()) {
                pending_.front().event.synchronize();
                free_.push_back(std::move(pending_.front().event));
                pending_.pop_front();
            }
        }

        /// @brief まだ完了回収されていない event/payload の数を返す。
        std::size_t PendingCount() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return pending_.size();
        }

        /// @brief 再利用可能な event の数を返す。
        std::size_t FreeCount() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return free_.size();
        }

    private:
        struct Pending {
            at::cuda::CUDAEvent event;
            Payload payload;
        };

        void ReclaimCompletedLocked()
        {
            while (!pending_.empty() && pending_.front().event.query()) {
                free_.push_back(std::move(pending_.front().event));
                pending_.pop_front();
            }
        }

        mutable std::mutex mutex_;
        std::vector<at::cuda::CUDAEvent> free_;
        std::deque<Pending> pending_;
    };

    /**
     * @brief sample オブジェクトを learner device へ転送した結果と、その完了 event を保持する。
     *
     * CPU constructor は同期的に `cpu_samples.To(device)` するだけで、`ready_event` を持たない。
     * CUDA constructor は source を pinned memory にそろえ、指定された copy stream 上で
     * `To(device, true)` を実行し、完了 event を `ready_event` に record する。
     *
     * CUDA path では `retained_source` が H2D 転送完了まで転送元 storage を保持する。
     * 利用側は `ready_event->block(consumer_stream)` 後に `RecordStreamOn(device_samples, consumer_stream)`
     * を呼び、consumer stream での使用中に allocator が storage を早期再利用しないようにする。
     */
    template <TransferableSamples Samples>
    struct DeviceTransfer {
        /// @brief 転送後の sample。CUDA path では copy stream 上で作られる。
        Samples device_samples;
        /// @brief 非同期転送完了まで転送元 CPU/pinned tensor の lifetime を保持する。
        Samples retained_source;
        /// @brief CUDA path の転送完了 event。CPU path では `std::nullopt`。
        std::optional<at::cuda::CUDAEvent> ready_event;

        /// @brief 空の transfer を作成する。後で move 代入される一時値用。
        DeviceTransfer() = default;

        /**
         * @brief sample を通常の `To(device)` で即時転送する。
         *
         * CUDA event を使わない同期/CPU 互換経路。`ready_event` は設定されない。
         */
        DeviceTransfer(Samples cpu_samples, torch::Device device)
            : device_samples(cpu_samples.To(device))
        {
        }

        /**
         * @brief sample を pinned source 経由で copy stream に非同期転送する。
         *
         * `cpu_samples` 内の tensor は `pin_memory()` 済みの `retained_source` に変換される。
         * `device_samples` は `copy_stream` guard 下で `non_blocking=true` により作られ、
         * 最後に `ready_event` が `copy_stream` へ record される。
         *
         * @param cpu_samples 転送元 sample。move され、完了まで `retained_source` に保持される。
         * @param device 転送先 device。
         * @param copy_stream H2D copy を enqueue する stream。
         * @param event_recycler ready event を取得する recycler。
         */
        DeviceTransfer(
            Samples cpu_samples,
            torch::Device device,
            at::cuda::CUDAStream copy_stream,
            EventRecycler<Samples>& event_recycler)
            : retained_source(std::move(cpu_samples))
        {
            PinRetainedSource();
            ready_event.emplace(event_recycler.Acquire());
            {
                at::cuda::CUDAStreamGuard guard(copy_stream);
                device_samples = retained_source.To(device, /*non_blocking=*/true);
            }
            ready_event->record(copy_stream);
        }

        /// @brief 非同期転送の所有権を移すため move 可能。
        DeviceTransfer(DeviceTransfer&&) = default;
        /// @brief 非同期転送の所有権を移すため move 代入可能。
        DeviceTransfer& operator=(DeviceTransfer&&) = default;
        /// @brief event と retained source の二重所有を避けるためコピー不可。
        DeviceTransfer(const DeviceTransfer&) = delete;
        /// @brief event と retained source の二重所有を避けるためコピー代入不可。
        DeviceTransfer& operator=(const DeviceTransfer&) = delete;

    private:
        void PinRetainedSource()
        {
            retained_source.ForEachTensor([](torch::Tensor& tensor) {
                if (!tensor.defined()) return;
                if (!tensor.device().is_cpu()) {
                    tensor = tensor.to(torch::kCPU);
                }
                if (tensor.is_pinned()) return;

                tensor = tensor.pin_memory();
            });
        }
    };

    /**
     * @brief sample 内の CUDA tensor を consumer stream に record する。
     *
     * copy stream 上で作られた `device_samples` を別の stream で使う場合、event wait だけでは
     * caching allocator が tensor storage を copy stream 完了後すぐ再利用できてしまう。
     * `ready_event->block(consumer_stream)` の直後にこの関数を呼び、consumer stream 上の使用が
     * 終わるまで storage が生きることを allocator に知らせる。
     *
     * CPU tensor と undefined tensor は無視される。
     *
     * @param samples record 対象の sample。
     * @param consumer_stream 転送結果を実際に消費する stream。
     */
    template <TransferableSamples Samples>
    void RecordStreamOn(Samples& samples, at::cuda::CUDAStream consumer_stream)
    {
        samples.ForEachTensor([&](torch::Tensor& tensor) {
            if (tensor.defined() && tensor.is_cuda()) {
                tensor.record_stream(consumer_stream);
            }
        });
    }

    /**
     * @brief 単一 tensor の D2H readback を enqueue し、CPU materialize を遅延する。
     *
     * CPU tensor を渡した場合は即時に `pinned_result` として保持し、event は使わない。
     * CUDA tensor を渡した場合は producer stream に source-ready event を record し、
     * copy stream がそれを待って pinned CPU tensor へ non-blocking copy を enqueue する。
     * copy 完了は `ready_event` に record され、利用側は `Wait()` 後に `pinned_result` を読む。
     *
     * `RetireEvents()` は `Wait()` 後に呼ぶ。source-ready event と done event を recycler へ返し、
     * `retained_source` を done event 完了まで保持する。
     */
    struct HostReadback {
        /// @brief CPU 側で読む結果 tensor。CUDA path では pinned CPU tensor。
        torch::Tensor pinned_result;
        /// @brief D2H copy 完了まで GPU source tensor の lifetime を保持する。
        torch::Tensor retained_source;
        /// @brief producer stream 上で source が読めることを示す event。
        std::optional<at::cuda::CUDAEvent> source_ready_event;
        /// @brief copy stream 上の D2H copy 完了 event。
        std::optional<at::cuda::CUDAEvent> ready_event;

        /// @brief 空の readback を作成する。後で move 代入される一時値用。
        HostReadback() = default;

        /**
         * @brief CPU tensor を即時 readback 結果として保持する。
         *
         * 入力が誤って CPU 以外の場合は同期的に CPU へ移す。通常は CPU path 用で、
         * `Wait()` と `RetireEvents()` は実質 no-op になる。
         */
        explicit HostReadback(torch::Tensor cpu_source)
            : pinned_result(std::move(cpu_source))
        {
            if (pinned_result.defined() && !pinned_result.device().is_cpu()) {
                pinned_result = pinned_result.to(torch::kCPU);
            }
        }

        /**
         * @brief CUDA tensor の D2H copy を copy stream へ非同期 enqueue する。
         *
         * `producer_stream` で source-ready event を record し、`copy_stream` はその event を待って
         * pinned CPU destination へ `copy_(..., non_blocking=true)` を積む。最後に done event を
         * `copy_stream` へ record する。
         *
         * @param gpu_source readback 対象 tensor。move され、完了まで `retained_source` に保持される。
         * @param copy_stream D2H copy を enqueue する stream。
         * @param producer_stream `gpu_source` を最後に生成・更新した stream。
         * @param event_recycler source-ready / done event を取得する recycler。
         */
        HostReadback(
            torch::Tensor gpu_source,
            at::cuda::CUDAStream copy_stream,
            at::cuda::CUDAStream producer_stream,
            EventRecycler<torch::Tensor>& event_recycler)
            : retained_source(std::move(gpu_source))
        {
            if (!retained_source.defined()) return;
            if (!retained_source.is_cuda()) {
                pinned_result = retained_source;
                return;
            }

            const auto options = retained_source.options().device(torch::kCPU).pinned_memory(true);
            pinned_result = torch::empty_strided(retained_source.sizes(), retained_source.strides(), options);

            source_ready_event.emplace(event_recycler.Acquire());
            source_ready_event->record(producer_stream);
            ready_event.emplace(event_recycler.Acquire());
            {
                at::cuda::CUDAStreamGuard guard(copy_stream);
                source_ready_event->block(copy_stream);
                pinned_result.copy_(retained_source, /*non_blocking=*/true);
            }
            ready_event->record(copy_stream);
        }

        /// @brief readback の所有権を移すため move 可能。
        HostReadback(HostReadback&&) = default;
        /// @brief readback の所有権を移すため move 代入可能。
        HostReadback& operator=(HostReadback&&) = default;
        /// @brief event と source lifetime の二重所有を避けるためコピー不可。
        HostReadback(const HostReadback&) = delete;
        /// @brief event と source lifetime の二重所有を避けるためコピー代入不可。
        HostReadback& operator=(const HostReadback&) = delete;

        /**
         * @brief D2H copy 完了まで待つ。
         *
         * CUDA path では `ready_event` を synchronize する。CPU path や空 readback では何もしない。
         * `pinned_result` の値を CPU から読む前の同期境界として使う。
         */
        void Wait()
        {
            if (ready_event) {
                ready_event->synchronize();
            }
        }

        /**
         * @brief readback で使った event を recycler へ返す。
         *
         * `Wait()` 後に呼ぶ想定。source-ready event は payload なしで返し、done event は
         * `retained_source` と一緒に返して、copy stream 完了まで GPU source storage を保持する。
         */
        void RetireEvents(EventRecycler<torch::Tensor>& event_recycler)
        {
            if (source_ready_event) {
                event_recycler.Retire(std::move(*source_ready_event), torch::Tensor{});
                source_ready_event.reset();
            }
            if (ready_event) {
                event_recycler.Retire(std::move(*ready_event), std::move(retained_source));
                ready_event.reset();
            }
        }
    };

}  // namespace anet::transfer
