// stacker.cpp

#include "anet/stacker.hpp"
#include "anet/log.hpp" 


using namespace anet::rl;


// =============================================================
// StackerActionContext
// =============================================================

StackerActionContext::StackerActionContext(RunMode run_mode, std::shared_ptr<FrameStacker> stacker, std::optional<seed_t> seed)
    : ActionContext(run_mode, seed), stacker_(std::move(stacker))
{
    ;
}

torch::Tensor StackerActionContext::PushObservation(const BatchState& state)
{
    if (stacker_) {
        // BatchState から obs と episode_start を抽出して委譲
        return stacker_->Stack(state.obs, state.episode_start);
    }
    return state.obs;
}

void StackerActionContext::Reset()
{
    if (stacker_) stacker_->Reset();
}


// =============================================================
// TensorFrameStacker
// =============================================================

TensorFrameStacker::TensorFrameStacker(int stack_count, int batch_size, torch::Device device)
    : stack_count_(stack_count)
    , batch_size_(batch_size)
    , device_(device)
{
    ANET_ASSERT(stack_count_ > 0);
    ANET_ASSERT(batch_size_ > 0);
    //buffers_.resize(batch_size_);
}

torch::Tensor TensorFrameStacker::Stack(const torch::Tensor& data, const torch::Tensor& resets)
{
    // 入力チェック
    if (data.size(0) != batch_size_ || resets.size(0) != batch_size_) {
        ANET_SYSTEM_ERROR("TensorFrameStacker dimension mismatch. Expected batch_size="
            << batch_size_ << ", Got data=" << data.size(0));
    }

    // 念のためCPUに確実にあるかチェック
    auto data_cpu = data.device().is_cpu() ? data : data.to(torch::kCPU);
    auto resets_cpu = resets.device().is_cpu() ? resets.contiguous() : resets.to(torch::kCPU).contiguous();

    // 初回のみバッファをメモリ確保 (N, S, F...)
    if (!buffer_.defined()) {
        auto shape = data_cpu.sizes().vec();
        shape.insert(shape.begin() + 1, stack_count_); // Stack次元を追加
        buffer_ = torch::empty(shape, data_cpu.options());

        // 初回は全てのスタックを現在のフレームで埋める
        for (int k = 0; k < stack_count_; ++k) {
            buffer_.select(/*dim=*/1, /*index=*/k).copy_(data_cpu);
        }
        return buffer_.clone().to(device_); // 外部からの破壊を防ぐためcloneして返す
    }

    // 履歴をインプレースでスライド (左に1つ詰める)
    if (stack_count_ > 1) {
        buffer_.slice(/*dim=*/1, /*start=*/0, /*end=*/stack_count_ - 1)
            .copy_(buffer_.slice(/*dim=*/1, /*start=*/1, /*end=*/stack_count_).clone());
    }

    // 最新フレームを末尾に挿入
    buffer_.select(/*dim=*/1, /*index=*/-1).copy_(data_cpu);

    // リセットされた環境の処理
    auto resets_ptr = resets_cpu.data_ptr<bool>();
    for (int i = 0; i < batch_size_; ++i) {
        if (resets_ptr[i]) {
            // リセット時は、過去のスタックもすべて最新フレーム(data_cpu[i])で上書き
            // (末尾の stack_count_ - 1 番目は既に更新済みなのでループから省く)
            for (int k = 0; k < stack_count_ - 1; ++k) {
                buffer_[i][k].copy_(data_cpu[i]);
            }
        }
    }

    // 外部でのテンソル破壊を防ぐため clone を返す。
    //return (buffer_.dim() > 2 ? buffer_.flatten(1, 2) : buffer_).clone().to(device_);
    return buffer_.clone().to(device_);
}

void TensorFrameStacker::Reset()
{
    // バッファ破棄
    buffer_ = torch::Tensor();
}
