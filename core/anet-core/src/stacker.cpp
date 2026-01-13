// stacker.cpp

#include "anet/stacker.hpp"
#include "anet/log.hpp" 


using namespace anet::rl;


// =============================================================
// StackerActionContext
// =============================================================

StackerActionContext::StackerActionContext(RunMode run_mode, std::shared_ptr<FrameStacker> stacker)
    : ActionContext(run_mode), stacker_(std::move(stacker))
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
    buffers_.resize(batch_size_);
}

torch::Tensor TensorFrameStacker::Stack(const torch::Tensor& data, const torch::Tensor& resets)
{
    // 入力チェック (不整合があれば即座に検知)
    if (data.size(0) != batch_size_ || resets.size(0) != batch_size_) {
        ANET_SYSTEM_ERROR("TensorFrameStacker dimension mismatch. Expected batch_size="
            << batch_size_ << ", Got data="  << data.size(0));
    }

    // デバイス操作 (deque管理のためCPUへ)
    auto data_cpu = data.to(torch::kCPU);
    auto resets_cpu = resets.to(torch::kCPU);
    auto resets_ptr = resets_cpu.data_ptr<bool>();

    std::vector<torch::Tensor> stacked_batch(batch_size_);

    for (int i = 0; i < batch_size_; ++i) {
        torch::Tensor current_frame = data_cpu[i];
        bool is_reset = resets_ptr[i];
        auto& buffer = buffers_[i];

        if (is_reset || buffer.empty()) {
            // Reset or Init -> Padding
            buffer.clear();
            for (int k = 0; k < stack_count_; ++k) {
                buffer.push_back(current_frame);
            }
        } else {
            // Update -> Slide
            buffer.pop_front();
            buffer.push_back(current_frame);
        }

        // Stack (S, F...)
        std::vector<torch::Tensor> frame_vec(buffer.begin(), buffer.end());
        stacked_batch[i] = torch::stack(frame_vec, 0);
    }

    // Batch Stack (N, S, F...)
    auto output = torch::stack(stacked_batch, 0);
    return output.to(device_);
}

void TensorFrameStacker::Reset()
{
    for (auto& buf : buffers_) buf.clear();
}
