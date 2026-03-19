// anet/stacker.hpp

#pragma once
#include <vector>
#include <deque>
#include <torch/torch.h>
#include "anet/rl.hpp"
#include "anet/agent.hpp"


namespace anet::rl {


    // ----------------------------------------------------------------------
    // FrameStacker Interface
    // ----------------------------------------------------------------------
    class FrameStacker {
    public:
        virtual torch::Tensor Stack(const torch::Tensor& data, const torch::Tensor& resets) = 0;
        virtual void Reset() = 0;
        virtual ~FrameStacker() = default;
    };


    // ----------------------------------------------------------------------
    // StackerActionContext
    // ----------------------------------------------------------------------

    /// AgentのCreateActionContextで生成され、Runnerに保持される
    class StackerActionContext : public ActionContext {
    public:
        explicit StackerActionContext(RunMode run_mode, std::shared_ptr<FrameStacker> stacker, std::optional<seed_t> seed = std::nullopt);

        torch::Tensor PushObservation(const BatchState& state) override;
        void Reset() override;
    private:
        std::shared_ptr<FrameStacker> stacker_;
    };


    // ----------------------------------------------------------------------
    // StateFrameStacker Implementation
    // ----------------------------------------------------------------------
    class TensorFrameStacker : public FrameStacker {
    public:
        TensorFrameStacker(int stack_count, int batch_size, torch::Device device);

        torch::Tensor Stack(const torch::Tensor& data, const torch::Tensor& resets) override;
        void Reset() override;
    private:
        int stack_count_;
        int batch_size_;
        torch::Device device_;
        torch::Tensor buffer_;
    };

}
