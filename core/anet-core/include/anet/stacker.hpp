// anet/stacker.hpp

#pragma once
#include <vector>
#include <deque>
#include <torch/torch.h>
#include "anet/tensor_util.hpp"
#include "anet/rl.hpp"
#include "anet/agent.hpp"


namespace anet::rl {


    // ----------------------------------------------------------------------
    // FrameStacker Interface
    // ----------------------------------------------------------------------
    class FrameStacker {
    public:
        virtual anet::TensorDict Stack(const anet::TensorDict& data, const torch::Tensor& resets) = 0;
        virtual void Reset() = 0;
        virtual ~FrameStacker() = default;
    };


    // ----------------------------------------------------------------------
    // DictFrameStacker
    // ----------------------------------------------------------------------
    class DictFrameStacker final : public FrameStacker {
    public:
        DictFrameStacker(int stack_count, int batch_size, torch::Device device, std::optional<std::vector<std::string>> stack_keys = std::nullopt);

        anet::TensorDict Stack(const anet::TensorDict& data, const torch::Tensor& resets) override;
        void Reset() override;
    private:
        int stack_count_;
        int batch_size_;
        std::optional<std::vector<std::string>> stack_keys_;
        torch::Device device_;
        std::unordered_map<std::string, torch::Tensor> buffers_;    // Keyごとに履歴バッファを保持
    };


    // ----------------------------------------------------------------------
    // StackerActionContext
    // ----------------------------------------------------------------------

    class StackerActionContext final : public ActionContext {
    public:
        explicit StackerActionContext(RunMode run_mode, std::shared_ptr<FrameStacker> stacker, std::optional<seed_t> seed = std::nullopt);

        anet::TensorDict PushObservation(const BatchState& state) override;
        void Reset() override;
    private:
        std::shared_ptr<FrameStacker> stacker_;
    };


}
