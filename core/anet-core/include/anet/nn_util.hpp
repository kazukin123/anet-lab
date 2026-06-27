#pragma once

#include <torch/torch.h>

#include <optional>
#include <unordered_map>
#include <vector>

namespace anet {

    template <typename HolderType>
    void ApplyHeNormal(HolderType& holder, double alpha = 0.0) {
        auto module = holder.ptr();
        if (!module) return;

        for (auto& p : module->named_parameters(/*recurse=*/false)) {
            const auto& k = p.key();
            auto& t = p.value();
            if (k == "weight") {
                torch::nn::init::kaiming_normal_(t, alpha, torch::kFanIn, torch::kReLU);
            }
            else if (k == "bias") {
                torch::nn::init::zeros_(t);
            }
        }
    }

    template <typename HolderType>
    void ApplyXavierUniform(HolderType& holder, double gain = 1.0) {
        auto module = holder.ptr();
        if (!module) return;

        for (auto& p : module->named_parameters(/*recurse=*/false)) {
            const auto& k = p.key();
            auto& t = p.value();
            if (k == "weight") {
                torch::nn::init::xavier_uniform_(t, gain);
            }
            else if (k == "bias") {
                torch::nn::init::zeros_(t);
            }
        }
    }

    std::vector<torch::Tensor> CollectDefinedGrads(const std::vector<torch::Tensor>& parameters);

    torch::Tensor ForeachGradNorm(const std::vector<torch::Tensor>& grads);

    void ForeachClipGradNorm_(const std::vector<torch::Tensor>& grads, const torch::Tensor& total_norm, float tau);


    // ======================================================
    // Autocast
    // ======================================================

    class Autocast {
    public:
        Autocast(torch::DeviceType device_type, bool enabled, torch::ScalarType dtype = torch::kHalf);
        ~Autocast();
    private:
        const torch::DeviceType device_type_;
        const bool prev_enabled_;
        const torch::ScalarType prev_dtype_;
    };


    // ======================================================
    // TrainingModeGuard
    // ======================================================

    class TrainingModeGuard {
    public:
        TrainingModeGuard(torch::nn::Module& module, bool training)
            : module_(module)
            , was_training_(module.is_training())
        {
            module_.train(training);
        }

        ~TrainingModeGuard()
        {
            module_.train(was_training_);
        }

        TrainingModeGuard(const TrainingModeGuard&) = delete;
        TrainingModeGuard& operator=(const TrainingModeGuard&) = delete;
        TrainingModeGuard(TrainingModeGuard&&) = delete;
        TrainingModeGuard& operator=(TrainingModeGuard&&) = delete;
    private:
        torch::nn::Module& module_;
        const bool was_training_;
    };


    // ======================================================
    // FusedAdamW
    // ======================================================

    class FusedAdamW : public torch::optim::AdamW {
    public:
        using torch::optim::AdamW::AdamW;

        torch::Tensor step(LossClosure closure = nullptr) override;
        void load(torch::serialize::InputArchive& archive) override;
    private:
        struct FusedAdamWStepGroup;
    private:
        std::unordered_map<void*, torch::Tensor> step_tensors_;
    };


    // ======================================================
    // GradScaler
    // ======================================================

    class GradScaler {
    public:
        GradScaler(double init_scale = 65536.0, double growth_factor = 2.0, double backoff_factor = 0.5, int64_t growth_interval = 2000);

        // Lossをスケール
        torch::Tensor Scale(const torch::Tensor& loss)
        {
            return loss * static_cast<float>(scale_);
        }

        // Step (直前のUnscale_で検出したInf/NaNに応じてスキップ)
        void Step(torch::optim::Optimizer& optimizer);

        // Step (Inf/NaNチェック付き)
        // found_inf: 外部でNaNチェックした場合に true を渡す
        void Step(torch::optim::Optimizer& optimizer, bool found_inf);

        // スケール更新
        void Update(std::optional<bool> found_inf_override = std::nullopt);

        // 手動Unscale (勾配の非有限値検出とscale除算をforeachで行う)
        void Unscale_(torch::optim::Optimizer& optimizer);
    private:
        double scale_;
        const double growth_factor_;
        const double backoff_factor_;
        const int64_t growth_interval_;
        int64_t growth_tracker_ = 0;
        bool found_inf_ = false;
        torch::Tensor found_inf_tensor_;
    };
}
