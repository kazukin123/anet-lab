#pragma once

#include <torch/torch.h>
#include <ATen/autocast_mode.h>

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

    class Autocast {
    public:
        Autocast(torch::DeviceType device_type, bool enabled, torch::ScalarType dtype = torch::kHalf)
            : device_type_(device_type)
        {
            // 現在の状態を保存
            prev_enabled_ = at::autocast::is_autocast_enabled(device_type_);
            prev_dtype_ = at::autocast::get_autocast_dtype(device_type_);

            // 新しい状態を設定
            at::autocast::set_autocast_dtype(device_type_, dtype);
            at::autocast::set_autocast_enabled(device_type_, enabled);
        }

        ~Autocast()
        {
            at::autocast::set_autocast_enabled(device_type_, prev_enabled_);    // 元に戻す
            at::autocast::set_autocast_dtype(device_type_, prev_dtype_);        // 元に戻す
        }
    private:
        torch::DeviceType device_type_;
        bool prev_enabled_;
        torch::ScalarType prev_dtype_;
    };


    class GradScaler {
    public:
        GradScaler(double init_scale = 65536.0, double growth_factor = 2.0, double backoff_factor = 0.5, int64_t growth_interval = 2000)
            : scale_(init_scale), growth_factor_(growth_factor), backoff_factor_(backoff_factor), growth_interval_(growth_interval)
        {
        }

        // Lossをスケール
        torch::Tensor Scale(const torch::Tensor& loss)
        {
            return loss * static_cast<float>(scale_);
        }

        // Step (Inf/NaNチェック付き)
        // found_inf: 外部でNaNチェックした場合に true を渡す
        void Step(torch::optim::Optimizer& optimizer, bool found_inf = false)
        {
            if (!found_inf) {
                // 簡易チェック: まだUnscaleされていない場合などに備えて念のため
                // (外部でclip_grad_norm_前にチェック済みならここはfalseで来るはず)
                optimizer.step();
            }

            // 次のupdateのためにフラグ保存
            found_inf_ = found_inf;
        }

        // スケール更新
        void Update(std::optional<bool> found_inf_override = std::nullopt)
        {
            bool has_inf = found_inf_override.value_or(found_inf_);

            if (has_inf) {
                scale_ *= backoff_factor_;
                growth_tracker_ = 0;
            } else {
                growth_tracker_++;
                if (growth_tracker_ >= growth_interval_) {
                    scale_ *= growth_factor_;
                    growth_tracker_ = 0;
                }
            }
            if (scale_ < 1.0) scale_ = 1.0;
            found_inf_ = false; // リセット
        }

        // 手動Unscale (Optimizerの全パラメータの勾配を scale で割る)
        void Unscale_(torch::optim::Optimizer& optimizer)
        {
            double inv_scale = 1.0 / scale_;
            for (auto& group : optimizer.param_groups()) {
                for (auto& param : group.params()) {
                    if (param.grad().defined()) {
                        // In-placeで割り算
                        param.grad().mul_(inv_scale);
                    }
                }
            }
        }

    private:
        double scale_;
        double growth_factor_;
        double backoff_factor_;
        int64_t growth_interval_;
        int64_t growth_tracker_ = 0;
        bool found_inf_ = false;
    };
}