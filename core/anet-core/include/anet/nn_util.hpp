#pragma once

#include <torch/torch.h>

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


}