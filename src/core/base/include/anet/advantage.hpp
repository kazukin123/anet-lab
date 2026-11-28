#pragma once

#include <torch/torch.h>

namespace anet::rl {

    /// advantage と、value の学習の目標の組。どちらも [時刻, lane] で、入力と同じ device にある。
    struct GaeResult {
        torch::Tensor advantages;
        torch::Tensor value_targets; ///< advantages に、入力の values を足した値。
    };

    /// Generalized Advantage Estimation（Schulman ら 2016）で、advantage と value の目標を作る。
    ///
    ///   delta[t]     = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
    ///   advantage[t] = delta[t] + gamma * lambda * (1 - episode_end[t]) * advantage[t + 1]
    ///
    /// 入力はどれも [時刻, lane] で、同じ device に置く。最後の時刻の次の advantage は 0 とする。
    /// @param rewards       その時刻の遷移の報酬。
    /// @param values        その時刻の観測の value。
    /// @param next_values   その時刻の遷移の次の観測の value。episode が終わる時刻では、Reset の前の観測の value を渡す。
    /// @param terminated    真の時刻では、次の観測の value を足さない（終端）。bool。
    /// @param episode_end   真の時刻では、次の時刻の advantage を伝えない（終端か打ち切り）。bool。
    /// @param gamma         割引率。[0, 1]。
    /// @param lambda        GAE の λ。[0, 1]。0 は 1 step の TD 誤差、1 は割引の和になる。
    GaeResult ComputeGae(const torch::Tensor& rewards, const torch::Tensor& values, const torch::Tensor& next_values,
        const torch::Tensor& terminated, const torch::Tensor& episode_end, double gamma, double lambda);

} // namespace anet::rl
