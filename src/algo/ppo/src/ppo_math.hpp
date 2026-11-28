#pragma once

#include <optional>
#include <torch/torch.h>

namespace anet::rl::ppo {

    /// 方策の損失と、診断に使う比。
    struct PolicyLoss {
        torch::Tensor loss;      ///< スカラー。clip した代理目的の符号を反転した平均。
        torch::Tensor ratio;     ///< [件数]。新しい方策と収集時の方策の確率の比。勾配を持つ。
        torch::Tensor log_ratio; ///< [件数]。比の対数。
    };

    /// value の損失と、clip が効いた割合。
    struct ValueLoss {
        torch::Tensor loss;          ///< スカラー。係数を掛ける前の値。
        torch::Tensor clip_fraction; ///< スカラー。clip を使わないときは未定義。
    };

    /// ミニバッチの advantage を、平均 0・標準偏差 1 に標準化する。入力は [件数] で、件数は 2 以上。
    torch::Tensor NormalizeAdvantages(const torch::Tensor& advantages);

    /// clip した代理目的（PPO-Clip）の損失を作る。入力はどれも [件数]。
    PolicyLoss ComputeClippedPolicyLoss(const torch::Tensor& new_log_probs, const torch::Tensor& old_log_probs,
        const torch::Tensor& advantages, float clip_range);

    /// value の損失を作る。入力はどれも [件数]。clip_range を渡すと、収集時の value からの変化を clip した損失と大きいほうを取る。
    ValueLoss ComputeValueLoss(const torch::Tensor& new_values, const torch::Tensor& old_values,
        const torch::Tensor& value_targets, std::optional<float> clip_range);

    /// 収集時の方策から新しい方策への KL の近似 mean((ratio − 1) − log ratio) を返す。常に 0 以上になる。
    torch::Tensor ComputeApproxKl(const torch::Tensor& ratio, const torch::Tensor& log_ratio);

    /// value が目標のばらつきをどれだけ説明するか 1 − Var(target − value) / Var(target) を返す。
    /// 目標の分散が 0 のときは NaN を返す。入力はどちらも [件数]。
    torch::Tensor ComputeExplainedVariance(const torch::Tensor& values, const torch::Tensor& value_targets);

} // namespace anet::rl::ppo
