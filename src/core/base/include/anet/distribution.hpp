#pragma once

#include <cstdint>
#include <torch/torch.h>

namespace anet::rl {

    /// 離散行動の分布。行ごとの logits から作り、計算と結果は logits と同じ device に置く。
    /// 値を CPU へ読み戻さないので、行動の番号が範囲に入っているかは検査しない。
    class CategoricalDistribution {
    public:
        /// logits は [行, 行動数] の浮動小数で、行動数は 2 以上。
        explicit CategoricalDistribution(const torch::Tensor& logits);

        /// 行ごとに 1 つの行動を抽選し、[行] の int64 で返す。generator は logits と同じ device のものを渡す。
        torch::Tensor Sample(torch::Generator generator) const;
        /// 行ごとに、確率が最大の行動を [行] の int64 で返す。
        torch::Tensor GetMode() const;
        /// [行] の整数の行動を受け取り、その行動の log 確率を [行] で返す。
        torch::Tensor ComputeLogProb(const torch::Tensor& actions) const;
        /// 行ごとの entropy を [行] で返す。
        torch::Tensor ComputeEntropy() const;

        const torch::Tensor& GetProbs() const { return probs_; }
        const torch::Tensor& GetLogProbs() const { return log_probs_; }
        int64_t GetActionCount() const { return log_probs_.size(1); }
    private:
        torch::Tensor log_probs_; ///< [行, 行動数]。logits を正規化した log 確率。
        torch::Tensor probs_;     ///< [行, 行動数]。
    };

} // namespace anet::rl
