#include "anet/distribution.hpp"
#include <string>
#include "anet/tensor_util.hpp"
#include "anet/diag.hpp"

using namespace anet::rl;

CategoricalDistribution::CategoricalDistribution(const torch::Tensor& logits)
{
    // 形を先に確かめる。行動が 1 つしか無い分布は、抽選も学習も意味を持たない。
    if (!logits.defined() || logits.dim() != 2 || logits.size(1) < 2 || !logits.is_floating_point()) {
        ANET_SYSTEM_ERROR("Invalid categorical logits="
            << (logits.defined() ? anet::ToDefString(logits) : std::string("undefined"))
            << "; expected a floating-point tensor [rows, actions] with actions >= 2.");
    }
    // log π(a|s) = logits_a − log Σ_b exp(logits_b)。
    // log_softmax は最大の logits を引いてから指数を取るので、大きな logits でも桁あふれしない。
    // 半精度の logits でも確率の和が 1 からずれないよう、float32 で正規化する。
    log_probs_ = torch::log_softmax(logits.to(torch::kFloat32), 1);
    probs_ = log_probs_.exp();
}

torch::Tensor CategoricalDistribution::Sample(torch::Generator generator) const
{
    // 行ごとに、確率に比例して 1 つ選ぶ。抽選は微分できないので、結果は勾配を持たない。
    return torch::multinomial(probs_, 1, /*replacement=*/true, generator).squeeze(1);
}

torch::Tensor CategoricalDistribution::GetMode() const
{
    return log_probs_.argmax(1);
}

torch::Tensor CategoricalDistribution::ComputeLogProb(const torch::Tensor& actions) const
{
    // 行動は行ごとに 1 つの番号で受け取る。broadcast で別の形が通らないよう、形と型を確かめる。
    if (!actions.defined() || actions.dim() != 1 || actions.size(0) != log_probs_.size(0)
        || !torch::isIntegralType(actions.scalar_type(), /*includeBool=*/false)) {
        ANET_SYSTEM_ERROR("Invalid categorical actions="
            << (actions.defined() ? anet::ToDefString(actions) : std::string("undefined"))
            << "; expected an integer tensor [" << log_probs_.size(0) << "].");
    }
    // 選ばれた行動の列だけを取り出す。log_probs_ が勾配を持てば、結果も勾配を持つ。
    return log_probs_.gather(1, actions.to(torch::kInt64).unsqueeze(1)).squeeze(1);
}

torch::Tensor CategoricalDistribution::ComputeEntropy() const
{
    // H(π) = −Σ_a π(a) log π(a)。xlogy は π(a) = 0 の項を 0 にするので、
    // logits が −∞ の行動（合法手のマスクなど）があっても NaN にならない。
    return torch::xlogy(probs_, probs_).sum(1).neg();
}
