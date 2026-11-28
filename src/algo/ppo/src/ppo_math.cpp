#include "ppo_math.hpp"
#include <limits>
#include <string>
#include <utility>
#include "anet/tensor_util.hpp"
#include "anet/diag.hpp"

using namespace anet::rl::ppo;

// 標準化の分母に足す値。参照実装（CleanRL・SB3）と同じ。
constexpr double kAdvantageNormEpsilon = 1e-8;
// value の損失は、二乗誤差の半分にする（勾配が、誤差そのものになる）。
constexpr double kValueLossScale = 0.5;

static void RequireSameRows(const torch::Tensor& tensor, const torch::Tensor& reference, const char* name)
{
    // 入力は件数の軸だけを持つ。[件数] と [件数, 1] が broadcast で [件数, 件数] になる誤りを、ここで止める。
    if (!tensor.defined() || tensor.dim() != 1 || tensor.size(0) != reference.size(0)) {
        ANET_SYSTEM_ERROR("Invalid PPO loss input " << name << "="
            << (tensor.defined() ? anet::ToDefString(tensor) : std::string("undefined"))
            << "; expected a tensor [" << reference.size(0) << "].");
    }
}

static void RequireRows(const torch::Tensor& tensor, const char* name, int64_t min_rows)
{
    if (!tensor.defined() || tensor.dim() != 1 || tensor.size(0) < min_rows) {
        ANET_SYSTEM_ERROR("Invalid PPO loss input " << name << "="
            << (tensor.defined() ? anet::ToDefString(tensor) : std::string("undefined"))
            << "; expected a tensor [rows] with rows >= " << min_rows << ".");
    }
}

torch::Tensor anet::rl::ppo::NormalizeAdvantages(const torch::Tensor& advantages)
{
    RequireRows(advantages, "advantages", 2);
    // (A − mean) / (std + eps)。advantage の大きさは報酬の尺度と学習の進みで変わるので、
    // ミニバッチごとにそろえて、方策の勾配の大きさを安定させる。
    return (advantages - advantages.mean()) / (advantages.std() + kAdvantageNormEpsilon);
}

PolicyLoss anet::rl::ppo::ComputeClippedPolicyLoss(const torch::Tensor& new_log_probs,
    const torch::Tensor& old_log_probs, const torch::Tensor& advantages, float clip_range)
{
    RequireRows(new_log_probs, "new_log_probs", 1);
    RequireSameRows(old_log_probs, new_log_probs, "old_log_probs");
    RequireSameRows(advantages, new_log_probs, "advantages");

    // r(θ) = π_θ(a|s) / π_old(a|s) を、log 確率の差の指数で作る。更新の前は r = 1 である。
    auto log_ratio = new_log_probs - old_log_probs;
    auto ratio = log_ratio.exp();

    // L = E[min(r·A, clip(r, 1 − ε, 1 + ε)·A)] を最大にする。
    // r が [1 − ε, 1 + ε] を出て目的が良くなる向きでは clip した側が選ばれ、勾配が 0 になる。
    // これで、1 つの rollout を何度も使っても、方策が収集時から離れすぎない。
    // 損失は符号を反転して最小にするので、min が max になる。
    const auto unclipped = -advantages * ratio;
    const auto clipped = -advantages * ratio.clamp(1.0f - clip_range, 1.0f + clip_range);
    auto loss = torch::max(unclipped, clipped).mean();
    return PolicyLoss{ .loss = std::move(loss), .ratio = std::move(ratio), .log_ratio = std::move(log_ratio) };
}

ValueLoss anet::rl::ppo::ComputeValueLoss(const torch::Tensor& new_values, const torch::Tensor& old_values,
    const torch::Tensor& value_targets, std::optional<float> clip_range)
{
    RequireRows(new_values, "new_values", 1);
    RequireSameRows(old_values, new_values, "old_values");
    RequireSameRows(value_targets, new_values, "value_targets");

    // 基本は、二乗誤差の半分の平均。
    const auto unclipped = (new_values - value_targets).square();
    if (!clip_range) return ValueLoss{ .loss = kValueLossScale * unclipped.mean(), .clip_fraction = torch::Tensor() };

    // value の clip: 収集時の value からの変化を ±clip_range に抑えた予測でも誤差を作り、大きいほうを取る。
    // 方策の clip と同じく、1 回の rollout で value が大きく動くのを抑えるねらいである（効果には議論がある）。
    const auto delta = new_values - old_values;
    const auto clipped_values = old_values + delta.clamp(-*clip_range, *clip_range);
    const auto clipped = (clipped_values - value_targets).square();
    auto loss = kValueLossScale * torch::max(unclipped, clipped).mean();
    auto clip_fraction = (delta.abs() > *clip_range).to(torch::kFloat32).mean();
    return ValueLoss{ .loss = std::move(loss), .clip_fraction = std::move(clip_fraction) };
}

torch::Tensor anet::rl::ppo::ComputeApproxKl(const torch::Tensor& ratio, const torch::Tensor& log_ratio)
{
    RequireRows(ratio, "ratio", 1);
    RequireSameRows(log_ratio, ratio, "log_ratio");
    // KL(π_old ‖ π_θ) = E_old[−log r]。r − 1 ≥ log r なので、(r − 1) − log r は各サンプルで 0 以上になり、
    // −log r の平均より分散が小さい（E_old[r − 1] = 0 を足しているだけなので、期待値は同じ）。
    return ((ratio - 1.0f) - log_ratio).mean();
}

torch::Tensor anet::rl::ppo::ComputeExplainedVariance(const torch::Tensor& values, const torch::Tensor& value_targets)
{
    RequireRows(values, "values", 2);
    RequireSameRows(value_targets, values, "value_targets");
    // 1 なら value が目標を完全に説明し、0 なら平均を出すのと同じ、負なら平均より悪い。
    const auto target_variance = value_targets.var();
    const auto explained = 1.0f - (value_targets - values).var() / target_variance;
    // 目標が全部同じ値のときは定義できないので、NaN にする。device の上で選び、CPU へ読み戻さない。
    return torch::where(target_variance > 0, explained,
        torch::full_like(explained, std::numeric_limits<float>::quiet_NaN()));
}
