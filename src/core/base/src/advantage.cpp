#include "anet/advantage.hpp"
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include "anet/profile.hpp"
#include "anet/tensor_util.hpp"
#include "anet/diag.hpp"

using namespace anet::rl;

static void RequireGaeInput(const torch::Tensor& tensor, const char* name, const torch::Tensor& reference, bool is_flag)
{
    // 入力はすべて [時刻, lane] でそろえる。broadcast で別の形が通ると、lane や時刻の対応がずれたまま計算が進む。
    const bool valid_type = tensor.defined()
        && (is_flag ? tensor.scalar_type() == torch::kBool : tensor.is_floating_point());
    if (!valid_type || tensor.dim() != 2 || tensor.sizes() != reference.sizes()
        || tensor.device() != reference.device()) {
        ANET_SYSTEM_ERROR("Invalid GAE input " << name << "="
            << (tensor.defined() ? anet::ToDefString(tensor) : std::string("undefined"))
            << "; expected a " << (is_flag ? "bool" : "floating-point") << " tensor with the shape "
            << reference.sizes() << " on the device of rewards (" << reference.device() << ").");
    }
}

GaeResult anet::rl::ComputeGae(const torch::Tensor& rewards, const torch::Tensor& values,
    const torch::Tensor& next_values, const torch::Tensor& terminated, const torch::Tensor& episode_end,
    double gamma, double lambda)
{
    ANET_PROFILE_FUNC();

    // 形・型・範囲を、計算の前に確かめる。
    if (!rewards.defined() || rewards.dim() != 2 || rewards.size(0) < 1 || rewards.size(1) < 1
        || !rewards.is_floating_point()) {
        ANET_SYSTEM_ERROR("Invalid GAE input rewards="
            << (rewards.defined() ? anet::ToDefString(rewards) : std::string("undefined"))
            << "; expected a non-empty floating-point tensor [time, lanes].");
    }
    RequireGaeInput(values, "values", rewards, /*is_flag=*/false);
    RequireGaeInput(next_values, "next_values", rewards, /*is_flag=*/false);
    RequireGaeInput(terminated, "terminated", rewards, /*is_flag=*/true);
    RequireGaeInput(episode_end, "episode_end", rewards, /*is_flag=*/true);
    if (!std::isfinite(gamma) || gamma < 0.0 || gamma > 1.0) {
        ANET_SYSTEM_ERROR("Invalid GAE gamma=" << gamma << "; expected a value in [0, 1].");
    }
    if (!std::isfinite(lambda) || lambda < 0.0 || lambda > 1.0) {
        ANET_SYSTEM_ERROR("Invalid GAE lambda=" << lambda << "; expected a value in [0, 1].");
    }

    // 推定は学習の目標を作るだけなので、勾配を作らない。
    torch::NoGradGuard no_grad;

    // 2 つの印は役割が違う。
    //   terminated  : その遷移で episode が本当に終わった。次の観測の value は 0 として扱う。
    //   episode_end : 終端か打ち切り。次の時刻は別の episode なので、advantage を時刻をまたいで伝えない。
    // 打ち切りの時刻は terminated が偽、episode_end が真で、次の観測の value（Reset の前の観測）だけを使う。
    const auto keeps_next_value = terminated.logical_not().to(values.scalar_type());
    const auto keeps_trace = episode_end.logical_not().to(values.scalar_type());

    // 1 step の TD 誤差 δ_t = r_t + γ V(s_{t+1}) (1 − terminated_t) − V(s_t) を、全時刻まとめて作る。
    const auto deltas = rewards + gamma * next_values * keeps_next_value - values;

    // A_t = δ_t + γ λ (1 − episode_end_t) A_{t+1} を、最後の時刻から前へたどる。
    // λ は、先の TD 誤差をどこまで足し込むかを決める（分散と偏りの釣り合い）。
    const auto time_count = rewards.size(0);
    std::vector<torch::Tensor> rows(static_cast<size_t>(time_count));
    auto next_advantage = torch::zeros_like(deltas.select(0, 0));
    for (int64_t t = time_count - 1; t >= 0; --t) {
        next_advantage = deltas.select(0, t) + gamma * lambda * keeps_trace.select(0, t) * next_advantage;
        rows[static_cast<size_t>(t)] = next_advantage;
    }
    auto advantages = torch::stack(rows, 0);

    // value の目標は、advantage に収集時の value を足した値（λ-return）。value の Network はこれへ近づける。
    auto value_targets = advantages + values;
    return GaeResult{ .advantages = std::move(advantages), .value_targets = std::move(value_targets) };
}
