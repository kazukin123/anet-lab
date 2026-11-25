#include "anet/test/catch.hpp"
#include <cmath>
#include <optional>
#include "ppo_math.hpp"

using namespace anet::rl::ppo;

// 診断のメッセージが名指す、入力の名前。
constexpr char kAdvantagesName[] = "advantages";
constexpr char kValueTargetsName[] = "value_targets";

TEST_CASE("PPO clipped policy loss matches hand-computed values and stops gradients outside the clip range",
    "[ppo][ppo_math][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // 収集時の確率は 0.5、新しい確率は 0.75 と 0.25 なので、比は 1.5 と 0.5 になる。
    const auto old_log_probs = torch::tensor({ 0.5f, 0.5f }).log().to(device);
    const auto new_log_probs = torch::tensor({ 0.75f, 0.25f }).log().to(device).requires_grad_(true);
    constexpr float kClipRange = 0.2f;

    SECTION("the clipped side is chosen when moving further would improve the objective")
    {
        // A = (1, −1)。r·A = (1.5, −0.5)、clip(r)·A = (1.2, −0.8)。小さいほう (1.2, −0.8) の平均 0.2 の符号を反転する。
        const auto advantages = torch::tensor({ 1.0f, -1.0f }).to(device);
        const auto result = ComputeClippedPolicyLoss(new_log_probs, old_log_probs, advantages, kClipRange);
        CHECK(result.loss.item<float>() == Catch::Approx(-0.2f).margin(1e-6));
        CHECK(torch::allclose(result.ratio.detach().cpu(), torch::tensor({ 1.5f, 0.5f }), 1e-5, 1e-6));
        CHECK(torch::allclose(result.log_ratio.detach().cpu(), torch::tensor({ 1.5f, 0.5f }).log(), 1e-5, 1e-6));
        // どちらの行も clip した側が選ばれるので、これ以上、比を動かす勾配が無い。
        result.loss.backward();
        CHECK(torch::allclose(new_log_probs.grad().cpu(), torch::zeros({ 2 }), 0.0, 1e-7));
    }

    SECTION("the unclipped side is chosen when the objective got worse")
    {
        // A = (−1, 1)。r·A = (−1.5, 0.5)、clip(r)·A = (−1.2, 0.8)。小さいほう (−1.5, 0.5) の平均 −0.5 の符号を反転する。
        const auto advantages = torch::tensor({ -1.0f, 1.0f }).to(device);
        const auto result = ComputeClippedPolicyLoss(new_log_probs, old_log_probs, advantages, kClipRange);
        CHECK(result.loss.item<float>() == Catch::Approx(0.5f).margin(1e-6));
        // clip しない側の勾配は −A·r / 件数 = (0.75, −0.25)。悪くなった向きは、戻す勾配が残る。
        result.loss.backward();
        CHECK(torch::allclose(new_log_probs.grad().cpu(), torch::tensor({ 0.75f, -0.25f }), 1e-5, 1e-6));
    }
}

TEST_CASE("PPO value loss matches hand-computed values with and without clipping", "[ppo][ppo_math][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    const auto new_values = torch::tensor({ 1.9f, 3.0f }).to(device);
    const auto old_values = torch::tensor({ 1.0f, 1.0f }).to(device);
    const auto value_targets = torch::tensor({ 2.0f, 3.0f }).to(device);

    SECTION("without clipping")
    {
        // 0.5 × mean((1.9 − 2)², (3 − 3)²) = 0.5 × 0.005
        const auto result = ComputeValueLoss(new_values, old_values, value_targets, std::nullopt);
        CHECK(result.loss.item<float>() == Catch::Approx(0.0025f).margin(1e-6));
        CHECK_FALSE(result.clip_fraction.defined());
    }

    SECTION("with clipping")
    {
        // 収集時の value 1.0 からの変化を ±0.2 に抑えると、予測は (1.2, 1.2)。誤差の二乗は (0.64, 3.24) で、
        // clip しない誤差 (0.01, 0) より大きいので、こちらが選ばれる。0.5 × mean(0.64, 3.24) = 0.97
        const auto result = ComputeValueLoss(new_values, old_values, value_targets, 0.2f);
        CHECK(result.loss.item<float>() == Catch::Approx(0.97f).margin(1e-5));
        // どちらの行も、変化が幅を超えている。
        CHECK(result.clip_fraction.item<float>() == Catch::Approx(1.0f));
    }

    SECTION("clipping keeps the unclipped error when the prediction moved away from the target")
    {
        // 目標 0 に対して、予測が 1.0 から 2.0 へ離れた行。clip した予測 1.2 の誤差 1.44 より、clip しない誤差 4 が大きい。
        const auto result = ComputeValueLoss(torch::tensor({ 2.0f, 1.1f }).to(device), old_values,
            torch::zeros({ 2 }).to(device), 0.2f);
        // 2 行目は幅の中（変化 0.1）で、誤差は 1.21。0.5 × mean(4, 1.21) = 1.3025
        CHECK(result.loss.item<float>() == Catch::Approx(1.3025f).margin(1e-5));
        CHECK(result.clip_fraction.item<float>() == Catch::Approx(0.5f));
    }
}

TEST_CASE("PPO approximate KL is zero for an unchanged policy and positive otherwise", "[ppo][ppo_math][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // 比が 1 なら、方策は収集時と同じで、KL は 0 である。
    const auto ones = torch::ones({ 3 }).to(device);
    CHECK(ComputeApproxKl(ones, ones.log()).item<float>() == Catch::Approx(0.0f).margin(1e-7));

    // mean((r − 1) − log r) = ((0.5 − ln 1.5) + (−0.5 − ln 0.5)) / 2
    const auto ratio = torch::tensor({ 1.5f, 0.5f }).to(device);
    const double expected = ((0.5 - std::log(1.5)) + (-0.5 - std::log(0.5))) / 2.0;
    const auto kl = ComputeApproxKl(ratio, ratio.log()).item<float>();
    CHECK(kl == Catch::Approx(expected).margin(1e-6));
    CHECK(kl > 0.0f);
}

TEST_CASE("PPO advantage normalization yields zero mean and unit standard deviation", "[ppo][ppo_math][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    const auto normalized = NormalizeAdvantages(torch::tensor({ 1.0f, 2.0f, 3.0f, 6.0f }).to(device));
    CHECK(normalized.mean().item<float>() == Catch::Approx(0.0f).margin(1e-6));
    CHECK(normalized.std().item<float>() == Catch::Approx(1.0f).margin(1e-5));
    // 並びの順は変わらない。
    CHECK(normalized[0].item<float>() < normalized[3].item<float>());

    // 1 件だけでは標準偏差を作れないので、止める。
    CHECK_THROWS_WITH(NormalizeAdvantages(torch::tensor({ 1.0f }).to(device)),
        Catch::Matchers::ContainsSubstring(kAdvantagesName));
}

TEST_CASE("PPO explained variance is 1 for exact values, 0 for a constant and NaN for constant targets",
    "[ppo][ppo_math][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    const auto targets = torch::tensor({ 1.0f, 2.0f, 3.0f, 4.0f }).to(device);
    // 予測が目標と同じなら 1、定数（目標のばらつきを何も説明しない）なら 0。
    CHECK(ComputeExplainedVariance(targets, targets).item<float>() == Catch::Approx(1.0f));
    CHECK(ComputeExplainedVariance(torch::full({ 4 }, 2.5f).to(device), targets).item<float>()
        == Catch::Approx(0.0f).margin(1e-6));
    // 目標と逆向きの予測は、定数より悪いので負になる。
    CHECK(ComputeExplainedVariance(-targets, targets).item<float>() < 0.0f);
    // 目標が全部同じ値なら、割る分散が 0 で定義できない。
    CHECK(std::isnan(ComputeExplainedVariance(targets, torch::ones({ 4 }).to(device)).item<float>()));
}

TEST_CASE("PPO loss functions reject inputs whose shapes would broadcast", "[ppo][ppo_math][rfd0202]")
{
    // [件数] と [件数, 1] を混ぜると、差が [件数, 件数] に広がって、誤った損失になる。形の違いは、計算の前に止める。
    const auto rows = torch::zeros({ 3 });
    const auto column = torch::zeros({ 3, 1 });
    CHECK_THROWS_WITH(ComputeClippedPolicyLoss(rows, column, rows, 0.2f),
        Catch::Matchers::ContainsSubstring("old_log_probs"));
    CHECK_THROWS_WITH(ComputeClippedPolicyLoss(rows, rows, torch::zeros({ 2 }), 0.2f),
        Catch::Matchers::ContainsSubstring(kAdvantagesName));
    CHECK_THROWS_WITH(ComputeValueLoss(column, rows, rows, std::nullopt),
        Catch::Matchers::ContainsSubstring("new_values"));
    CHECK_THROWS_WITH(ComputeValueLoss(rows, rows, column, 0.2f),
        Catch::Matchers::ContainsSubstring(kValueTargetsName));
    CHECK_THROWS_WITH(ComputeApproxKl(rows, column), Catch::Matchers::ContainsSubstring("log_ratio"));
    CHECK_THROWS_WITH(ComputeExplainedVariance(rows, column), Catch::Matchers::ContainsSubstring(kValueTargetsName));
}
