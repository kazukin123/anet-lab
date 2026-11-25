#include "anet/test/catch.hpp"
#include <cmath>
#include <limits>
#include "anet/random.hpp"
#include "anet/distribution.hpp"

TEST_CASE("Categorical distribution matches hand-computed probabilities, log probabilities and entropy",
    "[distribution][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // 1 行目は (0.5, 0.25, 0.25)、2 行目は一様。logits に行ごとの定数を足しても、分布は変わらない。
    const auto probs = torch::tensor({ { 0.5f, 0.25f, 0.25f }, { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f } });
    const auto logits = (probs.log() + torch::tensor({ { 2.0f }, { -5.0f } })).to(device);
    const anet::rl::CategoricalDistribution distribution(logits);

    CHECK(distribution.GetActionCount() == 3);
    CHECK(distribution.GetProbs().device() == logits.device());
    CHECK(torch::allclose(distribution.GetProbs().cpu(), probs, 1e-5, 1e-6));
    CHECK(torch::allclose(distribution.GetLogProbs().cpu(), probs.log(), 1e-5, 1e-6));

    // 行ごとに 1 つの行動の log 確率を取り出す。
    const auto actions = torch::tensor({ 0, 2 }, torch::kInt64).to(device);
    const auto log_probs = distribution.ComputeLogProb(actions).cpu();
    REQUIRE(log_probs.sizes() == torch::IntArrayRef({ 2 }));
    CHECK(log_probs[0].item<float>() == Catch::Approx(std::log(0.5f)).margin(1e-5));
    CHECK(log_probs[1].item<float>() == Catch::Approx(std::log(1.0f / 3.0f)).margin(1e-5));

    // H = −Σ p log p。1 行目は 0.5·ln2 + 0.5·ln4、2 行目は ln3。
    const auto entropy = distribution.ComputeEntropy().cpu();
    CHECK(entropy[0].item<float>() == Catch::Approx(0.5 * std::log(2.0) + 0.5 * std::log(4.0)).margin(1e-5));
    CHECK(entropy[1].item<float>() == Catch::Approx(std::log(3.0)).margin(1e-5));

    // 確率が最大の行動。2 行目は一様なので、1 行目だけを確かめる。
    CHECK(distribution.GetMode().cpu()[0].item<int64_t>() == 0);
}

TEST_CASE("Categorical distribution keeps entropy finite when an action has zero probability",
    "[distribution][rfd0202]")
{
    // logits が −∞ の行動は確率 0 になり、entropy の項は 0 として数える。
    const auto negative_infinity = -std::numeric_limits<float>::infinity();
    const anet::rl::CategoricalDistribution distribution(torch::tensor({ { 0.0f, 0.0f, negative_infinity } }));
    CHECK(distribution.ComputeEntropy()[0].item<float>() == Catch::Approx(std::log(2.0)).margin(1e-5));
}

TEST_CASE("Categorical distribution samples actions with frequencies that follow the probabilities",
    "[distribution][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // 同じ分布を 10 万行並べて 1 回抽選し、行動ごとの頻度を確率と比べる。
    constexpr int64_t kRows = 100000;
    const auto probs = torch::tensor({ 0.6f, 0.3f, 0.1f });
    const anet::rl::CategoricalDistribution distribution(probs.log().expand({ kRows, 3 }).to(device));
    anet::RandomGenerator random(12345);
    const auto actions = distribution.Sample(random.GetTorchGenerator(device));
    REQUIRE(actions.sizes() == torch::IntArrayRef({ kRows }));
    CHECK(actions.scalar_type() == torch::kInt64);
    CHECK(actions.device() == distribution.GetProbs().device());
    const auto frequency = torch::bincount(actions.cpu(), {}, 3).to(torch::kFloat32) / static_cast<float>(kRows);
    CHECK(torch::allclose(frequency, probs, 0.0, 0.01));

    // 同じ seed の生成器は、同じ行動の並びを返す。
    anet::RandomGenerator repeated(12345);
    CHECK(torch::equal(actions.cpu(), distribution.Sample(repeated.GetTorchGenerator(device)).cpu()));
}

TEST_CASE("Categorical distribution keeps gradients through log probabilities and entropy", "[distribution][rfd0202]")
{
    auto logits = torch::tensor({ { 0.1f, -0.2f, 0.3f } }).requires_grad_(true);
    const anet::rl::CategoricalDistribution distribution(logits);
    const auto loss = distribution.ComputeLogProb(torch::tensor({ 1 }, torch::kInt64)).sum()
        + distribution.ComputeEntropy().sum();
    loss.backward();
    REQUIRE(logits.grad().defined());
    CHECK(logits.grad().abs().sum().item<float>() > 0.0f);
}

TEST_CASE("Categorical distribution rejects invalid logits and actions", "[distribution][rfd0202]")
{
    // logits は [行, 行動数] の浮動小数で、行動数は 2 以上。
    CHECK_THROWS(anet::rl::CategoricalDistribution(torch::Tensor()));
    CHECK_THROWS(anet::rl::CategoricalDistribution(torch::zeros({ 3 })));
    CHECK_THROWS(anet::rl::CategoricalDistribution(torch::zeros({ 2, 1 })));
    CHECK_THROWS(anet::rl::CategoricalDistribution(torch::zeros({ 2, 3, 4 })));
    CHECK_THROWS(anet::rl::CategoricalDistribution(torch::zeros({ 2, 3 }, torch::kInt64)));

    // 行動は [行] の整数。行数の違い、[行, 1] の形、浮動小数は拒む。
    const anet::rl::CategoricalDistribution distribution(torch::zeros({ 2, 3 }));
    CHECK_NOTHROW(distribution.ComputeLogProb(torch::zeros({ 2 }, torch::kInt64)));
    CHECK_THROWS(distribution.ComputeLogProb(torch::zeros({ 3 }, torch::kInt64)));
    CHECK_THROWS(distribution.ComputeLogProb(torch::zeros({ 2, 1 }, torch::kInt64)));
    CHECK_THROWS(distribution.ComputeLogProb(torch::zeros({ 2 }, torch::kFloat32)));
}
