#include "anet/test/catch.hpp"
#include <cmath>
#include <initializer_list>
#include <vector>
#include "anet/advantage.hpp"

// 1 lane の値を [時刻, 1] の形にする。
static torch::Tensor MakeColumn(std::initializer_list<float> values)
{
    return torch::tensor(std::vector<float>(values)).unsqueeze(1);
}

static torch::Tensor MakeFlagColumn(std::initializer_list<int> values)
{
    return torch::tensor(std::vector<int>(values)).to(torch::kBool).unsqueeze(1);
}

TEST_CASE("GAE matches a hand-computed three step example", "[advantage][rfd0209]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // γ = 0.5、λ = 0.5、episode は続く。
    //   delta = r + γ·next_value − value = (1 + 1 − 1, 2 + 2 − 2, 3 + 2.5 − 4) = (1, 2, 1.5)
    //   A2 = 1.5、A1 = 2 + 0.25·1.5 = 2.375、A0 = 1 + 0.25·2.375 = 1.59375
    const auto rewards = MakeColumn({ 1.0f, 2.0f, 3.0f }).to(device);
    const auto values = MakeColumn({ 1.0f, 2.0f, 4.0f }).to(device);
    const auto next_values = MakeColumn({ 2.0f, 4.0f, 5.0f }).to(device);
    const auto off = MakeFlagColumn({ 0, 0, 0 }).to(device);
    const auto result = anet::rl::ComputeGae(rewards, values, next_values, off, off, 0.5, 0.5);

    REQUIRE(result.advantages.sizes() == torch::IntArrayRef({ 3, 1 }));
    CHECK(result.advantages.device() == rewards.device());
    CHECK_FALSE(result.advantages.requires_grad());
    CHECK(torch::allclose(result.advantages.cpu(), MakeColumn({ 1.59375f, 2.375f, 1.5f }), 1e-6, 1e-6));
    // value の目標は、advantage に value を足した値。
    CHECK(torch::allclose(result.value_targets.cpu(), MakeColumn({ 2.59375f, 4.375f, 5.5f }), 1e-6, 1e-6));
}

TEST_CASE("GAE drops the next value at termination and cuts the trace at an episode end", "[advantage][rfd0209]")
{
    const auto rewards = MakeColumn({ 1.0f, 1.0f, 1.0f });
    const auto values = MakeColumn({ 0.0f, 0.0f, 0.0f });
    const auto next_values = MakeColumn({ 10.0f, 10.0f, 10.0f });
    const auto off = MakeFlagColumn({ 0, 0, 0 });
    const auto middle = MakeFlagColumn({ 0, 1, 0 });

    SECTION("termination uses no next value and stops the trace") {
        // 時刻 1 が終端。delta = (1 + 10, 1, 1 + 10)。時刻 1 は時刻 2 の advantage を受け取らない。
        const auto result = anet::rl::ComputeGae(rewards, values, next_values, middle, middle, 1.0, 1.0);
        CHECK(torch::allclose(result.advantages, MakeColumn({ 12.0f, 1.0f, 11.0f })));
    }
    SECTION("truncation keeps the next value and stops the trace") {
        // 時刻 1 が打ち切り。次の観測の value（Reset の前の観測）は使い、時刻 2 の advantage は受け取らない。
        const auto result = anet::rl::ComputeGae(rewards, values, next_values, off, middle, 1.0, 1.0);
        CHECK(torch::allclose(result.advantages, MakeColumn({ 22.0f, 11.0f, 11.0f })));
    }
    SECTION("lanes are independent") {
        // lane 0 は続き、lane 1 は時刻 0 で終端。
        const auto two_rewards = torch::ones({ 2, 2 });
        const auto two_values = torch::zeros({ 2, 2 });
        const auto two_next = torch::full({ 2, 2 }, 10.0f);
        const auto ends = torch::tensor({ { 0, 1 }, { 0, 0 } }).to(torch::kBool);
        const auto result = anet::rl::ComputeGae(two_rewards, two_values, two_next, ends, ends, 1.0, 1.0);
        CHECK(torch::allclose(result.advantages, torch::tensor({ { 22.0f, 1.0f }, { 11.0f, 11.0f } })));
    }
}

TEST_CASE("GAE reduces to the one step TD error and to the discounted sum at the ends of lambda",
    "[advantage][rfd0209]")
{
    const auto rewards = MakeColumn({ 1.0f, 2.0f, 4.0f });
    const auto values = MakeColumn({ 0.5f, 1.0f, 2.0f });
    const auto next_values = MakeColumn({ 1.0f, 2.0f, 3.0f });
    const auto off = MakeFlagColumn({ 0, 0, 0 });
    const double gamma = 0.9;

    // λ = 0: advantage は 1 step の TD 誤差 r + γ·V(次) − V になる。
    const auto td = anet::rl::ComputeGae(rewards, values, next_values, off, off, gamma, 0.0);
    CHECK(torch::allclose(td.advantages, rewards + gamma * next_values - values));

    // λ = 1: value の目標は、最後の次の観測の value で bootstrap した割引の和になる。
    // 途中の next_values は次の時刻の values と同じにしておく（同じ lane の続きの観測）。
    const auto chained_next = MakeColumn({ 1.0f, 2.0f, 3.0f });
    const auto sum = anet::rl::ComputeGae(rewards, values, chained_next, off, off, gamma, 1.0);
    const float last = 4.0f + 0.9f * 3.0f;
    const float middle = 2.0f + 0.9f * last;
    const float first = 1.0f + 0.9f * middle;
    CHECK(torch::allclose(sum.value_targets, MakeColumn({ first, middle, last }), 1e-5, 1e-5));
}

TEST_CASE("GAE rejects mismatched shapes, wrong types and out-of-range coefficients", "[advantage][rfd0209]")
{
    const auto rewards = torch::zeros({ 3, 2 });
    const auto values = torch::zeros({ 3, 2 });
    const auto flags = torch::zeros({ 3, 2 }, torch::kBool);
    CHECK_NOTHROW(anet::rl::ComputeGae(rewards, values, values, flags, flags, 0.99, 0.95));

    // 形の違い。[時刻, lane] と [時刻] は broadcast できてしまうので、明示的に拒む。
    CHECK_THROWS(anet::rl::ComputeGae(rewards, torch::zeros({ 3 }), values, flags, flags, 0.99, 0.95));
    CHECK_THROWS(anet::rl::ComputeGae(rewards, values, torch::zeros({ 3, 1 }), flags, flags, 0.99, 0.95));
    CHECK_THROWS(anet::rl::ComputeGae(torch::zeros({ 6 }), torch::zeros({ 6 }), torch::zeros({ 6 }),
        torch::zeros({ 6 }, torch::kBool), torch::zeros({ 6 }, torch::kBool), 0.99, 0.95));
    // 印は bool、値は浮動小数。
    CHECK_THROWS(anet::rl::ComputeGae(rewards, values, values, torch::zeros({ 3, 2 }), flags, 0.99, 0.95));
    CHECK_THROWS(anet::rl::ComputeGae(rewards, values, values, flags, torch::zeros({ 3, 2 }), 0.99, 0.95));
    CHECK_THROWS(anet::rl::ComputeGae(torch::zeros({ 3, 2 }, torch::kInt64), values, values, flags, flags, 0.99, 0.95));
    // γ と λ は [0, 1]。
    CHECK_THROWS(anet::rl::ComputeGae(rewards, values, values, flags, flags, 1.01, 0.95));
    CHECK_THROWS(anet::rl::ComputeGae(rewards, values, values, flags, flags, -0.1, 0.95));
    CHECK_THROWS(anet::rl::ComputeGae(rewards, values, values, flags, flags, 0.99, 1.5));
    CHECK_THROWS(anet::rl::ComputeGae(rewards, values, values, flags, flags, 0.99, std::nan("")));
}
