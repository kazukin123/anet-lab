// stacker_test.cpp

#include "anet/catch_test.hpp"

#include "anet/stacker.hpp"

#include <string>
#include <vector>

using namespace anet::rl;

namespace stacker_test {

// (N, C) 形状で env e の値が base + e*100 となるフレームを作る (env間の取り違え検出用)
torch::Tensor MakeFrame(int num_envs, int channels, float base)
{
    auto frame = torch::empty({ num_envs, channels }, torch::kFloat32);
    for (int e = 0; e < num_envs; ++e) {
        frame[e].fill_(base + static_cast<float>(e) * 100.0f);
    }
    return frame;
}

// リセットなしの resets テンソル
torch::Tensor NoResets(int num_envs)
{
    return torch::zeros({ num_envs }, torch::kBool);
}

}

using namespace stacker_test;


TEST_CASE("DictFrameStacker fills all slots with the first frame", "[stacker]")
{
    const int kNumEnvs = 2;
    const int kStack = 4;
    const int kChannels = 3;
    DictFrameStacker stacker(kStack, kNumEnvs, torch::kCPU);

    auto frame = MakeFrame(kNumEnvs, kChannels, 1.0f);
    auto out = stacker.Stack(anet::TensorDict{ { "obs", frame } }, NoResets(kNumEnvs));

    auto stacked = out.At("obs");
    REQUIRE(stacked.sizes().vec() == std::vector<int64_t>{ kNumEnvs, kStack, kChannels });
    for (int k = 0; k < kStack; ++k) {
        REQUIRE(torch::allclose(stacked.select(1, k), frame));
    }
}

TEST_CASE("DictFrameStacker keeps frames in oldest-to-newest order", "[stacker]")
{
    const int kNumEnvs = 2;
    const int kStack = 3;
    const int kChannels = 2;
    DictFrameStacker stacker(kStack, kNumEnvs, torch::kCPU);

    // フレーム値 1..6 を順に投入 (リングバッファを一周以上させる)
    anet::TensorDict out;
    for (int t = 1; t <= 6; ++t) {
        out = stacker.Stack(
            anet::TensorDict{ { "obs", MakeFrame(kNumEnvs, kChannels, static_cast<float>(t)) } },
            NoResets(kNumEnvs));
    }

    // 直近 kStack フレーム (4,5,6) が古→新順に並ぶ
    auto stacked = out.At("obs");
    for (int k = 0; k < kStack; ++k) {
        auto expected = MakeFrame(kNumEnvs, kChannels, static_cast<float>(4 + k));
        REQUIRE(torch::allclose(stacked.select(1, k), expected));
    }
}

TEST_CASE("DictFrameStacker clears history only for reset envs", "[stacker]")
{
    const int kNumEnvs = 3;
    const int kStack = 4;
    const int kChannels = 2;
    DictFrameStacker stacker(kStack, kNumEnvs, torch::kCPU);

    stacker.Stack(anet::TensorDict{ { "obs", MakeFrame(kNumEnvs, kChannels, 1.0f) } }, NoResets(kNumEnvs));
    stacker.Stack(anet::TensorDict{ { "obs", MakeFrame(kNumEnvs, kChannels, 2.0f) } }, NoResets(kNumEnvs));

    // env=1 だけリセットして 3 フレーム目を投入
    auto resets = NoResets(kNumEnvs);
    resets[1] = true;
    auto out = stacker.Stack(anet::TensorDict{ { "obs", MakeFrame(kNumEnvs, kChannels, 3.0f) } }, resets);
    auto stacked = out.At("obs");

    // env=1: 全スロットが最新フレームで塗りつぶされる (データリーク防止)
    auto latest = MakeFrame(kNumEnvs, kChannels, 3.0f);
    for (int k = 0; k < kStack; ++k) {
        REQUIRE(torch::allclose(stacked[1][k], latest[1]));
    }

    // env=0, env=2: 履歴が維持される (初回充填 1,1 → 2 → 3 の順)
    const std::vector<float> expected_bases = { 1.0f, 1.0f, 2.0f, 3.0f };
    for (int k = 0; k < kStack; ++k) {
        auto expected = MakeFrame(kNumEnvs, kChannels, expected_bases[k]);
        REQUIRE(torch::allclose(stacked[0][k], expected[0]));
        REQUIRE(torch::allclose(stacked[2][k], expected[2]));
    }
}

TEST_CASE("DictFrameStacker passes through non-stack keys", "[stacker]")
{
    const int kNumEnvs = 2;
    const int kStack = 3;
    const int kChannels = 2;
    DictFrameStacker stacker(kStack, kNumEnvs, torch::kCPU, std::vector<std::string>{ "image" });

    auto image = MakeFrame(kNumEnvs, kChannels, 1.0f);
    auto vec = MakeFrame(kNumEnvs, kChannels, 5.0f);
    auto out = stacker.Stack(anet::TensorDict{ { "image", image }, { "vector", vec } }, NoResets(kNumEnvs));

    // image はスタックされ (N,S,C)、vector は形状そのまま (N,C)
    REQUIRE(out.At("image").sizes().vec() == std::vector<int64_t>{ kNumEnvs, kStack, kChannels });
    REQUIRE(out.At("vector").sizes().vec() == std::vector<int64_t>{ kNumEnvs, kChannels });
    REQUIRE(torch::allclose(out.At("vector"), vec));
}

TEST_CASE("DictFrameStacker with stack_count=1 keeps only the latest frame", "[stacker]")
{
    const int kNumEnvs = 2;
    const int kChannels = 3;
    DictFrameStacker stacker(1, kNumEnvs, torch::kCPU);

    stacker.Stack(anet::TensorDict{ { "obs", MakeFrame(kNumEnvs, kChannels, 1.0f) } }, NoResets(kNumEnvs));
    auto out = stacker.Stack(anet::TensorDict{ { "obs", MakeFrame(kNumEnvs, kChannels, 2.0f) } }, NoResets(kNumEnvs));

    auto stacked = out.At("obs");
    REQUIRE(stacked.sizes().vec() == std::vector<int64_t>{ kNumEnvs, 1, kChannels });
    REQUIRE(torch::allclose(stacked.select(1, 0), MakeFrame(kNumEnvs, kChannels, 2.0f)));
}

TEST_CASE("DictFrameStacker output is independent from internal buffer", "[stacker]")
{
    const int kNumEnvs = 1;
    const int kStack = 2;
    const int kChannels = 2;
    DictFrameStacker stacker(kStack, kNumEnvs, torch::kCPU);

    auto out1 = stacker.Stack(anet::TensorDict{ { "obs", MakeFrame(kNumEnvs, kChannels, 1.0f) } }, NoResets(kNumEnvs));
    out1.At("obs").fill_(-999.0f);   // 返却テンソルを外部で破壊

    auto out2 = stacker.Stack(anet::TensorDict{ { "obs", MakeFrame(kNumEnvs, kChannels, 2.0f) } }, NoResets(kNumEnvs));
    auto stacked = out2.At("obs");

    // 内部バッファは壊れておらず [1, 2] の順で出てくる
    REQUIRE(torch::allclose(stacked.select(1, 0), MakeFrame(kNumEnvs, kChannels, 1.0f)));
    REQUIRE(torch::allclose(stacked.select(1, 1), MakeFrame(kNumEnvs, kChannels, 2.0f)));
}
