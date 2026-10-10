#pragma once

#include "anet/nn.hpp"

namespace anet::test {
    /// 公開の準備済み経路から箱を観測し、実行オブジェクトの寿命と一緒に保持する。
    std::shared_ptr<anet::nn::NetworkForwardState> MakeTestForwardState(anet::seed_t seed);
}
