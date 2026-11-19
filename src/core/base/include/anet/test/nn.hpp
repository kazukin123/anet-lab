#pragma once

#ifndef ANET_TEST_SUPPORT
#error "anet/test/nn.hpp requires ANET_TEST_SUPPORT"
#endif

#include "anet/nn_head.hpp"

namespace anet::test {
    struct TestNetworkBranch {
        std::string name;
        std::vector<std::vector<std::string>> bind_terms;
        int64_t concat_dim;
        std::string module_name;
        std::shared_ptr<nn::NetworkModule> module;
    };

    std::shared_ptr<nn::Network> MakeTestNetwork(
        const nn::NetworkConfig& config, const TensorSpecMap& specs,
        std::shared_ptr<nn::NetworkHeadFactory> factory,
        std::shared_ptr<nn::NetworkHead> head,
        const std::vector<TestNetworkBranch>& branches = {});

    std::shared_ptr<nn::NetworkForwardState> MakeNetworkForwardState(seed_t seed);
    void RegisterStandardNetworkModules();
    torch::Tensor DropPath(const torch::Tensor& input, double probability, bool training);
    std::shared_ptr<nn::NetworkModule> MakeStandardNetworkModule(
        const std::string& type, const ConfigData& data);
}
