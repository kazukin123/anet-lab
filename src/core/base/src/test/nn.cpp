#include "anet/test/nn.hpp"
#include "nn_impl.hpp"

std::shared_ptr<anet::nn::Network> anet::test::MakeTestNetwork(
    const nn::NetworkConfig& config, const TensorSpecMap& specs,
    std::shared_ptr<nn::NetworkHeadFactory> factory,
    std::shared_ptr<nn::NetworkHead> head,
    const std::vector<TestNetworkBranch>& branches)
{
    std::vector<std::shared_ptr<nn::NetworkBranch>> body_branches;
    for (const auto& branch : branches) {
        auto block = std::make_shared<nn::NetworkBlock>(branch.module_name, branch.module);
        auto structure = std::make_shared<nn::NetworkStruct>(
            std::vector<std::shared_ptr<nn::NetworkBlock>>{ std::move(block) });
        body_branches.push_back(std::make_shared<nn::NetworkBranch>(
            branch.name, branch.bind_terms, branch.concat_dim, std::move(structure)));
    }
    auto body = std::make_shared<nn::NetworkBody>(
        std::move(body_branches), specs, std::vector<std::string>{}, config.output_keys);
    return std::make_shared<nn::Network>(config, specs, std::move(factory), std::move(body), std::move(head));
}

std::shared_ptr<anet::nn::NetworkForwardState> anet::test::MakeNetworkForwardState(seed_t seed)
{
    return std::make_shared<nn::NetworkForwardState>(seed);
}

void anet::test::RegisterStandardNetworkModules()
{
    nn::InitNN();
}

std::shared_ptr<anet::nn::NetworkModule> anet::test::MakeStandardNetworkModule(
    const std::string& type, const ConfigData& data)
{
    return nn::NetworkModuleRepository::Instance().GetFactory(type)->CreateModule(data, nn::ModuleContext{});
}

torch::Tensor anet::test::DropPath(const torch::Tensor& input, double probability, bool training)
{
    return nn::DropPath(input, probability, training);
}
