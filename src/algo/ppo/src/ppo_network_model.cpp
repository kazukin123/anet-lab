#include "ppo_network_model.hpp"
#include <string>
#include <unordered_set>
#include <utility>
#include "anet/diag.hpp"
#include "anet/profile.hpp"
#include "anet/nn_head.hpp"

using namespace anet::rl::ppo;

// Network は Head を "head" の名前で登録し、合成の Head は中の Head を組の名前で登録する。
constexpr char kHeadParameterPrefix[] = "head.";
constexpr char kParameterNameSeparator[] = ".";

NetworkModel::NetworkModel(const anet::nn::NetworkConfig& network_config, const anet::TensorSpecMap& obs_specs,
    int64_t action_count, const anet::nn::WeightInitConfig& policy_init,
    const anet::nn::WeightInitConfig& value_init, torch::Device device, anet::seed_t seed)
    : network_config_(network_config)
{
    ANET_PROFILE_FUNC();

    // Head が受け取る 2 つのキーが、設定で branch へ向けられていることを確かめる。
    // 2 つを同じ branch へ向ければ共有、別の branch へ向ければ別の Network になる。
    for (const auto* key : { kPolicyFeatureKey, kValueFeatureKey }) {
        if (!network_config_.output_keys.contains(key)) {
            ANET_SYSTEM_ERROR("Missing " << network_config_.config_prefix << ".body.output.[" << key
                << "]; expected the name of the branch that feeds the Head.");
        }
    }

    // 方策の Head は行動数ぶんの logits、価値の Head は 1 つの値を出す。どちらも最後の Linear 1 層である。
    auto head_factory = std::make_shared<anet::nn::CompositeHeadFactory>(std::vector<anet::nn::CompositeHeadEntry>{
        { .name = kPolicyHeadName, .feature_key = kPolicyFeatureKey,
          .factory = std::make_shared<anet::nn::LinearHeadFactory>(action_count, kLogitsKey, policy_init) },
        { .name = kValueHeadName, .feature_key = kValueFeatureKey,
          .factory = std::make_shared<anet::nn::LinearHeadFactory>(1, kValueKey, value_init) } });
    const auto input_specs = MakeInputSpecs(obs_specs);
    network_ = anet::nn::NetworkBuilder::BuildNetwork(network_config_, input_specs, head_factory, seed, device);
    network_->to(device);
    // 共有する Network は通常 eval にし、Learner の学習の区間だけ train にする。
    network_->eval();

    // 出力の形を、仮の入力 1 行で確かめる。stack 軸を畳まない Body だと、Head の出力の rank が増える。
    torch::NoGradGuard no_grad;
    anet::TensorDict dummy;
    for (const auto& [key, spec] : input_specs) {
        std::vector<int64_t> shape{ 1 };
        shape.insert(shape.end(), spec.shape.begin(), spec.shape.end());
        dummy.Set(key, torch::zeros(shape, torch::TensorOptions().dtype(spec.dtype).device(device)));
    }
    const auto outputs = network_->Forward(dummy);
    const auto logits = outputs.At(kLogitsKey);
    const auto value = outputs.At(kValueKey);
    if (logits.dim() != 2 || logits.size(1) != action_count || value.dim() != 2 || value.size(1) != 1) {
        ANET_SYSTEM_ERROR("Invalid " << network_config_.config_prefix << " output shapes logits=" << logits.sizes()
            << " value=" << value.sizes() << "; expected [rows, " << action_count << "] and [rows, 1]."
            << " Start each branch with a module that folds the stack axis, such as Flatten.");
    }
}

anet::TensorSpecMap NetworkModel::MakeInputSpecs(const anet::TensorSpecMap& obs_specs)
{
    // Perception の stack の対象のキーは、1 枚でも stack 軸を持つ（重ねる枚数を増やしても形の意味が変わらない）。
    // PPO 系は観測を重ねないので、全部のキーに長さ 1 の軸を足す。
    auto specs = obs_specs;
    for (auto& [key, spec] : specs) {
        if (spec.has_stack_axis) {
            ANET_SYSTEM_ERROR("Observation key='" << key << "' already has a stack axis; expected an observation without it.");
        }
        spec.shape.insert(spec.shape.begin(), 1);
        spec.has_stack_axis = true;
    }
    return specs;
}

anet::TensorDict NetworkModel::MakeNetworkInput(const anet::TensorDict& obs)
{
    anet::TensorDict input;
    for (const auto& [key, value] : obs) input.Set(key, value.unsqueeze(1));
    return input;
}

std::shared_ptr<anet::nn::Network> NetworkModel::CloneNetwork(torch::Device device) const
{
    auto clone = network_->Clone(device);
    clone->eval();
    return clone;
}

NetworkModel::ParameterGroups NetworkModel::MakeParameterGroups() const
{
    // Body の parameter を、どちらの Head へ届くかで分ける。
    // GetParameterSplit の feature は、指定した branch と、その branch が依存する branch の parameter である。
    const auto policy_body = network_->GetParameterSplit(network_config_.output_keys.at(kPolicyFeatureKey)).feature;
    const auto value_body = network_->GetParameterSplit(network_config_.output_keys.at(kValueFeatureKey)).feature;
    std::unordered_set<const torch::TensorImpl*> policy_ids;
    std::unordered_set<const torch::TensorImpl*> value_ids;
    for (const auto& parameter : policy_body) policy_ids.insert(parameter.unsafeGetTensorImpl());
    for (const auto& parameter : value_body) value_ids.insert(parameter.unsafeGetTensorImpl());

    ParameterGroups groups;
    for (const auto& parameter : policy_body) {
        (value_ids.contains(parameter.unsafeGetTensorImpl()) ? groups.shared : groups.policy).push_back(parameter);
    }
    for (const auto& parameter : value_body) {
        if (!policy_ids.contains(parameter.unsafeGetTensorImpl())) groups.value.push_back(parameter);
    }

    // Head の parameter は、合成の Head が付けた名前（head.<名前>.…）で分ける。
    const auto policy_prefix = std::string(kHeadParameterPrefix) + kPolicyHeadName + kParameterNameSeparator;
    const auto value_prefix = std::string(kHeadParameterPrefix) + kValueHeadName + kParameterNameSeparator;
    for (const auto& item : network_->named_parameters()) {
        if (!item.value().requires_grad()) continue;
        if (item.key().starts_with(policy_prefix)) groups.policy.push_back(item.value());
        else if (item.key().starts_with(value_prefix)) groups.value.push_back(item.value());
    }
    return groups;
}

int64_t NetworkModel::Save(anet::OutputArchive& archive) const
{
    return archive.WriteTorchObject(network_);
}

int64_t NetworkModel::Load(anet::InputArchive& archive)
{
    const auto size = archive.ReadTorchObject(network_);
    network_->eval();
    return size;
}
