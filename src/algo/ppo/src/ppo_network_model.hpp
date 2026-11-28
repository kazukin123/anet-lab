#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>
#include "anet/common.hpp"
#include "anet/random.hpp"
#include "anet/serialize.hpp"
#include "anet/nn.hpp"

namespace anet::rl::ppo {

    /// 方策と価値の Head を持つ Network の持ち主。
    /// Network は Agent の Resource で、Learner が更新し、Actor は共有するか写しを持つ。
    /// 方策と価値の Body を共有するか別にするかは、Head が受け取る 2 つのキーを向ける branch で決まる。
    class NetworkModel {
    public:
        /// 勾配の内訳に使う parameter の組。
        struct ParameterGroups {
            std::vector<torch::Tensor> shared; ///< 方策と価値の両方の Head へ届く Body の parameter。
            std::vector<torch::Tensor> policy; ///< 方策の Head と、方策の Head にだけ届く Body の parameter。
            std::vector<torch::Tensor> value;  ///< 価値の Head と、価値の Head にだけ届く Body の parameter。
        };
    public:
        static constexpr char kPolicyFeatureKey[] = "policy_feature"; ///< 方策の Head が受け取る Body の出力のキー。
        static constexpr char kValueFeatureKey[] = "value_feature";   ///< 価値の Head が受け取る Body の出力のキー。
        static constexpr char kPolicyHeadName[] = "policy";
        static constexpr char kValueHeadName[] = "value";
        static constexpr char kLogitsKey[] = "logits"; ///< Network の出力。[行, 行動数]。
        static constexpr char kValueKey[] = "value";   ///< Network の出力。[行, 1]。
    public:
        NetworkModel(const anet::nn::NetworkConfig& network_config, const anet::TensorSpecMap& obs_specs,
            int64_t action_count, const anet::nn::WeightInitConfig& policy_init,
            const anet::nn::WeightInitConfig& value_init, torch::Device device, anet::seed_t seed);

        /// 観測の spec から、Network の入力の spec を作る。各キーの先頭に長さ 1 の stack 軸を足す。
        static anet::TensorSpecMap MakeInputSpecs(const anet::TensorSpecMap& obs_specs);
        /// device 上の観測から、Network の入力を作る。各キーの先頭（batch 軸の次）に長さ 1 の stack 軸を足す。
        static anet::TensorDict MakeNetworkInput(const anet::TensorDict& obs);

        const std::shared_ptr<anet::nn::Network>& GetNetwork() const { return network_; }
        /// Actor が持つ写しを作る。呼ぶ側が、Network を守るロックを持つ。
        std::shared_ptr<anet::nn::Network> CloneNetwork(torch::Device device) const;
        ParameterGroups MakeParameterGroups() const;

        /// 方策の世代。Learner が 1 つの rollout の更新を終えるたびに 1 進む。
        uint64_t GetPolicyVersion() const { return policy_version_.load(); }
        void AdvancePolicyVersion() { ++policy_version_; }

        int64_t Save(anet::OutputArchive& archive) const;
        int64_t Load(anet::InputArchive& archive);
    private:
        const anet::nn::NetworkConfig network_config_;
        std::shared_ptr<anet::nn::Network> network_;
        std::atomic<uint64_t> policy_version_{ 0 }; ///< Learner が進め、Actor が読む。
    };

} // namespace anet::rl::ppo
