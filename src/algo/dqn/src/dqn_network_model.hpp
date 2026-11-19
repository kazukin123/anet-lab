#pragma once
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <limits>
#include <span>
#include <string_view>
#include <unordered_set>
#include <utility>
#include "anet/random.hpp"
#include "anet/stacker.hpp"
#include "anet/schedule.hpp"
#include "anet/nn_util.hpp"
#include "anet/nn.hpp"
#include "anet/rl.hpp"
#include "anet/scaler.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/agent.hpp"
#include "anet/transfer.hpp"
#include "dqn_math.hpp"

namespace anet::rl::dqn {

    struct SkipNetworkDefinition {
        SkipConfig config;
        anet::nn::NetworkConfig network_config;
        std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory;
        anet::seed_t seed;
    };

    class NetworkModel : public anet::Serializable {
    public:
        static constexpr const char* kKeyActionInput = "action";

        NetworkModel(
            const NetworkModelConfig& config,
            const torch::Device device,
            const anet::nn::NetworkConfig& network_config,
            const anet::TensorSpecMap& obs_spec,
            int64_t n_actions,
            std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory,
            bool distributional,
            anet::seed_t network_seed,
            const SkipNetworkDefinition* skip_definition = nullptr);
    protected:
        NetworkModel(
            const NetworkModelConfig& config,
            std::shared_ptr<anet::nn::Network> online_net,
            std::shared_ptr<anet::nn::Network> target_net,
            int64_t n_actions,
            bool distributional);
    public:

        /// 訓練モードを一時的に有効にし、復元して返す。
        anet::TensorDict ForwardOnlineWithTrain(const anet::TensorDict& obs, anet::nn::NetworkForwardExecutor& forward, const anet::nn::NetworkForwardCallInfo& info,
            anet::nn::NetworkForwardCapture* capture = nullptr) const;
        anet::TensorDict ForwardOnlineUpTo(const anet::TensorDict& obs, const std::string& branch_key,
            anet::nn::NetworkForwardCapture* capture = nullptr) const;

        // Network取得
        std::shared_ptr<anet::nn::Network> GetOnlineNetwork() { return online_net_; }
        std::shared_ptr<anet::nn::Network> GetTargetNetwork() { return target_net_; }
        std::shared_ptr<anet::nn::Network> GetSkipOnlineNetwork() { return skip_online_net_; }
        std::shared_ptr<anet::nn::Network> GetSkipTargetNetwork() { return skip_target_net_; }
        const SkipConfig& GetSkipConfig() const { return skip_config_; }

        /// 分布型Head判定
        bool IsDistributional() const;

        /// online_netのパラメータ取得
        std::vector<torch::Tensor> GetOnlineParameters() const;

        torch::OrderedDict<std::string, torch::Tensor> GetOnlineNamedParameters() const;

        /// target network 同期
        void UpdateTarget(anet::rl::step_t learn_step, float tau);
        const anet::ProfiledValueConfig<float>& GetSoftUpdateTauConfig() const { return config_.soft_update_tau; }
        int GetHardUpdateInterval() const { return config_.hard_update_interval; }

        /// メトリクス用：TensorDict
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key, const torch::Device& device);
    public:
        int64_t Save(OutputArchive& archive) const override;
        int64_t Load(InputArchive& archive) override;
        int64_t LoadCheckpoint(InputArchive& archive, bool has_skip);
    private:
        void ValidateSpectralNormSoftUpdateConfig() const;
        void SoftUpdate(float tau);
        void HardUpdate();
    private:
        const anet::rl::dqn::NetworkModelConfig config_;
        std::shared_ptr<anet::nn::Network> online_net_;
        std::shared_ptr<anet::nn::Network> target_net_;
        SkipConfig skip_config_;
        std::shared_ptr<anet::nn::Network> skip_online_net_;
        std::shared_ptr<anet::nn::Network> skip_target_net_;

        int64_t n_actions_;
        bool distributional_;
    };

    anet::TensorDict MakeSkipInput(const anet::TensorDict& obs, const anet::TensorDict& primary,
        const torch::Tensor& actions, bool stop_gradient);

    anet::TensorDict ReadSkipValues(const anet::TensorDict& output, const anet::TensorDict& primary,
        const torch::Tensor& actions, const SkipConfig& config);

} // namespace anet::rl::dqn
