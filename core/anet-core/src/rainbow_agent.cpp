// dqn_agent.cpp

#include "anet/rainbow_agent.hpp"
#include "dqn_based_agent.hpp"
#include <memory>
#include <torch/torch.h>
#include <tuple>
#include "anet/str_util.hpp"
#include "anet/nn_util.hpp"
#include "anet/tensor_util.hpp"
#include "anet/tensor_check.hpp"
#include "anet/log.hpp"
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/profile.hpp"
#include "dqn_based_heads.hpp"

using namespace anet::rl::dqn;
namespace LOG = anet::log;


// ======================================================
// RainbowAgent 本体
// ======================================================

RainbowAgent::RainbowAgent(
    const RainbowAgentConfig& config
    , const anet::nn::NetworkConfig& net_config
    , const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, const torch::Device& device
    , std::optional<seed_t> seed)
    : AgentBase(device, batch_env_spec, env_spec, seed)
    , config_(config)
{
    ANET_LOG_DEBUG("seed=" << GetSeed());

    // ログ：パラメータ記録
    LOG::info() << "RainbowAgent config=" << config_.ToString();
    anet::MetricsLogger::Instance()->Log("RainbowAgent", config_.ToJson());
    anet::MetricsLogger::Instance()->Flush();

    //seed
    anet::SeedMaker seed_maker(GetSeed());
    auto replay_seed = seed_maker.MakeNamedSeed("replaybuffer");
    //auto action_policy_seed = seed_maker.MakeNamedSeed("action_policy");

    // RuntimeVars生成
    this->vars_ = std::make_unique<RuntimeVars>();
    this->vars_->epsilon = config_.action_policy.eps_start;

    // QR-DQN設定確認 (use_qr フラグと num_quantiles の整合性)
    bool is_distributional = config_.use_qr;
    if (is_distributional && config_.num_quantiles <= 1) {
        LOG::warn() << "use_qr is true but num_quantiles <= 1. Treating as Scalar DQN.";
        is_distributional = false;
    }

    // ------------------------------------------------------------
	// HeadFactory の準備
	// ------------------------------------------------------------
	
	// Head用の初期化設定を作成 (AgentがConfigDataから読み取る)
    const anet::nn::WeightInitConfig& head_init_config = config_.head_init;

    std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory;

    // アルゴリズムに応じてFactoryを切り替え
    if (is_distributional) {
        if (config_.use_dueling_net) {
            head_factory = std::make_shared<QuantileDuelingHeadFactory>(
                n_actions_, config_.num_quantiles, head_init_config);
            LOG::info() << "Network Head: Quantile Dueling (N=" << config_.num_quantiles << ")";
        } else {
            head_factory = std::make_shared<QuantileHeadFactory>(
                n_actions_, config_.num_quantiles, head_init_config);
            LOG::info() << "Network Head: Quantile Plain (N=" << config_.num_quantiles << ")";
        }
    } else {
        if (config_.use_dueling_net) {
            head_factory = std::make_shared<DuelingHeadFactory>(
                n_actions_, head_init_config);
            LOG::info() << "Network Head: Dueling";
        } else {
            head_factory = std::make_shared<LinearHeadFactory>(
                n_actions_, head_init_config);
            LOG::info() << "Network Head: Plain Linear";
        }
    }

    // ------------------------------------------------------------
    // Network構築 (Builder)
    // ------------------------------------------------------------

    // NetworkModel生成
    this->model_ = std::make_unique<NetworkModel>(
        config_.model, device_, net_config, env_spec.state_spec.obs_spec, n_actions_, head_factory, config_.num_quantiles);

    // ActionPolicy生成
    this->action_policy_ = std::make_unique<EpsilonGreedyActionPolicy>(config_.action_policy);

    // Greedyは、EpsilonGreedyのノイズ0としてインスタンス化
    ActionPolicyConfig greedy_cfg;
    greedy_cfg.policy_type = "EpsilonGreedy";
    greedy_cfg.eps_start = 0.0f;
    greedy_cfg.eps_end = 0.0f;
    this->target_policy_ = std::make_shared<EpsilonGreedyActionPolicy>(greedy_cfg);

    // Learner生成
    if (is_distributional) {
        this->learner_ = std::make_unique<QRLearner>(config_.learner, *model_, *vars_, nullptr, batch_env_spec, env_spec, device_, replay_seed, target_policy_);
        LOG::info() << "Initialized QRLearner (Quantiles=" << config_.num_quantiles << ")";
    } else {
        this->learner_ = std::make_unique<TDLearner>(config_.learner, *model_, *vars_, nullptr, batch_env_spec, env_spec, device_, replay_seed, target_policy_);
        LOG::info() << "Initialized TDLearner";
    }
}

std::optional<anet::TensorFunction> RainbowAgent::GetTensorFunction(const std::string& key)
{
    auto fn = model_->GetTensorFunction(key, device_);
    if (fn == std::nullopt) return fn;

    auto self = shared_from_this();
    auto network_fn = *fn;

    anet::TensorFunction norm_fn = [self, network_fn](const torch::Tensor& obs) {
        std::shared_lock<std::shared_mutex> lock(*(self->mutex_));
        auto out = network_fn(obs);
        return out;
        };

    return norm_fn;
}

std::optional<float> RainbowAgent::GetScalar(const std::string& key, int64_t index) const
{
    if (key == "epsilon" || key == "uqe_tau") {
        return action_policy_->GetScalar(key, index);
    }
    if (key == "per_beta") {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return vars_->per_beta;
    }
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetScalar(key);
    }

    return std::nullopt;
}

std::optional<torch::Tensor> RainbowAgent::GetTensor(const std::string& key, int64_t index) const
{
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetTensor(key);
    }

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> RainbowAgent::GetTensorVector(const std::string& key, int64_t index) const
{
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetTensorVector(key);
    }

    return std::nullopt;
}

std::shared_ptr<anet::rl::ActionContext> RainbowAgent::CreateActionContext(
    const BatchEnvSpec& batch_env_spec, RunMode run_mode, std::optional<torch::Device> device) const
{
    // 専用のRNGから1つシードを払い出してコンテキストに渡す
    auto rng = GetRandomGenerator(run_mode);
    seed_t ctx_seed = rng->RandUint64();
    return std::make_shared<DefaultActionContext>(run_mode, ctx_seed, device_);
}

std::shared_ptr<anet::rl::Actor> RainbowAgent::CreateActor(const anet::rl::BatchEnvSpec& batch_env_spec, anet::rl::RunMode run_mode, bool clone_model, std::optional<torch::Device> device) const
{
    // Contextを生成
    auto ctx = this->CreateActionContext(batch_env_spec, run_mode, device_);

    // モードに応じて適切な Policy と Network を選択
    std::shared_ptr<ActionPolicy> policy;
    std::shared_ptr<anet::nn::Network> src_network;

    // 元ネタのPolicyとNetoworkを決定
    if (anet::rl::IsEval(run_mode)) {
        src_network = (run_mode == anet::rl::RunMode::Eval1) ? model_->GetTargetNetwork() : model_->GetOnlineNetwork();
    } else {
        src_network = model_->GetOnlineNetwork();
    }

    // 必要に応じてCloneしてActor向けネットワークとする
    auto network = (clone_model) ? src_network->Clone(device) : src_network;
    if (clone_model) {
        network->eval();
    }

    // Actor を生成
    auto actor = std::make_shared<Actor>(action_policy_, nullptr, ctx, this->mutex_, network, src_network);

    // 生成したActorを返す
    return actor;
}

std::shared_ptr<anet::rl::Learner> RainbowAgent::CreateLearner()
{
    return this->shared_from_this();
}

anet::rl::BatchUpdateResultList
RainbowAgent::UpdateFromBatch(const StepCounts& counts, const anet::rl::BatchExperience& batch_exp)
{
    ANET_PROFILE_FUNC();

    anet::rl::BatchUpdateResultList result_list;
    {
        // 排他ロック
        std::unique_lock<std::shared_mutex> lock(*mutex_);

        // Update実行
        auto result = this->learner_->UpdateFromBatch(counts, batch_exp);
        result_list = std::move(result);

        // Update後処理
        action_policy_->OnLearn(counts);
    }

    // BatchUpdateResultListを返す
    return result_list;
}

// ======================================================
// RainbowAgentFactory
// ======================================================

std::shared_ptr<anet::rl::Agent> RainbowAgentFactory::CreateAgent(
    const EnvSpec& env_spec, const BatchEnvSpec& batch_env_spec,
    const torch::Device& device, const anet::ConfigData& config_data,
    std::shared_ptr<anet::rl::Notifier> notifier, std::optional<anet::seed_t> seed) const
{
    RainbowAgentConfig config(config_data);
    anet::nn::NetworkConfig net_config(config_data);
    auto agent = std::make_shared<RainbowAgent>(config, net_config, batch_env_spec, env_spec, device, seed);
    return agent;

    /// @todo 引数の順番を統一
}

//ANET_REGISTER_AGENT_FACTORY(RainbowAgentFactory);
