// default_dqn_agent.cpp

#include "anet/default_dqn_agent.hpp"
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


using namespace anet::rl::dqn;
namespace LOG = anet::log;


// ======================================================
// DefaultDQNAgent 本体
// ======================================================

DefaultDQNAgent::DefaultDQNAgent(
    const DefaultDQNAgentConfig& config
    , const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, const torch::Device& device
    , std::shared_ptr<Notifier> notifier
    , std::optional<seed_t> seed)
    : FlatStateAgent(device, notifier, batch_env_spec, env_spec, seed)
    , config_(config)
{
    ANET_LOG_DEBUG("seed=" << GetSeed());

    // ログ：パラメータ記録
    LOG::info() << "DefaultDQNAgent config=" << config_;
    anet::MetricsLogger::Instance()->Log(config_);

    //seed
    anet::SeedMaker seed_maker(GetSeed());
    auto replay_seed = seed_maker.MakeNamedSeed("replaybuffer");
    auto action_policy_seed = seed_maker.MakeNamedSeed("action_policy");

    // RuntimeVars生成
    this->vars_ = std::make_unique<dqn::RuntimeVars>();

    // QR-DQN設定確認 (use_qr フラグと num_quantiles の整合性)
    bool is_distributional = config_.use_qr;
    if (is_distributional && config_.num_quantiles <= 1) {
        LOG::warn() << "use_qr is true but num_quantiles <= 1. Treating as Scalar DQN.";
        is_distributional = false;
    }

    // RewardScaler生成
    anet::rl::RewardScalerFactory reward_scaler_factory(config_.reward_scaler);
    this->reward_scaler_ = reward_scaler_factory.CreateRewardScaler(config_.learner.gamma);
    
    // ObservationNormalizer生成
    anet::rl::ObservationNormalizerFactory obs_norm_factory(config_.obs_norm);
    this->obs_norm_ = obs_norm_factory.CreateObservationNormalizer(env_spec.state_spec);

    // NN生成＆初期化
    std::shared_ptr<anet::rl::dqn::QNet> policy_net;
    std::shared_ptr<anet::rl::dqn::QNet> target_net;
    if (config_.use_dueling_net) {
        if (is_distributional) {
            policy_net = std::make_shared<anet::rl::dqn::QuantileDuelingQNet>(config_.qnet, state_dim_, n_actions_);
            target_net = std::make_shared<anet::rl::dqn::QuantileDuelingQNet>(config_.qnet, state_dim_, n_actions_);
        } else {
            policy_net = std::make_shared<anet::rl::dqn::DuelingQNet>(config_.qnet, state_dim_, n_actions_);
            target_net = std::make_shared<anet::rl::dqn::DuelingQNet>(config_.qnet, state_dim_, n_actions_);
        }
    } else {
        if (is_distributional) {
            policy_net = std::make_shared<anet::rl::dqn::QuantilePlainQNet>(config_.qnet, state_dim_, n_actions_);
            target_net = std::make_shared<anet::rl::dqn::QuantilePlainQNet>(config_.qnet, state_dim_, n_actions_);
        } else {
            policy_net = std::make_shared<anet::rl::dqn::PlainQNet>(config_.qnet, state_dim_, n_actions_);
            target_net = std::make_shared<anet::rl::dqn::PlainQNet>(config_.qnet, state_dim_, n_actions_);
        }
    }

    // Network生成
    this->network_ = std::make_unique<dqn::Network>(config_.network, device_, policy_net, target_net);

    // ActionPolicy生成
    this->action_policy_ = std::make_unique<dqn::ActionPolicy>(*network_, *vars_, action_policy_seed);

    // Learner生成
    if (is_distributional) {
        this->learner_ = std::make_unique<dqn::QRLearner>(
            config_.learner, *network_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed);
        LOG::info() << "Initialized QRLearner (Quantiles=" << config_.num_quantiles << ")";
    } else {
        this->learner_ = std::make_unique<dqn::TDLearner>(
            config_.learner, *network_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed);
        LOG::info() << "Initialized TDLearner";
    }
}

std::optional<anet::TensorFunction> DefaultDQNAgent::GetTensorFunction(const std::string& key)
{
    return network_->GetTensorFunction(key, device_, mutex_);
}

std::optional<float> DefaultDQNAgent::GetScalar(const std::string& key, int index) const
{
    if (key == "epsilon") {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return vars_->epsilon;
    }
    if (key == "per_beta") {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return vars_->per_beta;
    }
    if (key.find(ReplayBuffer::kKeyPrefix) == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetScalar(key);
    }
    if (key.find(RewardScaler::kKeyPrefix) == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return reward_scaler_->GetScalar(key);
    }
    if (key.find(ObservationNormalizer::kKeyPrefix) == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return obs_norm_->GetScalar(key);
    }

    return std::nullopt;
}

std::optional<torch::Tensor> DefaultDQNAgent::GetTensor(const std::string& key, int index) const
{
    if (key.find(ReplayBuffer::kKeyPrefix) == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetTensor(key);
    }
    if (key.find(RewardScaler::kKeyPrefix) == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return reward_scaler_->GetTensor(key);
    }
    if (key.find(ObservationNormalizer::kKeyPrefix) == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return obs_norm_->GetTensor(key);
    }

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> DefaultDQNAgent::GetTensorVector(const std::string& key, int index) const
{
    if (key.find(ReplayBuffer::kKeyPrefix) == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetTensorVector(key);
    }

    return std::nullopt;
}

anet::rl::BatchActionInfo DefaultDQNAgent::MakeAction(const StepCounts& step, const BatchState& state, RunMode runmode) const
{
    ProfileRange r1("DefaultDQNAgent::MakeAction");
    ANET_CHECK_SHAPE(state.obs, { ANET_SHAPE_ANY, state_dim_ });

    // 共有ロック＆Grad抑止
    std::shared_lock<std::shared_mutex> lock(*mutex_);
    torch::NoGradGuard ng;

    // Flatなobsを生成
    auto flat_obs = state.To(device_).Flatten().obs;

    // Normalize observations
    auto obs_norm = this->obs_norm_->Normalize(flat_obs);

    // 行動選択
    auto greedy_only = anet::rl::IsEval(runmode);
    auto use_target = (runmode == anet::rl::RunMode::Eval1);
    auto act_info = this->action_policy_->SelectAction(obs_norm, greedy_only, use_target);

    // ActionInfoを返す
    return act_info;
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
DefaultDQNAgent::UpdateFromBatch(const StepCounts& counts, const anet::rl::BatchExperience& batch_exp, const anet::rl::Runner& runner)
{
    ProfileRange r1("DefaultDQNAgent::UpdateFromBatch");

    std::shared_ptr<const anet::rl::BatchUpdateResult> update_result;

    if (true) {
        // 排他ロック
        std::unique_lock<std::shared_mutex> lock(*mutex_);

        // RewardScaler
        auto scaled_rewards = this->reward_scaler_->Scale(batch_exp.reward);

        // Normalize observations 統計更新
        this->obs_norm_->NormalizeAndUpdateStats(batch_exp.state.obs);

        // 【重要】next_state では更新しない (false)
        // 理由：next_stateには終端状態などが含まれ、入力分布を歪める可能性があるため
        // また、state だけで十分なサンプル数があるため

        // BatchExperience生成
        // 【重要】ReplayBufferには「生の観測」を渡す。 報酬だけはスケール済みを使う
        BatchExperience exp {
            batch_exp.state,
            batch_exp.action,
            scaled_rewards,
            batch_exp.next_state
        };

        // Update実行
        update_result = this->learner_->UpdateFromBatch(counts, exp, runner);
    } else {
        // 更新なしでResultだけ作る
        update_result = std::make_shared<dqn::BatchUpdateResult>(0);
    }

    // LearnEvent通知
    if (update_result->GetLearnStepDiff() > 0 && notifier_ != nullptr) {
        anet::rl::LearnEvent event{ batch_exp, runner, counts, shared_from_this(), update_result };
        notifier_->Notify(event);
    }

    // BatchUpdateResultを返す
    return update_result;
}

// ======================================================
// DefaultDQNAgentFactory
// ======================================================

std::shared_ptr<anet::rl::Agent> DefaultDQNAgentFactory::CreateAgent(
    const EnvSpec& env_spec, const BatchEnvSpec& batch_env_spec,
    const torch::Device& device, const anet::ConfigData& config_data,
    std::shared_ptr<anet::rl::Notifier> notifier, std::optional<anet::seed_t> seed) const
{
    DefaultDQNAgentConfig config(config_data);
    auto agent = std::make_shared<DefaultDQNAgent>(config, batch_env_spec, env_spec, device, notifier, seed);
    return agent;

    /// @todo 引数の順番を統一
}

//ANET_REGISTER_AGENT_FACTORY(DefaultDQNAgentFactory);

