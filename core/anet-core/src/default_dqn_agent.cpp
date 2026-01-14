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
#include "anet/stacker.hpp"
#include "nn_heads.hpp"


using namespace anet::rl::dqn;
namespace LOG = anet::log;


// ======================================================
// DefaultDQNAgent 本体
// ======================================================

DefaultDQNAgent::DefaultDQNAgent(
    const DefaultDQNAgentConfig& config
    , const anet::nn::NetworkConfig& net_config
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
    anet::MetricsLogger::Instance()->Log("net.body", net_config.ToJson());

    //seed
    anet::SeedMaker seed_maker(GetSeed());
    auto replay_seed = seed_maker.MakeNamedSeed("replaybuffer");
    auto action_policy_seed = seed_maker.MakeNamedSeed("action_policy");

    // RuntimeVars生成
    this->vars_ = std::make_unique<dqn::RuntimeVars>();

    // QR-DQN設定確認 (use_qrとの整合性)
    bool is_distributional = config_.use_qr;
    if (is_distributional && config_.num_quantiles <= 1) {
        LOG::error() << "use_qr is true but num_quantiles <= 1. Treating as Scalar DQN.";
        ANET_SYSTEM_ERROR("use_qr is true but num_quantiles <= 1. Treating as Scalar DQN.");
    }
    if (!is_distributional) {
        if (config_.action_policy.policy_type == kActionPolicyType_UQE || config_.action_policy.policy_type == kActionPolicyType_ThompsonSampling) {
            // UQEbベースではQR必須
            LOG::error() << "action_policy.policy_type is UQE but use_qr disabled.";
            ANET_SYSTEM_ERROR("action_policy.policy_type is UQE but use_qr disabled.");
        }
    }

    // RewardScaler生成
    anet::rl::RewardScalerFactory reward_scaler_factory(config_.reward_scaler);
    this->reward_scaler_ = reward_scaler_factory.CreateRewardScaler(config_.learner.gamma);
    
    // ObservationNormalizer生成
    anet::rl::ObservationNormalizerFactory obs_norm_factory(config_.obs_norm);
    this->obs_norm_ = obs_norm_factory.CreateObservationNormalizer(env_spec.state_spec);


    // ------------------------------------------------------------
    // HeadFactory の準備
    // ------------------------------------------------------------

    // Head用の初期化設定を作成 (AgentがConfigDataから読み取る)
    const anet::nn::WeightInitConfig& head_init_config = config_.head_init;

    std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory;

    // アルゴリズムに応じてFactoryを切り替え
    if (is_distributional) {
        if (config_.use_dueling_net) {
            head_factory = std::make_shared<anet::nn::QuantileDuelingHeadFactory>(
                n_actions_, config_.num_quantiles, head_init_config);
            LOG::info() << "Network Head: Quantile Dueling (N=" << config_.num_quantiles << ")";
        } else {
            head_factory = std::make_shared<anet::nn::QuantileHeadFactory>(
                n_actions_, config_.num_quantiles, head_init_config);
            LOG::info() << "Network Head: Quantile Plain (N=" << config_.num_quantiles << ")";
        }
    } else {
        if (config_.use_dueling_net) {
            head_factory = std::make_shared<anet::nn::DuelingHeadFactory>(
                n_actions_, head_init_config);
            LOG::info() << "Network Head: Dueling";
        } else {
            head_factory = std::make_shared<anet::nn::LinearHeadFactory>(
                n_actions_, head_init_config);
            LOG::info() << "Network Head: Plain Linear";
        }
    }

    // ------------------------------------------------------------
    // Network構築 (Builder)
    // ------------------------------------------------------------

    // 入力形状 (C, H, W) or (L,)
    auto input_shape = env_spec.state_spec.shape;

    // Stackの次元を先頭に追加（S, F)
    if (config_.stucker.use_stacker) {
        input_shape.insert(input_shape.begin(), config_.stucker.stack_count);
    }

    // Network構築
    auto policy_net = anet::nn::NetworkBuilder::BuildNetwork(net_config, input_shape, head_factory);
    auto target_net = anet::nn::NetworkBuilder::BuildNetwork(net_config, input_shape, head_factory);

    // デバイス転送
    policy_net->to(device_);
    target_net->to(device_);

    // 管理クラス (dqn::Network) に委譲
    this->network_ = std::make_unique<dqn::Network>(
        config_.network, device_, policy_net, target_net, n_actions_,
        config_.use_qr ? config_.num_quantiles : 0
    );

    // ActionPolicy生成
    if (config_.action_policy.policy_type == kActionPolicyType_EpsilonGreedy) {
        this->action_policy_ = std::make_unique<dqn::EpsilonGreedyActionPolicy>(config_.action_policy, *network_, *vars_, action_policy_seed);
    } else if (config_.action_policy.policy_type == kActionPolicyType_UQE) {
        this->action_policy_ = std::make_unique<dqn::UQEActionPolicy>(config_.action_policy, *network_, *vars_, action_policy_seed);
    }  else if (config_.action_policy.policy_type == kActionPolicyType_ThompsonSampling) {
        this->action_policy_ = std::make_unique<dqn::ThompsonSamplingActionPolicy>(config_.action_policy, *network_, *vars_, action_policy_seed);
    } else {
        ANET_SYSTEM_ERROR("Unknown action policy type. action_policy.policy_type=" << config_.action_policy.policy_type);
    }

    // Learner生成
    if (is_distributional) {
        this->learner_ = std::make_unique<dqn::QRLearner>(
            config_.learner, *network_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed, config_.stucker);
        LOG::info() << "Initialized QRLearner (Quantiles=" << config_.num_quantiles << ")";
    } else {
        this->learner_ = std::make_unique<dqn::TDLearner>(
            config_.learner, *network_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed, config_.stucker);
        LOG::info() << "Initialized TDLearner";
    }
}

std::optional<anet::TensorFunction> DefaultDQNAgent::GetTensorFunction(const std::string& key)
{
    auto fn = network_->GetTensorFunction(key, device_);
    if (fn == std::nullopt) return fn;

    auto self = shared_from_this();
    auto network_fn = *fn;
    bool use_stacker = config_.stucker.use_stacker;
    int stack_count = config_.stucker.stack_count;

    anet::TensorFunction norm_fn = [self, network_fn, use_stacker, stack_count](const torch::Tensor& obs) {

        std::shared_lock<std::shared_mutex> lock(*(self->mutex_));

        //ANET_LOG_DEBUG("obs=" << anet::ToDefString(obs));
        torch::Tensor proc_obs = obs;

        // Stacker有効なのに送られてきたデータが2次元(N, F)だった場合、時間方向に複製して3次元化する
        if (use_stacker && proc_obs.dim() == 2) {
            // (N, F) -> (N, 1, F) -> (N, Stack, F)
            proc_obs = proc_obs.unsqueeze(1).expand({ -1, stack_count, -1 });
        }
        //ANET_LOG_DEBUG("proc_obs=" << anet::ToDefString(proc_obs));

        // 正規化
        auto obs_norm = self->obs_norm_->Normalize(proc_obs);
        //ANET_LOG_DEBUG("obs_norm=" << anet::ToDefString(obs_norm));

		// ネットワーク実行 (stack有効の場合は(N, S, F)、無効の場合は(N, F)
        auto out = network_fn(obs_norm);
        return out;
        };

    return norm_fn;
}

std::optional<float> DefaultDQNAgent::GetScalar(const std::string& key, int64_t index) const
{
    if (key == "epsilon") {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return vars_->epsilon;
    }
    if (key == "uqe_tau") {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return vars_->uqe_tau;
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

std::optional<torch::Tensor> DefaultDQNAgent::GetTensor(const std::string& key, int64_t index) const
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

std::optional<std::vector<torch::Tensor>> DefaultDQNAgent::GetTensorVector(const std::string& key, int64_t index) const
{
    if (key.find(ReplayBuffer::kKeyPrefix) == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetTensorVector(key);
    }

    return std::nullopt;
}

std::shared_ptr<anet::rl::ActionContext> DefaultDQNAgent::CreateActionContext(
    const BatchEnvSpec& batch_env_spec, RunMode run_mode) const
{
    if (config_.stucker.use_stacker) {
        // Stackerを作成して包んで返す
        auto stacker = std::make_shared<TensorFrameStacker>(
            config_.stucker.stack_count, batch_env_spec.batch_size, this->device_ );
        return std::make_shared<StackerActionContext>(run_mode, stacker);
    }

    // Stacker無効ならデフォルト
    return std::make_shared<DefaultActionContext>(run_mode);
}

anet::rl::BatchActionInfo DefaultDQNAgent::MakeAction(const StepCounts& step, const BatchState& state, std::shared_ptr<ActionContext> ctx) const
{
    ProfileRange r1("DefaultDQNAgent::MakeAction");
    ANET_ASSERT_SHAPE(state.obs, { ANET_SHAPE_ANY, state_dim_ });

    // 共有ロック＆Grad抑止
    std::shared_lock<std::shared_mutex> lock(*mutex_);
    torch::NoGradGuard ng;

    // obsを生成
    torch::Tensor obs = state.obs;
    if (ctx) obs = ctx->PushObservation(state);

    // Normalize observations
    auto obs_norm = this->obs_norm_->Normalize(obs);

    // 行動選択
	auto run_mode = (ctx != nullptr) ? ctx->GetRunMode(): anet::rl::RunMode::Train;
    auto greedy_only = anet::rl::IsEval(run_mode);
    auto use_target = (run_mode == anet::rl::RunMode::Eval1);
    auto act_info = this->action_policy_->SelectAction(obs_norm, greedy_only, use_target);

    // ActionInfoを返す
    return act_info;
}

anet::rl::BatchUpdateResultList
DefaultDQNAgent::UpdateFromBatch(const StepCounts& counts, const anet::rl::BatchExperience& batch_exp, const anet::rl::Runner& runner)
{
    ProfileRange r1("DefaultDQNAgent::UpdateFromBatch");

    BatchUpdateResultList result_list;

    if (true) {
        // 排他ロック
        std::unique_lock<std::shared_mutex> lock(*mutex_);

        // RewardScaler
        auto scaled_rewards = this->reward_scaler_->Scale(batch_exp.reward);

        // Normalize observations 統計更新
        this->obs_norm_->NormalizeAndUpdateStats(batch_exp.state.obs);
        this->obs_norm_->NormalizeAndUpdateStats(batch_exp.next_state.obs); // 二重計上になるがエピソード終端（着地/墜落）の状態も統計に反映させる

        // BatchExperience生成
        // ReplayBufferには「生の観測」を渡す。 報酬だけはスケール済みを使う
        BatchExperience exp {
            batch_exp.state,
            batch_exp.action,
            scaled_rewards,
            batch_exp.next_state
        };

        // Update実行
        auto result = this->learner_->UpdateFromBatch(counts, exp, runner);
        result_list = std::move(result);

        // Update後処理
        action_policy_->OnLearn(counts);
    }

    // LearnEvent通知（排他解除後でないとデッドロックになる）
    if (result_list.size() > 0 && notifier_ != nullptr) {
        for (auto result : result_list) {
            anet::rl::LearnEvent event{ batch_exp, runner, counts, shared_from_this(), result_list };
            notifier_->Notify(event);
        }
    }

    // BatchUpdateResultを返す
    return result_list;
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
	anet::nn::NetworkConfig net_config(config_data);
    auto agent = std::make_shared<DefaultDQNAgent>(config, net_config, batch_env_spec, env_spec, device, notifier, seed);
    return agent;

    /// @todo 引数の順番を統一
}

//ANET_REGISTER_AGENT_FACTORY(DefaultDQNAgentFactory);
