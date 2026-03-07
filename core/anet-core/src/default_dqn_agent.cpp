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
#include "anet/serialize.hpp"

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
    LOG::info() << "DefaultDQNAgent config=" << config_.ToString();
    anet::MetricsLogger::Instance()->Log(config_);
    anet::MetricsLogger::Instance()->Log("net.body", net_config.ToJson());

	// 割引率gammaと実効ホライズンのログ
    {
        // 実効ホライズン (Effective Horizon) = 1 / (1 - gamma)
        // gamma=0.9 -> 10 steps, 0.99 -> 100 steps, 0.995 -> 200 steps
        float g = config_.learner.gamma;
        float horizon = (std::abs(g - 1.0f) < 1e-6) ? -1.0f : (1.0f / (1.0f - g));

        //時系列の重みが約37%に減衰するまでのSTEP数
        if (horizon > 0) {
            LOG::info() << "gamma=" << g << " (EffectiveHorizon: " << static_cast<int>(horizon) << " steps)";
        } else {
            LOG::info() << "gamma=" << g << " (Effective Horizon: Infinite)";
        }
    }

    // seed
    anet::SeedMaker seed_maker(GetSeed());
    auto replay_seed = seed_maker.MakeNamedSeed("replaybuffer");
    auto learner_seed = seed_maker.MakeNamedSeed("learner");
    this->action_context_seed_ = seed_maker.MakeNamedSeed("action_context");

    // RuntimeVars生成
    this->vars_ = std::make_unique<dqn::RuntimeVars>();

    // QR-DQN設定確認 (use_qrとの整合性)
    bool is_distributional = config_.use_qr;
    if (is_distributional && config_.num_quantiles <= 1) {
        LOG::error() << "use_qr is true but num_quantiles <= 1. Treating as Scalar DQN.";
        ANET_SYSTEM_ERROR("use_qr is true but num_quantiles <= 1. Treating as Scalar DQN.");
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
    this->train_policy_ = CreateActionPolicy(config_.train_policy);
    this->eval_policy_ = CreateActionPolicy(config_.eval_policy);
    this->target_policy_ = CreateActionPolicy(config_.target_policy);

    // Target Policyの妥当性チェック
    if (config_.target_policy.policy_type == "EpsilonGreedy" && config_.target_policy.eps_start > 0.0f) {
        // TargetActionPolicy（学習用）はUQE/ThompsonSamplingもしくはGreedyである必要がある(ランダム要素はNG)
        ANET_SYSTEM_ERROR("target_policy cannot be EpsilonGreedy with eps > 0. It must be deterministic or optimistic.");
    }

    // Learner生成
    if (is_distributional) {
        this->learner_ = std::make_unique<dqn::QRLearner>(
            config_.learner, *network_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed, target_policy_, config_.stucker, learner_seed);
        LOG::info() << "Initialized QRLearner (Quantiles=" << config_.num_quantiles << ")";
    } else {
        this->learner_ = std::make_unique<dqn::TDLearner>(
            config_.learner, *network_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed, target_policy_, config_.stucker, learner_seed);
        LOG::info() << "Initialized TDLearner";
    }
}

std::shared_ptr<anet::rl::dqn::ActionPolicy> DefaultDQNAgent::CreateActionPolicy(const ActionPolicyConfig& policy_config)
{
    if (policy_config.policy_type == "EpsilonGreedy" || policy_config.policy_type == "0") {
        // ε-Greedy
        return std::make_shared<dqn::EpsilonGreedyActionPolicy>(policy_config, *network_);
    } else if (policy_config.policy_type == "UQE" || policy_config.policy_type == "1") {
        // UQE
        ANET_CHECK(config_.use_qr);
        return std::make_shared<dqn::UQEActionPolicy>(policy_config, *network_);
    } else if (policy_config.policy_type == "ThompsonSampling" || policy_config.policy_type == "2") {
        //ThompsonSampling
        ANET_CHECK(config_.use_qr);
        return std::make_shared<dqn::ThompsonSamplingActionPolicy>(policy_config, *network_);
    } else if (policy_config.policy_type == "Greedy" || policy_config.policy_type == "3") {
        // Greedyは、EpsilonGreedyのノイズ0としてインスタンス化
        ActionPolicyConfig greedy_cfg = policy_config;
        greedy_cfg.eps_start = 0.0f;
        greedy_cfg.eps_end = 0.0f;
        return std::make_shared<dqn::EpsilonGreedyActionPolicy>(greedy_cfg, *network_);
    }

    // 不明なtype
    ANET_SYSTEM_ERROR("Unknown action policy type: " << policy_config.policy_type);
    return nullptr;
}

int64_t DefaultDQNAgent::Save(anet::OutputArchive& archive) const
{
    ProfileRange r1("DefaultDQNAgent::Save");

	int64_t total_size = 0;

    // ヘッダ
    anet::ArchiveHeader header;
    header.info = "DefaultDQNAgent";
    auto header_size = archive.Write(header);
	total_size += header_size;

    // Config
    auto config_size = archive.Write(this->config_.ToString());
	total_size += config_size;

    /// @todo state_dim_やn_actions_の永続化対応(NN整合チェックで必要)
    /// @todo RewardScalerの永続化対応
    /// @todo ObservationNormalizerの永続化対応

    // Network(policy_net/target_net)
	auto network_size = network_->Save(archive);
	total_size += network_size;

    // Learner(Adam)
    auto adam_size = learner_->Save(archive);
	total_size += adam_size;

    // ログ
    LOG::info() << "DefaultDQNAgent Serialized. total_size=" << anet::FormatWithCommas(total_size)
        << " config_size=" << anet::FormatWithCommas(config_size)
        << " adam_size=" << anet::FormatWithCommas(adam_size)
        << " network_size=" << network_size;

    return total_size;
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

std::optional<anet::TensorDictFunction> DefaultDQNAgent::GetTensorDictFunction(const std::string& key)
{
    // dqn::Network に委譲してベース関数を取得
    auto fn = network_->GetTensorDictFunction(key, device_);
    if (fn == std::nullopt) return std::nullopt;

    auto self = shared_from_this();
    auto network_fn = *fn;
    bool use_stacker = config_.stucker.use_stacker;
    int stack_count = config_.stucker.stack_count;

    // ロックと前処理（正規化等）をラップした関数を作成
    anet::TensorDictFunction norm_fn = [self, network_fn, use_stacker, stack_count](const torch::Tensor& obs) {

        // 排他制御（他スレッドでのパラメータ更新と競合しないように）
        std::shared_lock<std::shared_mutex> lock(*(self->mutex_));
        torch::NoGradGuard grad_guard;

        torch::Tensor proc_obs = obs;

        // Stacker有効なのに送られてきたデータが2次元(N, F)だった場合、時間方向に複製して3次元化する
        if (use_stacker && proc_obs.dim() == 2) {
            proc_obs = proc_obs.unsqueeze(1).expand({ -1, stack_count, -1 });
        }

        // Agentが持っているObservationNormalizerを通す
        auto obs_norm = self->obs_norm_->Normalize(proc_obs);

        // ネットワーク(policy_net or target_net)から抽出して返す
        return network_fn(obs_norm);
        };

    return norm_fn;
}

std::optional<float> DefaultDQNAgent::GetScalar(const std::string& key, int64_t index) const
{
    // プレフィックスで各 Policy インスタンスへ処理を委譲
    if (anet::StartsWith(key, "train_policy.")) return train_policy_->GetScalar(anet::RemovePrefix(key, "train_policy."), index);
    if (anet::StartsWith(key, "eval_policy.")) return eval_policy_->GetScalar(anet::RemovePrefix(key, "eval_policy."), index);
    if (anet::StartsWith(key, "target_policy.")) return target_policy_->GetScalar(anet::RemovePrefix(key, "target_policy."), index);

    //（後方互換）単なる "epsilon" などの指定は train_policy の情報を返す
    if (key == "epsilon" || key == "uqe_tau") {
        return train_policy_->GetScalar(key, index);
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

std::shared_ptr<anet::rl::ActionContext> DefaultDQNAgent::CreateActionContext(const BatchEnvSpec& batch_env_spec, RunMode run_mode) const
{
    seed_t ctx_seed = 0;
    {
        std::lock_guard<std::mutex> lock(rng_mutex_);

        // RunMode別のRNGからシードを1つ引く
        if (context_seed_rngs_.find(run_mode) == context_seed_rngs_.end()) {
            // そのRunModeのRNGがまだ無ければ、AgentのベースシードとRunModeで初期化して作る
            seed_t mode_base_seed = anet::splitmix64(this->action_context_seed_ ^ static_cast<uint64_t>(run_mode));
            context_seed_rngs_[run_mode] = std::make_shared<anet::RandomGenerator>(mode_base_seed);
        }

        // 要求が来るたびに、そのRunMode専用のRNG内部状態が進み、新しいシードが払い出される
        ctx_seed = context_seed_rngs_[run_mode]->RandUint64();
    }

    if (config_.stucker.use_stacker) {
        // Stackerを作成して包んで返す
        auto stacker = std::make_shared<TensorFrameStacker>(
            config_.stucker.stack_count, batch_env_spec.batch_size, this->device_ );
        return std::make_shared<StackerActionContext>(run_mode, stacker, ctx_seed);
    }

    // Stacker無効ならデフォルト
    return std::make_shared<DefaultActionContext>(run_mode, ctx_seed);
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
    auto obs_norm = obs_norm_->Normalize(obs);

    // 行動選択
    BatchActionInfo act_info;

    // RunMode に応じて Policy を切り替える
    auto rnd = ctx->GetRandomGenerator();
    auto run_mode = (ctx != nullptr) ? ctx->GetRunMode() : anet::rl::RunMode::Train;
    if (anet::rl::IsEval(run_mode)) {
        auto use_target = (run_mode == anet::rl::RunMode::Eval1);
        act_info = eval_policy_->SelectAction(obs_norm, false, use_target, rnd);
    } else {
        act_info = train_policy_->SelectAction(obs_norm, false, false, rnd);
    }

    // スタック済み・正規化前の観測テンソルをAuxに詰める
    act_info.GetAuxData()["raw_obs"] = obs;

    // ActionInfoを返す
    return act_info;
}

anet::rl::BatchUpdateResultList
DefaultDQNAgent::UpdateFromBatch(const StepCounts& counts, const anet::rl::BatchExperience& batch_exp, std::shared_ptr<const anet::rl::Runner> runner)
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
        train_policy_->OnLearn(counts);
        eval_policy_->OnLearn(counts);
        target_policy_->OnLearn(counts);
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
