// default_dqn_agent.cpp

#include "anet/default_dqn_agent.hpp"
#include "dqn_based_agent.hpp"
#include <memory>
#include <torch/torch.h>
#include <cmath>
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
#include "dqn_based_heads.hpp"
#include "nn_heads.hpp"
#include "anet/serialize.hpp"


using namespace anet::rl::dqn;
namespace LOG = anet::log;

namespace {

bool IsSameSharedNetworkDevice(const torch::Device& lhs, const torch::Device& rhs)
{
    if (lhs.type() != rhs.type()) {
        return false;
    }
    if (lhs.type() != torch::kCUDA) {
        return true;
    }
    const auto lhs_index = lhs.has_index() && lhs.index() >= 0 ? lhs.index() : 0;
    const auto rhs_index = rhs.has_index() && rhs.index() >= 0 ? rhs.index() : 0;
    return lhs_index == rhs_index;
}

} // namespace


// ======================================================
// DefaultDQNAgent
// ======================================================

DefaultDQNAgent::DefaultDQNAgent(
    const DefaultDQNAgentConfig& config
    , const anet::nn::NetworkConfig& net_config
    , const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, const torch::Device device
    , std::optional<seed_t> seed)
    : AgentBase(device, batch_env_spec, env_spec, seed)
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
    this->vars_ = std::make_unique<RuntimeVars>();

    const bool is_distributional = config_.quantile_mode != "none";

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

    // 分布表現とDuelingの直積でHeadを選択する。
    if (config_.quantile_mode == "iqn") {
        if (config_.use_dueling_net) {
            head_factory = std::make_shared<IQNDuelingHeadFactory>(n_actions_, head_init_config);
            LOG::info() << "Network Head: IQN Dueling";
        } else {
            head_factory = std::make_shared<IQNHeadFactory>(n_actions_, head_init_config);
            LOG::info() << "Network Head: IQN Plain";
        }
    } else if (config_.quantile_mode == "qr") {
        if (config_.use_dueling_net) {
            head_factory = std::make_shared<QuantileDuelingHeadFactory>(
                n_actions_, config_.qr.num_quantiles, head_init_config);
            LOG::info() << "Network Head: Quantile Dueling (N=" << config_.qr.num_quantiles << ")";
        } else {
            head_factory = std::make_shared<QuantileHeadFactory>(
                n_actions_, config_.qr.num_quantiles, head_init_config);
            LOG::info() << "Network Head: Quantile Plain (N=" << config_.qr.num_quantiles << ")";
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

    // オリジナルのStateSpecをコピー
    auto network_obs_spec = env_spec.state_spec.obs_spec;

    // Stackerが有効な場合、NN入力specへ実データと同じstack軸を追加
    if (config_.stucker.use_stacker && config_.stucker.stack_count > 1) {
        for (auto& kv : network_obs_spec) {

            // このKeyがStack対象として設定されているかチェック
            bool is_stacked_target = true;
            if (!config_.stucker.stack_keys.empty()) {
                auto it = std::find(config_.stucker.stack_keys.begin(), config_.stucker.stack_keys.end(), kv.first);
                is_stacked_target = (it != config_.stucker.stack_keys.end());
            }

            // Stack対象のKeyのみ、batch直後にstack軸が来るshapeへ変換
            if (is_stacked_target) {
                kv.second.shape.insert(kv.second.shape.begin(), config_.stucker.stack_count);
            }
        }
    }

    // tausはObservationではなくAgentが注入するNN入力なので、Stacker調整後にspecを追加する。
    if (config_.quantile_mode == "iqn") {
        anet::TensorSpec tau_spec;
        tau_spec.type = anet::SpaceType::Vector;
        tau_spec.shape = { config_.train_policy.tau_rule.num_taus };
        tau_spec.dtype = torch::kFloat32;
        tau_spec.num_classes = 0;
        network_obs_spec[anet::nn::kKey_Taus] = tau_spec;
    }

    // NetworkModel生成
    this->model_ = std::make_unique<NetworkModel>(
        config_.model, device_,
        net_config, network_obs_spec, n_actions_, head_factory,
        is_distributional
    );

    // Network グラフ可視化
    {
        const auto& net = *model_->GetOnlineNetwork();
        auto structure_view = net.MakeGraphViz(anet::nn::NetworkGraphVizConfig{});
        anet::MetricsLogger::Instance()->Log("net.structure", *structure_view);
        auto detail_view = net.MakeGraphViz(config_.nn_viz);
        anet::MetricsLogger::Instance()->Log("net.detail", *detail_view);
    }

    // Target Policyの妥当性チェック
    if (config_.target_policy.policy_type == "EpsilonGreedy" &&
        (config_.target_policy.eps_start > 0.0f || config_.target_policy.eps_end > 0.0f)) {
        // TargetActionPolicy（学習用）はUQE/ThompsonSamplingもしくはGreedyである必要がある(ランダム要素はNG)
        ANET_SYSTEM_ERROR("target_policy cannot be EpsilonGreedy with eps > 0. It must be deterministic or optimistic.");
    }

    // ActionPolicy生成
    this->train_policy_ = CreateActionPolicy(config_.train_policy, config_.train_policy.use_spatial_exploration, num_envs_, device_);
    this->eval_policy_ = CreateActionPolicy(config_.eval_policy, false, num_envs_, device_);
    this->target_policy_ = CreateActionPolicy(config_.target_policy, false, num_envs_, device_);

    // Learner生成
    if (config_.quantile_mode == "iqn") {
        this->learner_ = std::make_unique<IQNLearner>(
            config_.learner, *model_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed, target_policy_, config_.stucker, learner_seed);
        LOG::info() << "Initialized IQNLearner (current_taus=" << config_.learner.iqn.current_taus.num_taus
            << ", target_taus=" << config_.learner.iqn.target_taus.num_taus << ")";
    } else if (config_.quantile_mode == "qr") {
        this->learner_ = std::make_unique<QRLearner>(
            config_.learner, *model_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed, target_policy_, config_.stucker, learner_seed);
        LOG::info() << "Initialized QRLearner (Quantiles=" << config_.qr.num_quantiles << ")";
    } else {
        this->learner_ = std::make_unique<TDLearner>(
            config_.learner, *model_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed, target_policy_, config_.stucker, learner_seed);
        LOG::info() << "Initialized TDLearner";
    }

    // load
    if (!config_.auto_load_file.empty()) {
        LOG::info() << "Auto-loading network from file: " << config_.auto_load_file;
        LoadNetwork(config_.auto_load_file);
	}
}

std::shared_ptr<ActionPolicy> DefaultDQNAgent::CreateActionPolicy(
    const ActionPolicyConfig& policy_config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device)
{
    if (policy_config.policy_type == "EpsilonGreedy" || policy_config.policy_type == "0") {
        // ε-Greedy
        return std::make_shared<EpsilonGreedyActionPolicy>(policy_config, enable_spatial_exploration, num_envs, device);
    } else if (policy_config.policy_type == "UQE" || policy_config.policy_type == "1") {
        // UQE
        if (config_.quantile_mode == "none") {
            ANET_SYSTEM_ERROR("Invalid action policy: policy_type='" << policy_config.policy_type
                << "' requires quantile_mode=qr or iqn, actual='" << config_.quantile_mode << "'");
        }
        return std::make_shared<UQEActionPolicy>(policy_config, enable_spatial_exploration, num_envs, device);
    } else if (policy_config.policy_type == "ThompsonSampling" || policy_config.policy_type == "2") {
        //ThompsonSampling
        if (config_.quantile_mode == "none") {
            ANET_SYSTEM_ERROR("Invalid action policy: policy_type='" << policy_config.policy_type
                << "' requires quantile_mode=qr or iqn, actual='" << config_.quantile_mode << "'");
        }
        return std::make_shared<ThompsonSamplingActionPolicy>(policy_config, enable_spatial_exploration, num_envs, device);
    } else if (policy_config.policy_type == "Greedy" || policy_config.policy_type == "3") {
        // Greedyは、EpsilonGreedyのノイズ0としてインスタンス化
        ActionPolicyConfig greedy_cfg = policy_config;
        greedy_cfg.eps_start = 0.0f;
        greedy_cfg.eps_end = 0.0f;
		greedy_cfg.eps_decay_steps = 0;
        greedy_cfg.use_spatial_exploration = false;
        return std::make_shared<EpsilonGreedyActionPolicy>(greedy_cfg, false, num_envs, device);
    }

    // 不明なtype
    ANET_SYSTEM_ERROR("Unknown action policy type: " << policy_config.policy_type);
    return nullptr;
}

int64_t DefaultDQNAgent::Save(anet::OutputArchive& archive) const
{
    ANET_PROFILE_FUNC();
	std::shared_lock<std::shared_mutex> lock(*mutex_);

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
	auto model_size = model_->Save(archive);
	total_size += model_size;

    // Learner(Adam)
    auto adam_size = learner_->Save(archive);
	total_size += adam_size;

    // ログ
    LOG::info() << "DefaultDQNAgent Serialized. total_size=" << anet::FormatWithCommas(total_size)
        << " config_size=" << anet::FormatWithCommas(config_size)
        << " adam_size=" << anet::FormatWithCommas(adam_size)
        << " model_size=" << anet::FormatWithCommas(model_size);

    return total_size;
}

void DefaultDQNAgent::LoadNetwork(const std::string& filename)
{
	std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        LOG::info() << "cwd=" << std::filesystem::current_path();
        ANET_SYSTEM_ERROR("Failed to open file for loading: " << filename);
	}
	anet::InputArchive in(ifs);

    // Header
    anet::ArchiveHeader header;
	in.Read(header);
	LOG::verbose() << "LoadNetwork: Archive Header: " << header.kMagicWord << " " << header.kFormatVersion << " "  << header.info;

    // Config
	std::string config_str;
	auto config_len = in.Read(config_str);
    LOG::info() << "LoadNetwork: config: len=" << config_len;
    LOG::verbose() << "LoadNetwork: config: config_str=\n" << config_str;

	// Network
	auto model_size = model_->Load(in);
    LOG::info() << "LoadNetwork: model_size=" << model_size;

	// Learner
	auto learner_size = learner_->Load(in);
    LOG::info() << "LoadNetwork: learner_size=" << learner_size;
}

std::optional<anet::TensorDictFunction> DefaultDQNAgent::GetTensorDictFunction(const std::string& key)
{
    // NetworkModel に委譲してベース関数を取得
    auto fn = model_->GetTensorDictFunction(key, device_);
    if (fn == std::nullopt) return std::nullopt;

    auto self = shared_from_this();
    auto network_fn = *fn;
    bool use_stacker = config_.stucker.use_stacker;
    int stack_count = config_.stucker.stack_count;

    // ロックと前処理（正規化等）をラップした関数を作成
    anet::TensorDictFunction norm_fn = [self, network_fn, use_stacker, stack_count](const anet::TensorDict& obs) {

        // 排他制御（他スレッドでのパラメータ更新と競合しないように）
        std::shared_lock<std::shared_mutex> lock(*(self->mutex_));
        torch::NoGradGuard grad_guard;

        anet::TensorDict proc_obs;
        anet::TensorDict device_obs = obs.To(self->device_);

        for (const auto& kv : device_obs) {
            auto k = kv.first;
            auto t = kv.second;

            bool is_stacked_target = true;
            if (!self->config_.stucker.stack_keys.empty()) {
                auto it = std::find(self->config_.stucker.stack_keys.begin(), self->config_.stucker.stack_keys.end(), k);
                is_stacked_target = (it != self->config_.stucker.stack_keys.end());
            }

            // 状態スイープは実フレーム履歴を持たない合成1フレーム入力なので、
            // stacker 有効時は同じフレームを stack 全域へ複製して通常の network 入力形状に合わせる。
            if (use_stacker && is_stacked_target) {
                // (B, C, H, W) -> (B, 1, C, H, W)
                t = t.unsqueeze(1);
                auto sizes = t.sizes().vec();
                sizes[1] = stack_count;
                // .expand はメモリを余分に確保せずポインタだけ増やすので高速
                t = t.expand(sizes);
            }
            proc_obs.Set(k, t);
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

std::shared_ptr<anet::rl::ActionContext> DefaultDQNAgent::CreateActionContext(
    const BatchEnvSpec& batch_env_spec, RunMode run_mode, std::optional<torch::Device> device) const
{
    // Seed生成
    auto rnd = GetRandomGenerator(run_mode);
    auto ctx_seed = rnd->RandUint64();

    // ActionContext向けのDeviceを取得
    auto target_device = device.value_or(this->device_);

    if (config_.stucker.use_stacker) {
        // Stackerを作成して包んで返す
        std::optional<std::vector<std::string>> stack_keys;
        if (!config_.stucker.stack_keys.empty()) stack_keys = config_.stucker.stack_keys;
        auto stacker = std::make_shared<DictFrameStacker>(
			config_.stucker.stack_count, batch_env_spec.num_envs, target_device, stack_keys);
        return std::make_shared<StackerActionContext>(run_mode, stacker, ctx_seed);
    }

    // Stacker無効ならデフォルト
    return std::make_shared<DefaultActionContext>(run_mode, ctx_seed, target_device);
}

static bool IsForTarget(anet::rl::RunMode run_mode)
{
    // Train
    if (!anet::rl::IsEval(run_mode)) {
        return false;
    }

    // Eval
    return run_mode == anet::rl::RunMode::Eval1;
}

std::shared_ptr<anet::rl::Actor> DefaultDQNAgent::CreateActor(
    const anet::rl::BatchEnvSpec& batch_env_spec,
    const anet::rl::EnvSpec& env_spec,
    anet::rl::RunMode run_mode,
    std::optional<bool> clone_model_override,
    std::optional<torch::Device> device) const
{
    env_spec_.CheckSameStateActionSpec(env_spec);
    const bool is_train_actor = !anet::rl::IsEval(run_mode);
    const bool clone_model = clone_model_override.value_or(
        is_train_actor ? config_.train_actor.clone_model : false);
    const auto actor_device = device.value_or(device_);
    ANET_CHECK_MSG(
        clone_model || IsSameSharedNetworkDevice(actor_device, device_),
        "DefaultDQNAgent shared Actor device mismatch: actor_device=" << actor_device.str()
        << " agent_device=" << device_.str()
        << ". Use clone_model_override=true or the Agent device.");

    // Contextを生成
    auto ctx = this->CreateActionContext(batch_env_spec, run_mode, actor_device);

    // モードに応じて適切な Policy と Network を選択
    std::shared_ptr<ActionPolicy> policy;
    std::shared_ptr<anet::nn::Network> src_network;

    // 元ネタのPolicyとNetoworkを決定
    if (anet::rl::IsEval(run_mode)) {
        policy = eval_policy_;
        src_network = IsForTarget(run_mode) ? model_->GetTargetNetwork() : model_->GetOnlineNetwork();
    } else {
        policy = train_policy_;
        src_network = model_->GetOnlineNetwork();
    }

    // 必要に応じてCloneしてActor向けネットワークとする
    auto network = src_network;
    if (clone_model) {
        // Clone構築時のparameter・buffer copyもLearner更新と同じmutexで保護する。
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        network = src_network->Clone(actor_device);
        network->eval();
    }

    // Actor を生成
    const bool emit_actor_q_hint = !anet::rl::IsEval(run_mode)
        && config_.learner.use_per
        && ParseReplayInitialPriorityMode(config_.learner) == ReplayInitialPriorityMode::ACTOR_APPROX;
    const auto snapshot_sync_interval = is_train_actor && clone_model
        ? std::optional<anet::ProfiledValueConfig<step_t>>(config_.train_actor.sync_interval)
        : std::nullopt;
    auto actor = std::make_shared<Actor>(
        policy, obs_norm_, ctx, this->mutex_, network, src_network, emit_actor_q_hint, snapshot_sync_interval, true);

    // 生成したActorを返す
    return actor;
}

std::shared_ptr<anet::rl::Learner> DefaultDQNAgent::CreateLearner()
{
    return this->shared_from_this();
}

anet::rl::BatchUpdateResultList
DefaultDQNAgent::UpdateFromBatch(const StepCounts& counts, const anet::rl::BatchExperience& batch_exp)
{
    ANET_PROFILE_FUNC();

    BatchUpdateResultList result_list;

    {
        // 排他ロック（Writeロック)
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
        auto result = this->learner_->UpdateFromBatch(counts, exp);
        result_list = std::move(result);

        // Update後処理
        train_policy_->OnLearn(counts);
        eval_policy_->OnLearn(counts);
        target_policy_->OnLearn(counts);
    }

    // BatchUpdateResultを返す
    return result_list;
}


// ======================================================
// DefaultDQNAgentFactory
// ======================================================

void DefaultDQNAgentFactory::ValidateQuantileNetworkContract(
    const DefaultDQNAgentConfig& config,
    const anet::nn::NetworkConfig& net_config,
    const std::string& net_config_prefix)
{
    // 各 branch の解析済み bind factor から、taus を直接要求する設定キーを列挙する。
    std::vector<std::string> taus_bind_keys;
    for (const auto& [branch_name, branch_config] : net_config.branches) {
        bool binds_taus = false;
        for (const auto& term : branch_config.bind_terms) {
            if (std::find(term.begin(), term.end(), anet::nn::kKey_Taus) != term.end()) {
                binds_taus = true;
                break;
            }
        }
        if (binds_taus) {
            taus_bind_keys.push_back(
                net_config_prefix + ".branch.[" + branch_name + "].bind");
        }
    }

    // quantile_mode と自分の net サブツリーとの局所契約を、モデル構築前に検証する。
    if (config.quantile_mode == "iqn" && taus_bind_keys.empty()) {
        ANET_SYSTEM_ERROR(
            "Invalid DefaultDQNAgent.quantile_mode='iqn': expected at least one 'taus' factor in "
            << net_config_prefix << ".branch.[*].bind.");
    }
    if (config.quantile_mode != "iqn" && !taus_bind_keys.empty()) {
        std::string joined_keys;
        for (const auto& key : taus_bind_keys) {
            if (!joined_keys.empty()) {
                joined_keys += ", ";
            }
            joined_keys += key;
        }
        ANET_SYSTEM_ERROR(
            "Invalid DefaultDQNAgent.quantile_mode='" << config.quantile_mode
            << "': unexpected 'taus' factor in " << joined_keys << ".");
    }
}

std::shared_ptr<anet::rl::Agent> DefaultDQNAgentFactory::CreateAgent(
    const EnvSpec& env_spec, const BatchEnvSpec& batch_env_spec,
    const torch::Device& device, const anet::ConfigData& config_data,
    std::shared_ptr<anet::rl::Notifier> notifier, std::optional<anet::seed_t> seed) const
{
    DefaultDQNAgentConfig config(config_data);
    const auto net_config_prefix = GetTargetAgentClassId() + ".net";
    anet::nn::NetworkConfig net_config(config_data, net_config_prefix);
    ValidateQuantileNetworkContract(config, net_config, net_config_prefix);
    auto agent = std::make_shared<DefaultDQNAgent>(config, net_config, batch_env_spec, env_spec, device, seed);
    return agent;

    /// @todo 引数の順番を統一
}

//ANET_REGISTER_AGENT_FACTORY(DefaultDQNAgentFactory);
