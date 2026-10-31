// default_dqn_agent.cpp

#include "anet/default_dqn_agent.hpp"
#include <memory>
#include <sstream>
#include <torch/torch.h>
#include <cmath>
#include "anet/profile.hpp"
#include "anet/str_util.hpp"
#include "anet/nn_util.hpp"
#include "anet/tensor_util.hpp"
#include "anet/tensor_check.hpp"
#include "anet/log.hpp"
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/stacker.hpp"
#include "anet/serialize.hpp"
#include "nn_heads.hpp"
#include "dqn_based_heads.hpp"
#include "dqn_based_agent.hpp"


using namespace anet::rl::dqn;
namespace LOG = anet::log;

namespace {
// Agent全体の保存形式。第1版の識別子は既存primary checkpointのため保持する。
constexpr char kCheckpointFormatV1[] = "DefaultDQNAgent";
constexpr char kCheckpointFormatV2[] = "DefaultDQNAgent-2";
constexpr const char* kSkipStructureTag = "skip.net.structure";
constexpr const char* kSkipDetailTag = "skip.net.detail";
constexpr const char* kSkipNetworkSeed = "skip_network";
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
    // 直接DTOを渡す構築経路でも、有効化時は未指定のJを資源構築前に拒否する。
    if (config_.skip.enabled && config_.skip.max_length < 1) {
        ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.skip.max_length=" << config_.skip.max_length
            << "; expected explicit value >= 1 when skip.enabled=true.");
    }
    ANET_LOG_DEBUG("seed=" << GetSeed());

    // ログ：パラメータ記録
    LOG::info() << "DefaultDQNAgent config=" << config_.ToString();
    anet::MetricsLogger::Instance()->Log(config_);
    anet::MetricsLogger::Instance()->Log("net.body", net_config.ToJson());

	// 割引率gammaと実効ホライズンのログ
    {
        // 実効ホライズン (Effective Horizon) = 1 / (1 - gamma)
        // gamma=0.9 -> 10 steps, 0.99 -> 100 steps, 0.995 -> 200 steps
        float g = anet::ProfiledValue<float>(config_.learner.gamma).Evaluate(0);
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
    auto network_seed = seed_maker.MakeNamedSeed("network");
    this->plasticity_probe_random_ = std::make_shared<anet::RandomGenerator>(
        seed_maker.MakeNamedSeed("plasticity_probe"));
    this->policy_churn_probe_random_ = std::make_shared<anet::RandomGenerator>(
        seed_maker.MakeNamedSeed("policy_churn_probe"));

    // RuntimeVars生成
    this->vars_ = std::make_unique<RuntimeVars>();

    const bool is_distributional = config_.quantile_mode != ActionPolicyConfig::kQuantileModeNone;

    // RewardScaler生成
    anet::rl::RewardScalerFactory reward_scaler_factory(config_.reward_scaler);
    this->reward_scaler_ = reward_scaler_factory.CreateRewardScaler(anet::ProfiledValue<float>(config_.learner.gamma).Evaluate(0));

    // ObservationNormalizer生成
    anet::rl::ObservationNormalizerFactory obs_norm_factory(config_.obs_norm);
    this->obs_norm_ = obs_norm_factory.CreateObservationNormalizer(env_spec.state_spec);


    // ------------------------------------------------------------
    // HeadFactory の準備
    // ------------------------------------------------------------

    // Head用の初期化設定を作成 (AgentがConfigDataから読み取る)
    const anet::nn::WeightInitConfig& head_init_config = config_.head.init;

    std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory;

    // 分布表現とDuelingの直積でHeadを選択する。
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        if (config_.head.use_dueling) {
            head_factory = std::make_shared<IQNDuelingHeadFactory>(n_actions_, head_init_config, config_.head.projection);
            LOG::info() << "Network Head: IQN Dueling";
        } else {
            head_factory = std::make_shared<IQNHeadFactory>(n_actions_, head_init_config, config_.head.projection);
            LOG::info() << "Network Head: IQN Plain";
        }
    } else if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeQr) {
        if (config_.head.use_dueling) {
            head_factory = std::make_shared<QuantileDuelingHeadFactory>(
                n_actions_, config_.qr.num_quantiles, head_init_config, config_.head.projection);
            LOG::info() << "Network Head: Quantile Dueling (N=" << config_.qr.num_quantiles << ")";
        } else {
            head_factory = std::make_shared<QuantileHeadFactory>(
                n_actions_, config_.qr.num_quantiles, head_init_config, config_.head.projection);
            LOG::info() << "Network Head: Quantile Plain (N=" << config_.qr.num_quantiles << ")";
        }
    } else {
        if (config_.head.use_dueling) {
            head_factory = std::make_shared<DuelingHeadFactory>(
                n_actions_, head_init_config, config_.head.projection);
            LOG::info() << "Network Head: Dueling";
        } else {
            head_factory = std::make_shared<LinearHeadFactory>(
                n_actions_, head_init_config, config_.head.projection);
            LOG::info() << "Network Head: Plain Linear";
        }
    }

    // DTO にある設定データから、用途が参照する不変 Resource を構築する。
    for (const auto& [key, data] : config_.nn_forward) {
        forward_configs_.emplace(key, std::make_shared<const anet::nn::NetworkForwardConfig>(
            config_.MakeForwardConfigKey(key), data));
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
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        anet::TensorSpec tau_spec;
        tau_spec.type = anet::SpaceType::Vector;
        tau_spec.shape = { config_.learner.iqn.current_taus.num_taus };
        tau_spec.dtype = torch::kFloat32;
        tau_spec.num_classes = 0;
        network_obs_spec[anet::nn::kKey_Taus] = tau_spec;
    }

    // NetworkModel生成
    std::optional<SkipNetworkDefinition> skip_definition;
    if (config_.skip.enabled) {
        skip_definition.emplace(SkipNetworkDefinition{
            .config = config_.skip,
            .network_config = anet::nn::NetworkConfig(config_.skip_net, "DefaultDQNAgent.skip.net"),
            .head_factory = std::make_shared<SkipHeadFactory>(n_actions_, config_.skip.max_length,
                config_.skip.network_mode, config_.quantile_mode, config_.qr.num_quantiles,
                config_.skip.head.init, config_.skip.head.projection),
            .seed = seed_maker.MakeNamedSeed(kSkipNetworkSeed)});
    }
    this->model_ = std::make_unique<NetworkModel>(
        config_.model, device_,
        net_config, network_obs_spec, n_actions_, head_factory,
        is_distributional, network_seed, skip_definition ? &*skip_definition : nullptr
    );

    // Network グラフ可視化
    {
        const auto& net = *model_->GetOnlineNetwork();
        auto structure_view = net.MakeGraphViz(anet::nn::NetworkGraphVizConfig{});
        anet::MetricsLogger::Instance()->Log("net.structure", *structure_view);
        auto detail_view = net.MakeGraphViz(config_.nn_viz);
        anet::MetricsLogger::Instance()->Log("net.detail", *detail_view);
        if (const auto skip = model_->GetSkipOnlineNetwork()) {
            anet::MetricsLogger::Instance()->Log(kSkipStructureTag, *skip->MakeGraphViz(anet::nn::NetworkGraphVizConfig{}));
            anet::MetricsLogger::Instance()->Log(kSkipDetailTag, *skip->MakeGraphViz(config_.nn_viz));
        }
    }

    // target用のランダム探索は全有効端点で禁止する。
    if (config_.target_policy.policy_type == "EpsilonGreedy") {
        auto epsilon = config_.target_policy.epsilon;
        epsilon.max_value = 0.0f;
        anet::detail::ValidateProfiledValueConfig(epsilon, "target_policy.epsilon");
    }

    // ActionPolicy生成
    this->target_policy_ = CreateActionPolicy(config_.target_policy, false, num_envs_, device_);

    // 解決済みのbootstrap方策をログへ示し、平均と経験分位のsoft targetを区別する。
    const auto risk_score = target_policy_->GetRiskScoreSpec();
    std::ostringstream munchausen_info;
    munchausen_info << " munchausen=" << (config_.learner.munchausen.enabled ? "on" : "off")
        << " log_policy_mode=" << config_.learner.munchausen.log_policy_mode
        << " score_source=" << (risk_score ? "risk_biased" : "mean");
    if (risk_score) munchausen_info << " risk_tau=" << risk_score->tau << " use_tail_mean=" << risk_score->use_tail_mean;

    // Learner生成
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        this->learner_ = std::make_unique<IQNLearner>(
            config_.learner, *model_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed,
            target_policy_,
            LearnerForwardResources{
                .current = *forward_configs_.at(config_.learner.current.nn_forward_key),
                .target = *forward_configs_.at(config_.learner.target.nn_forward_key)}, config_.stucker, learner_seed,
            plasticity_probe_random_.get(), policy_churn_probe_random_.get());
        LOG::info() << "Initialized IQNLearner (current_taus=" << config_.learner.iqn.current_taus.num_taus
            << ", target_taus=" << config_.learner.iqn.target_taus.num_taus << ")" << munchausen_info.str();
    } else if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeQr) {
        this->learner_ = std::make_unique<QRLearner>(
            config_.learner, *model_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed,
            target_policy_,
            LearnerForwardResources{
                .current = *forward_configs_.at(config_.learner.current.nn_forward_key),
                .target = *forward_configs_.at(config_.learner.target.nn_forward_key)}, config_.stucker, learner_seed,
            plasticity_probe_random_.get(), policy_churn_probe_random_.get());
        LOG::info() << "Initialized QRLearner (Quantiles=" << config_.qr.num_quantiles << ")" << munchausen_info.str();
    } else {
        this->learner_ = std::make_unique<TDLearner>(
            config_.learner, *model_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed,
            target_policy_,
            LearnerForwardResources{
                .current = *forward_configs_.at(config_.learner.current.nn_forward_key),
                .target = *forward_configs_.at(config_.learner.target.nn_forward_key)}, config_.stucker, learner_seed,
            plasticity_probe_random_.get(), policy_churn_probe_random_.get());
        LOG::info() << "Initialized TDLearner" << munchausen_info.str();
    }

    // load
    if (!config_.auto_load_file.empty()) {
        LOG::info() << "Auto-loading network from file: " << config_.auto_load_file;
        LoadNetwork(config_.auto_load_file);
	}
}

std::shared_ptr<ActionPolicy> DefaultDQNAgent::CreateActionPolicy(
    const ActionPolicyConfig& policy_config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device) const
{
    if (policy_config.policy_type == "EpsilonGreedy" || policy_config.policy_type == "0") {
        // ε-Greedy
        return std::make_shared<EpsilonGreedyActionPolicy>(policy_config, enable_spatial_exploration, num_envs, device);
    } else if (policy_config.policy_type == "UQE" || policy_config.policy_type == "1") {
        // UQE
        if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeNone) {
            ANET_SYSTEM_ERROR("Invalid action policy: policy_type='" << policy_config.policy_type
                << "' requires quantile_mode=" << ActionPolicyConfig::kQuantileModeQr
                << " or " << ActionPolicyConfig::kQuantileModeIqn << ", actual='" << config_.quantile_mode << "'");
        }
        return std::make_shared<UQEActionPolicy>(policy_config, enable_spatial_exploration, num_envs, device);
    } else if (policy_config.IsThompsonSampling()) {
        //ThompsonSampling
        if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeNone) {
            ANET_SYSTEM_ERROR("Invalid action policy: policy_type='" << policy_config.policy_type
                << "' requires quantile_mode=" << ActionPolicyConfig::kQuantileModeQr
                << " or " << ActionPolicyConfig::kQuantileModeIqn << ", actual='" << config_.quantile_mode << "'");
        }
        return std::make_shared<ThompsonSamplingActionPolicy>(policy_config, enable_spatial_exploration, num_envs, device);
    } else if (policy_config.policy_type == kActionPolicyTypeStr_Greedy || policy_config.policy_type == "3") {
        // Greedyは、EpsilonGreedyのノイズ0としてインスタンス化
        ActionPolicyConfig greedy_cfg = policy_config;
        greedy_cfg.epsilon = {.value = 0.0f, .min_value = 0.0f, .max_value = 1.0f};
        greedy_cfg.spatial.enabled = false;
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
    header.info = config_.skip.enabled ? kCheckpointFormatV2 : kCheckpointFormatV1;
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

    const bool has_skip = header.info == kCheckpointFormatV2;
    if (header.info != kCheckpointFormatV1 && !has_skip) {
        ANET_SYSTEM_ERROR("Invalid DefaultDQN checkpoint header='" << header.info
            << "'; expected " << kCheckpointFormatV1 << " or " << kCheckpointFormatV2 << ".");
    }
    if (has_skip && !config_.skip.enabled) ANET_SYSTEM_ERROR("Checkpoint contains skip networks but DefaultDQNAgent.skip.enabled=false.");

	// Network
	auto model_size = model_->LoadCheckpoint(in, has_skip);
    LOG::info() << "LoadNetwork: model_size=" << model_size;

	// Learner
	auto learner_size = learner_->LoadCheckpoint(in, has_skip);
    LOG::info() << "LoadNetwork: learner_size=" << learner_size;
    if (config_.skip.enabled && !has_skip) {
        LOG::warn() << "Checkpoint has no skip networks; restored primary networks and optimizer state, initialized skip networks and optimizer groups. file=" << filename;
    }
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
    if (anet::StartsWith(key, "target_policy.")) return target_policy_->GetScalar(anet::RemovePrefix(key, "target_policy."), index);

    for (const auto* schedule_key : kLearnerScheduleMetricKeys) {
        if (key == schedule_key) {
            std::shared_lock<std::shared_mutex> lock(*mutex_);
            return learner_->GetScalar(key);
        }
    }
    if (key == "per_beta") {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return vars_->per_beta;
    }
    if (key.starts_with("plasticity_probe_") || key.starts_with("noisy_") || key.find(ReplayBuffer::kKeyPrefix) == 0) {
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

void DefaultDQNAgent::ConfigureScalarMetricSubscriptions(
    const std::vector<ScalarMetricSubscription>& subscriptions)
{
    std::unique_lock<std::shared_mutex> lock(*mutex_);
    // 購読内容からResourceの必要性を決め、母数・実PERのみの購読では専用RNGを作らない。
    std::array<bool, dqn::kReplayFitMetricKeys.size()> request{};
    bool replay_fit_enabled = false;
    if (config_.learner.enabled) {
        for (const auto& subscription : subscriptions) {
            if (subscription.scope != RunnerScope::TRAIN || subscription.event != EventType::LEARN
                || subscription.target != EventField::UPDATE_RESULT) continue;
            const auto metric = dqn::ParseReplayFitMetric(subscription.source_key);
            if (!metric) continue;
            replay_fit_enabled = true;
            request[*metric] = true;
        }
    }
    if (replay_fit_enabled && config_.target_policy.IsThompsonSampling()) {
        ANET_SYSTEM_ERROR("Replay fit requires a deterministic target policy; target_policy.policy_type="
            << config_.target_policy.policy_type << " expected " << kActionPolicyTypeStr_Greedy << ", "
            << kActionPolicyTypeStr_EpsilonGreedy << ", or " << kActionPolicyTypeStr_UQE << ".");
    }
    const bool needs_sampling = dqn::ResolveReplayFitRequirements(request, config_.learner.use_per).NeedsSampling();
    if (needs_sampling && !replay_fit_probe_random_) {
        anet::SeedMaker seed_maker(GetSeed());
        replay_fit_probe_random_ = std::make_shared<anet::RandomGenerator>(seed_maker.MakeNamedSeed("replay_fit_probe"));
    }
    // Noisy 診断の用途を固定の sample/call/batch として選び、他 Module の設定は opaque に保持する。
    const bool needs_noisy_probe = config_.learner.enabled && std::ranges::any_of(subscriptions, [](const auto& entry) {
        const auto metric = dqn::ParseNoisyMetric(entry.source_key);
        return entry.scope == RunnerScope::TRAIN && entry.event == EventType::LEARN && metric && *metric >= 8;
    }) && !model_->GetOnlineNetwork()->GetNoisyParameterEntries().empty();
    if (needs_noisy_probe && !noisy_probe_random_) {
        anet::SeedMaker seed_maker(GetSeed());
        noisy_probe_random_ = std::make_shared<anet::RandomGenerator>(seed_maker.MakeNamedSeed("noisy_probe.sample"));
        anet::ConfigData data;
        const std::string prefix = MakeNetworkForwardConfigKey("noisy_probe");
        for (const auto& [key, value] : forward_configs_.at(config_.learner.current.nn_forward_key)->GetConfigData().Map()) {
            data.Set(key, value);
        }
        data.Set("noisy.mode", "sample");
        data.Set("noisy.hold", "call");
        data.Set("noisy.share", "batch");
        noisy_probe_forward_config_ = std::make_shared<const anet::nn::NetworkForwardConfig>(prefix, data);
    }
    // attentionも購読された場合だけ専用抽選器を作り、他診断の乱数列と分離する。
    const bool needs_attention = config_.learner.enabled && std::ranges::any_of(subscriptions, [](const auto& entry) {
        return entry.scope == RunnerScope::TRAIN && entry.event == EventType::LEARN
            && entry.target == EventField::UPDATE_RESULT && dqn::ParseAttentionEntropyMetric(entry.source_key).has_value();
    });
    if (needs_attention && !attention_entropy_probe_random_) {
        anet::SeedMaker seed_maker(GetSeed());
        attention_entropy_probe_random_ = std::make_shared<anet::RandomGenerator>(seed_maker.MakeNamedSeed("attention_entropy_probe"));
    }
    learner_->ConfigureScalarMetricSubscriptions(subscriptions, needs_sampling ? replay_fit_probe_random_.get() : nullptr,
        needs_noisy_probe ? noisy_probe_random_.get() : nullptr,
        needs_noisy_probe ? noisy_probe_forward_config_.get() : nullptr,
        needs_attention ? attention_entropy_probe_random_.get() : nullptr);
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

std::unique_ptr<anet::rl::FrameStacker> DefaultDQNAgent::CreateFrameStacker(const ActorRequest& request) const
{
    // laneごとの履歴をActorが専有し、推論device上でstackする。
    if (config_.stucker.use_stacker) {
        std::optional<std::vector<std::string>> keys;
        if (!config_.stucker.stack_keys.empty()) keys = config_.stucker.stack_keys;
        return std::make_unique<DictFrameStacker>(config_.stucker.stack_count,
            request.batch_env_spec.num_envs, request.device, keys);
    }
    return nullptr;
}

std::shared_ptr<anet::rl::Actor> DefaultDQNAgent::CreateActor(const ActorRequest& request) const
{
    ANET_PROFILE_FUNC();
    env_spec_.CheckSameStateActionSpec(request.env_spec);
    const auto& cfg = FindActorConfig(config_.actor, request.actor_key);
    if (!forward_configs_.contains(cfg.nn_forward_key)) {
        ANET_SYSTEM_ERROR("Missing resolved actor.[" << request.actor_key << "].nn_forward_key; expected a declared forward configuration.");
    }
    ValidateActorDevice(cfg.clone_model, request.device);
    auto policy = CreateActionPolicy(cfg.policy, cfg.policy.spatial.enabled,
        request.batch_env_spec.num_envs, request.device);
    auto source = cfg.network == "target" ? model_->GetTargetNetwork() : model_->GetOnlineNetwork();
    auto network = source;
    std::optional<SkipActorResources> skip;
    if (config_.skip.enabled) {
        auto skip_source = cfg.network == "target" ? model_->GetSkipTargetNetwork() : model_->GetSkipOnlineNetwork();
        // greedy評価は両方をgreedyに揃え、既定では同じPolicyの現在値を使う。
        const bool shared_policy = !cfg.has_skip_policy || cfg.policy.policy_type == kActionPolicyTypeStr_Greedy;
        // モードは設定の選択だけを分け、skip の専用 seed と時計は Actor に任せる。
        const auto& skip_forward_key = cfg.skip.nn_forward_mode == DQNActorConfig::kForwardDedicated
            ? *cfg.skip.nn_forward_key : cfg.nn_forward_key;
        skip.emplace(SkipActorResources{
            .config = config_.skip,
            .policy = shared_policy ? policy : CreateActionPolicy(cfg.skip_policy, cfg.skip_policy.spatial.enabled,
                request.batch_env_spec.num_envs, request.device),
            .network = skip_source, .source = skip_source,
            .forward_config = forward_configs_.at(skip_forward_key),
            .use_amp = cfg.policy.use_amp, .use_amp_bf16 = cfg.policy.use_amp_bf16,
            .shared_policy = shared_policy, .max_length_mode = cfg.skip.max_length_mode,
            .max_length = cfg.skip.max_length});
    }
    if (cfg.clone_model) {
        // sourceのparameterとbufferを学習更新から保護して複製する。
        std::shared_lock lock(*mutex_);
        network = source->Clone(request.device);
        network->eval();
        if (skip) {
            skip->network = skip->source->Clone(request.device);
            skip->network->eval();
        }
    }
    const bool emit_hint = config_.learner.use_per
        && ParseReplayInitialPriorityMode(config_.learner) == ReplayInitialPriorityMode::ACTOR_APPROX;
    const ActorQHintConfig hint_config{
        .munchausen = config_.learner.munchausen, .use_tbo = config_.learner.use_tbo,
        .tbo_epsilon = config_.learner.tbo_epsilon, .gamma = config_.learner.gamma};
    return std::make_shared<Actor>(policy, obs_norm_, CreateFrameStacker(request), request.device, request.seed,
        mutex_, network, source, *forward_configs_.at(cfg.nn_forward_key), emit_hint, cfg.clone_model ? cfg.sync_interval : std::nullopt, true, hint_config, std::move(skip));
}

std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> DefaultDQNAgent::CreateStepDiagnosticsProvider(int64_t num_lanes) const
{
    // 診断部品がAgentを共有所有し、scalerが公開した変換だけをロックなしで読む。
    auto self = shared_from_this();
    return anet::rl::dqn::CreateStepDiagnosticsProvider(config_.learner, num_lanes,
        [self](const torch::Tensor& reward) {
            return self->reward_scaler_->ScaleWithoutUpdate(reward);
        });
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
        target_policy_->UpdateSchedule(counts);
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
    if (config.quantile_mode == ActionPolicyConfig::kQuantileModeIqn && taus_bind_keys.empty()) {
        ANET_SYSTEM_ERROR(
            "Invalid DefaultDQNAgent.quantile_mode='" << ActionPolicyConfig::kQuantileModeIqn
            << "': expected at least one 'taus' factor in "
            << net_config_prefix << ".branch.[*].bind.");
    }
    if (config.quantile_mode != ActionPolicyConfig::kQuantileModeIqn && !taus_bind_keys.empty()) {
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
