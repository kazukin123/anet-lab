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

    // オリジナルのStateSpecをコピー
    auto network_obs_spec = env_spec.state_spec.obs_spec;

    // Stackerが有効な場合、NNが期待する入力チャンネル（次元）数を調整
    if (config_.stucker.use_stacker && config_.stucker.stack_count > 1) {
        for (auto& kv : network_obs_spec) {

            // このKeyがStack対象として設定されているかチェック
            bool is_stacked_target = true;
            if (!config_.stucker.stack_keys.empty()) {
                auto it = std::find(config_.stucker.stack_keys.begin(), config_.stucker.stack_keys.end(), kv.first);
                is_stacked_target = (it != config_.stucker.stack_keys.end());
            }

            // Stack対象のKeyのみ、最初の次元をstack_count倍する
            if (is_stacked_target && !kv.second.shape.empty()) {
                kv.second.shape[0] *= config_.stucker.stack_count;
            }
        }
    }

    // NetworkModel生成
    this->model_ = std::make_unique<dqn::NetworkModel>(
        config_.model, device_,
        net_config, network_obs_spec, n_actions_, head_factory,
        config_.use_qr ? config_.num_quantiles : 0
    );

    // Target Policyの妥当性チェック
    if (config_.target_policy.policy_type == "EpsilonGreedy" && config_.target_policy.eps_start > 0.0f) {
        // TargetActionPolicy（学習用）はUQE/ThompsonSamplingもしくはGreedyである必要がある(ランダム要素はNG)
        ANET_SYSTEM_ERROR("target_policy cannot be EpsilonGreedy with eps > 0. It must be deterministic or optimistic.");
    }

    // ActionPolicy生成
    this->train_policy_ = CreateActionPolicy(config_.train_policy);
    this->eval_policy_ = CreateActionPolicy(config_.eval_policy);
    this->target_policy_ = CreateActionPolicy(config_.target_policy);

    // Learner生成
    if (is_distributional) {
        this->learner_ = std::make_unique<dqn::QRLearner>(
            config_.learner, *model_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed, target_policy_, config_.stucker, learner_seed);
        LOG::info() << "Initialized QRLearner (Quantiles=" << config_.num_quantiles << ")";
    } else {
        this->learner_ = std::make_unique<dqn::TDLearner>(
            config_.learner, *model_, *vars_, obs_norm_, batch_env_spec, env_spec, device_, replay_seed, target_policy_, config_.stucker, learner_seed);
        LOG::info() << "Initialized TDLearner";
    }

    // load
    if (!config_.auto_load_file.empty()) {
        LOG::info() << "Auto-loading network from file: " << config_.auto_load_file;
        LoadNetwork(config_.auto_load_file);
	}
}

std::shared_ptr<anet::rl::dqn::ActionPolicy> DefaultDQNAgent::CreateActionPolicy(const ActionPolicyConfig& policy_config)
{
    if (policy_config.policy_type == "EpsilonGreedy" || policy_config.policy_type == "0") {
        // ε-Greedy
        return std::make_shared<dqn::EpsilonGreedyActionPolicy>(policy_config);
    } else if (policy_config.policy_type == "UQE" || policy_config.policy_type == "1") {
        // UQE
        ANET_CHECK(config_.use_qr);
        return std::make_shared<dqn::UQEActionPolicy>(policy_config);
    } else if (policy_config.policy_type == "ThompsonSampling" || policy_config.policy_type == "2") {
        //ThompsonSampling
        ANET_CHECK(config_.use_qr);
        return std::make_shared<dqn::ThompsonSamplingActionPolicy>(policy_config);
    } else if (policy_config.policy_type == "Greedy" || policy_config.policy_type == "3") {
        // Greedyは、EpsilonGreedyのノイズ0としてインスタンス化
        ActionPolicyConfig greedy_cfg = policy_config;
        greedy_cfg.eps_start = 0.0f;
        greedy_cfg.eps_end = 0.0f;
        return std::make_shared<dqn::EpsilonGreedyActionPolicy>(greedy_cfg);
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

std::optional<anet::TensorFunction> DefaultDQNAgent::GetTensorFunction(const std::string& key)
{
    auto fn = model_->GetTensorFunction(key, device_);
    if (fn == std::nullopt) return fn;

    return std::nullopt;

  //  auto self = shared_from_this();
  //  auto network_fn = *fn;
  //  bool use_stacker = config_.stucker.use_stacker;
  //  int stack_count = config_.stucker.stack_count;

  //  anet::TensorFunction norm_fn = [self, network_fn, use_stacker, stack_count](const torch::Tensor& obs) {

  //      std::shared_lock<std::shared_mutex> lock(*(self->mutex_));

  //      //ANET_LOG_DEBUG("obs=" << anet::ToDefString(obs));
  //      torch::Tensor proc_obs = obs;

  //      // Stacker有効なのに送られてきたデータが2次元(N, F)だった場合、時間方向に複製して3次元化する
  //      if (use_stacker && proc_obs.dim() == 2) {
  //          // (N, F) -> (N, 1, F) -> (N, Stack, F)
  //          proc_obs = proc_obs.unsqueeze(1).expand({ -1, stack_count, -1 });
  //      }
  //      //ANET_LOG_DEBUG("proc_obs=" << anet::ToDefString(proc_obs));

  //      // 正規化
  //      auto obs_norm = self->obs_norm_->Normalize(proc_obs);
  //      //ANET_LOG_DEBUG("obs_norm=" << anet::ToDefString(obs_norm));

		//// ネットワーク実行 (stack有効の場合は(N, S, F)、無効の場合は(N, F)
  //      auto out = network_fn(obs_norm);
  //      return out;
  //      };

  //  return norm_fn;
}

std::optional<anet::TensorDictFunction> DefaultDQNAgent::GetTensorDictFunction(const std::string& key)
{
    // dqn::Network に委譲してベース関数を取得
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

        for (const auto& kv : obs) {
            auto k = kv.first;
            auto t = kv.second;

            bool is_stacked_target = true;
            if (!self->config_.stucker.stack_keys.empty()) {
                auto it = std::find(self->config_.stucker.stack_keys.begin(), self->config_.stucker.stack_keys.end(), k);
                is_stacked_target = (it != self->config_.stucker.stack_keys.end());
            }

            // Observerから来る生データ(1フレーム)を、Stack次元を追加して複製(Expand)する
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
			config_.stucker.stack_count, batch_env_spec.batch_size, target_device, stack_keys);
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

anet::rl::BatchActionInfo DefaultDQNAgent::MakeAction(const StepCounts& step, const BatchState& state, std::shared_ptr<ActionContext> ctx) const
{
    ProfileRange r1("DefaultDQNAgent::MakeAction");
    //ANET_ASSERT_SHAPE(state.obs, { ANET_SHAPE_ANY, state_dim_ });

    // 共有ロック＆Grad抑止
    std::shared_lock<std::shared_mutex> lock(*mutex_);
    torch::NoGradGuard ng;

    // obsを生成
    auto obs = state.obs.To(device_);
    if (ctx) obs = ctx->PushObservation(state);

    // Normalize observations
    auto norm_obs = obs_norm_->Normalize(obs);
    //ANET_LOG_DEBUG("norm_obs=" << norm_obs.ToDefString());

    // 行動選択
    BatchActionInfo act_info;

    // RunMode に応じて Policy を切り替える
    auto rnd = ctx->GetRandomGenerator();
    auto run_mode = (ctx != nullptr) ? ctx->GetRunMode() : anet::rl::RunMode::Train;
    if (anet::rl::IsEval(run_mode)) {
        auto use_target = IsForTarget(run_mode);
        auto network = use_target ? model_->GetTargetNetwork() : model_->GetMainNetwork();
        act_info = eval_policy_->SelectAction(norm_obs, false, network, rnd);
    } else {
        // Train向けでは train_policy_ と MainNetwork で固定
        act_info = train_policy_->SelectAction(norm_obs, false, model_->GetMainNetwork(), rnd);
    }

    // ObservationをAuxに詰める
    act_info.GetAuxData()["raw_obs"] = anet::rl::ToUnifiedObservation(obs);     // スタック済・正規化前のObservation
    if (obs_norm_ != nullptr) {
        act_info.GetAuxData()["norm_obs"] = anet::rl::ToUnifiedObservation(norm_obs);       // スタック済・正規化済のObservation
    }

    // ActionInfoを返す
    return act_info;
}

std::shared_ptr<anet::rl::Actor> DefaultDQNAgent::CreateActor(const anet::rl::BatchEnvSpec& batch_env_spec, anet::rl::RunMode run_mode, bool clone_model, std::optional<torch::Device> device) const
{
    // Contextを生成
    auto ctx = this->CreateActionContext(batch_env_spec, run_mode, device);

    // モードに応じて適切な Policy と Network を選択
    std::shared_ptr<anet::rl::dqn::ActionPolicy> policy;
    std::shared_ptr<anet::nn::Network> src_network;

    // 元ネタのPolicyとNetoworkを決定
    if (anet::rl::IsEval(run_mode)) {
        policy = eval_policy_;
        src_network = IsForTarget(run_mode) ? model_->GetTargetNetwork() : model_->GetMainNetwork();
    } else {
        policy = train_policy_;
        src_network = model_->GetMainNetwork();
    }

    // 必要に応じてCloneしてActor向けネットワークとする
    auto network = (clone_model) ? src_network->Clone(device) : src_network;

    // Actor を生成
    auto actor = std::make_shared<dqn::Actor>(policy, obs_norm_, ctx, this->mutex_, network, src_network);

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
    ProfileRange r1("DefaultDQNAgent::UpdateFromBatch");

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

std::shared_ptr<anet::rl::Agent> DefaultDQNAgentFactory::CreateAgent(
    const EnvSpec& env_spec, const BatchEnvSpec& batch_env_spec,
    const torch::Device& device, const anet::ConfigData& config_data,
    std::shared_ptr<anet::rl::Notifier> notifier, std::optional<anet::seed_t> seed) const
{
    DefaultDQNAgentConfig config(config_data);
	anet::nn::NetworkConfig net_config(config_data);
    auto agent = std::make_shared<DefaultDQNAgent>(config, net_config, batch_env_spec, env_spec, device, seed);
    return agent;

    /// @todo 引数の順番を統一
}

//ANET_REGISTER_AGENT_FACTORY(DefaultDQNAgentFactory);
