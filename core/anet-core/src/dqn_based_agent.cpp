#include "dqn_based_agent.hpp"
#include <algorithm>
#include <tuple>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include "anet/log.hpp"
#include "anet/profile.hpp"
#include "anet/tensor_check.hpp"
#include "anet/tensor_util.hpp"
#include "anet/str_util.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/image.hpp"
#include "anet/metrics_logger.hpp"
#include "nn_heads.hpp"

using namespace anet::rl::dqn;
namespace LOG = anet::log;

static void ValidateTauGeneratorArguments(int64_t batch_size, int64_t num_taus, const std::string& sample_mode)
{
    if (batch_size < 0) {
        ANET_SYSTEM_ERROR("TauGenerator requires batch_size>=0, but batch_size=" << batch_size << ".");
    }
    if (num_taus <= 0) {
        ANET_SYSTEM_ERROR("TauGenerator requires num_taus>0, but num_taus=" << num_taus << ".");
    }
    if (sample_mode != "random" && sample_mode != "fixed") {
        ANET_SYSTEM_ERROR("TauGenerator received sample_mode=" << sample_mode
            << "; expected random or fixed.");
    }
}

static torch::Tensor MakeTauMidpointPositions(int64_t num_taus, const torch::TensorOptions& options)
{
    return (torch::arange(num_taus, options) + 0.5f) / static_cast<float>(num_taus);
}

torch::Tensor anet::rl::dqn::GenerateTaus(
    int64_t batch_size,
    int64_t num_taus,
    const std::string& sample_mode,
    float tau_min,
    float tau_max,
    const torch::Device& device,
    anet::RandomGenerator& rnd)
{
    ANET_PROFILE_FUNC();
    ValidateTauGeneratorArguments(batch_size, num_taus, sample_mode);

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    if (sample_mode == "random") {
        // 乱数を対象 device 上で一括生成し、CPU 同期を挟まず指定範囲へ写像する。
        auto gen = rnd.GetTorchGenerator(device);
        const auto unit = torch::rand({ batch_size, num_taus }, gen, options);
        return tau_min + unit * (tau_max - tau_min);
    }

    // midpoint 配置では generator に触れず、全 batch へ同じ配置を共有する。
    const auto positions = MakeTauMidpointPositions(num_taus, options);
    const auto taus = tau_min + positions * (tau_max - tau_min);
    return taus.unsqueeze(0).expand({ batch_size, num_taus });
}

torch::Tensor anet::rl::dqn::GenerateTaus(
    const torch::Tensor& tau_min_per_env,
    float tau_max,
    int64_t num_taus,
    const std::string& sample_mode,
    anet::RandomGenerator& rnd)
{
    ANET_PROFILE_FUNC();
    if (tau_min_per_env.dim() != 1) {
        ANET_SYSTEM_ERROR("TauGenerator requires tau_min_per_env rank=1 (B), but shape="
            << tau_min_per_env.sizes() << ".");
    }
    ValidateTauGeneratorArguments(tau_min_per_env.size(0), num_taus, sample_mode);

    const auto device = tau_min_per_env.device();
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto lower = tau_min_per_env.to(torch::kFloat32).unsqueeze(1);
    if (sample_mode == "random") {
        // env ごとの下限を broadcast し、各行を独立にサンプリングする。
        auto gen = rnd.GetTorchGenerator(device);
        const auto unit = torch::rand({ tau_min_per_env.size(0), num_taus }, gen, options);
        return lower + unit * (tau_max - lower);
    }

    // midpoint の相対位置だけを共有し、絶対位置は env ごとの下限で決める。
    const auto positions = MakeTauMidpointPositions(num_taus, options).unsqueeze(0);
    return lower + positions * (tau_max - lower);
}

anet::rl::ReplayInitialPriorityMode anet::rl::dqn::ParseReplayInitialPriorityMode(const LearnerConfig& config)
{
    // 文字列表現をReplayBufferへ渡す内部modeへ変換し、未知値は設定ミスとして停止する。
    if (config.per_initial_priority_mode == "fixed") {
        return ReplayInitialPriorityMode::FIXED;
    } else if (config.per_initial_priority_mode == "max") {
        return ReplayInitialPriorityMode::MAX;
    } else if (config.per_initial_priority_mode == "actor_approx") {
        return ReplayInitialPriorityMode::ACTOR_APPROX;
    }

    ANET_SYSTEM_ERROR("Invalid learner.per_initial_priority_mode='" << config.per_initial_priority_mode
        << "'. Expected one of: fixed, max, actor_approx.");
    return ReplayInitialPriorityMode::FIXED; // ANET_SYSTEM_ERROR後のMSVC制御経路警告を抑止する。
}

void anet::rl::dqn::ValidateReplayPriorityConfig(
    const LearnerConfig& config, ReplayInitialPriorityMode initial_priority_mode)
{
    // modeを切り替えても同じconfigを再利用できるよう、関連する数値設定を一括検証する。
    if (!std::isfinite(config.per_initial_priority) || config.per_initial_priority < 0.0f) {
        ANET_SYSTEM_ERROR("Invalid learner.per_initial_priority=" << config.per_initial_priority
            << ". Expected a finite value >= 0; use learner.per_initial_priority_mode=max for maximum initialization.");
    }
    if (!std::isfinite(config.per_eps) || config.per_eps < 0.0f) {
        ANET_SYSTEM_ERROR("Invalid learner.per_eps=" << config.per_eps << ". Expected a finite value >= 0.");
    }
    if (!config.use_per && initial_priority_mode != ReplayInitialPriorityMode::FIXED) {
        ANET_SYSTEM_ERROR("learner.per_initial_priority_mode=" << config.per_initial_priority_mode
            << " requires learner.use_per=true; set the mode to fixed or enable PER.");
    }
    if (config.use_per && config.per_eps == 0.0f) {
        LOG::warn() << "learner.per_eps=0 is allowed with learner.use_per=true, but zero-TD-error transitions may "
            << "receive zero sampling priority. Set learner.per_eps to a finite value > 0 to keep a positive priority floor.";
    }
    if (initial_priority_mode == ReplayInitialPriorityMode::ACTOR_APPROX && config.per_alpha == 0.0f) {
        LOG::warn() << "learner.per_initial_priority_mode=actor_approx with learner.per_alpha=0 is allowed, but Actor "
            << "priority does not affect sampling. Set learner.per_alpha to a finite value > 0 to enable prioritized sampling.";
    }
    if (config.use_per_prio_clip) {
        if (!std::isfinite(config.per_prio_clip_value) || config.per_prio_clip_value < 0.0f) {
            ANET_SYSTEM_ERROR("Invalid learner.per_prio_clip_value=" << config.per_prio_clip_value
                << ". Expected a finite value >= 0 when learner.use_per_prio_clip=true.");
        }
        if (config.per_prio_clip_value == 0.0f) {
            LOG::warn() << "learner.use_per_prio_clip=true with learner.per_prio_clip_value=0 is allowed, but all PER "
                << "priorities are clipped to zero. Set learner.per_prio_clip_value to a finite value > 0 or disable "
                << "learner.use_per_prio_clip.";
        } else if (config.per_prio_clip_value <= config.per_eps) {
            LOG::warn() << "learner.per_prio_clip_value=" << config.per_prio_clip_value
                << " <= learner.per_eps=" << config.per_eps
                << " may collapse priority differences. Set learner.per_prio_clip_value > learner.per_eps or disable "
                << "learner.use_per_prio_clip.";
        }
    }
}

torch::Tensor anet::rl::dqn::TransformH(const torch::Tensor& x, float epsilon)
{
    return x.sign() * (torch::sqrt(x.abs() + 1.0f) - 1.0f) + epsilon * x;
}

float anet::rl::dqn::TransformH(float x, float epsilon)
{
    const float sign = static_cast<float>((x > 0.0f) - (x < 0.0f));
    return sign * (std::sqrt(std::abs(x) + 1.0f) - 1.0f) + epsilon * x;
}

torch::Tensor anet::rl::dqn::TransformHInv(const torch::Tensor& x, float epsilon)
{
    auto abs_x = x.abs();
    auto inner = (torch::sqrt(1.0f + 4.0f * epsilon * (abs_x + 1.0f + epsilon)) - 1.0f) / (2.0f * epsilon);
    return x.sign() * (inner * inner - 1.0f);
}

float anet::rl::dqn::TransformHInv(float x, float epsilon)
{
    const float sign = static_cast<float>((x > 0.0f) - (x < 0.0f));
    const float inner = (std::sqrt(1.0f + 4.0f * epsilon * (std::abs(x) + 1.0f + epsilon)) - 1.0f)
        / (2.0f * epsilon);
    return sign * (inner * inner - 1.0f);
}

PerRawPriorityBatchResult anet::rl::dqn::MakePerRawPriorityBatch(
    const torch::Tensor& td_error, float per_eps, bool use_clip, float clip_value)
{
    // clip前の値で変更件数を判定し、上限と元から等しいpriorityは数えない。
    auto unclipped_priorities = td_error.abs() + per_eps;
    auto clipped_count = use_clip
        ? unclipped_priorities.gt(clip_value).sum()
        : torch::zeros({}, unclipped_priorities.options().dtype(torch::kInt64));
    auto priorities = use_clip
        ? unclipped_priorities.clamp_max(clip_value)
        : unclipped_priorities;
    return {
        .priorities = std::move(priorities),
        .clipped_count = std::move(clipped_count),
    };
}

torch::Tensor anet::rl::dqn::MakePerRawPriority(
    const torch::Tensor& td_error, float per_eps, bool use_clip, float clip_value)
{
    return MakePerRawPriorityBatch(td_error, per_eps, use_clip, clip_value).priorities;
}

float anet::rl::dqn::MakePerRawPriority(
    float td_error, float per_eps, bool use_clip, float clip_value)
{
    const float priority = std::abs(td_error) + per_eps;
    return use_clip ? std::min(priority, clip_value) : priority;
}

torch::Tensor anet::rl::dqn::PackActorQHint(
    const torch::Tensor& actor_q_sa, const torch::Tensor& actor_state_value)
{
    // ActorとWithActionが同じ列順でpackできるよう、DQN payloadの構築を一元化する。
    if (!actor_q_sa.defined() || !actor_state_value.defined()) {
        ANET_SYSTEM_ERROR("PackActorQHint requires defined tensors.");
    }
    if (actor_q_sa.dim() != 1 || actor_state_value.dim() != 1
        || actor_q_sa.sizes() != actor_state_value.sizes()) {
        ANET_SYSTEM_ERROR("PackActorQHint expected matching rank-1 tensors. actor_q_sa="
            << actor_q_sa.sizes() << " actor_state_value=" << actor_state_value.sizes());
    }
    if (actor_q_sa.device() != actor_state_value.device()) {
        ANET_SYSTEM_ERROR("PackActorQHint expected tensors on the same device. actor_q_sa="
            << actor_q_sa.device() << " actor_state_value=" << actor_state_value.device());
    }
    return torch::stack({ actor_q_sa, actor_state_value }, 1).to(torch::kFloat32).contiguous();
}

ActorQHintBatch anet::rl::dqn::DecodeActorQHint(const torch::Tensor& payload)
{
    // tensor carrierからDQN列を取り出す境界で、共通層が扱わないschemaを検証する。
    if (!payload.defined()) {
        ANET_SYSTEM_ERROR("DQN Actor Q hint payload must be defined.");
    }
    if (payload.dim() != 2 || payload.scalar_type() != torch::kFloat32
        || payload.size(1) != kActorQHintColumnCount) {
        ANET_SYSTEM_ERROR("DQN Actor Q hint must be float32 [B,2]. actual=" << payload.sizes());
    }
    return ActorQHintBatch{
        .actor_q_sa = payload.select(1, kActorQSaColumn),
        .actor_state_value = payload.select(1, kActorStateValueColumn),
    };
}

ActorQHintRow anet::rl::dqn::DecodeActorQHint(std::span<const float> payload)
{
    // ReplayBufferから渡されたopaque rowは、DQN推定器に入る時点でだけ列へdecodeする。
    if (payload.size() != kActorQHintColumnCount) {
        ANET_SYSTEM_ERROR("DQN Actor Q hint must have K=2. actual=" << payload.size());
    }
    return ActorQHintRow{
        .actor_q_sa = payload[kActorQSaColumn],
        .actor_state_value = payload[kActorStateValueColumn],
    };
}

namespace {

class DqnInitialPriorityEstimator final : public anet::rl::InitialPriorityEstimator {
public:
    explicit DqnInitialPriorityEstimator(const LearnerConfig& config)
        : use_tbo_(config.use_tbo), tbo_epsilon_(config.tbo_epsilon), per_eps_(config.per_eps)
        , use_clip_(config.use_per_prio_clip), clip_value_(config.per_prio_clip_value)
    {
    }

    bool ValidateHint(std::span<const float> hint) const override
    {
        const auto decoded = DecodeActorQHint(hint);
        return std::isfinite(decoded.actor_q_sa) && std::isfinite(decoded.actor_state_value);
    }

    std::optional<float> Estimate(const anet::rl::InitialPriorityEstimateInput& input) const override
    {
        if (!ValidateHint(input.start_hint)
            || (!input.terminal && !ValidateHint(input.bootstrap_hint))) {
            return std::nullopt;
        }
        const auto start_hint = DecodeActorQHint(input.start_hint);
        // TBOではbootstrap値を実空間へ戻してn-step収益と加算し、Learnerと同じh空間へ再変換する。
        float target = input.target_return;
        if (!input.terminal) {
            const float bootstrap_state_value = DecodeActorQHint(input.bootstrap_hint).actor_state_value;
            const float bootstrap = use_tbo_
                ? TransformHInv(bootstrap_state_value, tbo_epsilon_)
                : bootstrap_state_value;
            target += input.discount * bootstrap;
        }
        if (use_tbo_) target = TransformH(target, tbo_epsilon_);
        // Learnerと同じraw priority確定policyをscalar overloadで適用する。
        const float priority = MakePerRawPriority(
            target - start_hint.actor_q_sa, per_eps_, use_clip_, clip_value_);
        if (!std::isfinite(priority)) return std::nullopt;
        return priority;
    }
private:
    bool use_tbo_;
    float tbo_epsilon_;
    float per_eps_;
    bool use_clip_;
    float clip_value_;
};

} // namespace

std::unique_ptr<anet::rl::InitialPriorityEstimator> anet::rl::dqn::CreateInitialPriorityEstimator(
    const LearnerConfig& config)
{
    return std::make_unique<DqnInitialPriorityEstimator>(config);
}

// ======================================================
// NetworkModel
// ======================================================

NetworkModel::NetworkModel(
    const NetworkModelConfig& config, const torch::Device device,
    const anet::nn::NetworkConfig& network_config, const anet::TensorSpecMap& obs_spec, int64_t n_actions, std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory,
    bool distributional)
    : config_(config)
    , n_actions_(n_actions)
    , distributional_(distributional)
{
    ANET_ASSERT(n_actions_ > 0);

    // メインネットワークを作る
    online_net_ = anet::nn::NetworkBuilder::BuildNetwork(network_config, obs_spec, head_factory, device);
    online_net_->to(device);
    online_net_->eval();

#if 0
    {
        torch::NoGradGuard no_grad;
        for (auto& param : online_net_->parameters()) {
            torch::nn::init::constant_(param, 0.01f);
        }
    }
#endif

    // メインネットワークをコピーしてターゲットネットワークを作る
    target_net_ = online_net_->Clone(device);
    target_net_->eval();

    LOG::info() << "========== MODEL SHAPE DUMP ==========";
    for (const auto& pair : online_net_->named_parameters()) {
        LOG::info() << pair.key() << " : " << pair.value().sizes();
    }
    LOG::info() << "======================================";

    LOG::info() << "Number of online network parameter tensors: " << online_net_->parameters().size();
    LOG::info() << "Number of target network parameter tensors: " << target_net_->parameters().size();
}

NetworkModel::NetworkModel(
    const NetworkModelConfig& config,
    std::shared_ptr<anet::nn::Network> online_net,
    std::shared_ptr<anet::nn::Network> target_net,
    int64_t n_actions,
    bool distributional)
    : config_(config)
    , online_net_(std::move(online_net))
    , target_net_(std::move(target_net))
    , n_actions_(n_actions)
    , distributional_(distributional)
{
    ANET_ASSERT(n_actions_ > 0);
    ANET_ASSERT(online_net_ != nullptr);
    ANET_ASSERT(target_net_ != nullptr);
    online_net_->eval();
    target_net_->eval();
}

anet::TensorDict NetworkModel::ForwardOnline(const anet::TensorDict& obs) const
{
    return online_net_->Forward(obs);
}

anet::TensorDict NetworkModel::ForwardOnlineWithTrain(const anet::TensorDict& obs) const
{
    anet::TrainingModeGuard guard(*online_net_, true);
    return online_net_->Forward(obs);
}

anet::TensorDict NetworkModel::ForwardTarget(const anet::TensorDict& obs) const
{
    return target_net_->Forward(obs);
}

bool NetworkModel::IsDistributional() const
{
    return distributional_;
}

std::vector<torch::Tensor> NetworkModel::GetOnlineParameters() const
{
    return online_net_->parameters();
}

torch::OrderedDict<std::string, torch::Tensor> NetworkModel::GetOnlineNamedParameters() const
{
    return online_net_->named_parameters();
}

void NetworkModel::UpdateTarget(step_t learn_step)
{
    torch::NoGradGuard grad_guard;

    if (config_.hard_update_interval > 0) {
        if (learn_step % config_.hard_update_interval == 0) {
            HardUpdate();
        }
        return;
    }
    SoftUpdate();
}

void NetworkModel::SoftUpdate()
{
    online_net_->SoftCopyTo(*target_net_, config_.soft_update_tau);
}

void NetworkModel::HardUpdate()
{
    online_net_->CopyTo(*target_net_);
}

std::optional<anet::TensorDictFunction> NetworkModel::GetTensorDictFunction(const std::string& key, const torch::Device& device)
{
    // 入力 TensorDict の device 正規化は Agent wrapper の責務。
    // ここでは key routing だけを行い、network 本体の関数を返す。
    (void)device;
    static constexpr const char* POLICY_PREFIX = "policy-net.";
    static constexpr const char* TARGET_PREFIX = "target-net.";

    // Keyの指定に応じて、対象のネットワークを切り替える
    std::shared_ptr<anet::nn::Network> net = nullptr;
    std::string subkey;

    if (anet::StartsWith(key, POLICY_PREFIX)) {
        net = online_net_;
        subkey = anet::RemovePrefix(key, POLICY_PREFIX);
    } else if (anet::StartsWith(key, TARGET_PREFIX)) {
        net = target_net_;
        subkey = anet::RemovePrefix(key, TARGET_PREFIX);
    }

    // 対象が見つからなければ nullopt
    if (!net) {
        return std::nullopt;
    }

    // 汎用 head ベース抽出へ委譲
    return net->GetTensorDictFunction(subkey);
}

int64_t NetworkModel::Save(OutputArchive& archive) const
{
	int64_t size = 0;
    size += archive.WriteTorchObject(online_net_);
    size += archive.WriteTorchObject(target_net_);
    return size;
}

int64_t NetworkModel::Load(InputArchive& archive)
{
    int64_t size = 0;
    size += archive.ReadTorchObject(online_net_);
    size += archive.ReadTorchObject(target_net_);
    online_net_->eval();
    target_net_->eval();
    return size;
}


// ======================================================
// ActionPolicy
// ======================================================

std::shared_ptr<anet::rl::BatchActionInfo> DQNActionInfo::To(torch::Device device) const
{
    auto result = std::make_shared<DQNActionInfo>(
        GetAction(device), info_.To(device), aux_, replay_initial_priority_hint_);
    result->train_actor_snapshot_metrics_ = train_actor_snapshot_metrics_;
    return result;
}

std::shared_ptr<anet::rl::BatchActionInfo> DQNActionInfo::WithAction(torch::Tensor action) const
{
    auto hint = replay_initial_priority_hint_;
    if (hint.has_value()) {
        // 実行行動の差し替え後も、既存forwardの全行動QからQ(s,a)だけを再取得してhint契約を保つ。
        const auto it = aux_.find("q_values");
        if (it == aux_.end() || !it->second.defined()) {
            ANET_SYSTEM_ERROR(
                "DQNActionInfo::WithAction requires aux[q_values] when an Actor Q hint is present.");
        }
        const auto& q_values = it->second;
        if (action.device() != q_values.device()) {
            ANET_SYSTEM_ERROR("DQNActionInfo::WithAction cannot replace a hinted action across devices. action="
                << action.device() << " q_values=" << q_values.device());
        }
        auto q_sa = q_values.gather(1, action.to(torch::kInt64).unsqueeze(1)).squeeze(1).to(torch::kFloat32);
        const auto decoded = DecodeActorQHint(hint->GetPayload());
        auto packed = PackActorQHint(q_sa, decoded.actor_state_value);
        hint = ReplayInitialPriorityHint(std::move(packed));
    }
    auto result = std::make_shared<DQNActionInfo>(std::move(action), info_, aux_, std::move(hint));
    result->train_actor_snapshot_metrics_ = train_actor_snapshot_metrics_;
    return result;
}

namespace {

enum class ActionUqeScalarKind {
    kWinRate,
    kMargin,
};

struct ActionUqeScalarKey {
    ActionUqeScalarKind kind;
    int64_t action_index;
};

int64_t ParseActionScalarIndex(const std::string& key, const char* prefix)
{
    const std::string prefix_text(prefix);
    if (!anet::StartsWith(key, prefix_text) || !anet::EndsWith(key, "]")) {
        ANET_SYSTEM_ERROR("DQNActionInfo: invalid scalar key format: " << key);
    }

    const std::string index_text = key.substr(
        prefix_text.size(),
        key.size() - prefix_text.size() - 1);
    if (index_text.empty()) {
        ANET_SYSTEM_ERROR("DQNActionInfo: action index is empty in scalar key: " << key);
    }

    int64_t action_index = -1;
    try {
        size_t parsed_len = 0;
        action_index = std::stoll(index_text, &parsed_len);
        if (parsed_len != index_text.size()) {
            ANET_SYSTEM_ERROR("DQNActionInfo: invalid action index in scalar key: " << key);
        }
    } catch (const std::exception&) {
        ANET_SYSTEM_ERROR("DQNActionInfo: invalid action index in scalar key: " << key);
    }
    if (action_index < 0) {
        ANET_SYSTEM_ERROR("DQNActionInfo: action index must be non-negative in scalar key: " << key);
    }
    return action_index;
}

std::optional<ActionUqeScalarKey> ParseActionUqeScalarKey(const std::string& key)
{
    static constexpr const char* kWinRatePrefix = "action_uqe_win_rate.[";
    static constexpr const char* kWinRateBase = "action_uqe_win_rate";
    static constexpr const char* kMarginPrefix = "action_uqe_margin.[";
    static constexpr const char* kMarginBase = "action_uqe_margin";

    if (anet::StartsWith(key, kWinRateBase)) {
        return ActionUqeScalarKey{ ActionUqeScalarKind::kWinRate, ParseActionScalarIndex(key, kWinRatePrefix) };
    }
    if (anet::StartsWith(key, kMarginBase)) {
        return ActionUqeScalarKey{ ActionUqeScalarKind::kMargin, ParseActionScalarIndex(key, kMarginPrefix) };
    }
    return std::nullopt;
}

enum class EpisodeStartActionMarginValueKind {
    kUqe,
    kQ,
};

struct EpisodeStartActionMarginKey {
    EpisodeStartActionMarginValueKind value_kind;
    int64_t action_index;
};

std::optional<EpisodeStartActionMarginKey> ParseEpisodeStartActionMarginKey(const std::string& key)
{
    static constexpr const char* kUqePrefix = "episode_start_action_uqe_margin.[";
    static constexpr const char* kUqeBase = "episode_start_action_uqe_margin";
    static constexpr const char* kQPrefix = "episode_start_action_q_margin.[";
    static constexpr const char* kQBase = "episode_start_action_q_margin";

    if (anet::StartsWith(key, kUqeBase)) {
        return EpisodeStartActionMarginKey{
            EpisodeStartActionMarginValueKind::kUqe,
            ParseActionScalarIndex(key, kUqePrefix)
        };
    }
    if (anet::StartsWith(key, kQBase)) {
        return EpisodeStartActionMarginKey{
            EpisodeStartActionMarginValueKind::kQ,
            ParseActionScalarIndex(key, kQPrefix)
        };
    }
    return std::nullopt;
}

} // namespace

std::optional<float> DQNActionInfo::GetScalar(const std::string& key, int64_t) const
{
    if (key == "train_actor_snapshot_interval" || key == "train_actor_snapshot_age") {
        if (!train_actor_snapshot_metrics_.has_value()) {
            return std::nullopt;
        }
        return key == "train_actor_snapshot_interval"
            ? train_actor_snapshot_metrics_->interval
            : train_actor_snapshot_metrics_->age;
    }

    const auto episode_start_key = ParseEpisodeStartActionMarginKey(key);
    if (episode_start_key.has_value()) {
        // 指定された価値表現を取得し、非UQE Policyだけは未対応値としてNaNを返す。
        const bool is_uqe = episode_start_key->value_kind == EpisodeStartActionMarginValueKind::kUqe;
        const char* values_key = is_uqe ? "uqe_values" : "q_values";
        const auto values_it = aux_.find(values_key);
        if (values_it == aux_.end() || !values_it->second.defined()) {
            if (is_uqe) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            ANET_SYSTEM_ERROR("DQNActionInfo: required aux value is missing. key=" << key << " aux=" << values_key);
        }

        const auto& values = values_it->second;
        if (values.dim() != 2) {
            ANET_SYSTEM_ERROR("DQNActionInfo: " << values_key << " must have shape [B,A]. actual=" << values.sizes());
        }
        const int64_t batch_size = values.size(0);
        const int64_t num_actions = values.size(1);
        if (batch_size <= 0 || num_actions < 2) {
            ANET_SYSTEM_ERROR("DQNActionInfo: " << values_key
                << " must have non-empty batch and at least two actions. actual=" << values.sizes());
        }
        const int64_t action_index = episode_start_key->action_index;
        if (action_index >= num_actions) {
            ANET_SYSTEM_ERROR("DQNActionInfo: action index out of range. key=" << key << " num_actions=" << num_actions);
        }

        // 現在actionと同じbatchに対応するepisode開始maskを検証する。
        const auto episode_start_it = aux_.find("episode_start");
        if (episode_start_it == aux_.end() || !episode_start_it->second.defined()) {
            ANET_SYSTEM_ERROR("DQNActionInfo: required aux value is missing. key=" << key << " aux=episode_start");
        }
        const auto& episode_start = episode_start_it->second;
        if (episode_start.scalar_type() != torch::kBool) {
            ANET_SYSTEM_ERROR("DQNActionInfo: episode_start must have dtype bool. actual=" << episode_start.scalar_type());
        }
        if (episode_start.dim() != 1 || episode_start.size(0) != batch_size) {
            ANET_SYSTEM_ERROR("DQNActionInfo: episode_start must have shape [" << batch_size
                << "]. actual=" << episode_start.sizes());
        }

        // resetがない通常stepではdevice転送せず、Observerが出力をskipするNaNを返す。
        const auto episode_start_cpu = episode_start.device().is_cpu()
            ? episode_start
            : episode_start.to(torch::kCPU);
        if (!episode_start_cpu.any().item<bool>()) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        // reset laneだけを抽出し、指定actionと最良の他actionとの差を平均する。
        const auto episode_start_on_device = episode_start.device() == values.device()
            ? episode_start
            : episode_start.to(values.device());
        const auto reset_values = values.index({ episode_start_on_device });
        const auto selected = reset_values.select(1, action_index);
        torch::Tensor other_values;
        if (action_index == 0) {
            other_values = reset_values.slice(1, 1, num_actions);
        } else if (action_index == num_actions - 1) {
            other_values = reset_values.slice(1, 0, num_actions - 1);
        } else {
            other_values = torch::cat({
                reset_values.slice(1, 0, action_index),
                reset_values.slice(1, action_index + 1, num_actions)
            }, 1);
        }
        const auto max_other = std::get<0>(other_values.max(1));
        return (selected - max_other).mean().item<float>();
    }

    const auto parsed_key = ParseActionUqeScalarKey(key);
    if (!parsed_key.has_value()) {
        return std::nullopt;
    }
    const int64_t action_index = parsed_key->action_index;

    auto it = aux_.find("uqe_values");
    if (it == aux_.end() || !it->second.defined()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const auto& uqe_values = it->second;
    if (uqe_values.dim() != 2) {
        ANET_SYSTEM_ERROR("DQNActionInfo: uqe_values must have shape [B,A]. actual=" << uqe_values.sizes());
    }
    const int64_t batch_size = uqe_values.size(0);
    const int64_t num_actions = uqe_values.size(1);
    if (batch_size <= 0 || num_actions < 2) {
        ANET_SYSTEM_ERROR("DQNActionInfo: uqe_values must have non-empty batch and at least two actions. actual=" << uqe_values.sizes());
    }
    if (action_index >= num_actions) {
        ANET_SYSTEM_ERROR("DQNActionInfo: action index out of range. key=" << key << " num_actions=" << num_actions);
    }

    const auto selected = uqe_values.select(1, action_index);
    torch::Tensor other_values;
    if (action_index == 0) {
        other_values = uqe_values.slice(1, 1, num_actions);
    } else if (action_index == num_actions - 1) {
        other_values = uqe_values.slice(1, 0, num_actions - 1);
    } else {
        other_values = torch::cat({
            uqe_values.slice(1, 0, action_index),
            uqe_values.slice(1, action_index + 1, num_actions)
        }, 1);
    }

    const auto max_other = std::get<0>(other_values.max(1));
    if (parsed_key->kind == ActionUqeScalarKind::kWinRate) {
        const auto win_rate = selected.ge(max_other).to(torch::kFloat32).mean();
        return win_rate.item<float>();
    }
    const auto margin = (selected - max_other).mean();
    return margin.item<float>();
}

anet::rl::dqn::ActionPolicy::ActionPolicy(
    const ActionPolicyConfig& config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device)
    : config_(config)
    , use_spatial_exploration_(enable_spatial_exploration)
    , spatial_num_envs_(enable_spatial_exploration ? num_envs : 0)
{
    if (!use_spatial_exploration_) return;

    if (num_envs <= 0) {
        ANET_SYSTEM_ERROR("ActionPolicy spatial exploration requires num_envs > 0. num_envs=" << num_envs);
    }
    if (config_.spatial_scale_type != "log" && config_.spatial_scale_type != "linear") {
        ANET_SYSTEM_ERROR("Invalid spatial_scale_type: " << config_.spatial_scale_type);
    }
    if (num_envs < 32) {
        LOG::warn() << "ActionPolicy spatial exploration is enabled with num_envs < 32. num_envs=" << num_envs;
    }

    current_epsilon_ = std::numeric_limits<float>::quiet_NaN();
    current_uqe_tau_ = std::numeric_limits<float>::quiet_NaN();
}

anet::TensorDict anet::rl::dqn::ActionPolicy::ForwardForAction(
    const anet::TensorDict& obs, std::shared_ptr<anet::nn::Network> network, const anet::TraceSink& sink) const
{
    return network->Forward(obs, sink);
}

anet::TensorDict anet::rl::dqn::ActionPolicy::ForwardForActionWithTaus(
    const anet::TensorDict& obs, const torch::Tensor& taus,
    std::shared_ptr<anet::nn::Network> network, const anet::TraceSink& sink) const
{
    // 呼び出し元やReplayBufferが所有する辞書を変更せず、forward専用入力へtausを足す。
    anet::TensorDict network_input = obs;
    network_input.Set(anet::nn::kKey_Taus, taus);
    return ForwardForAction(network_input, std::move(network), sink);
}

torch::Tensor anet::rl::dqn::ActionPolicy::CreateSpatialTensor(
    int64_t num_envs, float start_val, float end_val, const std::string& scale_type, const torch::Device& device)
{
    if (num_envs <= 0) {
        ANET_SYSTEM_ERROR("CreateSpatialTensor requires num_envs > 0. num_envs=" << num_envs);
    }
    if (scale_type != "log" && scale_type != "linear") {
        ANET_SYSTEM_ERROR("Invalid spatial_scale_type: " << scale_type);
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    if (num_envs == 1) {
        return torch::full({ 1 }, start_val, options);
    }

    auto base = torch::linspace(0.0f, 1.0f, num_envs, options);
    if (scale_type == "linear") {
        return start_val + base * (end_val - start_val);
    }

    constexpr float kMinLogValue = 1.0e-4f;
    float start_clamped = start_val;
    float end_clamped = end_val;
    if (start_clamped <= 0.0f || end_clamped <= 0.0f) {
        static std::once_flag warn_once;
        std::call_once(warn_once, []() {
            LOG::warn() << "ActionPolicy spatial log scale received non-positive boundary value. "
                << "Clamping to 1e-4 for log-space calculation.";
        });
        start_clamped = std::max(start_clamped, kMinLogValue);
        end_clamped = std::max(end_clamped, kMinLogValue);
    }

    const float log_ratio = std::log(start_clamped / end_clamped);
    return end_clamped * torch::exp((1.0f - base) * log_ratio);
}

torch::Tensor anet::rl::dqn::ActionPolicy::CreateSpatialLaneTensor(
    int64_t num_envs, float start_val, float end_val, const std::string& scale_type, const torch::Device& device)
{
    // 補間値と探索パラメータの組を変えず、env[0]がend側になるようlane割り当てだけを反転する。
    return CreateSpatialTensor(num_envs, start_val, end_val, scale_type, device).flip({ 0 });
}

static torch::Tensor PrepareSpatialTensorForAction(
    const torch::Tensor& tensor, int64_t expected_num_envs, int64_t actual_num_envs, const torch::Device& device, const char* name)
{
    if (!tensor.defined()) {
        ANET_SYSTEM_ERROR("Spatial exploration tensor is not initialized: " << name);
    }
    if (expected_num_envs != actual_num_envs) {
        ANET_SYSTEM_ERROR("Spatial exploration batch size mismatch for " << name
            << ". expected num_envs=" << expected_num_envs << " actual=" << actual_num_envs);
    }
    if (tensor.dim() != 1 || tensor.size(0) != actual_num_envs) {
        ANET_SYSTEM_ERROR("Spatial exploration tensor shape mismatch for " << name
            << ". expected=[" << actual_num_envs << "] actual=" << tensor.sizes());
    }
    return tensor.device() == device ? tensor : tensor.to(device);
}

torch::Tensor anet::rl::dqn::ActionPolicy::GetSpatialEpsilonTensor(int64_t num_envs, const torch::Device& device, bool is_uqe) const
{
    return PrepareSpatialTensorForAction(
        is_uqe ? spatial_uqe_eps_tensor_ : spatial_eps_tensor_,
        spatial_num_envs_,
        num_envs,
        device,
        is_uqe ? "uqe_epsilon" : "epsilon");
}

torch::Tensor anet::rl::dqn::ActionPolicy::GetSpatialTauTensor(int64_t num_envs, const torch::Device& device) const
{
    return PrepareSpatialTensorForAction(spatial_tau_tensor_, spatial_num_envs_, num_envs, device, "uqe_tau");
}

torch::Tensor anet::rl::dqn::ActionPolicy::MakeEpsilonGreedyAction(const torch::Tensor& greedy_action, float epsilon, int64_t num_envs, int64_t n_actions, std::shared_ptr<anet::RandomGenerator> rnd) const
{
    ANET_PROFILE_FUNC();

    // epsilongがゼロ(十分小さい)場合はGreedy（乱数生成影響を排除）
    if (epsilon <= std::numeric_limits<float>::epsilon()) {
        return greedy_action;
    }

    auto device = greedy_action.device();
    auto gen = rnd->GetTorchGenerator(device);

    // ランダム選択するActionをマスクとして選択（εを使って）
    auto mask = torch::rand({ num_envs }, gen, torch::TensorOptions().device(device)).lt(epsilon);    // mask: (N) bool, GPU上で生成

    // ランダム選択対象のアクションについて、乱数でアクション決定
    auto random_actions = torch::randint(/*low=*/0, /*high=*/n_actions, { num_envs }, gen, // random actions(N) int64
        torch::TensorOptions().dtype(torch::kInt64).device(device));

    // actions: where(mask, random_actions, greedy)
    auto actions = torch::where(mask, random_actions, greedy_action);

    return actions;
}

torch::Tensor anet::rl::dqn::ActionPolicy::MakeEpsilonGreedyAction(
    const torch::Tensor& greedy_action, const torch::Tensor& epsilon_tensor, int64_t num_envs, int64_t n_actions, std::shared_ptr<anet::RandomGenerator> rnd) const
{
    ANET_PROFILE_FUNC();

    if (epsilon_tensor.dim() != 1 || epsilon_tensor.size(0) != num_envs) {
        ANET_SYSTEM_ERROR("Vectorized epsilon shape mismatch. expected=[" << num_envs << "] actual=" << epsilon_tensor.sizes());
    }

    auto device = greedy_action.device();
    auto gen = rnd->GetTorchGenerator(device);
    auto epsilon = epsilon_tensor.device() == device ? epsilon_tensor : epsilon_tensor.to(device);

    auto mask = torch::rand({ num_envs }, gen, torch::TensorOptions().device(device)).lt(epsilon);
    auto random_actions = torch::randint(/*low=*/0, /*high=*/n_actions, { num_envs }, gen,
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    return torch::where(mask, random_actions, greedy_action);
}

std::shared_ptr<DQNActionInfo> anet::rl::dqn::ActionPolicy::MakeActionInfo(const torch::Tensor& action_values, const torch::Tensor& q_values, const torch::Tensor& q_quantiles) const
{
    ANET_PROFILE_FUNC();

    auto action_info = std::make_shared<DQNActionInfo>(action_values);
    auto& aux = action_info->GetAuxData();

    // aux[max_q]
    auto max_pair = q_values.max(1);
    auto max_q = std::get<0>(max_pair).detach();
    aux["max_q"] = max_q;
    aux["q_values"] = q_values;
    aux["q_quantiles"] = q_quantiles;
    aux["raw_actions"] = action_values;

    return action_info;
}

void anet::rl::dqn::ActionPolicy::UpdateEpsilon(step_t step, bool is_uqe)
{
    if (IsSpatialExplorationEnabled()) return;

    if (is_uqe) {
        if (config_.uqe_eps_decay_steps <= 0) return;
        if (step >= config_.uqe_eps_decay_steps) {
            current_epsilon_ = config_.uqe_eps_end;
        } else {
            const float t = static_cast<float>(step) / static_cast<float>(config_.uqe_eps_decay_steps);
            current_epsilon_ = config_.uqe_eps_start + t * (config_.uqe_eps_end - config_.uqe_eps_start);
        }
    } else {
        if (config_.eps_decay_steps <= 0) return;
        if (step >= config_.eps_decay_steps) {
            current_epsilon_ = config_.eps_end;
        } else {
            const float t = static_cast<float>(step) / static_cast<float>(config_.eps_decay_steps);
            current_epsilon_ = config_.eps_start + t * (config_.eps_end - config_.eps_start);
        }
    }
}

std::optional<float> anet::rl::dqn::ActionPolicy::GetScalar(const std::string& key, int64_t index) const
{
    if (key == "epsilon") return current_epsilon_;
    if (key == "uqe_tau") return current_uqe_tau_;
    return std::nullopt;
}


// ======================================================
// EpsilonGreedyActionPolicy 
// ======================================================

anet::rl::dqn::EpsilonGreedyActionPolicy::EpsilonGreedyActionPolicy(
    const ActionPolicyConfig& config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device)
    : ActionPolicy(config, enable_spatial_exploration, num_envs, device)
{
    if (IsSpatialExplorationEnabled()) {
        spatial_eps_tensor_ = CreateSpatialLaneTensor(num_envs, config_.eps_start, config_.eps_end, config_.spatial_scale_type, device);
        return;
    }
    current_epsilon_ = config_.eps_start;
}

void anet::rl::dqn::EpsilonGreedyActionPolicy::OnLearn(const StepCounts& counts)
{
    if (IsSpatialExplorationEnabled()) return;
    UpdateEpsilon(counts.exp_step);
}

std::shared_ptr<DQNActionInfo> EpsilonGreedyActionPolicy::SelectAction(const anet::TensorDict& obs, bool greedy_only,
    std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd, const anet::TraceSink& sink) const
{
    ANET_PROFILE_FUNC();

    torch::NoGradGuard grad_guard;
    torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;
    anet::Autocast cast_guard(obs.device(), config_.use_amp, amp_dtype);

    // IQNではpolicy自身のtau ruleでサンプルし、QR/TDは従来入力をそのまま使う。
    anet::TensorDict out;
    if (config_.quantile_mode == "iqn") {
        auto taus = GenerateTaus(
            obs.Size(0), config_.tau_rule.num_taus, config_.tau_rule.sample_mode,
            0.0f, 1.0f, obs.device(), *rnd);
        out = ForwardForActionWithTaus(obs, taus, network, sink);
    } else {
        out = ForwardForAction(obs, network, sink);
    }
    auto q_values = out.At("q");
    torch::Tensor q_quantiles;
    if (out.Contains("q_dist")) {
        q_quantiles = out.At("q_dist");
    }

    // Q値GreedyなActionを生成
    auto greedy_action = q_values.argmax(1, /*keepdim=*/false);        // greedy = argmax(q_values, dim=1)

    // Greedy指定ならargmxを返す
    if (greedy_only) return MakeActionInfo(greedy_action, q_values, q_quantiles);

    // EpsilonGreedy
    const int64_t N = q_values.sizes()[0];      // shape 読み取りは TensorOptions 経由で同期を回避
    const int64_t A = q_values.sizes()[1];
    torch::Tensor actions;
    if (IsSpatialExplorationEnabled()) {
        actions = MakeEpsilonGreedyAction(greedy_action, GetSpatialEpsilonTensor(N, q_values.device(), false), N, A, rnd);
    } else {
        actions = MakeEpsilonGreedyAction(greedy_action, current_epsilon_, N, A, rnd);
    }
    auto action_info = MakeActionInfo(actions, q_values, q_quantiles);
    return action_info;
}


// ======================================================
// UQEActionPolicy
// ======================================================

anet::rl::dqn::UQEActionPolicy::UQEActionPolicy(
    const ActionPolicyConfig& config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device)
    : ActionPolicy(config, enable_spatial_exploration, num_envs, device)
{
    if (IsSpatialExplorationEnabled()) {
        spatial_uqe_eps_tensor_ = CreateSpatialLaneTensor(num_envs, config_.uqe_eps_start, config_.uqe_eps_end, config_.spatial_scale_type, device);
        spatial_tau_tensor_ = CreateSpatialLaneTensor(num_envs, config_.uqe_tau_start, config_.uqe_tau_end, config_.spatial_scale_type, device);
        return;
    }
    current_epsilon_ = config_.uqe_eps_start;
    current_uqe_tau_ = config_.uqe_tau_start;
}

void anet::rl::dqn::UQEActionPolicy::UpdateTau(step_t step)
{
    if (IsSpatialExplorationEnabled()) return;

    if (config_.uqe_tau_decay_steps <= 0) return;
    if (step >= config_.uqe_tau_decay_steps) {
        current_uqe_tau_ = config_.uqe_tau_end;
        return;
    }
    const float t = static_cast<float>(step) / static_cast<float>(config_.uqe_tau_decay_steps);
    current_uqe_tau_ = config_.uqe_tau_start + t * (config_.uqe_tau_end - config_.uqe_tau_start);
}

void anet::rl::dqn::UQEActionPolicy::OnLearn(const StepCounts& counts)
{
    if (IsSpatialExplorationEnabled()) return;
    UpdateEpsilon(counts.exp_step, true);
    UpdateTau(counts.exp_step);
}

/// Q分布の上位tau%(上振れ時)Q値が最大になる行動を選択する。
/// tauが大きい(1.0に近い)程、不確実だけど化ける可能性がある行動が相対的に選択されやすくなる
torch::Tensor UQEActionPolicy::MakeUQEValues(float tau, const torch::Tensor& q_quantiles) const
{
    ANET_PROFILE_FUNC();

    // GPU同期を避けるため、sizes() からメタデータのみ取得
    const int64_t n_quantiles = q_quantiles.size(-1);

    // インデックス決定: floor(tau * (N-1)) clampして範囲外アクセスを防止
    int64_t tau_idx = static_cast<int64_t>(tau * (n_quantiles - 1));
    tau_idx = std::max<int64_t>(0, std::min<int64_t>(tau_idx, n_quantiles - 1));
    ANET_LOG_DEBUG("tau_idx=" << tau_idx);

    // 最後の次元(分位点)を確実に昇順(小さい順)にソートする
    torch::Tensor sorted_quantiles = std::get<0>(q_quantiles.sort(-1, /*descending=*/false));

    torch::Tensor uqe_values;
    if (config_.uqe_use_tail_mean) {
        // 上位分位点すべての平均を使う場合

        // tau_idx から 最後まで (N-1) の範囲を切り出す
        auto tail_values = sorted_quantiles.slice(-1, tau_idx, n_quantiles);  // (B, A, N - tau_idx)
        //ANET_LOG_DEBUG("tail_values=" << anet::ToString(tail_values));

        // 切り出した範囲の平均をとる
        uqe_values = tail_values.mean(-1);
    } else {
        // 特定の分位点におけるQ値を取得
        uqe_values = sorted_quantiles.select(-1, tau_idx);  // (B, A, N) -> (B, A)
    }
    //ANET_LOG_DEBUG("uqe_values=" << anet::ToString(uqe_values));

    return uqe_values;
}

torch::Tensor UQEActionPolicy::MakeVectorizedUQEValues(const torch::Tensor& tau_tensor, const torch::Tensor& q_quantiles) const
{
    // q_quantiles: (N, A, n_quantiles)
    // tau_tensor:  (N, 1)  (0.0 ~ 1.0)

    if (q_quantiles.dim() != 3) {
        ANET_SYSTEM_ERROR("UQEActionPolicy::MakeVectorizedUQEValues expected q_quantiles shape [N,A,Q], actual=" << q_quantiles.sizes());
    }

    const int64_t N = q_quantiles.size(0);
    const int64_t A = q_quantiles.size(1);
    const int64_t n_quantiles = q_quantiles.size(2);
    auto device = q_quantiles.device();

    if (!tau_tensor.defined()) {
        ANET_SYSTEM_ERROR("UQEActionPolicy::MakeVectorizedUQEValues received undefined tau_tensor.");
    }
    if (tau_tensor.dim() != 2 || tau_tensor.size(0) != N || tau_tensor.size(1) != 1) {
        ANET_SYSTEM_ERROR("UQEActionPolicy::MakeVectorizedUQEValues expected tau_tensor shape [" << N << ",1], actual="
            << tau_tensor.sizes());
    }
    auto tau = tau_tensor.device() == device ? tau_tensor : tau_tensor.to(device);
    if (tau.dtype() != q_quantiles.dtype()) {
        tau = tau.to(q_quantiles.dtype());
    }

    // 最後の次元(分位点)を確実に昇順(小さい順)にソートする
    torch::Tensor sorted_quantiles = std::get<0>(q_quantiles.sort(-1, /*descending=*/false));

    // インデックスの計算 (N, 1)
    auto tau_idx = (tau * (n_quantiles - 1)).to(torch::kLong);
    tau_idx = tau_idx.clamp(0, n_quantiles - 1);

    torch::Tensor uqe_values;

    if (config_.uqe_use_tail_mean) {
        // マスクを使って平均を計算する

        // range: (1, 1, n_quantiles) -> [0, 1, 2, ...]
        auto range = torch::arange(n_quantiles, device).view({ 1, 1, -1 });

        // mask: (N, 1, n_quantiles)
        // range >= tau_idx の部分が True (1.0)
        // tau_idx は (N, 1) なので (N, 1, 1) に view して broadcast
        auto mask = range.ge(tau_idx.view({ N, 1, 1 })).to(q_quantiles.dtype());

        // 平均計算: (sum(Q * mask) / sum(mask))
        // small_epsilon を足して 0除算防止
        uqe_values = (sorted_quantiles * mask).sum(-1) / (mask.sum(-1) + 1e-6);
    } else {
        // tau_idx を (N, A, 1) に拡張
        // バッチ(N)ごとに違うインデックスだが、Action(A)に対しては同じインデックスを使う
        auto gather_idx = tau_idx.view({ N, 1, 1 }).expand({ N, A, 1 });

        // gather: dim=-1 (quantiles次元) に沿って収集
        // output: (N, A, 1) -> squeeze -> (N, A)
        uqe_values = sorted_quantiles.gather(-1, gather_idx).squeeze(-1);
    }

    return uqe_values;
}

std::shared_ptr<DQNActionInfo> UQEActionPolicy::MakeUQEActionInfo(float tau, const torch::Tensor& tau_tensor, const anet::TensorDict& obs,
    bool greedy_only, std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd, const anet::TraceSink& sink,
    bool iqn_use_full_range) const
{
    ANET_PROFILE_FUNC();

    torch::NoGradGuard grad_guard;
    torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;
    anet::Autocast cast_guard(obs.device(), config_.use_amp, amp_dtype);

    // policy更新で変化する探索率と、呼び出し側が決めたrisk基準tauを今回のaction queryへ反映する。
    float effective_epsilon = current_epsilon_;
    float effective_tau = tau;
    bool use_vectorized_tau = tau_tensor.defined();

    //greedy_onlyが指定された場合は、ランダムノイズ(ε)を強制的にゼロにする
    // (Tauによる楽観的選択の基準は維持しつつ、デタラメな行動を防ぐ)
    if (greedy_only) {
        effective_epsilon = 0.0f;
    }

    // IQNでは行動scoreを作るrisk queryを先頭へ置き、任意のfull queryを後ろへ連結して1回のforwardで処理する。
    // QRではネットワークが固定quantileを返すため、tausを観測へ追加せず従来のforwardを使う。
    ANET_PROFILE_SCOPE(generate_taus);
    anet::TensorDict out;
    int64_t risk_num_taus = 0;
    if (config_.quantile_mode == "iqn") {
        const int64_t N = obs.Size(0);
        torch::Tensor risk_taus;
        if (iqn_use_full_range) {
            // ThompsonSamplingの非spatial IQNは、各環境でfull rangeをサンプルして分布から直接scoreを作る。
            risk_taus = GenerateTaus(
                N, config_.tau_rule.num_taus, config_.tau_rule.sample_mode,
                0.0f, 1.0f, obs.device(), *rnd);
        } else if (use_vectorized_tau) {
            // spatial UQEでは環境ごとの下限tauを使い、tail meanなら下限から1までをサンプルする。
            auto lower = tau_tensor.reshape({ N });
            if (config_.uqe_use_tail_mean) {
                risk_taus = GenerateTaus(
                    lower, 1.0f, config_.tau_rule.num_taus, config_.tau_rule.sample_mode, *rnd);
            } else {
                // point UQEは基準分位そのものを使う。full query併用時は重複計算を避けて1点だけ渡す。
                const int64_t point_count = config_.full_distribution_query.enabled ? 1 : config_.tau_rule.num_taus;
                risk_taus = lower.unsqueeze(1).expand({ N, point_count });
            }
        } else if (config_.uqe_use_tail_mean) {
            // scalar tail meanは[effective_tau, 1]を問い合わせる。
            risk_taus = GenerateTaus(
                N, config_.tau_rule.num_taus, config_.tau_rule.sample_mode,
                effective_tau, 1.0f, obs.device(), *rnd);
        } else {
            // scalar point UQEはeffective_tauだけを問い合わせる。
            const int64_t point_count = config_.full_distribution_query.enabled ? 1 : config_.tau_rule.num_taus;
            risk_taus = torch::full(
                { N, point_count }, effective_tau,
                torch::TensorOptions().dtype(torch::kFloat32).device(obs.device()));
        }
        risk_num_taus = risk_taus.size(1);

        // 可視化・統計用のfull rangeはrisk queryと混ぜず、後段で切り分けられる順序で連結する。
        torch::Tensor iqn_taus = risk_taus;
        if (config_.full_distribution_query.enabled) {
            const auto& full_rule = config_.full_distribution_query.tau_rule;
            auto full_taus = GenerateTaus(
                N, full_rule.num_taus, full_rule.sample_mode,
                0.0f, 1.0f, obs.device(), *rnd);
            iqn_taus = torch::cat({ risk_taus, full_taus }, 1);
        }

        ANET_PROFILE_SCOPE_NEXT(forward);
        out = ForwardForActionWithTaus(obs, iqn_taus, network, sink);
    } else {
        ANET_PROFILE_SCOPE_NEXT(forward);
        out = ForwardForAction(obs, network, sink);
    }
    // Headのqは連結した全tausの平均になるため、full query有効時は使わず各領域から平均を再計算する。
    ANET_PROFILE_SCOPE_NEXT(split_outputs);
    auto q_values = out.At("q");
    auto q_quantiles = out.At("q_dist");
    torch::Tensor full_q_values;
    torch::Tensor full_q_quantiles;
    if (config_.quantile_mode == "iqn" && config_.full_distribution_query.enabled) {
        full_q_quantiles = q_quantiles.slice(2, risk_num_taus);
        q_quantiles = q_quantiles.slice(2, 0, risk_num_taus);
        q_values = q_quantiles.mean(2);
        full_q_values = full_q_quantiles.mean(2);
    }

    // IQNのrisk部分は問い合わせたい分位だけで構成済みなので、その平均をそのまま行動scoreにする。
    // QRは固定quantile列から基準分位またはupper-tailを抽出する既存処理を維持する。
    torch::Tensor uqe_values;
    if (config_.quantile_mode == "iqn") {
        uqe_values = q_values;
    } else if (use_vectorized_tau) {
        // 学習時: バッチごとに異なるTauが渡されている場合（hompsonSampling向け）
        uqe_values = MakeVectorizedUQEValues(tau_tensor, q_quantiles);
    } else {
        // Eval時 または 学習時(固定Tau): スカラーTauを使う
        uqe_values = MakeUQEValues(effective_tau, q_quantiles);
    }
    auto uqe_action_values = uqe_values.argmax(1);

    // risk-biased scoreでgreedy actionを決めた後、通常またはspatialなepsilon探索を適用する。
    const int64_t N = q_values.sizes()[0];      // shape 読み取りは TensorOptions 経由で同期を回避
    const int64_t A = q_values.sizes()[1];
    torch::Tensor actions;
    if (IsSpatialExplorationEnabled() && !greedy_only) {
        actions = MakeEpsilonGreedyAction(uqe_action_values, GetSpatialEpsilonTensor(N, q_values.device(), true), N, A, rnd);
    } else {
        actions = MakeEpsilonGreedyAction(uqe_action_values, effective_epsilon, N, A, rnd);
    }

    // q_quantilesは行動選択に使ったrisk分布、full_q_quantilesは独立した観測用分布として公開する。
    // どちらも同一forward由来なので、QValuePanelやmetricのための追加forwardは発生しない。
    auto action_info = MakeActionInfo(actions, q_values, q_quantiles);
    action_info->GetAuxData()["uqe_values"] = uqe_values.detach();
    if (full_q_quantiles.defined()) {
        action_info->GetAuxData()["full_q_values"] = full_q_values.detach();
        action_info->GetAuxData()["full_q_quantiles"] = full_q_quantiles.detach();
    }
    return action_info;
}

std::shared_ptr<DQNActionInfo> UQEActionPolicy::SelectAction(const anet::TensorDict& obs, bool greedy_only,
    std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd, const anet::TraceSink& sink) const
{
    if (IsSpatialExplorationEnabled()) {
        const int64_t N = obs.Size(0);
        auto tau_tensor = GetSpatialTauTensor(N, obs.device()).view({ N, 1 });
        return MakeUQEActionInfo(0.0f, tau_tensor, obs, greedy_only, network, rnd, sink);
    }
    return MakeUQEActionInfo(current_uqe_tau_, torch::Tensor(), obs, greedy_only, network, rnd, sink);
}


// ======================================================
// ThompsonSamplingActionPolicy
// ======================================================

anet::rl::dqn::ThompsonSamplingActionPolicy::ThompsonSamplingActionPolicy(
    const ActionPolicyConfig& config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device)
    : UQEActionPolicy(config, enable_spatial_exploration, num_envs, device)
{
    ;
}

void anet::rl::dqn::ThompsonSamplingActionPolicy::OnLearn(const StepCounts& counts)
{
    if (IsSpatialExplorationEnabled()) return;
    UpdateEpsilon(counts.exp_step, true);
    //UpdateTau(counts.exp_step);
}

std::shared_ptr<DQNActionInfo> ThompsonSamplingActionPolicy::SelectAction(const anet::TensorDict& obs, bool greedy_only,
    std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd, const anet::TraceSink& sink) const
{
    // ランダムな Tau をバッチサイズ分生成 (N, 1)
    const int64_t N = obs.Size(0);
    auto device = obs.device();
    if (IsSpatialExplorationEnabled()) {
        auto tau_tensor = GetSpatialTauTensor(N, device).view({ N, 1 });
        return MakeUQEActionInfo(0.0f, tau_tensor, obs, greedy_only, network, rnd, sink);
    }

    if (config_.quantile_mode == "iqn") {
        return MakeUQEActionInfo(
            0.0f, torch::Tensor(), obs, greedy_only, network, rnd, sink,
            /*iqn_use_full_range=*/true);
    }

    auto gen = rnd->GetTorchGenerator(device);
    auto tau_tensor = torch::rand({ N, 1 }, gen, torch::TensorOptions().device(device));

    // tau_tensor(ランダム)でUQE適用
    return MakeUQEActionInfo(0.0f, tau_tensor, obs, greedy_only, network, rnd, sink);
}


// ======================================================
// Actor
// ======================================================

Actor::Actor(std::shared_ptr<ActionPolicy> policy,
    std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm,
    std::shared_ptr<ActionContext> context,
    std::shared_ptr<std::shared_mutex> mutex,
    std::shared_ptr<anet::nn::Network> network,
    std::shared_ptr<anet::nn::Network> src_network,
    bool emit_actor_q_hint,
    std::optional<anet::ProfiledValueConfig<step_t>> snapshot_sync_interval,
    bool emit_snapshot_metrics)
    : policy_(std::move(policy)), obs_norm_(std::move(obs_norm)), context_(std::move(context)), mutex_(std::move(mutex))
    , network_(std::move(network)), src_network_(std::move(src_network)), emit_actor_q_hint_(emit_actor_q_hint)
    , emit_snapshot_metrics_(emit_snapshot_metrics)
{
    if (snapshot_sync_interval.has_value()) {
        snapshot_sync_interval_.emplace(std::move(*snapshot_sync_interval));
    }
}

void Actor::CopySourceNetwork() const
{
    if (network_ == src_network_) {
        return;
    }

    // Learner更新と同じAgent mutexでsourceを保護してsnapshotへ複製する。
    std::shared_lock<std::shared_mutex> lock(*mutex_);
    src_network_->CopyTo(*network_);
    network_->eval();
}

void Actor::UpdateSnapshot(const StepCounts& step) const
{
    if (!snapshot_sync_interval_.has_value()) {
        return;
    }

    // exp_stepで現在周期を評価してから、強制同期後のage基準を現在actionへ合わせる。
    snapshot_sync_interval_->Update(step.exp_step);
    if (reset_snapshot_age_on_next_action_) {
        last_snapshot_sync_train_step_ = step.train_step;
        reset_snapshot_age_on_next_action_ = false;
    }

    // 現在ageが評価済み周期へ到達したaction境界でsnapshotを更新する。
    const auto snapshot_age = step.train_step - last_snapshot_sync_train_step_;
    if (snapshot_age >= snapshot_sync_interval_->Value()) {
        CopySourceNetwork();
        last_snapshot_sync_train_step_ = step.train_step;
    }
}

std::shared_ptr<anet::rl::BatchActionInfo> Actor::MakeAction(const StepCounts& step, const anet::rl::BatchState& state) const
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard ng;

    // action forwardより前にTrain Actor network snapshotを必要な場合だけ同期する。
    UpdateSnapshot(step);

    // FrameStacking
    auto obs = state.obs;
    if (context_ != nullptr) {
        obs = context_->PushObservation(state);
    }
    ANET_LOG_DEBUG("obs=" << obs.ToDefString());
    //ANET_LOG_DEBUG("obs=" << obs.ToString());

    // Observation正規化
    auto norm_obs = obs;
    if (obs_norm_ != nullptr) {
        norm_obs = obs_norm_->Normalize(obs);
    }
    //norm_obs = norm_obs.to(torch::kCPU);
    ANET_LOG_DEBUG("norm_obs=" << norm_obs.ToDefString());
    //ANET_LOG_DEBUG("norm_obs=" << norm_obs.ToString());

    // 行動選択
    auto rnd = context_->GetRandomGenerator();
    anet::TensorDict trace;
    anet::TraceSink sink = anet::rl::MakeActionTraceSink(trace);
    std::shared_ptr<DQNActionInfo> act_info;
    if (network_ != src_network_) {
        // Clone済み: 自分専用のネットワークなので排他不要
        act_info = policy_->SelectAction(norm_obs, false, network_, rnd, sink);
    } else {
        // Clone無し（直列モード）: Learnerの更新と競合しないようSharedLock
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        act_info = policy_->SelectAction(norm_obs, false, network_, rnd, sink);
    }

    // 学習Actorだけが、既存forwardの平均Qから初期優先度用ヒントをpackする。
    if (emit_actor_q_hint_) {
        const auto q_it = act_info->GetAuxData().find("q_values");
        if (q_it == act_info->GetAuxData().end() || !q_it->second.defined()) {
            ANET_SYSTEM_ERROR("actor_approx requires ActionPolicy aux[q_values].");
        }
        const auto& q_values = q_it->second;
        auto action = act_info->GetAction();
        if (action.device() != q_values.device()) {
            ANET_SYSTEM_ERROR("actor_approx action and q_values must share a device. action="
                << action.device() << " q_values=" << q_values.device());
        }
        auto q_sa = q_values.gather(1, action.to(torch::kInt64).unsqueeze(1)).squeeze(1);
        auto state_value = std::get<0>(q_values.max(1));
        auto packed = PackActorQHint(q_sa, state_value);
        act_info->SetReplayInitialPriorityHint(ReplayInitialPriorityHint(std::move(packed)));
    }

    // AuxData の詰め込み
    // 現在actionがepisode開始直後かを、device転送せず診断用auxへ保持する。
    act_info->GetAuxData()["episode_start"] = state.episode_start;
    act_info->GetAuxData()["raw_obs"] = anet::rl::ToUnifiedObservation(obs);
    if (obs_norm_ != nullptr) {
        act_info->GetAuxData()["norm_obs"] = anet::rl::ToUnifiedObservation(norm_obs);
    }
    anet::rl::AppendTraceAux(act_info->GetAuxData(), trace);

    // DefaultDQNでは無効時もNaN、周期snapshot有効時は同期後の現在値を公開する。
    if (emit_snapshot_metrics_) {
        TrainActorSnapshotMetrics metrics;
        if (snapshot_sync_interval_.has_value()) {
            metrics.interval = static_cast<float>(snapshot_sync_interval_->Value());
            metrics.age = static_cast<float>(step.train_step - last_snapshot_sync_train_step_);
        }
        act_info->SetTrainActorSnapshotMetrics(metrics);
    }

    return act_info;
}

void Actor::Sync()
{
    CopySourceNetwork();
    if (snapshot_sync_interval_.has_value()) {
        // Syncはstepを受け取らないため、次actionのtrain_stepを新しいage基準にする。
        reset_snapshot_age_on_next_action_ = true;
    }
}


// ======================================================
// Learner
// ======================================================

Learner::Learner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed,
    std::shared_ptr<ActionPolicy> target_policy, std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed)
    : RandomHolder(target_seed), config_(config), stucker_config_(stucker_config), model_(model), vars_(vars), obs_norm_(std::move(obs_norm))
    , num_envs_(batch_env_spec.num_envs)
    , n_actions_(env_spec.action_spec.GetNumActions())//, state_dim_(env_spec.state_spec.CalcFlattenDim())
    , device_(std::move(device))
    , target_policy_(std::move(target_policy))
{
    // Credit計算
    if (config_.replay_ratio > 0) {
        // RRモード:  U = (N * RR) / B
        earned_credit_ = static_cast<float>(num_envs_) * config_.replay_ratio / config_.replay_batch_size;
    } else {
        // Intervalモード: U = 1.0 / Interval
        earned_credit_ = 1.0f / static_cast<float>(std::max(1, config_.update_interval));
    }

    LOG::info() << "Learner: U = " << earned_credit_;

    if (device_.is_cuda()) {
        per_priority_copy_stream_ = at::cuda::getStreamFromPool(false);
    }
}

std::optional<float> Learner::GetScalar(const std::string& key, int64_t index) const
{
    if (key.find(ReplayBuffer::kKeyPrefix) == 0 && replay_buffer_ != nullptr) {
        return replay_buffer_->GetScalar(key);
    }

    return std::nullopt;
}

std::optional<torch::Tensor> Learner::GetTensor(const std::string& key, int64_t index) const
{
    if (key.find(ReplayBuffer::kKeyPrefix) == 0 && replay_buffer_ != nullptr)
        return replay_buffer_->GetTensor(key);

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> Learner::GetTensorVector(const std::string& key, int64_t index) const
{
    if (key.find(ReplayBuffer::kKeyPrefix) == 0 && replay_buffer_ != nullptr)
        return replay_buffer_->GetTensorVector(key);

    return std::nullopt;
}

void Learner::SetupOptimizer()
{
    //auto opts = torch::optim::AdamOptions(config_.alpha).eps(config_.adam_eps);
    auto opts = torch::optim::AdamWOptions(config_.alpha).weight_decay(config_.weight_decay).eps(config_.adam_eps);
    LOG::verbose() << "Learner: lr=" << opts.lr() << " weight_decay=" << opts.weight_decay()
        << " eps=" << opts.eps() << " fused=" << config_.use_fused_optimizer;
    if (config_.use_fused_optimizer) {
        this->optimizer_ = std::make_unique<anet::FusedAdamW>(model_.GetOnlineParameters(), opts);
    } else {
        this->optimizer_ = std::make_unique<torch::optim::AdamW>(model_.GetOnlineParameters(), opts);
    }
}

void Learner::SetupReplayBuffer(const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, seed_t seed)
{
    const auto initial_priority_mode = ParseReplayInitialPriorityMode(config_);
    ValidateReplayPriorityConfig(config_, initial_priority_mode);
    anet::rl::ReplayBufferConfig rep_config{};
    rep_config.capacity = config_.replay_capacity;
    rep_config.gamma = config_.gamma;
    rep_config.n_step = config_.use_n_step ? config_.n_step : 1;

    if (config_.use_per) {
        rep_config.sampler_type = anet::rl::ReplaySamplerType::PRIORITIZED;
        rep_config.per_alpha = config_.per_alpha;
        rep_config.per_initial_priority = config_.per_initial_priority;
        rep_config.per_initial_priority_mode = initial_priority_mode;
        vars_.per_beta = config_.per_beta_start;
    } else {
        rep_config.sampler_type = ReplaySamplerType::UNIFORM;
    }
    if (stucker_config_.has_value()) {
        //rep_config.use_stacker = stucker_config_->use_stacker;
        rep_config.stack_count = stucker_config_->use_stacker ? stucker_config_->stack_count : 1;
        rep_config.stack_keys = stucker_config_->stack_keys;
    } else {
        rep_config.stack_count = 1;
    }
    
    //const ReplayBufferConfig& config, const EnvSpec& env_spec, int64_t num_envs, torch::Device storage_device, bool pin_memory, std::optional<uint64_t> seed)

    //this->replay_buffer_ = anet::rl::CreateReplayBuffer(rep_config, env_spec, batch_env_spec.num_envs, device_, false, seed);
    //this->replay_buffer_ = anet::rl::CreateReplayBuffer(rep_config, env_spec, batch_env_spec.num_envs, device_, true, seed);
    std::unique_ptr<InitialPriorityEstimator> estimator;
    if (initial_priority_mode == ReplayInitialPriorityMode::ACTOR_APPROX) {
        estimator = CreateInitialPriorityEstimator(config_);
    }
    this->replay_buffer_ = anet::rl::CreateReplayBuffer(
        rep_config, env_spec, batch_env_spec.num_envs, torch::kCPU, false, seed, std::move(estimator));
    if (config_.use_rb_prefetch) {
        this->replay_buffer_ = std::make_shared<anet::rl::PrefetchingReplayBuffer>(this->replay_buffer_, device_);
    }
}

NormalizedSampleObservations Learner::NormalizeSampleObservations(const anet::rl::ExperienceSamples& samples) const
{
    auto obs = samples.obs;
    auto next_obs = samples.next_state.next_obs;
    if (obs_norm_) {
        // 統計更新は Agent 側の収集フェーズで行うためここでは適用のみ
        obs = obs_norm_->Normalize(obs);
        next_obs = obs_norm_->Normalize(next_obs);
    }
    return { obs, next_obs };
}

OptimizerStepResult Learner::Optimize(const torch::Tensor& loss)
{
	ANET_PROFILE_FUNC();
	
    auto parameters = model_.GetOnlineParameters();

    OptimizerStepResult result;
    result.grad_clip_tau = config_.use_grad_clip ? config_.grad_clip_tau : std::numeric_limits<float>::infinity();

    ANET_PROFILE_SCOPE(zero_grad);
    optimizer_->zero_grad();

    // ------------------------------------------------------------
    // AMP + BF16: scalerを使わず、旧QRLearnerと同じ順序で更新する
    // ------------------------------------------------------------
    if (config_.use_amp && config_.use_amp_bf16) {
        ANET_PROFILE_SCOPE_NEXT(backward);
        loss.backward();

        // 勾配ノルムとclipをforeach化し、CPU同期をoptimizer step後まで遅延する
        ANET_PROFILE_SCOPE_NEXT(grad_norm);
        auto grads = anet::CollectDefinedGrads(parameters);
        result.grad_norm = std::nullopt;
        result.grad_norm_tensor = anet::ForeachGradNorm(grads);

        if (config_.use_grad_clip) {
            ANET_PROFILE_SCOPE_NEXT(grad_clip);
            anet::ForeachClipGradNorm_(grads, result.grad_norm_tensor, config_.grad_clip_tau);

            ANET_PROFILE_SCOPE_NEXT(optimizer_step);
            optimizer_->step();
            return result;
        } else {
            ANET_PROFILE_SCOPE_NEXT(optimizer_step);
            optimizer_->step();
            return result;
        }
    }

    // ------------------------------------------------------------
    // AMP + FP16: scale -> backward -> unscale -> clip/norm -> step/update
    // ------------------------------------------------------------
    if (config_.use_amp) {
        ANET_PROFILE_SCOPE_NEXT(backward);
        grad_scaler_.Scale(loss).backward();

        ANET_PROFILE_SCOPE_NEXT(unscale);
        grad_scaler_.Unscale_(*optimizer_);

        // Unscale後の勾配に対してノルム計算とclipを行う
        auto grads = anet::CollectDefinedGrads(parameters);
        ANET_PROFILE_SCOPE_NEXT(grad_norm);
        result.grad_norm = std::nullopt;
        result.grad_norm_tensor = anet::ForeachGradNorm(grads);

        if (config_.use_grad_clip) {
            ANET_PROFILE_SCOPE_NEXT(grad_clip);
            anet::ForeachClipGradNorm_(grads, result.grad_norm_tensor, config_.grad_clip_tau);

            ANET_PROFILE_SCOPE_NEXT(scaler_step);
            grad_scaler_.Step(*optimizer_);

            ANET_PROFILE_SCOPE_NEXT(scaler_update);
            grad_scaler_.Update();
            return result;
        } else {
            ANET_PROFILE_SCOPE_NEXT(scaler_step);
            grad_scaler_.Step(*optimizer_);

            ANET_PROFILE_SCOPE_NEXT(scaler_update);
            grad_scaler_.Update();
            return result;
        }
    }

    // ------------------------------------------------------------
    // FP32: grad_norm_tensorをTensorのまま保持し、CPU同期を避けて手動clipする
    // ------------------------------------------------------------
    ANET_PROFILE_SCOPE_NEXT(backward);
    loss.backward();

#if ANET_ENABLE_TENSOR_NAN_CHECK
    ANET_PROFILE_SCOPE_NEXT(nan_check);
    for (auto& param : parameters) {
        if (param.grad().defined()) {
            ANET_ASSERT_NAN(param.grad());
        }
    }
    ANET_PROFILE_SCOPE_NEXT(grad_norm);
#else
    ANET_PROFILE_SCOPE_NEXT(grad_norm);
#endif

    auto grads = anet::CollectDefinedGrads(parameters);

    // BatchUpdateResult側でgrad_norm_tensorへフォールバックさせるため、FP32ではgrad_normを値化しない
    result.grad_norm = std::nullopt;
    result.grad_norm_tensor = anet::ForeachGradNorm(grads);

    if (config_.use_grad_clip) {
        ANET_PROFILE_SCOPE_NEXT(grad_clip);

        // clip_grad_norm_のCPU同期を避けるため、foreach演算のままスケールを適用する
        anet::ForeachClipGradNorm_(grads, result.grad_norm_tensor, config_.grad_clip_tau);

        ANET_PROFILE_SCOPE_NEXT(optimizer_step);
        optimizer_->step();
        return result;
    }

    ANET_PROFILE_SCOPE_NEXT(optimizer_step);
    optimizer_->step();
    return result;
}

PerPriorityUpdatePending Learner::PreparePerPriorityUpdate(const anet::rl::ExperienceSamples& samples, const torch::Tensor& td_error)
{
    PerPriorityUpdatePending pending;

    if (!config_.use_per) {
        return pending;
    }

    torch::NoGradGuard grad_guard;
    const int64_t batch_size = td_error.size(0);
    pending.enabled = true;
    pending.per_minibatch_size = batch_size;
    if (samples.is_weights.defined()) {
        pending.per_is_weights = samples.is_weights;
    }
    if (samples.per_priority_sources.defined()) {
        auto sources = samples.per_priority_sources.to(torch::kInt8).to(torch::kCPU).contiguous();
        auto initial_flags = sources.ne(static_cast<int8_t>(ReplayPrioritySource::NONE))
            & sources.ne(static_cast<int8_t>(ReplayPrioritySource::LEARNER_UPDATED));
        ANET_ASSERT_SHAPE(initial_flags, { batch_size });
        pending.per_sample_initial_count = initial_flags.to(torch::kFloat32).sum();
        pending.per_sample_fixed_initial_count = sources.eq(
            static_cast<int8_t>(ReplayPrioritySource::FIXED_INITIAL)).to(torch::kFloat32).sum();
        pending.per_sample_max_initial_count = sources.eq(
            static_cast<int8_t>(ReplayPrioritySource::MAX_INITIAL)).to(torch::kFloat32).sum();
        pending.per_sample_actor_initial_count = sources.eq(
            static_cast<int8_t>(ReplayPrioritySource::ACTOR_INITIAL)).to(torch::kFloat32).sum();
    }

    {
        ANET_PROFILE_SCOPE_FULL(indices_cpu, "Learner::UpdatePerPriorities.indices_cpu");

        if (!samples.replay_item_keys.device().is_cpu()) {
            ANET_SYSTEM_ERROR("Learner::PreparePerPriorityUpdate expected samples.replay_item_keys on CPU, actual="
                << samples.replay_item_keys.device());
        }
        ANET_ASSERT_DEVICE_CPU(samples.replay_item_keys);
        ANET_ASSERT_SHAPE(samples.replay_item_keys, { batch_size });
        ANET_ASSERT_DTYPE(samples.replay_item_keys, torch::kInt64);
        auto indices_tensor_cpu = samples.replay_item_keys.contiguous();
        auto indices_ptr = indices_tensor_cpu.data_ptr<int64_t>();
        pending.indices.assign(indices_ptr, indices_ptr + batch_size);
    }

    // TD error由来のpriorityとclip件数をGPU上で確定し、1本のD2Hへpackする。
    auto priority_result = MakePerRawPriorityBatch(
        td_error.detach(), config_.per_eps, config_.use_per_prio_clip, config_.per_prio_clip_value);
    auto raw_priorities = priority_result.priorities.contiguous();
    ANET_ASSERT_SHAPE(raw_priorities, { batch_size });
    ANET_ASSERT_NAN(raw_priorities);
    auto packed_readback = torch::cat({
        raw_priorities,
        priority_result.clipped_count.to(raw_priorities.options()).reshape({ 1 }),
    }).contiguous();
    ANET_ASSERT_SHAPE(packed_readback, { batch_size + 1 });
    ANET_ASSERT_NAN(packed_readback);

    if (packed_readback.is_cuda() && per_priority_copy_stream_.has_value()) {
        ANET_PROFILE_SCOPE_FULL(launch, "Learner::PerPriorityD2H.launch");
        const auto producer_stream = at::cuda::getCurrentCUDAStream();
        pending.priority_readback = anet::transfer::HostReadback(
            std::move(packed_readback),
            per_priority_copy_stream_.value(),
            producer_stream,
            per_priority_event_recycler_);
    } else {
        pending.priority_readback = anet::transfer::HostReadback(
            packed_readback.device().is_cpu() ? std::move(packed_readback) : packed_readback.cpu());
    }

    return pending;
}

PerPriorityUpdateInfo Learner::ApplyPerPriorityUpdate(PerPriorityUpdatePending pending)
{
    PerPriorityUpdateInfo info;
    if (!pending.enabled) {
        return info;
    }

    const int64_t batch_size = pending.per_minibatch_size;
    info.per_minibatch_size = batch_size;
    info.per_is_weights = pending.per_is_weights;
    info.per_sample_initial_count = pending.per_sample_initial_count;
    info.per_sample_fixed_initial_count = pending.per_sample_fixed_initial_count;
    info.per_sample_max_initial_count = pending.per_sample_max_initial_count;
    info.per_sample_actor_initial_count = pending.per_sample_actor_initial_count;

    {
        ANET_PROFILE_SCOPE_FULL(wait, "Learner::PerPriorityD2H.wait");
        pending.priority_readback.Wait();
    }

    std::vector<float> priorities_vec;
    {
        ANET_PROFILE_SCOPE_FULL(vector_copy, "Learner::PerPriorityD2H.vector_copy");

        auto packed_cpu = pending.priority_readback.pinned_result.to(torch::kFloat32).contiguous();
        ANET_ASSERT_DEVICE_CPU(packed_cpu);
        ANET_ASSERT_SHAPE(packed_cpu, { batch_size + 1 });

        // pack末尾の件数をCPU int64 scalarへ戻し、先頭B要素だけをReplayBufferへ渡す。
        const int64_t clipped_count = static_cast<int64_t>(packed_cpu[batch_size].item<float>());
        info.per_clipped_count = torch::tensor(
            clipped_count, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto priorities_cpu = packed_cpu.narrow(0, 0, batch_size).contiguous();

        info.per_priorities = priorities_cpu;
        auto priorities_ptr = priorities_cpu.data_ptr<float>();
        priorities_vec.assign(priorities_ptr, priorities_ptr + batch_size);
    }

    pending.priority_readback.RetireEvents(per_priority_event_recycler_);

    {
        ANET_PROFILE_SCOPE_FULL(update_tree, "Learner::UpdatePerPriorities.update_tree");
        info.per_update_result = replay_buffer_->UpdatePriorities(pending.indices, priorities_vec);
    }

    return info;
}

PerPriorityUpdateInfo Learner::UpdatePerPriorities(const anet::rl::ExperienceSamples& samples, const torch::Tensor& td_error)
{
    auto pending = PreparePerPriorityUpdate(samples, td_error);
    return ApplyPerPriorityUpdate(std::move(pending));
}

torch::Tensor Learner::TransformH(const torch::Tensor& x) const
{
    // h 空間へ圧縮し、大きなBellmanターゲットの影響を抑える。
    return dqn::TransformH(x, config_.tbo_epsilon);
}

torch::Tensor Learner::TransformHInv(const torch::Tensor& x) const
{
    // h 空間の値を、報酬と足し合わせる前に実空間へ戻す。
    return dqn::TransformHInv(x, config_.tbo_epsilon);
}

std::shared_ptr<anet::rl::dqn::BatchUpdateResult> Learner::MakeBatchUpdateResult(
    const torch::Tensor& loss,
    const torch::Tensor& td_error,
    const OptimizerStepResult& opt_result,
    const torch::Tensor& max_q,
    const torch::Tensor& q_sa,
    const PerPriorityUpdateInfo& per_info,
    const torch::Tensor& q_std,
    const torch::Tensor& q_gap,
    const torch::Tensor& q_gap_rel) const
{
    auto result = std::make_shared<BatchUpdateResult>();
    result->loss = loss.detach();
    result->td_error = td_error.detach();
    result->grad_norm = opt_result.grad_norm;
    result->grad_norm_tensor = opt_result.grad_norm_tensor;
    result->grad_clip_ratio = opt_result.grad_clip_ratio;
    result->grad_clip_tau = opt_result.grad_clip_tau;
    result->max_q = max_q;
    result->q_sa = q_sa;
    if (config_.use_tbo) {
        result->max_q_real = TransformHInv(max_q);
        result->q_sa_real = TransformHInv(q_sa);
    }
    result->q_std = q_std;
    result->q_gap = q_gap;
    result->q_gap_rel = q_gap_rel;
    if (per_info.per_minibatch_size > 0) {
        result->per_minibatch_size = per_info.per_minibatch_size;
        result->per_clipped_count = per_info.per_clipped_count;
        result->per_priorities = per_info.per_priorities;
        result->per_is_weights = per_info.per_is_weights;
        result->per_sample_initial_count = per_info.per_sample_initial_count;
        result->per_sample_fixed_initial_count = per_info.per_sample_fixed_initial_count;
        result->per_sample_max_initial_count = per_info.per_sample_max_initial_count;
        result->per_sample_actor_initial_count = per_info.per_sample_actor_initial_count;
        result->per_update_result = per_info.per_update_result;
    }
    return result;
}

void Learner::UpdateTargetNetwork(step_t step)
{
    ANET_PROFILE_FUNC();
    model_.UpdateTarget(step);
}

void Learner::UpdatePerBeta(step_t step)
{
    ANET_PROFILE_FUNC();

    if (!config_.use_per) return;
    if (config_.per_beta_step <= 0) return;
    if (step < config_.per_beta_step) {
        float progress = static_cast<float>(step) / static_cast<float>(config_.per_beta_step);
        vars_.per_beta = config_.per_beta_start + progress * (config_.per_beta_end - config_.per_beta_start);
    } else {
        vars_.per_beta = config_.per_beta_end;
    }
}

bool Learner::CanUpdate(step_t exp_step)
{
    // warmup
    if (config_.update_warmup_steps > 0 && exp_step < config_.update_warmup_steps)
        return false;

    // ReplayBufferのサイズがminibatchサイズに満たない場合はスキップ（N-STEPであり得る）
    // 一度 minibatch サイズに到達した後は sampleable count が閾値未満へ戻らない前提で、
    // 高コストな Size() 再走査を避ける。
    if (!has_enough_replay_samples_) {
        if (replay_buffer_->Size() < config_.replay_batch_size)
            return false;
        has_enough_replay_samples_ = true;
    }

    return true;
}

void Learner::ValidateDeviceSamples(const anet::rl::ExperienceSamples& samples, int64_t batch_size) const
{
    ANET_ASSERT_DEVICE(samples.obs, device_);
    ANET_ASSERT_DEVICE(samples.actions, device_);
    ANET_ASSERT_DEVICE(samples.target_returns, device_);
    ANET_ASSERT_DEVICE(samples.next_state.next_obs, device_);
    ANET_ASSERT_DEVICE(samples.next_state.terminals, device_);
    ANET_ASSERT_DEVICE(samples.n_steps, device_);
    ANET_ASSERT_DEVICE_CPU(samples.replay_item_keys);
    ANET_ASSERT_SHAPE(samples.actions, { batch_size });    // 離散アクション
    ANET_ASSERT_SHAPE(samples.target_returns, { batch_size });
    ANET_ASSERT_SHAPE(samples.next_state.terminals, { batch_size });
    ANET_ASSERT_SHAPE(samples.n_steps, { batch_size });
    ANET_ASSERT_SHAPE(samples.replay_item_keys, { batch_size });
    ANET_ASSERT_DTYPE(samples.actions, torch::kInt64);    // 離散アクション
    ANET_ASSERT_DTYPE(samples.target_returns, torch::kFloat32);
    ANET_ASSERT_DTYPE(samples.next_state.terminals, torch::kBool);
    ANET_ASSERT_DTYPE(samples.n_steps, torch::kInt64);
    ANET_ASSERT_DTYPE(samples.replay_item_keys, torch::kInt64);
}

anet::rl::BatchUpdateResultList
Learner::UpdateFromBatch(const anet::rl::StepCounts& counts, const anet::rl::BatchExperience& experiences)
{
    ANET_PROFILE_FUNC();

    // ReplayBuffer へ push
    replay_buffer_->Push(experiences);

    // 戻り値のUpdateResultを準備
    BatchUpdateResultList result_list;

    // Update不可なら空の結果を返す
    if (!CanUpdate(counts.exp_step)) {
        return result_list;  // 空配列
    }

    // Credit加算
    update_credit_ += earned_credit_;

    // update_credit が十分な間、学習ループを回す
    while (update_credit_ >= 1.0f) {
        const int64_t B = config_.replay_batch_size;
        //const int S = state_dim_;

        // Sample
        float current_beta = config_.use_per ? vars_.per_beta : 0.0f;
        ExperienceSamples samples;
        replay_buffer_->Sample(samples, B, current_beta);
        ExperienceSamples dev_samples = samples.To(device_);
//LOG::info() << "\n========\nUpdateFromBatch() exp_step=" << counts.exp_step << "\n========";
//LOG::verbose() << "indexes=" << anet::ToString(samples.indices);
//LOG::verbose() << "samples=" << samples.ToString();

        //ANET_LOG_DEBUG("raw_samples.obs=" << anet::ToDefString(raw_samples.obs));
        //ANET_LOG_DEBUG("raw_samples.next_states.obs=" << anet::ToDefString(raw_samples.next_states.obs));
//LOG::info() << "[ " << counts.exp_step << "] samples=\n" << samples.ToString() << "\n[" << counts.exp_step << "] ---- END OF SAMPLES ----";
//if (counts.exp_step > 120) {
//    LOG::info() << "DEUBG";
//}

        // Check shapes & dtypes
        ValidateDeviceSamples(dev_samples, B);

        // 固有処理呼び出し
        //auto samples = raw_samples.FlattenStates();
        auto result = UpdateFromSamples(dev_samples);
        result_list.push_back(result);

        // 更新後処理
        UpdateTargetNetwork(vars_.learn_step);
        UpdatePerBeta(counts.exp_step);

        // カウント系更新
        vars_.learn_step++;
        update_credit_ -= 1.0f;
    }

    return result_list;
}

int64_t Learner::Save(OutputArchive& archive) const
{
    int64_t size = 0;
    size += archive.WriteTorchObject(*optimizer_);
    return size;
}

int64_t Learner::Load(InputArchive& archive)
{
    int64_t size = 0;
    size += archive.ReadTorchObject(*optimizer_);
    return size;
}


// ======================================================
// QuantileLearnerBase
// ======================================================

QuantileLearnerBase::QuantileLearnerBase(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed,
    std::shared_ptr<ActionPolicy> target_policy, std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed)
    : Learner(config, model, vars, std::move(obs_norm), batch_env_spec, env_spec, std::move(device), replay_seed, std::move(target_policy), stucker_config, target_seed)
{
}

torch::Tensor QuantileLearnerBase::GatherActionQuantiles(const torch::Tensor& quantiles, const torch::Tensor& actions) const
{
    const int64_t B = actions.size(0);
    const int64_t N = quantiles.size(2);

    // actions(B)をquantile次元にbroadcastし、Z(s, a)だけを取り出す
    torch::Tensor idx_actions = actions.view({ B, 1, 1 }).expand({ B, 1, N });
    ANET_ASSERT_SHAPE(idx_actions, { B, 1, N });
    ANET_ASSERT_NAN(idx_actions);

    auto selected = quantiles.gather(1, idx_actions).squeeze(1);
    ANET_ASSERT_SHAPE(selected, { B, N });
    ANET_ASSERT_NAN(selected);
    return selected;
}

torch::Tensor QuantileLearnerBase::SelectTargetActions(const anet::TensorDict& next_obs)
{
    // Double DQN有効時はPolicyNetで行動選択し、価値評価は呼び出し側でTargetNetを使う
    auto network = (config_.use_double_dqn) ? model_.GetOnlineNetwork() : model_.GetTargetNetwork();
    auto target_action_info = target_policy_->SelectAction(next_obs, /*greedy_only=*/true, network, this->GetRandomGenerator());
    return target_action_info->GetAction(device_);
}

torch::Tensor QuantileLearnerBase::CalcTargetQuantiles(const anet::rl::ExperienceSamples& samples, const torch::Tensor& next_dist) const
{
    const int64_t B = samples.target_returns.size(0);
    const int64_t N = next_dist.size(1);

    auto returns = samples.target_returns.view({ B, 1 });
    auto not_terminal = (1.0f - samples.next_state.terminals.to(torch::kFloat32)).view({ B, 1 });
    ANET_ASSERT_SHAPE(returns, { B, 1 });
    ANET_ASSERT_SHAPE(not_terminal, { B, 1 });

    // N-step target: r^(n) + gamma^n * (1 - done) * Z_target(s', a*)
    auto gamma_n = torch::pow(config_.gamma, samples.n_steps.to(torch::kFloat32)).view({ B, 1 });
    auto next = config_.use_tbo ? TransformHInv(next_dist) : next_dist;
    auto raw_target_dist = returns + gamma_n * not_terminal * next;
    auto target_dist = config_.use_tbo ? TransformH(raw_target_dist) : raw_target_dist;
    ANET_ASSERT_SHAPE(target_dist, { B, N });
    ANET_ASSERT_NAN(target_dist);
    return target_dist;
}

QuantileMetrics QuantileLearnerBase::MakeQuantileMetrics(const torch::Tensor& current_dist, const torch::Tensor& q_values_mean) const
{
    QuantileMetrics metrics;
    const int64_t B = current_dist.size(0);
    auto current_dist_metrics = current_dist.detach();
    auto q_values_metrics = q_values_mean.detach();

    // 実際に選択された行動の期待値Qと、全行動中の最大期待値Qを記録する
    metrics.q_sa = current_dist_metrics.mean(1);
    metrics.max_q = std::get<0>(q_values_metrics.max(1));
    ANET_ASSERT_SHAPE(metrics.max_q, { B });

    // 分布の広がりをQ stdとして保持する。N=1は不偏分散が未定義なので明示的に0とする。
    if (current_dist_metrics.size(1) == 1) {
        metrics.q_std = torch::zeros({}, current_dist_metrics.options());
    } else {
        metrics.q_std = current_dist_metrics.std(1).mean();
    }
    ANET_ASSERT_SHAPE(metrics.q_std, {});

    if (n_actions_ >= 2) {
        // top-1/top-2の差を、絶対値と相対値の両方で監視する
        auto top2 = std::get<0>(q_values_metrics.topk(2, 1));
        auto q_best = top2.select(1, 0);
        auto q_second = top2.select(1, 1);
        auto gap_batch = q_best - q_second;
        metrics.q_gap = gap_batch.mean();

        auto denom = q_best.abs() + 1e-6f;
        metrics.q_gap_rel = (gap_batch / denom).mean();
    }

    return metrics;
}

torch::Tensor QuantileLearnerBase::ComputeQuantileHuberLoss(
    const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus, float kappa)
{
    ANET_PROFILE_FUNC();

    const auto B = current_dist.size(0);
    const auto N = current_dist.size(1);
    const auto M = target_dist.size(1);

    ANET_ASSERT_SHAPE(current_dist, { B, N });
    ANET_ASSERT_SHAPE(target_dist, { B, M });

    // current: (B, N) -> (B, N, 1), target: (B, M) -> (B, 1, M)
    // QR-DQNではN==M、IQNではtarget側のサンプル数Mが異なる可能性を許容する
    auto cur = current_dist.unsqueeze(2);
    auto tgt = target_dist.unsqueeze(1);
    ANET_ASSERT_SHAPE(cur, { B, N, 1 });
    ANET_ASSERT_SHAPE(tgt, { B, 1, M });

    // pair-wise差分 δ_ij = z'_j - z_i
    auto diff = tgt - cur;
    ANET_ASSERT_SHAPE(diff, { B, N, M });

    // Huber loss。|δ| < κ では二乗、外側では線形にする
    auto abs_diff = diff.abs();
    auto huber = torch::where(abs_diff < kappa, 0.5f * diff.pow(2), kappa * (abs_diff - 0.5f * kappa));
    ANET_ASSERT_SHAPE(huber, { B, N, M });

    // quantile regressionの重み |τ - I(δ < 0)| を掛ける
    auto indicator = (diff.detach() < 0).to(torch::kFloat);
    auto quantile_weight = torch::abs(taus - indicator);
    ANET_ASSERT_SHAPE(quantile_weight, { B, N, M });

    // target quantile方向を合計し、current quantile方向を平均してサンプル単位のlossにする
    auto loss_per_pair = quantile_weight * huber;
    auto element_wise_loss = loss_per_pair.sum(2).mean(1);
    ANET_ASSERT_SHAPE(element_wise_loss, { B });

    return element_wise_loss;
}

torch::Tensor QuantileLearnerBase::ComputeQuantileHuberLoss(
    const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus) const
{
    return ComputeQuantileHuberLoss(current_dist, target_dist, taus, config_.quantile_huber_kappa);
}

torch::Tensor QuantileLearnerBase::ComputeIqnQuantileHuberLoss(
    const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus, float kappa)
{
    ANET_PROFILE_FUNC();

    const auto B = current_dist.size(0);
    const auto N = current_dist.size(1);
    const auto M = target_dist.size(1);

    ANET_ASSERT_SHAPE(current_dist, { B, N });
    ANET_ASSERT_SHAPE(target_dist, { B, M });
    ANET_ASSERT_SHAPE(taus, { B, N, 1 });

    // Eq. 3 の全 (current tau_i, target tau'_j) ペアを作る。
    auto diff = target_dist.unsqueeze(1) - current_dist.unsqueeze(2);
    ANET_ASSERT_SHAPE(diff, { B, N, M });

    // IQN論文どおりHuberをkappaで正規化し、quantile regression重みを掛ける。
    auto abs_diff = diff.abs();
    auto huber = torch::where(
        abs_diff < kappa,
        0.5f * diff.pow(2),
        kappa * (abs_diff - 0.5f * kappa));
    auto indicator = (diff.detach() < 0).to(torch::kFloat);
    auto loss_per_pair = torch::abs(taus - indicator) * huber / kappa;
    ANET_ASSERT_SHAPE(loss_per_pair, { B, N, M });

    // current側Nをsum、target側Mをmeanし、サンプル単位のlossを返す。
    auto element_wise_loss = loss_per_pair.sum(1).mean(1);
    ANET_ASSERT_SHAPE(element_wise_loss, { B });
    return element_wise_loss;
}


// ======================================================
// TDLearner
// ======================================================


TDLearner::TDLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed,
    std::shared_ptr<ActionPolicy> target_policy, std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed)
    : Learner(config, model, vars, std::move(obs_norm), batch_env_spec, env_spec, device, replay_seed, std::move(target_policy), stucker_config, target_seed)
{
    SetupReplayBuffer(batch_env_spec, env_spec, replay_seed);
    SetupOptimizer();
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
TDLearner::UpdateFromSamples(const anet::rl::ExperienceSamples& samples)
{
    ANET_PROFILE_FUNC();

    const int B = config_.replay_batch_size;
    const int A = n_actions_;
    const auto& target_returns = samples.target_returns;
    const auto& terminals = samples.next_state.terminals;
    const torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;

    auto norm_samples = NormalizeSampleObservations(samples);
    const auto& obs = norm_samples.obs;
    const auto& next_obs = norm_samples.next_obs;

    // 結果変数（スコープ外で宣言）
    torch::Tensor loss;
    torch::Tensor td_error;
    torch::Tensor max_q;
    torch::Tensor q_sa;
    torch::Tensor gap_abs;
    torch::Tensor gap_rel;

    // ============================================================
    // Forward & Loss Calculation (Autocast Scope)
    // ============================================================
    {
        // AMP Guard (Forward計算のみfloat16化)
        anet::Autocast amp_guard(device_, config_.use_amp, amp_dtype);

        // ------------------------------------------------------------
        // Q(s, a)
        // ------------------------------------------------------------
        auto q_out = model_.ForwardOnlineWithTrain(obs);
        auto q_all = q_out.At("q"); // (B,A)
        ANET_ASSERT_SHAPE(q_all, { B, A });

        torch::Tensor idx_actions = samples.actions.view({ B, 1 });   // (B,1)
        ANET_ASSERT_SHAPE(idx_actions, { B, 1 });
        ANET_ASSERT_DTYPE(idx_actions, torch::kInt64);

        q_sa = q_all.gather(1, idx_actions).squeeze(1);          // (B)
        ANET_ASSERT_SHAPE(q_sa, { B });
        ANET_ASSERT_DTYPE(q_sa, torch::kFloat32);

        auto q_metrics = q_all.detach();
        max_q = std::get<0>(q_metrics.max(1));     // (B)

        // Gap Metrics
        if (n_actions_ >= 2) {  // 念の為
            auto top2 = std::get<0>(q_metrics.topk(2, 1));  //  (B, 2) 上位2つのQ値
            auto q_best = top2.select(1, 0);       // 1位 (B)
            auto q_second = top2.select(1, 1);     // 2位 (B)

            // 絶対値差分
            auto gap_batch = q_best - q_second;
            gap_abs = gap_batch.mean();

            // Relative Gap (相対差分: Gap / (|MaxQ| + eps))
            //   Q値は負になることもあるので abs() が必要
            //   学習初期は 0 になるので 1e-6 で割るのを防ぐ
            auto denom = q_best.abs() + 1e-6f;
            gap_rel = (gap_batch / denom).mean();
        }

        // ------------------------------------------------------------
        // max_a' Q(s', a')
        // ------------------------------------------------------------
        torch::Tensor max_next_q;
        {
            torch::NoGradGuard no_grad;

            // 行動 a' を決定
            // DDQN有効の場合は PolicyNetで行動選択、DDQN無効の場合はTargetNetで行動選択
            auto network = (config_.use_double_dqn) ? model_.GetOnlineNetwork() : model_.GetTargetNetwork();

            // target_policy_ に行動を選ばせる
            // ※ここで greedy_only=false にすることで、UQE設定時は楽観的に選ばれる
            auto target_action_info = target_policy_->SelectAction(next_obs, /*greedy_only=*/true, network, this->GetRandomGenerator());
            torch::Tensor next_actions = target_action_info->GetAction(device_);

            // 選んだ行動の価値を TargetNet で評価する (価値評価は常に TargetNet)
            auto next_q_out = model_.ForwardTarget(next_obs);
            auto next_q_target = next_q_out.At("q"); // (B, A)
            ANET_ASSERT_SHAPE(next_q_target, { B, A });

            torch::Tensor next_actions_b = next_actions.view({ B, 1 }); // (B,1)
            max_next_q = next_q_target.gather(1, next_actions_b).squeeze(1);
        }
        ANET_ASSERT_SHAPE(max_next_q, { B });
        ANET_ASSERT_DTYPE(max_next_q, torch::kFloat32);

        // ------------------------------------------------------------
        // TD target & TD Error
        // ------------------------------------------------------------
        auto not_terminal = 1.0f - terminals.to(torch::kFloat32); // (B,)
        auto gamma_n = torch::pow(config_.gamma, samples.n_steps.to(torch::kFloat32)); // (B,)
        auto bootstrap = config_.use_tbo ? TransformHInv(max_next_q.detach()) : max_next_q.detach();
        auto raw_td_target = target_returns.detach() + not_terminal * gamma_n * bootstrap; // (B,)
        auto td_target = config_.use_tbo ? TransformH(raw_td_target) : raw_td_target;
        td_error = q_sa - td_target; // (B,)
        ANET_ASSERT_SHAPE(td_error, { B });
        ANET_ASSERT_DTYPE(td_error, torch::kFloat32);

        // ------------------------------------------------------------
        // Loss Calculation
        // ------------------------------------------------------------
        torch::Tensor td_error_for_loss = td_error;
        if (config_.use_td_clip && config_.td_clip_value > 0.0f)
            td_error_for_loss = torch::clamp(td_error_for_loss, -config_.td_clip_value, config_.td_clip_value);

        if (config_.use_per) {
            auto element_loss = torch::nn::functional::smooth_l1_loss(
                td_error_for_loss,
                torch::zeros_like(td_error_for_loss),
                torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone));
            auto weights = samples.is_weights.to(element_loss.device());
            loss = (element_loss * weights).mean();
        } else {
            loss = torch::nn::functional::smooth_l1_loss(
                td_error_for_loss,
                torch::zeros_like(td_error_for_loss),
                torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kMean));
        }
    } // End of Autocast Scope

    ANET_PROFILE_SCOPE(prepare_per);
    auto per_pending = PreparePerPriorityUpdate(samples, td_error);

    ANET_PROFILE_SCOPE_NEXT(optimize);
    auto opt_result = Optimize(loss);

    ANET_PROFILE_SCOPE_NEXT(update_per);
    auto per_result = ApplyPerPriorityUpdate(std::move(per_pending));

    ANET_PROFILE_SCOPE_NEXT(make_result);
    return MakeBatchUpdateResult(loss, td_error, opt_result, max_q, q_sa.detach(), per_result, torch::Tensor(), gap_abs, gap_rel);
}


// ======================================================
// QRLearner
// ======================================================

QRLearner::QRLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed,
    std::shared_ptr<ActionPolicy> target_policy, std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed)
    : QuantileLearnerBase(config, model, vars, std::move(obs_norm), batch_env_spec, env_spec, std::move(device), replay_seed, std::move(target_policy), stucker_config, target_seed)
{
    SetupReplayBuffer(batch_env_spec, env_spec, replay_seed);
    SetupOptimizer();

    // tau_iの事前計算
    const int N = config_.num_quantiles;
    tau_i_ = torch::arange(0.5f / N, 1.0f, 1.0f / N, device).view({ 1, N, 1 });
    ANET_ASSERT_SHAPE(tau_i_, { 1, N, 1 });
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
QRLearner::UpdateFromSamples(const anet::rl::ExperienceSamples& samples)
{
    ANET_PROFILE_FUNC();

    const int B = config_.replay_batch_size;
    const int A = n_actions_;
    const int N = config_.num_quantiles;
    const torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;

    // 入力チェック
    ANET_ASSERT_SHAPE(samples.actions, { B });
    ANET_ASSERT_SHAPE(samples.target_returns, { B });
    ANET_ASSERT_SHAPE(samples.next_state.terminals, { B });
    //ANET_LOG_DEBUG("obs=" << samples.obs.ToString());

    ANET_PROFILE_SCOPE(normalize);
    auto norm_samples = NormalizeSampleObservations(samples);
    const auto& obs = norm_samples.obs;
    const auto& next_obs = norm_samples.next_obs;
    ANET_ASSERT_NAN(obs);
    ANET_ASSERT_NAN(next_obs);

    torch::Tensor loss;
    torch::Tensor td_error_tensor;
    QuantileMetrics metrics;

    // ============================================================
    // Forward & Loss Calculation (Autocast Scope)
    // ============================================================
    {
        // AMP Guard
        anet::Autocast amp_guard(device_, config_.use_amp, amp_dtype);

        // ------------------------------------------------------------
        // 分布計算
        // ------------------------------------------------------------

        ANET_PROFILE_SCOPE_NEXT(forward_current);

        // 現在の分布計算: Z(s, a)、ForwardQuantiles は (B, A, N) を返す
        auto current_out = model_.ForwardOnlineWithTrain(obs);
        auto current_dist_all = current_out.At("q_dist"); // (B, A, N)
        ANET_ASSERT_SHAPE(current_dist_all, { B, A, N });
        ANET_ASSERT_NAN(current_dist_all);

        auto current_dist = GatherActionQuantiles(current_dist_all, samples.actions);

        // メトリクス用: 平均値をmax_qとして報告
        auto q_values_mean = current_out.At("q"); // すでに計算済みの平均Q値 (B, A)
        metrics = MakeQuantileMetrics(current_dist, q_values_mean);

        // ------------------------------------------------------------
        // ターゲット分布計算: r + gamma * Z(s', a*)
        // ------------------------------------------------------------
        torch::Tensor target_dist;  // (B, N)
        {
            torch::NoGradGuard grad_guard;

            torch::Tensor next_actions;
            ANET_PROFILE_SCOPE_NEXT(select_target_action);
            next_actions = SelectTargetActions(next_obs);
            ANET_ASSERT_SHAPE(next_actions, { B });

            ANET_PROFILE_SCOPE_NEXT(forward_target);

            // 次状態のターゲット分布: Z_target(s', :)
            auto next_out = model_.ForwardTarget(next_obs);
            auto next_dist_all = next_out.At("q_dist"); // (B, A, N)
            ANET_ASSERT_SHAPE(next_dist_all, { B, A, N });

            auto next_dist = GatherActionQuantiles(next_dist_all, next_actions);
            ANET_ASSERT_SHAPE(next_dist, { B, N });

            target_dist = CalcTargetQuantiles(samples, next_dist);
        }

        // ------------------------------------------------------------
        // Loss Calculation
        // ------------------------------------------------------------
        ANET_PROFILE_SCOPE(loss);

        // target_dist: (B, N) -> mean -> (B)
        auto target_mean = target_dist.mean(1).detach();

        // TD誤差
        td_error_tensor = metrics.q_sa - target_mean;

        // 要素ごとのLoss (B) を取得  ※ここで重い計算を一回だけ行う
        auto element_loss = ComputeQuantileHuberLoss(current_dist, target_dist, tau_i_); // (B)
        ANET_LOG_DEBUG("element_loss=" << anet::ToString(element_loss));
        ANET_ASSERT_SHAPE(element_loss, { B });
        ANET_ASSERT_NAN(element_loss);

        // 最適化用Loss(Scalar) ※ PERの重み (IS Weights) を適用
        torch::Tensor weights = config_.use_per ? samples.is_weights : torch::ones({ B }, device_);
        ANET_LOG_DEBUG("weights=" << anet::ToString(weights));
        ANET_ASSERT_NAN(weights);
        loss = (element_loss * weights).mean();
        ANET_ASSERT_SHAPE(loss, {});
        ANET_ASSERT_NAN(loss);

    } // End of Autocast Scope

    ANET_PROFILE_SCOPE(prepare_per);
    auto per_pending = PreparePerPriorityUpdate(samples, td_error_tensor);

    ANET_PROFILE_SCOPE_NEXT(optimize);
    auto opt_result = Optimize(loss);

    ANET_PROFILE_SCOPE_NEXT(update_per);
    auto per_result = ApplyPerPriorityUpdate(std::move(per_pending));

    ANET_PROFILE_SCOPE_NEXT(make_result);
    return MakeBatchUpdateResult(
        loss, td_error_tensor, opt_result,
        metrics.max_q, metrics.q_sa, per_result,
        metrics.q_std, metrics.q_gap, metrics.q_gap_rel);
}


// ======================================================
// IQNLearner
// ======================================================

IQNLearner::IQNLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed,
    std::shared_ptr<ActionPolicy> target_policy, std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed)
    : QuantileLearnerBase(config, model, vars, std::move(obs_norm), batch_env_spec, env_spec, std::move(device), replay_seed,
        std::move(target_policy), stucker_config, target_seed)
{
    SetupReplayBuffer(batch_env_spec, env_spec, replay_seed);
    SetupOptimizer();
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
IQNLearner::UpdateFromSamples(const anet::rl::ExperienceSamples& samples)
{
    ANET_PROFILE_FUNC();

    const int B = config_.replay_batch_size;
    const int A = n_actions_;
    const int N = config_.iqn.current_taus.num_taus;
    const int M = config_.iqn.target_taus.num_taus;
    const torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;

    // ReplayBufferから受け取ったbatch契約を、forwardより前に検証する。
    ANET_ASSERT_SHAPE(samples.actions, { B });
    ANET_ASSERT_SHAPE(samples.target_returns, { B });
    ANET_ASSERT_SHAPE(samples.next_state.terminals, { B });

    ANET_PROFILE_SCOPE(normalize);
    auto norm_samples = NormalizeSampleObservations(samples);
    const auto& obs = norm_samples.obs;
    const auto& next_obs = norm_samples.next_obs;
    ANET_ASSERT_NAN(obs);
    ANET_ASSERT_NAN(next_obs);

    torch::Tensor loss;
    torch::Tensor td_error_tensor;
    QuantileMetrics metrics;

    // current・target-policy・target-valueで異なるtau集合を使い、行動選択と分布回帰の役割を分離する。
    {
        anet::Autocast amp_guard(device_, config_.use_amp, amp_dtype);

        // 現在ネットワークをN本のtauで評価し、経験に記録されたactionのquantile列だけを損失対象へ集める。
        ANET_PROFILE_SCOPE_NEXT(generate_current_taus);
        auto current_taus = GenerateTaus(B, N, config_.iqn.current_taus.sample_mode, 0.0f, 1.0f, device_, *GetRandomGenerator());
        ANET_ASSERT_SHAPE(current_taus, { B, N });

        // ReplayBuffer所有のobservationを変更せず、current forward専用の浅いcopyへtausを注入する。
        anet::TensorDict current_input = obs;
        current_input.Set(anet::nn::kKey_Taus, current_taus);

        ANET_PROFILE_SCOPE_NEXT(forward_current);
        auto current_out = model_.ForwardOnlineWithTrain(current_input);
        auto current_dist_all = current_out.At("q_dist");
        ANET_ASSERT_SHAPE(current_dist_all, { B, A, N });
        ANET_ASSERT_NAN(current_dist_all);

        auto current_dist = GatherActionQuantiles(current_dist_all, samples.actions);
        auto q_values_mean = current_out.At("q");
        metrics = MakeQuantileMetrics(current_dist, q_values_mean);

        torch::Tensor target_dist;
        {
            torch::NoGradGuard grad_guard;

            // Double DQNと同様にtarget-policy側で次actionを選ぶ。policy自身のtau ruleはlearnerのN/Mから独立する。
            ANET_PROFILE_SCOPE_NEXT(select_target_action);
            auto next_actions = SelectTargetActions(next_obs);
            ANET_ASSERT_SHAPE(next_actions, { B });

            // 選択済みactionの価値はtarget networkをM本のtauで評価し、bootstrap済みtarget分布へ変換する。
            // target-value用tauは行動選択時のtauから独立に生成し、NとMが異なる設定も許容する。
            ANET_PROFILE_SCOPE_NEXT(generate_target_taus);
            auto target_taus = GenerateTaus(B, M, config_.iqn.target_taus.sample_mode, 0.0f, 1.0f, device_, *GetRandomGenerator());
            ANET_ASSERT_SHAPE(target_taus, { B, M });

            anet::TensorDict target_input = next_obs;
            target_input.Set(anet::nn::kKey_Taus, target_taus);

            ANET_PROFILE_SCOPE_NEXT(forward_target);
            auto next_out = model_.ForwardTarget(target_input);
            auto next_dist_all = next_out.At("q_dist");
            ANET_ASSERT_SHAPE(next_dist_all, { B, A, M });
            ANET_ASSERT_NAN(next_dist_all);

            auto next_dist = GatherActionQuantiles(next_dist_all, next_actions);
            target_dist = CalcTargetQuantiles(samples, next_dist);
        }

        // ------------------------------------------------------------
        // Loss Calculation
        // ------------------------------------------------------------

        ANET_PROFILE_SCOPE(loss);

        // PER priority更新用に、選択actionの現在Q期待値とtarget分布の期待値との差をサンプル単位で求める。
        // quantile回帰lossとは定義が異なるため、最適化には使わずReplayBufferのpriority更新へ渡す。
        auto target_mean = target_dist.mean(1).detach();
        td_error_tensor = metrics.q_sa - target_mean;
        ANET_ASSERT_SHAPE(td_error_tensor, { B });
        ANET_ASSERT_NAN(td_error_tensor);

        // 各current tau_iへquantile regression重みを対応付け、全N×MペアのIQN Huber lossを計算する。
        // ComputeIqnQuantileHuberLossはcurrent側Nをsum、target側Mをmeanし、サンプル単位(B)で返す。
        auto tau_weights = current_taus.unsqueeze(2);
        auto element_loss = ComputeIqnQuantileHuberLoss(current_dist, target_dist, tau_weights, config_.quantile_huber_kappa);
        ANET_ASSERT_SHAPE(element_loss, { B });
        ANET_ASSERT_NAN(element_loss);

        // PER有効時はimportance sampling weightでsampling biasを補正し、最後にbatch平均してscalar lossへ集約する。
        // PER無効時は全サンプルを等重みとし、同じ集約経路を使う。
        auto weights = config_.use_per ? samples.is_weights : torch::ones({ B }, device_);
        ANET_ASSERT_SHAPE(weights, { B });
        ANET_ASSERT_NAN(weights);
        loss = (element_loss * weights).mean();
        ANET_ASSERT_SHAPE(loss, {});
        ANET_ASSERT_NAN(loss);
    }

    // backward前にTD errorからPER更新内容を準備し、optimizer完了後にReplayBufferへ反映する。
    ANET_PROFILE_SCOPE(prepare_per);
    auto per_pending = PreparePerPriorityUpdate(samples, td_error_tensor);

    ANET_PROFILE_SCOPE_NEXT(optimize);
    auto opt_result = Optimize(loss);

    ANET_PROFILE_SCOPE_NEXT(update_per);
    auto per_result = ApplyPerPriorityUpdate(std::move(per_pending));

    ANET_PROFILE_SCOPE_NEXT(make_result);
    return MakeBatchUpdateResult(
        loss, td_error_tensor, opt_result,
        metrics.max_q, metrics.q_sa, per_result,
        metrics.q_std, metrics.q_gap, metrics.q_gap_rel);
}
