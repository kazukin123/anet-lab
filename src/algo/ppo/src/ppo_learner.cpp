#include "ppo_learner.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include "anet/profile.hpp"
#include "anet/tensor_util.hpp"
#include "anet/diag.hpp"
#include "anet/advantage.hpp"
#include "anet/distribution.hpp"
#include "anet/nn_util.hpp"
#include "ppo_actor.hpp"
#include "ppo_math.hpp"

using namespace anet::rl::ppo;

// rollout の器の中の鍵。Network の入力は、Actor が付けた接頭辞つきの鍵のまま入れる。
constexpr char kStoredActionKey[] = "action";
constexpr char kStoredLogProbKey[] = "log_prob";
constexpr char kStoredValueKey[] = "value";
constexpr char kStoredSignalsKey[] = "signals";

// CPU の器にためる 1 本の中の並び。
constexpr int64_t kSignalReward = 0;
constexpr int64_t kSignalTerminated = 1;
constexpr int64_t kSignalTruncated = 2;
// 印は 0 か 1 の float でためるので、中間の値で真偽に戻す。
constexpr float kSignalTrueThreshold = 0.5f;

// Pipeline の Runner では、rollout の最初の 1 step だけ、更新の前の Network で行動が選ばれる（古さ 1）。
// それより古い経験は、写しを持つ Actor が写し直されないまま Train に使われていることを示す。
constexpr uint64_t kMaxPolicyLag = 1;

static float GetNaN()
{
    return std::numeric_limits<float>::quiet_NaN();
}

// lane ごとに 1 つの値を持つ tensor（[lane 数]）であることを確かめる。形だけを見るので、device の同期は起きない。
static void RequireLaneValues(const torch::Tensor& tensor, int64_t lane_count, const char* name)
{
    if (!tensor.defined() || tensor.dim() != 1 || tensor.size(0) != lane_count) {
        ANET_SYSTEM_ERROR("Invalid PPO experience " << name << " shape="
            << (tensor.defined() ? anet::ToDefString(tensor) : std::string("undefined")) << "; expected a tensor ["
            << lane_count << "] (one value per lane).");
    }
}

// 行動ごとの鍵（接頭辞のあとが数字だけ）なら、行動の番号を返す。
static std::optional<int64_t> ParseActionIndex(const std::string& key, std::string_view prefix)
{
    if (!key.starts_with(prefix) || key.size() == prefix.size()) return std::nullopt;
    int64_t index = 0;
    const auto* first = key.data() + prefix.size();
    const auto* last = key.data() + key.size();
    const auto result = std::from_chars(first, last, index);
    if (result.ec != std::errc() || result.ptr != last || index < 0) return std::nullopt;
    return index;
}

template<size_t Count>
static std::optional<size_t> FindKey(const std::array<const char*, Count>& keys, const std::string& key)
{
    for (size_t i = 0; i < Count; ++i) if (key == keys[i]) return i;
    return std::nullopt;
}

// ======================================================
// UpdateResult
// ======================================================

const std::array<const char*, UpdateResult::kMinibatchStatCount>& UpdateResult::GetMinibatchStatKeys()
{
    // 並びは MinibatchStat と同じにする。
    static const std::array<const char*, kMinibatchStatCount> keys{
        "loss_total", "loss_policy", "loss_value", "entropy", "entropy_ratio", "max_prob_mean", "approx_kl",
        "ratio_clip_fraction", "value_clip_fraction", "ratio_mean", "ratio_std", "ratio_max",
        "advantage_norm_abs_max", "grad_norm", "grad_norm_shared", "grad_norm_policy", "grad_norm_value",
        "grad_clipped" };
    return keys;
}

const std::array<const char*, UpdateResult::kRolloutStatCount>& UpdateResult::GetRolloutStatKeys()
{
    // 並びは RolloutStat と同じにする。
    static const std::array<const char*, kRolloutStatCount> keys{
        "kl_first_epoch", "kl_last_epoch", "first_ratio_deviation", "explained_variance", "value_mean", "value_std",
        "value_target_mean", "value_target_std", "value_error_std", "bootstrap_value_mean", "advantage_mean",
        "advantage_std", "terminated_ratio", "truncated_ratio", "episode_end_count" };
    return keys;
}

const std::array<const char*, UpdateResult::kHostStatCount>& UpdateResult::GetHostStatKeys()
{
    // 並びは HostStat と同じにする。
    static const std::array<const char*, kHostStatCount> keys{
        "learning_rate", "ratio_clip_range", "entropy_coef", "policy_lag", "epochs_run" };
    return keys;
}

bool UpdateResult::IsDeviceStatKey(const std::string& key)
{
    return FindKey(GetMinibatchStatKeys(), key) || FindKey(GetRolloutStatKeys(), key)
        || ParseActionIndex(key, kActionProbKeyPrefix) || ParseActionIndex(key, kActionFreqKeyPrefix);
}

const std::vector<float>& UpdateResult::RolloutStatistics::GetDeviceValuesOnHost() const
{
    // device から CPU への転送は、rollout につき、この 1 回だけにする。鍵ごとに読み戻さない。
    std::call_once(readback_once_, [this] { ReadBackDeviceValues(); });
    return readback_;
}

void UpdateResult::RolloutStatistics::ReadBackDeviceValues() const
{
    // 計測範囲に関数の名前が付くよう、読み戻しを、名前を持つ関数に分けてある。
    ANET_PROFILE_FUNC();
    const auto values = device_values.detach().to(torch::kCPU, torch::kFloat32).contiguous();
    readback_.assign(values.data_ptr<float>(), values.data_ptr<float>() + values.numel());
}

UpdateResult::UpdateResult(std::shared_ptr<const RolloutStatistics> statistics, int64_t row)
    : statistics_(std::move(statistics)), row_(row)
{
}

std::optional<float> UpdateResult::GetScalar(const std::string& key, int64_t index) const
{
    // CPU で分かっている値は、そのまま返す。
    if (const auto host = FindKey(GetHostStatKeys(), key)) return statistics_->host_values[*host];

    // device の統計の鍵を、行の中の位置に直す。知らない鍵は、値なしを返す。
    const auto row_width = GetRowWidth(statistics_->action_count);
    std::optional<int64_t> position;
    if (const auto stat = FindKey(GetMinibatchStatKeys(), key)) {
        position = row_ * row_width + static_cast<int64_t>(*stat);
    } else if (const auto stat = FindKey(GetRolloutStatKeys(), key)) {
        position = statistics_->row_count * row_width + static_cast<int64_t>(*stat);
    } else if (const auto action = ParseActionIndex(key, kActionProbKeyPrefix)) {
        if (*action >= statistics_->action_count) return GetNaN();
        position = row_ * row_width + kMinibatchStatCount + *action;
    } else if (const auto action = ParseActionIndex(key, kActionFreqKeyPrefix)) {
        if (*action >= statistics_->action_count) return GetNaN();
        position = row_ * row_width + kMinibatchStatCount + statistics_->action_count + *action;
    } else {
        return std::nullopt;
    }
    // 知っている鍵でも、統計を計算しなかった rollout では値が成立しない。
    if (!statistics_->device_values.defined()) return GetNaN();
    return statistics_->GetDeviceValuesOnHost().at(static_cast<size_t>(*position));
}

// ======================================================
// Learner
// ======================================================

// 平らにした rollout から取り出した、1 ミニバッチぶんの入力。
struct Learner::Minibatch {
    anet::TensorDict input;      ///< Network の入力。
    torch::Tensor actions;       ///< [件数]。
    torch::Tensor old_log_probs; ///< [件数]。収集時の log 確率。
    torch::Tensor old_values;    ///< [件数]。収集時の value。
    torch::Tensor advantages;    ///< [件数]。標準化の前。
    torch::Tensor value_targets; ///< [件数]。
};

struct Learner::MinibatchOutcome {
    torch::Tensor statistics;      ///< [行の幅]。統計を計算しないときは未定義。
    torch::Tensor approx_kl;       ///< スカラー。計算しないときは未定義。
    torch::Tensor ratio_deviation; ///< スカラー。|ratio − 1| の最大。計算しないときは未定義。
};

Learner::Learner(const LearnerConfig& config, int rollout_length, int64_t lane_count, int64_t action_count,
    std::shared_ptr<std::shared_mutex> mutex, std::shared_ptr<NetworkModel> model, torch::Device device,
    anet::seed_t seed)
    : config_(config)
    , rollout_length_(rollout_length)
    , lane_count_(lane_count)
    , action_count_(action_count)
    , mutex_(std::move(mutex))
    , model_(std::move(model))
    , device_(device)
    , random_(seed)
    , learning_rate_(config.learning_rate)
    , ratio_clip_range_(config.ratio_clip_range)
    , entropy_coef_(config.entropy_coef)
    , device_store_(rollout_length)
    , host_store_(rollout_length)
{
    // 学習する parameter と、勾配の内訳に使う組を固定する。
    for (const auto& parameter : model_->GetNetwork()->parameters()) {
        if (parameter.requires_grad()) parameters_.push_back(parameter);
    }
    parameter_groups_ = model_->MakeParameterGroups();

    // optimizer は、方策と価値で 1 つにする。weight_decay が 0 の AdamW は Adam と同じ更新になる。
    // eps は参照実装に合わせて 1e-5 を既定にする（PyTorch の既定は 1e-8）。
    auto options = torch::optim::AdamWOptions(learning_rate_.Value());
    options.eps(config_.adam_eps).weight_decay(config_.weight_decay);
    if (config_.use_fused_optimizer) {
        optimizer_ = std::make_unique<anet::FusedAdamW>(parameters_, options);
    } else {
        optimizer_ = std::make_unique<torch::optim::AdamW>(parameters_, options);
    }
}

void Learner::ConfigureScalarMetricSubscriptions(const std::vector<ScalarMetricSubscription>& subscriptions)
{
    // 統計は、Train の LEARN の購読に鍵があるときだけ計算する。勾配の内訳は、その鍵の購読があるときだけにする。
    statistics_enabled_ = false;
    gradient_groups_enabled_ = false;
    const auto& keys = UpdateResult::GetMinibatchStatKeys();
    for (const auto& subscription : subscriptions) {
        if (subscription.scope != RunnerScope::TRAIN || subscription.event != EventType::LEARN) continue;
        const auto& key = subscription.source_key;
        if (!UpdateResult::IsDeviceStatKey(key)) continue;
        statistics_enabled_ = true;
        if (key == keys[UpdateResult::kGradNormShared] || key == keys[UpdateResult::kGradNormPolicy]
            || key == keys[UpdateResult::kGradNormValue]) {
            gradient_groups_enabled_ = true;
        }
    }
}

anet::rl::BatchUpdateResultList Learner::UpdateFromBatch(const StepCounts& step,
    const anet::rl::BatchExperience& experience)
{
    ANET_PROFILE_FUNC();

    // 経験 1 step ぶんを rollout にためる。長さに届くまでは学ばず、空の結果を返す。
    // on-policy の学習は、今の方策で集めた経験だけを使うので、ためた経験は 1 回の更新で使い切る。
    ANET_PROFILE_SCOPE(append);
    AppendIntake(experience);
    if (device_store_.GetTimeCount() < rollout_length_) return {};

    // 届いたら、ためた経験で方策と価値を更新し、器を空にして次の rollout に備える。
    ANET_PROFILE_SCOPE_NEXT(update);
    auto results = UpdateFromRollout(step, experience);
    device_store_.Clear();
    host_store_.Clear();
    truncation_values_ = torch::Tensor();
    policy_versions_.clear();
    return results;
}

void Learner::AppendIntake(const anet::rl::BatchExperience& experience)
{
    ANET_PROFILE_FUNC();

    // Actor が行動の情報に載せた値を取り出す。PPO 系の Actor が選んだ行動でなければ、ここで止める。
    if (!experience.action) ANET_SYSTEM_ERROR("PPO Learner received an experience without action information.");
    const auto& info = experience.action->GetInfo();
    const auto log_prob = info.Get(Actor::kLogProbKey);
    const auto value = info.Get(Actor::kValueKey);
    const auto policy_version = info.Get(Actor::kPolicyVersionKey);
    if (!log_prob || !value || !policy_version) {
        ANET_SYSTEM_ERROR("PPO Learner received action information without the keys '" << Actor::kLogProbKey << "', '"
            << Actor::kValueKey << "' and '" << Actor::kPolicyVersionKey
            << "'; expected actions selected by a PPO Actor.");
    }
    // Actor の値は、device の上のままためる。CPU と CUDA をまたぐ Actor（写しを別の device に置いたものなど）の値は、
    // ためる前に止める（あとの計算で、device の違う tensor が混ざる）。
    if (log_prob->device().type() != device_.type()) {
        ANET_SYSTEM_ERROR("PPO Learner on device=" << device_.str() << " received Actor values on device="
            << log_prob->device().str() << "; expected a Train Actor on the Agent device.");
    }
    // どの値も lane ごとに 1 つである。lane 数が Agent を作ったときと違う経験は、器に入れる前に止める。
    const auto& next = experience.next_state;
    RequireLaneValues(*log_prob, lane_count_, Actor::kLogProbKey);
    RequireLaneValues(*value, lane_count_, Actor::kValueKey);
    RequireLaneValues(experience.reward, lane_count_, "reward");
    RequireLaneValues(next.done, lane_count_, "next_state.done");
    RequireLaneValues(next.truncated, lane_count_, "next_state.truncated");

    // device の器へ、Network の入力・行動・log 確率・value をためる。どれも Actor が device の上で作った値で、転送は起きない。
    anet::TensorDict row;
    for (const auto& [key, tensor] : info) {
        if (key.starts_with(Actor::kInputKeyPrefix)) row.Set(key, tensor);
    }
    if (row.empty()) {
        ANET_SYSTEM_ERROR("PPO Learner received action information without Network inputs (keys starting with '"
            << Actor::kInputKeyPrefix << "'); expected actions selected by a PPO Actor.");
    }
    row.Set(kStoredActionKey, experience.action->GetAction(device_));
    row.Set(kStoredLogProbKey, *log_prob);
    row.Set(kStoredValueKey, *value);
    device_store_.Append(row);

    // 報酬・終端・打ち切りは CPU の器へ 1 本にまとめてためる。device へは、rollout の終わりに 1 回で送る。
    host_store_.Append({ { kStoredSignalsKey, torch::stack({ experience.reward.to(torch::kFloat32),
        next.done.to(torch::kFloat32), next.truncated.to(torch::kFloat32) }, 1) } });
    policy_versions_.push_back(static_cast<uint64_t>(policy_version->item<int64_t>()));

    // 打ち切り（時間切れなど）で終わった lane は、本当は episode が続くはずだったので、
    // Reset の前の観測の value を「次の観測の value」として使う。終端と同じに 0 とすると、value の目標が低く偏る。
    // 印は CPU にあるので、ここで読んでも device の同期は起きない。
    if (config_.truncation_mode == LearnerConfig::kTruncationBootstrap) {
        const auto bootstrap_mask = next.truncated.logical_and(next.done.logical_not());
        if (bootstrap_mask.any().item<bool>()) {
            if (!truncation_values_.defined()) {
                truncation_values_ = torch::zeros({ rollout_length_, lane_count_ },
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            }
            const auto values = ComputeValues(next.obs);
            const auto time = device_store_.GetTimeCount() - 1;
            truncation_values_.select(0, time).copy_(values * bootstrap_mask.to(device_).to(values.scalar_type()));
        }
    }
}

torch::Tensor Learner::ComputeValues(const anet::TensorDict& obs) const
{
    // 学習の目標に使う value なので、勾配を作らない。Network を書き換えるのはこの Learner だけなので、ロックは要らない。
    torch::NoGradGuard no_grad;
    const auto input = NetworkModel::MakeNetworkInput(obs.To(device_));
    return model_->GetNetwork()->Forward(input).At(NetworkModel::kValueKey).squeeze(1);
}

void Learner::SetLearningRate(double learning_rate)
{
    for (auto& group : optimizer_->param_groups()) {
        static_cast<torch::optim::AdamWOptions&>(group.options()).lr(learning_rate);
    }
}

anet::rl::BatchUpdateResultList Learner::UpdateFromRollout(const StepCounts& step,
    const anet::rl::BatchExperience& last_experience)
{
    ANET_PROFILE_FUNC();
    const int64_t time_count = rollout_length_;
    const int64_t transition_count = time_count * lane_count_;
    const int64_t minibatch_size = transition_count / config_.minibatch_count;

    // ---- 1. 方策の古さを確かめる ----
    // 経験の行動を選んだ方策の世代と、今の世代の差を数える。共有の Network と Serial の Runner なら全部 0 になる。
    const auto current_version = model_->GetPolicyVersion();
    uint64_t max_lag = 0;
    double lag_sum = 0.0;
    for (const auto version : policy_versions_) {
        const auto lag = current_version >= version ? current_version - version : 0;
        max_lag = std::max(max_lag, lag);
        lag_sum += static_cast<double>(lag);
    }
    if (max_lag > kMaxPolicyLag) {
        ANET_SYSTEM_ERROR("PPO rollout contains actions selected by a policy " << max_lag
            << " updates old; expected at most " << kMaxPolicyLag
            << ". Use an Actor that shares the Learner network for training.");
    }

    // ---- 2. advantage と value の目標を作る ----
    ANET_PROFILE_SCOPE(advantage);
    const auto stored = device_store_.GetAll();
    const auto values = stored.At(kStoredValueKey);
    // 報酬と印を、device へ 1 回で送る。
    const auto signals = host_store_.GetAll().At(kStoredSignalsKey).to(device_);
    const auto rewards = signals.select(2, kSignalReward);
    const auto done = signals.select(2, kSignalTerminated) > kSignalTrueThreshold;
    const auto truncated = signals.select(2, kSignalTruncated) > kSignalTrueThreshold;
    const auto episode_end = done.logical_or(truncated);

    // 次の観測の value を決める。
    //   rollout の途中 : 次の step でためた value（同じ lane の次の観測の value）。
    //   rollout の最後 : 最後の経験の次の観測の value を、今の Network で計算する。
    //   打ち切り       : bootstrap なら Reset の前の観測の value、terminal なら終端と同じ（0）。
    //   終端           : 0（ComputeGae が、終端の印で次の観測の value を落とす）。
    const auto last_value = ComputeValues(last_experience.next_state.obs);
    auto next_values = torch::cat({ values.narrow(0, 1, time_count - 1), last_value.unsqueeze(0) }, 0);
    auto terminated = done;
    if (config_.truncation_mode == LearnerConfig::kTruncationBootstrap) {
        if (truncation_values_.defined()) {
            next_values = torch::where(truncated.logical_and(done.logical_not()), truncation_values_, next_values);
        }
    } else {
        terminated = episode_end;
    }
    const auto gae = anet::rl::ComputeGae(rewards, values, next_values, terminated, episode_end,
        config_.gamma, config_.gae_lambda);

    // ---- 3. rollout を平らにする ----
    // 時刻と lane の区別を捨てて、遷移の並びにする。PPO の損失は遷移ごとに独立なので、並べ替えて使える。
    anet::TensorDict flat_input;
    for (const auto& [key, tensor] : stored) {
        if (!key.starts_with(Actor::kInputKeyPrefix)) continue;
        flat_input.Set(key.substr(std::string_view(Actor::kInputKeyPrefix).size()), tensor.flatten(0, 1));
    }
    const auto flat_actions = stored.At(kStoredActionKey).flatten();
    const auto flat_log_probs = stored.At(kStoredLogProbKey).flatten();
    const auto flat_values = values.flatten();
    const auto flat_advantages = gae.advantages.flatten();
    const auto flat_value_targets = gae.value_targets.flatten();

    // ---- 4. 時間の値を、この rollout の値に固定する ----
    learning_rate_.Update(step.exp_step);
    ratio_clip_range_.Update(step.exp_step);
    entropy_coef_.Update(step.exp_step);
    SetLearningRate(learning_rate_.Value());

    // ---- 5. 複数 epoch のミニバッチ更新 ----
    // 同じ rollout を epochs 回使う。epoch ごとに並べ替え、同じ大きさのミニバッチに分けて、全部の遷移を 1 回ずつ使う。
    ANET_PROFILE_SCOPE_NEXT(optimize);
    std::vector<torch::Tensor> rows;
    std::vector<torch::Tensor> kl_values;
    torch::Tensor first_ratio_deviation;
    int epochs_run = 0;
    const auto generator = random_.GetTorchGenerator(device_);
    const auto index_options = torch::TensorOptions().dtype(torch::kInt64).device(device_);
    int64_t step_count = 0;
    for (int epoch = 0; epoch < config_.epochs; ++epoch) {
        const auto permutation = torch::randperm(transition_count, generator, index_options);
        torch::Tensor last_kl;
        for (int m = 0; m < config_.minibatch_count; ++m) {
            const auto indices = permutation.narrow(0, m * minibatch_size, minibatch_size);
            anet::TensorDict input;
            for (const auto& [key, tensor] : flat_input) input.Set(key, tensor.index_select(0, indices));
            const Minibatch minibatch{
                .input = std::move(input),
                .actions = flat_actions.index_select(0, indices),
                .old_log_probs = flat_log_probs.index_select(0, indices),
                .old_values = flat_values.index_select(0, indices),
                .advantages = flat_advantages.index_select(0, indices),
                .value_targets = flat_value_targets.index_select(0, indices) };
            auto outcome = UpdateMinibatch(minibatch, ratio_clip_range_.Value(), entropy_coef_.Value());
            if (step_count == 0) first_ratio_deviation = outcome.ratio_deviation;
            if (outcome.statistics.defined()) rows.push_back(std::move(outcome.statistics));
            if (outcome.approx_kl.defined()) kl_values.push_back(outcome.approx_kl);
            last_kl = outcome.approx_kl;
            ++step_count;
        }
        ++epochs_run;
        // KL で止める: epoch の最後のミニバッチの approx KL が閾値を超えたら、残りの epoch をやめる。
        // 方策が収集時から離れすぎた rollout を、それ以上使わないためである。判定のために、epoch に 1 回だけ値を CPU へ読む。
        if (config_.kl_stop.mode == LearnerConfig::kKlStopEpoch
            && last_kl.item<float>() > config_.kl_stop.epoch.threshold) {
            break;
        }
    }
    // 重みが変わったので、方策の世代を進める。次の rollout の経験は、この世代で選ばれる。
    model_->AdvancePolicyVersion();

    // ---- 6. 更新結果を作る ----
    ANET_PROFILE_SCOPE_NEXT(result);
    auto statistics = std::make_shared<UpdateResult::RolloutStatistics>();
    statistics->row_count = step_count;
    statistics->action_count = action_count_;
    statistics->host_values[UpdateResult::kLearningRate] = static_cast<float>(learning_rate_.Value());
    statistics->host_values[UpdateResult::kRatioClipRange] = ratio_clip_range_.Value();
    statistics->host_values[UpdateResult::kEntropyCoef] = entropy_coef_.Value();
    statistics->host_values[UpdateResult::kPolicyLag] =
        static_cast<float>(lag_sum / static_cast<double>(policy_versions_.size()));
    statistics->host_values[UpdateResult::kEpochsRun] = static_cast<float>(epochs_run);
    if (statistics_enabled_) {
        // rollout の単位の統計を device の上で作り、ミニバッチの行と 1 本にまとめる。CPU へは、読まれたときに 1 回で写す。
        torch::NoGradGuard no_grad;
        const auto nan = torch::full({}, GetNaN(), torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        const auto kl = torch::stack(kl_values);
        const auto first_epoch_count = std::min<int64_t>(config_.minibatch_count, step_count);
        const bool has_pair = transition_count >= 2;
        const auto errors = flat_value_targets - flat_values;
        std::array<torch::Tensor, UpdateResult::kRolloutStatCount> rollout_values;
        rollout_values[UpdateResult::kKlFirstEpoch] = kl.narrow(0, 0, first_epoch_count).mean();
        rollout_values[UpdateResult::kKlLastEpoch] = kl.narrow(0, step_count - first_epoch_count, first_epoch_count).mean();
        rollout_values[UpdateResult::kFirstRatioDeviation] = first_ratio_deviation;
        rollout_values[UpdateResult::kExplainedVariance] =
            has_pair ? ComputeExplainedVariance(flat_values, flat_value_targets) : nan;
        rollout_values[UpdateResult::kValueMean] = flat_values.mean();
        rollout_values[UpdateResult::kValueStd] = has_pair ? flat_values.std() : nan;
        rollout_values[UpdateResult::kValueTargetMean] = flat_value_targets.mean();
        rollout_values[UpdateResult::kValueTargetStd] = has_pair ? flat_value_targets.std() : nan;
        rollout_values[UpdateResult::kValueErrorStd] = has_pair ? errors.std() : nan;
        rollout_values[UpdateResult::kBootstrapValueMean] = last_value.mean();
        rollout_values[UpdateResult::kAdvantageMean] = flat_advantages.mean();
        rollout_values[UpdateResult::kAdvantageStd] = has_pair ? flat_advantages.std() : nan;
        rollout_values[UpdateResult::kTerminatedRatio] = done.to(torch::kFloat32).mean();
        rollout_values[UpdateResult::kTruncatedRatio] =
            truncated.logical_and(done.logical_not()).to(torch::kFloat32).mean();
        rollout_values[UpdateResult::kEpisodeEndCount] = episode_end.to(torch::kFloat32).sum();
        std::vector<torch::Tensor> rollout_list;
        for (const auto& value : rollout_values) rollout_list.push_back(value.to(torch::kFloat32).reshape({}));
        statistics->device_values = torch::cat({ torch::stack(rows).flatten(), torch::stack(rollout_list) });
    }
    anet::rl::BatchUpdateResultList results;
    results.reserve(static_cast<size_t>(step_count));
    for (int64_t row = 0; row < step_count; ++row) {
        results.push_back(std::make_shared<UpdateResult>(statistics, row));
    }
    return results;
}

Learner::MinibatchOutcome Learner::UpdateMinibatch(const Minibatch& minibatch, float ratio_clip_range,
    float entropy_coef)
{
    ANET_PROFILE_FUNC();
    const auto& network = model_->GetNetwork();
    const bool needs_kl = statistics_enabled_ || config_.kl_stop.mode == LearnerConfig::kKlStopEpoch;
    MinibatchOutcome outcome;
    torch::Tensor loss, entropy, grad_norm, shared_norm, policy_norm, value_norm, max_prob, action_probs;
    torch::Tensor normalized_advantages;
    PolicyLoss policy;
    ValueLoss value;
    {
        // Network を書き換える区間。共有する Network は通常 eval なので、この区間だけ train にする。
        // 排他のロックで、Actor の forward と、評価の Actor の写し直しを待たせる。
        ANET_PROFILE_SCOPE(lock_wait);
        std::unique_lock lock(*mutex_);
        anet::TrainingModeGuard train_guard(*network, true);

        // 今の重みで、ミニバッチの log 確率・entropy・value を計算し直す（勾配つき）。
        ANET_PROFILE_SCOPE_NEXT(forward);
        optimizer_->zero_grad();
        const auto outputs = network->Forward(minibatch.input);
        const anet::rl::CategoricalDistribution distribution(outputs.At(NetworkModel::kLogitsKey));
        const auto new_log_probs = distribution.ComputeLogProb(minibatch.actions);
        const auto new_values = outputs.At(NetworkModel::kValueKey).squeeze(1);
        entropy = distribution.ComputeEntropy().mean();

        // 損失を作る。L = L_policy − c_entropy · H + c_value · L_value
        //   L_policy : clip した代理目的。収集時の方策から離れすぎない範囲で、advantage が正の行動の確率を上げる。
        //   H        : 方策の entropy。大きいほど探索が残るので、引いて（最大化して）早すぎる決め打ちを防ぐ。
        //   L_value  : value を GAE の目標へ近づける。
        ANET_PROFILE_SCOPE_NEXT(loss);
        normalized_advantages = config_.advantage_norm_mode == LearnerConfig::kAdvantageNormMinibatch
            ? NormalizeAdvantages(minibatch.advantages) : minibatch.advantages;
        policy = ComputeClippedPolicyLoss(new_log_probs, minibatch.old_log_probs, normalized_advantages,
            ratio_clip_range);
        value = ComputeValueLoss(new_values, minibatch.old_values, minibatch.value_targets,
            config_.use_value_clip ? std::optional<float>(config_.value_clip_range) : std::nullopt);
        loss = policy.loss - entropy_coef * entropy + config_.value_coef * value.loss;

        // 勾配を作り、全体のノルムを上限に収めてから、重みを更新する。
        // ノルムの clip は、まれな大きな勾配で方策が壊れるのを防ぐ。
        ANET_PROFILE_SCOPE_NEXT(backward);
        loss.backward();
        const auto grads = anet::CollectDefinedGrads(parameters_);
        grad_norm = anet::ForeachGradNorm(grads);
        if (statistics_enabled_ && gradient_groups_enabled_) {
            // 内訳は clip の前の値で測る。組が空のときは、値が成立しない。
            const auto measure = [this](const std::vector<torch::Tensor>& parameters) {
                const auto group = anet::CollectDefinedGrads(parameters);
                return group.empty() ? torch::Tensor() : anet::ForeachGradNorm(group);
            };
            shared_norm = measure(parameter_groups_.shared);
            policy_norm = measure(parameter_groups_.policy);
            value_norm = measure(parameter_groups_.value);
        }
        if (config_.use_grad_clip) anet::ForeachClipGradNorm_(grads, grad_norm, config_.grad_clip_max_norm);
        ANET_PROFILE_SCOPE_NEXT(optimizer_step);
        optimizer_->step();

        if (statistics_enabled_) {
            torch::NoGradGuard no_grad;
            max_prob = std::get<0>(distribution.GetProbs().max(1)).mean();
            action_probs = distribution.GetProbs().mean(0);
        }
    }

    // ---- 統計（購読があるときだけ） ----
    torch::NoGradGuard no_grad;
    const auto ratio = policy.ratio.detach();
    if (needs_kl) outcome.approx_kl = ComputeApproxKl(ratio, policy.log_ratio.detach());
    if (!statistics_enabled_) return outcome;
    ANET_PROFILE_SCOPE(statistics);
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    const auto nan = torch::full({}, GetNaN(), options);
    const auto or_nan = [&nan](const torch::Tensor& tensor) { return tensor.defined() ? tensor : nan; };
    const auto deviation = (ratio - 1.0f).abs();
    outcome.ratio_deviation = deviation.max();
    std::array<torch::Tensor, UpdateResult::kMinibatchStatCount> stats;
    stats[UpdateResult::kLossTotal] = loss.detach();
    stats[UpdateResult::kLossPolicy] = policy.loss.detach();
    stats[UpdateResult::kLossValue] = value.loss.detach();
    stats[UpdateResult::kEntropy] = entropy.detach();
    stats[UpdateResult::kEntropyRatio] = entropy.detach() / std::log(static_cast<double>(action_count_));
    stats[UpdateResult::kMaxProbMean] = max_prob;
    stats[UpdateResult::kApproxKl] = outcome.approx_kl;
    stats[UpdateResult::kRatioClipFraction] = (deviation > ratio_clip_range).to(torch::kFloat32).mean();
    stats[UpdateResult::kValueClipFraction] = or_nan(value.clip_fraction);
    stats[UpdateResult::kRatioMean] = ratio.mean();
    stats[UpdateResult::kRatioStd] = ratio.numel() >= 2 ? ratio.std() : nan;
    stats[UpdateResult::kRatioMax] = ratio.max();
    stats[UpdateResult::kAdvantageNormAbsMax] = normalized_advantages.abs().max();
    stats[UpdateResult::kGradNorm] = grad_norm;
    stats[UpdateResult::kGradNormShared] = or_nan(shared_norm);
    stats[UpdateResult::kGradNormPolicy] = or_nan(policy_norm);
    stats[UpdateResult::kGradNormValue] = or_nan(value_norm);
    stats[UpdateResult::kGradClipped] = config_.use_grad_clip
        ? (grad_norm > config_.grad_clip_max_norm).to(torch::kFloat32) : nan;
    std::vector<torch::Tensor> scalars;
    for (const auto& stat : stats) scalars.push_back(stat.to(torch::kFloat32).reshape({}));
    // 行動ごとの値: 新しい方策の確率の平均と、ミニバッチの行動の頻度。
    const auto action_freqs = torch::nn::functional::one_hot(minibatch.actions, action_count_)
        .to(torch::kFloat32).mean(0);
    outcome.statistics = torch::cat({ torch::stack(scalars), action_probs.to(torch::kFloat32), action_freqs });
    return outcome;
}

int64_t Learner::Save(anet::OutputArchive& archive) const
{
    return archive.WriteTorchObject(*optimizer_);
}

int64_t Learner::Load(anet::InputArchive& archive)
{
    return archive.ReadTorchObject(*optimizer_);
}
