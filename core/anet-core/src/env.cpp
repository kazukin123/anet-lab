//env.cpp

#include "anet/env.hpp"
#include <algorithm>
#include <functional>
#include <deque>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include "anet/profile.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/thread.hpp"
#include "anet/tensor_util.hpp"
#include "anet/log.hpp"

using namespace anet::rl;
namespace LOG = anet::log;


//----------------------------------------------
// Env base classes
//----------------------------------------------

SingleDiscreteEnvBase::SingleDiscreteEnvBase(
    std::string name, RunMode run_mode, std::optional<ConfigData> config_data)
    : name_(std::move(name))
    , run_mode_(run_mode)
    , config_data_(std::move(config_data))
    , log(name_ + ": ")
{
    ANET_CHECK_MSG(!name_.empty(), "Env name must not be empty.");
}

BatchEnvBase::BatchEnvBase(
    std::string name, int num_envs, RunMode run_mode, std::optional<ConfigData> config_data)
    : name_(std::move(name))
    , run_mode_(run_mode)
    , config_data_(std::move(config_data))
    , log(name_ + ": ")
{
    ANET_CHECK_MSG(!name_.empty(), "Env name must not be empty.");
    ANET_CHECK_MSG(num_envs > 0,
        "BatchEnv num_envs must be positive. name='" << name_ << "' num_envs=" << num_envs);

    // lane名は構築時に一度だけ確定し、実行中には再構築しない。
    lane_names_.reserve(static_cast<size_t>(num_envs));
    for (int lane_index = 0; lane_index < num_envs; ++lane_index) {
        lane_names_.push_back(name_ + "[" + std::to_string(lane_index) + "]");
    }
}

const std::string& BatchEnvBase::GetEnvName(int64_t lane_index) const
{
    ANET_CHECK_MSG(lane_index >= 0 && lane_index < static_cast<int64_t>(lane_names_.size()),
        "Invalid BatchEnv lane index. name='" << name_ << "' lane_index=" << lane_index
        << " num_envs=" << lane_names_.size());
    return lane_names_[static_cast<size_t>(lane_index)];
}

void BatchEnvBase::SetConfigData(ConfigData config_data)
{
    ANET_CHECK_MSG(!config_data_.has_value(),
        "BatchEnv config snapshot must be assigned only once. name='" << name_ << "'");
    config_data_ = std::move(config_data);
}


//----------------------------------------------
// Episode group validation and aggregation
//----------------------------------------------

namespace anet::rl::episode_structure_detail {

std::vector<bool> ReadBoolMask(
    const std::string& env_name,
    const std::string& mask_name,
    const torch::Tensor& mask,
    int64_t num_envs)
{
    if (mask.dim() != 1 || mask.size(0) != num_envs || mask.scalar_type() != torch::kBool) {
        ANET_SYSTEM_ERROR("Invalid episode mask. env='" << env_name << "' mask=" << mask_name
            << " expected=Bool[" << num_envs << "] actual=" << anet::ToString(mask) << ".");
    }
    auto cpu = mask.to(torch::kCPU).contiguous();
    auto accessor = cpu.accessor<bool, 1>();
    std::vector<bool> values(static_cast<size_t>(num_envs));
    for (int64_t lane = 0; lane < num_envs; ++lane) {
        values[static_cast<size_t>(lane)] = accessor[lane];
    }
    return values;
}

void ValidateSharedMask(
    const std::string& env_name,
    const std::string& mask_name,
    const std::vector<bool>& values)
{
    const bool expected = values.front();
    for (size_t lane = 1; lane < values.size(); ++lane) {
        if (values[lane] != expected) {
            ANET_SYSTEM_ERROR("Episode group structure mismatch. env='" << env_name
                << "' group=0 lane=" << lane << " mask=" << mask_name
                << " expected=" << expected << " actual=" << values[lane] << ".");
        }
    }
}

} // namespace anet::rl::episode_structure_detail

void anet::rl::ValidateEpisodeStructure(
    const std::string& env_name,
    const BatchEnvSpec& batch_spec,
    const BatchResetResult& result)
{
    const auto episode_start = episode_structure_detail::ReadBoolMask(
        env_name, "state.episode_start", result.state.episode_start, batch_spec.num_envs);
    if (batch_spec.episode_scope == EpisodeScope::SHARED) {
        episode_structure_detail::ValidateSharedMask(
            env_name, "state.episode_start", episode_start);
    }
    for (size_t lane = 0; lane < episode_start.size(); ++lane) {
        if (!episode_start[lane]) {
            const int64_t group = batch_spec.episode_scope == EpisodeScope::PER_LANE
                ? static_cast<int64_t>(lane) : 0;
            ANET_SYSTEM_ERROR("Reset must return a fresh episode group. env='" << env_name
                << "' group=" << group << " lane=" << lane
                << " mask=state.episode_start expected=true actual=false.");
        }
    }
}

std::vector<int64_t> anet::rl::ValidateEpisodeStructure(
    const std::string& env_name,
    const BatchEnvSpec& batch_spec,
    const BatchStepResult& result)
{
    const auto done = episode_structure_detail::ReadBoolMask(
        env_name, "done", result.next_state.done, batch_spec.num_envs);
    const auto truncated = episode_structure_detail::ReadBoolMask(
        env_name, "truncated", result.next_state.truncated, batch_spec.num_envs);
    const auto episode_start = episode_structure_detail::ReadBoolMask(
        env_name, "continue_state.episode_start",
        result.continue_state.episode_start, batch_spec.num_envs);

    if (batch_spec.episode_scope == EpisodeScope::SHARED) {
        episode_structure_detail::ValidateSharedMask(env_name, "done", done);
        episode_structure_detail::ValidateSharedMask(env_name, "truncated", truncated);
        episode_structure_detail::ValidateSharedMask(
            env_name, "continue_state.episode_start", episode_start);
    }

    std::vector<int64_t> completed_groups;
    const int64_t group_count = batch_spec.episode_scope == EpisodeScope::PER_LANE
        ? batch_spec.num_envs : 1;
    completed_groups.reserve(static_cast<size_t>(group_count));
    for (int64_t group = 0; group < group_count; ++group) {
        const int64_t lane = batch_spec.episode_scope == EpisodeScope::PER_LANE ? group : 0;
        const bool completed = done[static_cast<size_t>(lane)] || truncated[static_cast<size_t>(lane)];
        if (episode_start[static_cast<size_t>(lane)] != completed) {
            ANET_SYSTEM_ERROR("Episode continuation structure mismatch. env='" << env_name
                << "' group=" << group << " lane=" << lane
                << " mask=continue_state.episode_start expected=" << completed
                << " actual=" << episode_start[static_cast<size_t>(lane)] << ".");
        }
        if (completed) completed_groups.push_back(group);
    }

    if (result.n_episode_end != completed_groups.size()) {
        ANET_SYSTEM_ERROR("Episode completion count mismatch. env='" << env_name
            << "' n_episode_end expected=" << completed_groups.size()
            << " actual=" << result.n_episode_end << ".");
    }
    return completed_groups;
}

EpisodeReturnAccumulator::EpisodeReturnAccumulator(BatchEnvSpec batch_spec)
    : batch_spec_(batch_spec)
    , current_returns_(static_cast<size_t>(
        batch_spec.episode_scope == EpisodeScope::PER_LANE ? batch_spec.num_envs : 1), 0.0f)
{
    ANET_CHECK_MSG(batch_spec_.num_envs > 0,
        "EpisodeReturnAccumulator num_envs must be positive. actual=" << batch_spec_.num_envs);
}

void EpisodeReturnAccumulator::Reset()
{
    std::fill(current_returns_.begin(), current_returns_.end(), 0.0f);
}

std::vector<CompletedEpisodeReturn> EpisodeReturnAccumulator::Add(const BatchStepResult& result)
{
    auto rewards = result.reward.to(torch::kCPU).contiguous();
    ANET_ASSERT_SHAPE(rewards, { batch_spec_.num_envs });
    auto rewards_acc = rewards.accessor<float, 1>();
    auto finished = (result.next_state.done.to(torch::kBool)
        | result.next_state.truncated.to(torch::kBool)).to(torch::kCPU).contiguous();
    ANET_ASSERT_SHAPE(finished, { batch_spec_.num_envs });
    auto finished_acc = finished.accessor<bool, 1>();

    // scope に応じて lane reward を episode group へ畳み込む。
    if (batch_spec_.episode_scope == EpisodeScope::PER_LANE) {
        for (int64_t lane = 0; lane < batch_spec_.num_envs; ++lane) {
            current_returns_[static_cast<size_t>(lane)] += rewards_acc[lane];
        }
    } else {
        for (int64_t lane = 0; lane < batch_spec_.num_envs; ++lane) {
            current_returns_[0] += rewards_acc[lane];
        }
    }

    std::vector<CompletedEpisodeReturn> completed;
    const int64_t group_count = batch_spec_.episode_scope == EpisodeScope::PER_LANE
        ? batch_spec_.num_envs : 1;
    for (int64_t group = 0; group < group_count; ++group) {
        const int64_t lane = batch_spec_.episode_scope == EpisodeScope::PER_LANE ? group : 0;
        if (!finished_acc[lane]) continue;
        completed.push_back({
            .group_index = group,
            .episode_return = current_returns_[static_cast<size_t>(group)],
        });
        current_returns_[static_cast<size_t>(group)] = 0.0f;
    }
    return completed;
}

EvalSessionEnv::EvalSessionEnv(
    std::shared_ptr<BatchEnv> inner,
    int eval_episodes,
    std::vector<std::string> subscribed_env_keys)
    : BatchEnvBase(
        inner != nullptr ? inner->GetName() : std::string(),
        inner != nullptr ? inner->GetBatchSpec().num_envs : 0,
        inner != nullptr ? inner->GetRunMode() : RunMode::Eval,
        inner != nullptr ? inner->GetConfigData() : std::nullopt)
    , inner_(std::move(inner))
    , batch_spec_(inner_->GetBatchSpec())
    , eval_episodes_(eval_episodes)
    , group_count_(batch_spec_.episode_scope == EpisodeScope::PER_LANE
        ? batch_spec_.num_envs : 1)
    , return_accumulator_(batch_spec_)
    , adopted_groups_(static_cast<size_t>(group_count_), false)
{
    ANET_CHECK(inner_ != nullptr);
    if (eval_episodes_ <= 0) {
        ANET_SYSTEM_ERROR("EvalSessionEnv eval_episodes must be positive. env='"
            << GetName() << "' actual=" << eval_episodes_ << ".");
    }

    subscriptions_.reserve(subscribed_env_keys.size());
    for (const auto& source_key : subscribed_env_keys) {
        const auto parsed = anet::ParseScalarAggregationKey(source_key);
        if (!parsed.has_value() && eval_episodes_ > 1) {
            ANET_SYSTEM_ERROR("EvalSessionEnv requires an aggregation prefix for multi-episode "
                "ENV scalar. env='" << GetName() << "' eval_episodes=" << eval_episodes_
                << " source_key='" << source_key << "' expected=mean|max|min|std.");
        }
        subscriptions_.push_back({
            .source_key = source_key,
            .base_key = parsed.has_value() ? parsed->base_key : source_key,
            .aggregation = parsed.has_value() ? parsed->aggregation : anet::ScalarAggregation::MEAN,
        });
    }
}

bool EvalSessionEnv::HasAllFreshGroups(const BatchState& state) const
{
    auto episode_start = state.episode_start.to(torch::kCPU).contiguous();
    ANET_ASSERT_SHAPE(episode_start, { batch_spec_.num_envs });
    auto accessor = episode_start.accessor<bool, 1>();
    for (int64_t lane = 0; lane < batch_spec_.num_envs; ++lane) {
        if (!accessor[lane]) return false;
    }
    return true;
}

void EvalSessionEnv::BeginSession()
{
    return_accumulator_.Reset();
    for (auto& subscription : subscriptions_) subscription.accumulator.Reset();
    std::fill(adopted_groups_.begin(), adopted_groups_.end(), false);
    issued_episodes_ = 0;
    completed_episodes_ = 0;
    captured_episode_returns_.clear();
    session_result_.reset();
    session_started_ = true;

    // group index 順で最初の採用権を発行する。
    const int initial_grants = std::min(eval_episodes_, static_cast<int>(group_count_));
    for (int group = 0; group < initial_grants; ++group) {
        adopted_groups_[static_cast<size_t>(group)] = true;
        issued_episodes_++;
    }
}

std::shared_ptr<const BatchResetResult> EvalSessionEnv::Reset()
{
    BeginSession();

    // 前回 Step の全 group が fresh なら、余分な Reset を避けて継続状態を再利用する。
    if (cached_step_result_ != nullptr
        && HasAllFreshGroups(cached_step_result_->continue_state)) {
        return std::make_shared<PlainBatchResetResult>(
            cached_step_result_->continue_state,
            std::vector<AuxData>(static_cast<size_t>(batch_spec_.num_envs)));
    }

    cached_step_result_.reset();
    auto result = inner_->Reset();
    ANET_CHECK(result != nullptr);
    ValidateEpisodeStructure(GetName(), batch_spec_, *result);
    return result;
}

void EvalSessionEnv::CaptureScalars(int64_t group_index)
{
    ANET_PROFILE_SCOPE(capture);

    const int64_t scalar_index = batch_spec_.episode_scope == EpisodeScope::PER_LANE
        ? group_index : -1;
    for (auto& subscription : subscriptions_) {
        subscription.accumulator.Add(inner_->GetScalar(subscription.base_key, scalar_index));
    }
}

std::shared_ptr<const BatchStepResult> EvalSessionEnv::Step(
    std::shared_ptr<BatchActionInfo> action_info)
{
    ANET_PROFILE_FUNC();
    ANET_CHECK_MSG(session_started_,
        "EvalSessionEnv::Reset must be called before Step. env='" << GetName() << "'.");
    ANET_CHECK_MSG(!session_result_.has_value(),
        "EvalSessionEnv session is already complete. env='" << GetName() << "'.");

    auto result = inner_->Step(std::move(action_info));
    ANET_CHECK(result != nullptr);
    const auto completed_groups = ValidateEpisodeStructure(GetName(), batch_spec_, *result);
    const auto completed_returns = return_accumulator_.Add(*result);
    ANET_CHECK(completed_groups.size() == completed_returns.size());

    // 完了 group を index 順に処理し、採用 episode だけを snapshot する。
    for (size_t i = 0; i < completed_groups.size(); ++i) {
        const int64_t group = completed_groups[i];
        ANET_CHECK(completed_returns[i].group_index == group);
        if (!adopted_groups_[static_cast<size_t>(group)]) continue;

        CaptureScalars(group);
        adopted_groups_[static_cast<size_t>(group)] = false;
        completed_episodes_++;
        captured_episode_returns_.push_back(completed_returns[i].episode_return);

        // auto-reset 後の同じ group へ、残りがある場合だけ次の採用権を渡す。
        if (issued_episodes_ < eval_episodes_) {
            adopted_groups_[static_cast<size_t>(group)] = true;
            issued_episodes_++;
        }
    }

    if (completed_episodes_ == eval_episodes_) {
        session_result_ = EvalSessionResult{ .episode_returns = captured_episode_returns_ };
    }
    cached_step_result_ = result;
    return result;
}

std::optional<float> EvalSessionEnv::GetScalar(const std::string& key, int64_t index) const
{
    if (index < 0) {
        for (const auto& subscription : subscriptions_) {
            if (subscription.source_key == key) {
                return subscription.accumulator.Get(subscription.aggregation);
            }
        }
    }
    return inner_->GetScalar(key, index);
}

std::optional<torch::Tensor> EvalSessionEnv::GetTensor(
    const std::string& key, int64_t index) const
{
    return inner_->GetTensor(key, index);
}

std::optional<std::vector<torch::Tensor>> EvalSessionEnv::GetTensorVector(
    const std::string& key, int64_t index) const
{
    return inner_->GetTensorVector(key, index);
}


//----------------------------------------------
// DiscreteBatchEnvBase Result
//----------------------------------------------

class DiscreteBatchEnvBase::Result : virtual public BatchEnvResult {
public:
    Result(int num_envs)
    {
        single_results.resize(num_envs);
    }

    std::vector<AuxData> GetAuxDataList(int env_index = -1) const override
    {
        std::vector<AuxData> auxs;
        if (env_index >= 0) {
            auto result = single_results[env_index];
            if (result != nullptr) {
                auto aux = result->GetAuxData();
                auxs.push_back(aux);
            }
        } else {
            for (auto result : single_results) {
                if (result != nullptr) {
                    auto aux = result->GetAuxData();
                    auxs.push_back(aux);
                }
            }
        }

        return auxs;
    }
public:
    std::vector<std::shared_ptr<const SingleEnvResult>> single_results;

};

class DiscreteBatchEnvBase::ResetResult : virtual public DiscreteBatchEnvBase::Result, public anet::rl::BatchResetResult {
public:
    ResetResult(int num_envs, anet::rl::BatchState state)
        : Result(num_envs), BatchResetResult(std::move(state))
    {
        ;
    }
};

class DiscreteBatchEnvBase::StepResult : virtual public DiscreteBatchEnvBase::Result, public anet::rl::BatchStepResult {
public:
    StepResult(int num_envs,
        torch::Tensor reward, BatchState next_state, BatchState continue_state, uint32_t n_transitions, uint32_t n_done)
        : Result(num_envs), BatchStepResult(std::move(reward), std::move(next_state), std::move(continue_state), n_transitions, n_done)
    {
        ;
    }
};


//----------------------------------------------
// DiscreteBatchEnvBase
//----------------------------------------------

DiscreteBatchEnvBase::DiscreteBatchEnvBase(
    const ConfigData& config_data,
    std::shared_ptr<SingleDiscreteEnvFactory> factory, const std::string& name, int num_envs,
    const torch::Device& device, std::optional<seed_t> seed, RunMode run_mode,
    const std::string& config_prefix)
    : BatchEnvBase(name, num_envs, run_mode)
    , RandomHolder(seed)
    , batch_spec_({ num_envs, 1 })
    , num_envs_(num_envs)
    , device_(device)
{
    ANET_ASSERT(num_envs_ > 0);
    ANET_LOG_DEBUG_PREFIXED("seed=" << this->GetSeed());

    // ベースのシードを準備
    anet::SeedMaker seed_maker(seed);

    //  num_envs 個のENVを生成
    envs_.reserve(num_envs_);
    for (int i = 0; i < num_envs; ++i) {
        anet::seed_t env_seed = seed_maker.MakeIndexedSeed(i);
        auto env = factory->CreateSingleEnv(config_data, device, GetEnvName(i), env_seed, run_mode, config_prefix);
        envs_.push_back(std::move(env));
    }

    // wrapper 自身と全 lane の設定を、注入スコープを保った一つのスナップショットへ統合する。
    ConfigData effective_config = BatchEnvBuilderConfig(config_data).GetScopedConfigData();
    bool all_children_supported = true;
    for (const auto& env : envs_) {
        const auto child_config = env->GetConfigData();
        if (!child_config.has_value()) {
            all_children_supported = false;
            break;
        }
        effective_config.MergeFromChecked(*child_config);
    }
    if (all_children_supported) {
        SetConfigData(std::move(effective_config));
    }

    // EnvSpec取得
    spec_ = std::make_unique<EnvSpec>(envs_[0]->GetSpec());
    spec_->state_spec.AssertSanity();

    // 初期化
    float_opt_ = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    bool_opt_ = torch::TensorOptions().dtype(torch::kBool).device(device);
    if (device_.is_cpu()) {
        //float_opt_ = float_opt_.pinned_memory(true);
        //bool_opt_ = bool_opt_.pinned_memory(true);
    }
    reset_result_ = createEmptyResetResult();
    step_result_ = createEmptyStepResult();
}

EnvSpec DiscreteBatchEnvBase::GetSpec() const
{
    return *spec_;
}

BatchEnvSpec DiscreteBatchEnvBase::GetBatchSpec() const
{
    return batch_spec_;
}

anet::TensorDict DiscreteBatchEnvBase::createEmptyObsDict() const
{
    anet::TensorDict dict;

    // EnvSpec に定義された全てのobsについて、バッチ次元を付与した空テンソルを作る
    for (const auto& kv : spec_->state_spec.obs_spec) {
        const auto& key = kv.first;
        const auto& t_spec = kv.second;

        // [N, shape...] の次元を作成
        std::vector<int64_t> dims = { num_envs_ };
        dims.insert(dims.end(), t_spec.shape.begin(), t_spec.shape.end());

        // Specに定義された dtype と、環境の device で確保
        auto opt = torch::TensorOptions().dtype(t_spec.dtype).device(device_);
        dict.Set(key, torch::empty(dims, opt));
    }
    return dict;
}

std::shared_ptr<DiscreteBatchEnvBase::ResetResult> DiscreteBatchEnvBase::createEmptyResetResult() const
{
    auto result = std::make_shared<DiscreteBatchEnvBase::ResetResult>(
        num_envs_,
        anet::rl::BatchState {
            createEmptyObsDict(),
            torch::empty({ num_envs_ }, bool_opt_),
            torch::empty({ num_envs_ }, bool_opt_),
            torch::empty({ num_envs_ }, bool_opt_)
        });
    return result;
}

std::shared_ptr<DiscreteBatchEnvBase::StepResult> DiscreteBatchEnvBase::createEmptyStepResult() const
{
    auto result = std::make_shared<DiscreteBatchEnvBase::StepResult>(
        num_envs_,

        torch::empty({ num_envs_ }, float_opt_),       // reward        (N) kFloat32
        BatchState{    // next_state
            createEmptyObsDict(),
            torch::empty({ num_envs_ }, bool_opt_),    // done          (N) kBool
            torch::empty({ num_envs_ }, bool_opt_),    // truncated     (N) kBool
            torch::empty({ num_envs_ }, bool_opt_)     // episode_start (N) kBool
        },
        BatchState{    // continue_state
            createEmptyObsDict(),
            torch::empty({ num_envs_ }, bool_opt_),    // done          (N) kBool
            torch::empty({ num_envs_ }, bool_opt_),    // truncated     (N) kBool
            torch::empty({ num_envs_ }, bool_opt_)     // episode_start (N) kBool
        },
        0,  // n_transitions
        0   // n_done
    );

    return result;
}

std::shared_ptr<DiscreteBatchEnvBase::ResetResult> DiscreteBatchEnvBase::getResetResult() const
{
    /// @todo どこかでDeepCopyしてない処理があるみたいので特定＆修正し、使いまわし&pinnedで高速化

    std::shared_ptr<DiscreteBatchEnvBase::ResetResult> result = this->reset_result_;    // 使い回す
    //std::shared_ptr<DiscreteBatchEnvBase::ResetResult> result = createEmptyResetResult();
    return result;
}

std::shared_ptr<DiscreteBatchEnvBase::StepResult> DiscreteBatchEnvBase::getStepResult() const
{
    //std::shared_ptr<DiscreteBatchEnvBase::StepResult> result = this->step_result_;    // 使い回す
    std::shared_ptr<DiscreteBatchEnvBase::StepResult> result = createEmptyStepResult();
    return result;
}


std::optional<float> DiscreteBatchEnvBase::GetScalar(const std::string& key, int64_t index) const
{
    ANET_ASSERT(index < 0 || index < envs_.size());

    // 特定の環境(index)が指定されている場合は、その環境の値を直接返す
    if (index >= 0) {
        ANET_ASSERT(index < envs_.size());
        return envs_[index]->GetScalar(key, -1);
    }

    // バッチ全体(index == -1)の場合、共通 parser と accumulator で集約する。
    const auto parsed = anet::ParseScalarAggregationKey(key);
    if (!parsed.has_value()) {
        // 集約方法が指定されない要求は、意味を暗黙に補わず設定箇所で検出させる。
        ANET_SYSTEM_ERROR("DiscreteBatchEnvBase scalar key requires an aggregation prefix. key='"
            << key << "' expected_prefix='max.|mean.|min.|std.'");
    }

    // 全ENVのデータを収集
    anet::ScalarSampleAccumulator samples;
    for (const auto& env : envs_) {
        samples.Add(env->GetScalar(parsed->base_key, -1));
    }
    return samples.Get(parsed->aggregation);
}

std::optional<torch::Tensor> DiscreteBatchEnvBase::GetTensor(const std::string& key, int64_t index) const
{
    ANET_ASSERT(index >= 0 && index < envs_.size());
    return envs_[index]->GetTensor(key, -1);
    /// @todo vec対応
}

std::optional<std::vector<torch::Tensor>> DiscreteBatchEnvBase::GetTensorVector(const std::string& key, int64_t index) const
{
    ANET_ASSERT(index >= 0 && index < envs_.size());
    return envs_[index]->GetTensorVector(key, -1);
    /// @todo vec対応
}


//----------------------------------------------
// VectorizedDiscreteBatchEnv
//----------------------------------------------

VectorizedDiscreteBatchEnv::VectorizedDiscreteBatchEnv(
    const ConfigData& configData,
    std::shared_ptr<SingleDiscreteEnvFactory> factory,
    const std::string& name,
    int num_envs,
    const torch::Device& device,
    std::optional<seed_t> seed,
    RunMode run_mode,
    const std::string& config_prefix)
    : DiscreteBatchEnvBase(configData, factory, name, num_envs, device, seed, run_mode, config_prefix)
{
    ANET_LOG_DEBUG_PREFIXED("seed=" << this->GetSeed());
}

std::shared_ptr<const BatchResetResult> VectorizedDiscreteBatchEnv::Reset()
{
    ANET_PROFILE_FUNC();

    // 戻りの枠生成
    auto result = getResetResult();

    // 全環境を初期化し、state_ バッファに書き込む（バッファは constructor 確保済み）
    for (int i = 0; i < num_envs_; ++i) {
        auto reset_result = envs_[i]->Reset();
        ANET_ASSERT_DEVICE(reset_result->state.obs, device_);
        result->state.obs.CopyBatchItem(i, reset_result->state.obs);
        result->state.done[i].fill_(reset_result->state.done);
        result->state.truncated[i].fill_(reset_result->state.truncated);
        result->state.episode_start[i].fill_(reset_result->state.episode_start);
        result->single_results[i] = reset_result;
    }

    return result;
}

std::shared_ptr<const BatchStepResult> VectorizedDiscreteBatchEnv::Step(std::shared_ptr<BatchActionInfo> action_info)
{
    ANET_PROFILE_FUNC();

    const int64_t N = batch_spec_.num_envs;

    ANET_ASSERT_DTYPE_MSG(action_info->GetAction(), torch::kInt64,
        "VectorizedDiscreteBatchEnv supports discrete action only. actions should be kInt64.");
    ANET_ASSERT_SHAPE(action_info->GetAction(), {N});

    // 戻りの枠生成
    auto result = getStepResult();

    auto actions = action_info->GetAction(device_);
    // ----- 環境を順次実行して埋める -----
    for (int i = 0; i < N; ++i) {
        auto a = actions[i].item<int64_t>();
        std::shared_ptr<const SingleStepResult> single_result = envs_[i]->Step(a);
        ANET_ASSERT_DEVICE(single_result->next_state.obs, device_);

        result->single_results[i] = single_result;

        result->next_state.obs.CopyBatchItem(i, single_result->next_state.obs);
        result->next_state.done[i].fill_(single_result->next_state.done);
        result->next_state.truncated[i].fill_(single_result->next_state.truncated);
        result->next_state.episode_start[i].fill_(single_result->next_state.episode_start);

        result->reward.select(0, i).fill_(single_result->reward);

        // Auto reset
        if (single_result->next_state.done || single_result->next_state.truncated) {
            auto reset_result = envs_[i]->Reset();
            ANET_ASSERT_DEVICE(reset_result->state.obs, device_);

            result->continue_state.obs.CopyBatchItem(i, reset_result->state.obs);
            result->continue_state.done[i].fill_(reset_result->state.done);
            result->continue_state.truncated[i].fill_(reset_result->state.truncated);
            result->continue_state.episode_start[i].fill_(reset_result->state.episode_start);
            result->n_episode_end++;
        } else {
            result->continue_state.obs.CopyBatchItem(i, single_result->next_state.obs);
            result->continue_state.done[i].fill_(false);
            result->continue_state.truncated[i].fill_(false);
            result->continue_state.episode_start[i].fill_(single_result->next_state.episode_start);
        }
    }

    result->n_transitions = N;

    return result;
}


//----------------------------------------------
// ThreadPoolDiscreteEnv
//----------------------------------------------

ThreadPoolDiscreteEnv::ThreadPoolDiscreteEnv(
    const ConfigData& configData,
    std::shared_ptr<SingleDiscreteEnvFactory> factory,
    const std::string& name,
    int num_envs,
    const torch::Device& device,
    std::shared_ptr<ThreadPool> pool,
    std::optional<seed_t> seed,
    RunMode run_mode,
    const std::string& config_prefix)
    : DiscreteBatchEnvBase(configData, factory, name, num_envs, device, seed, run_mode, config_prefix)
    , pool_(std::move(pool))
{
    ANET_ASSERT(pool_ != nullptr);
    ANET_LOG_DEBUG_PREFIXED("seed=" << this->GetSeed());
    this->batch_spec_.num_threads = pool_->GetWorkerCount();
}

void ThreadPoolDiscreteEnv::Shutdown()
{
    if (pool_) {
        pool_->Stop();
        //pool = nullptr;
    }
}

ThreadPoolDiscreteEnv::~ThreadPoolDiscreteEnv()
{
    if (pool_) {
        pool_->Stop();
    }
}

std::shared_ptr<const BatchResetResult>  ThreadPoolDiscreteEnv::Reset()
{
    ANET_PROFILE_FUNC();

    ANET_ASSERT(pool_ != nullptr);
    const int worker_count = pool_->GetWorkerCount();
    ANET_ASSERT(worker_count > 0);

    const int N = num_envs_;

    // 戻りの枠生成
    auto result = getResetResult();

    // 全ENV分のResetタスクをキューに積む
    for (int i = 0; i < N; ++i) {
        const int worker_id = i % worker_count;

        pool_->Enqueue(worker_id, [this, &result, i]()
            {
                // ENV Reset 実行
                auto single_result = envs_[i]->Reset();
                ANET_ASSERT_DEVICE(single_result->state.obs, device_);

                // 結果書き込み(i番目の行だけを書くので他 Worker と race しない)
                result->state.obs.CopyBatchItem(i, single_result->state.obs);
                result->state.done[i].fill_(single_result->state.done);
                result->state.truncated[i].fill_(single_result->state.truncated);
                result->state.episode_start[i].fill_(single_result->state.episode_start);

                // GetAuxDataList()向けにsingle_resultを詰める
                result->single_results[i] = single_result;
            });
    }

    // 全タスク終了待ち
    pool_->WaitAll();

    return result;
}

std::shared_ptr<const BatchStepResult> ThreadPoolDiscreteEnv::Step(std::shared_ptr<BatchActionInfo> action_info)
{
    ANET_PROFILE_FUNC();

    const int N = num_envs_;
    ANET_LOG_DEBUG_PREFIXED("action=" << anet::ToString(action_info->GetAction()));
    ANET_ASSERT_DTYPE_MSG(action_info->GetAction(), torch::kInt64,
        "ThreadPoolDiscreteEnv supports discrete action only. actions should be kInt64.");
    ANET_ASSERT_SHAPE(action_info->GetAction(), {N});
    const int worker_count = pool_->GetWorkerCount();
    ANET_ASSERT(worker_count > 0);

    // --- 返却バッファ ---
    auto result = getStepResult();

    auto actions = action_info->GetAction(this->device_);

    // --- 並列に Step + 結果書き込み ---
    for (int i = 0; i < N; ++i) {
        const int worker_id = i % worker_count;
        const int64_t action_i = actions[i].item<int64_t>();

        pool_->Enqueue(worker_id,
            [this, &result, i, action_i]()
            {
                // --- SingleEnv の Step 実行 ---
                auto r = envs_[i]->Step(action_i);
                ANET_ASSERT_DEVICE(r->next_state.obs, device_);

                // --- single_result ---
                result->single_results[i] = r;

                // --- next_state ---
                result->next_state.obs.CopyBatchItem(i, r->next_state.obs);
                result->next_state.done[i].fill_(r->next_state.done);
                result->next_state.truncated[i].fill_(r->next_state.truncated);
                result->next_state.episode_start[i].fill_(r->next_state.episode_start);

                // --- reward ---
                result->reward.select(0, i).fill_(r->reward);

                // --- continue_state ---
                if (r->next_state.done || r->next_state.truncated) {
                    // Reset_required
                    auto reset_result = envs_[i]->Reset();
                    ANET_ASSERT_DEVICE(reset_result->state.obs, device_);
                    result->continue_state.obs.CopyBatchItem(i, reset_result->state.obs);
                    result->continue_state.done[i].fill_(reset_result->state.done);
                    result->continue_state.truncated[i].fill_(reset_result->state.truncated);
                    result->continue_state.episode_start[i].fill_(reset_result->state.episode_start);
                } else {
                    // Continue as-is
                    result->continue_state.obs.CopyBatchItem(i, r->next_state.obs);
                    result->continue_state.done[i].fill_(false);
                    result->continue_state.truncated[i].fill_(false);
                    result->continue_state.episode_start[i].fill_(r->next_state.episode_start);
                }
            });

    }

    // --- すべてのタスク完了を待つ ---
    pool_->WaitAll();

    // カウント
    auto done = result->next_state.done;
    auto truncated = result->next_state.truncated;
    auto episode_end = done.logical_or(truncated);
    if (episode_end.device().is_cuda()) {
        episode_end = episode_end.to(torch::kCPU);
    }
    result->n_episode_end = episode_end.sum().item<int>();
    result->n_transitions = N;

    // 返す
    return result;
}


//----------------------------------------------
// DiscreteBatchEnvBase
//----------------------------------------------

int WorkerThreadResolver::GetLogicalCores()
{
	/// @todo 物理コア数も考慮する？
    unsigned int n = std::thread::hardware_concurrency();
    return (n == 0 ? 4 : (int)n);
}

int WorkerThreadResolver::Resolve(int worker_threads, int work_items)
{
    int wt = worker_threads;

    if (wt > 0) return wt;  // 明示指定

    const int logical = GetLogicalCores();

    switch (wt) {
    case WorkerThreadAuto::AUTO: {
        int safe = std::max(1, logical - 2);
        return std::min(work_items, safe);
    }
    case WorkerThreadAuto::ENV_COUNT:
        return work_items;

    case WorkerThreadAuto::LOGICAL_CORES: {
        int safe = std::max(1, logical - 2);
        return safe;
    }
    case WorkerThreadAuto::LOGICAL_CORES_EXACT:
        return logical;

    default:
        ANET_SYSTEM_ERROR("Invalid env.worker_threads=" << worker_threads
            << ". Expected a positive value or one of -1, -2, -3, -4.");
    }
    return 1;
}


//----------------------------------------------
// BatchEnvBuilder
//----------------------------------------------

BatchEnvBuilder::BatchEnvBuilder(
    const BatchEnvBuilderConfig& config,
    const ConfigData& config_data,
    int num_envs,
    std::optional<const torch::Device> device)
	: config_data_(config_data)
    , config_(config)
    , num_envs_(num_envs)
    , device_(device.value_or(anet::MakeDevice(config_.device_type, config_.device_index)))
{
    /// @todo deviceの指定方法が設定ファイル、config、device、三箇所あるのを整理
    ANET_ASSERT(num_envs_ > 0);

    // ログ：パラメータ記録
    LOG::info() << "BatchEnvBuilder config=" << config_.ToString();
    anet::MetricsLogger::Instance()->Log(config_);
}

int BatchEnvBuilder::ResolveWorkerThreads(int num_envs) const
{
    return WorkerThreadResolver::Resolve(config_.worker_threads, num_envs);
}

std::shared_ptr<anet::ThreadPool> BatchEnvBuilder::CreatePool(int worker_threads) const
{
    return std::make_shared<PinnedThreadPool>(worker_threads);
}

std::shared_ptr<BatchEnv> BatchEnvBuilder::CreateBatchEnv(
    const std::string& name, std::optional<seed_t> seed, int num_envs_in,
    RunMode run_mode, const std::string& config_prefix)
{
    const int num_envs = num_envs_in < 0 ? num_envs_ : num_envs_in;
    auto batch_factory = EnvRepository::Instance().GetBatchEnvFactory(config_.class_id);
    if (batch_factory != nullptr) {
        return batch_factory->CreateBatchEnv(
            config_data_, device_, name, seed, num_envs, run_mode, config_prefix);
    }

    auto factory = GetSingleFactory();
    if (factory == nullptr) return nullptr;

    auto env_class_id = factory->GetTargetEnvClassId();

    // num_envs == 1 では VectorizedDiscreteBatchEnv の方が有利
    if (num_envs == 1 || config_.worker_type == WorkerType::SINGLE_THREAD) {
        return std::make_shared<VectorizedDiscreteBatchEnv>(
            config_data_, factory, name, num_envs, device_, seed, run_mode, config_prefix);
    }

    int workers = ResolveWorkerThreads(num_envs);

    // workers == 0 の理論的ケース防止
    if (workers <= 0) workers = 1;

    // ThreadPool の生成
    auto pool = CreatePool(workers);

    // ThreadPoolDiscreteEnv の生成
    return std::make_shared<ThreadPoolDiscreteEnv>(
        config_data_, factory, name, num_envs, device_, pool, seed, run_mode, config_prefix);
}

void BatchEnvBuilder::ValidateConfig(RunMode run_mode, const std::string& config_prefix) const
{
    auto batch_factory = EnvRepository::Instance().GetBatchEnvFactory(config_.class_id);
    if (batch_factory != nullptr) {
        batch_factory->ValidateConfig(config_data_, run_mode, config_prefix);
    }
}

std::shared_ptr<SingleDiscreteEnvFactory> BatchEnvBuilder::GetSingleFactory() const
{
    auto factory = EnvRepository::Instance().GetSingleDiscreteEnvFactory(config_.class_id);
    if (factory == nullptr)
        return nullptr;
    return factory;
}


//----------------------------------------------
// EnvRepository
//----------------------------------------------

void EnvRepository::Regist(std::shared_ptr<SingleDiscreteEnvFactory> factory)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto class_id = factory->GetTargetEnvClassId();
    if (factories_.contains(class_id)) {
        ANET_SYSTEM_ERROR("Env factory is already registered. class_id='" << class_id << "'");
    }
    factories_.emplace(class_id, std::move(factory));
}

void EnvRepository::Regist(std::shared_ptr<BatchEnvFactory> factory)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto class_id = factory->GetTargetEnvClassId();
    if (factories_.contains(class_id)) {
        ANET_SYSTEM_ERROR("Env factory is already registered. class_id='" << class_id << "'");
    }
    factories_.emplace(class_id, std::move(factory));
}

std::shared_ptr<SingleDiscreteEnvFactory> EnvRepository::GetSingleDiscreteEnvFactory(const std::string& id) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = factories_.find(id);
    if (it == factories_.end()) return nullptr;
    const auto* factory = std::get_if<std::shared_ptr<SingleDiscreteEnvFactory>>(&it->second);
    return factory == nullptr ? nullptr : *factory;
}

std::shared_ptr<BatchEnvFactory> EnvRepository::GetBatchEnvFactory(const std::string& id) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = factories_.find(id);
    if (it == factories_.end()) return nullptr;
    const auto* factory = std::get_if<std::shared_ptr<BatchEnvFactory>>(&it->second);
    return factory == nullptr ? nullptr : *factory;
}
