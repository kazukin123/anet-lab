#include <limits>
#include <format>
#include <unordered_set>
#include "anet/trainer.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/profile.hpp"
#include "anet/observers.hpp"
#include "anet/tensor_util.hpp"
#include "anet/log.hpp"
#include "anet/env.hpp"
#include "anet/agent.hpp"


using namespace anet::rl;
namespace LOG = anet::log;

static std::string SanitizeEnvConfigFilename(std::string filename)
{
    // MetricsLogger と同じ置換規則で、書き込み前に異なる Env 名の衝突を検出する。
    for (char& c : filename) {
        switch (c) {
        case '/': case '\\': case ':': case '*': case '?':
        case '"': case '<': case '>': case '|':
            c = '-';
            break;
        default:
            break;
        }
    }
    return filename;
}


// ======================================================
// RunnerBase
// ======================================================

RunnerBase::RunnerBase(
    std::shared_ptr<anet::rl::BatchEnv> env, std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Notifier> notifier, const ActorRequest& request, std::string name)
    : name_(std::move(name))
    , env_(env)
    , agent_(agent)
    , notifier_(notifier)
    , reward_ema_(0.001)
{
    InitializeMetrics();
    status_ = anet::rl::RunnerStatus::RUNNING;
    // 参照元のRunner名を付け、カタログ診断を利用者の設定へ結び付ける。
    try {
        actor_ = agent_->CreateActor(request);
    } catch (const std::exception& error) {
        ANET_SYSTEM_ERROR("Actor creation failed for Runner '" << name_ << "', actor_key='"
            << request.actor_key << "': " << error.what());
    }
}

void RunnerBase::InitializeMetrics()
{
    ANET_CHECK(env_ != nullptr);

    auto batch_env_spec = env_->GetBatchSpec();

    // メトリクス初期化
    episode_stats_accumulator_ = std::make_unique<EpisodeStatsAccumulator>(batch_env_spec);
    completed_episode_returns_.Reset();
    completed_episode_steps_.Reset();
    last_step_had_episode_end_ = false;
}

void RunnerBase::UpdateMetrics(std::shared_ptr<const BatchStepResult> result)
{
    // 平均報酬更新
    float step_reward_mean = result->reward.mean().item<float>();
    last_reward_ = step_reward_mean;
    reward_ema_.Update(last_reward_);

}

bool RunnerBase::AccumulateAndNotifyEpisodeEnd(
    std::shared_ptr<const Runner> self, std::shared_ptr<const BatchStepResult> result, const StepCounts& event_counts)
{
    ANET_PROFILE_FUNC();

    ANET_CHECK(self != nullptr);
    ANET_CHECK(result != nullptr);

    const auto batch_env_spec = env_->GetBatchSpec();
    ValidateEpisodeStructure(env_->GetName(), batch_env_spec, *result);
    const auto completed = episode_stats_accumulator_->Add(*result);
    completed_episode_returns_.Reset();
    completed_episode_steps_.Reset();
    last_step_had_episode_end_ = !completed.empty();

    // 完了 group を昇順で通知し、直近 Step の return と steps を保持する。
    for (const auto& episode : completed) {
        completed_episode_returns_.Add(episode.episode_return);
        completed_episode_steps_.Add(static_cast<float>(episode.episode_steps));
        const int env_index = batch_env_spec.episode_scope == EpisodeScope::PER_LANE
            ? static_cast<int>(episode.group_index) : -1;
        EpisodeEndEvent event{ self, event_counts, agent_, env_, env_index };
        notifier_->Notify(event);
    }

    return last_step_had_episode_end_;
}

void RunnerBase::SetCompletedEpisodes(
    const std::vector<float>& returns, const std::vector<int64_t>& steps)
{
    // 対応が崩れた結果は公開せず、両集約を同じ採用 episode 群で確定する。
    ANET_CHECK_MSG(returns.size() == steps.size(),
        "Completed episode size mismatch. runner='" << name_
        << "' returns=" << returns.size() << " steps=" << steps.size()
        << " expected=equal sizes.");
    completed_episode_returns_.Reset();
    completed_episode_steps_.Reset();
    for (size_t i = 0; i < returns.size(); ++i) {
        completed_episode_returns_.Add(returns[i]);
        completed_episode_steps_.Add(static_cast<float>(steps[i]));
    }
}

StepCounts RunnerBase::DoUpdateFrame(int max_steps, ControlFunction pre_step_func, ControlFunction post_step_func)
{
    ANET_PROFILE_FUNC();

    int frame_step = 0;
    //StepCounts step_counts_;

    // --- 学習ステップを複数回回す ---
    while (max_steps < 0 || frame_step < max_steps) {
        ANET_PROFILE_SCOPE(step);

        // ステップ前制御
        if (pre_step_func != nullptr) {
            auto control_signal_pre = pre_step_func(step_counts_);
            if (control_signal_pre == anet::rl::ControlSignal::STOP) {
                status_ = anet::rl::RunnerStatus::COMPLETED;
                break;
            }
            if (control_signal_pre == anet::rl::ControlSignal::BREAK) {
                break;
            }
        }

        // Step実行
        step_counts_ = DoStep();

        // ステップ後処理
        if (post_step_func != nullptr) {
            auto control_signal_post = post_step_func(step_counts_);
            if (control_signal_post == anet::rl::ControlSignal::STOP) {
                status_ = anet::rl::RunnerStatus::COMPLETED;
                break;
            }
            if (control_signal_post == anet::rl::ControlSignal::BREAK) {
                break;
            }
        }

        // Stepカウント
        frame_step++;
    }

    // ログflush
    anet::MetricsLogger::Instance()->Flush();

    return step_counts_;
}

std::optional<float> RunnerBase::GetScalar(const std::string& key, int64_t index) const
{
    if (key == TRAIN_STEP) return static_cast<float>(step_counts_.train_step);
    if (key == EXP_STEP) return static_cast<float>(step_counts_.exp_step);
    if (key == LEARN_STEP) return static_cast<float>(step_counts_.learn_step);
    if (key == EPISODE_COUNT) return static_cast<float>(step_counts_.episode_count);
    if (key == SIM_STEP) return static_cast<float>(step_counts_.sim_step);

    if (key == REWARD) return last_reward_;
    if (key == REWARD_EMA) return reward_ema_.Value();
    const auto aggregation_key = anet::ParseScalarAggregationKey(key);
    if (aggregation_key.has_value() && aggregation_key->base_key == "episode_return") {
        return completed_episode_returns_.Get(aggregation_key->aggregation);
    }
    if (aggregation_key.has_value() && aggregation_key->base_key == "episode_steps") {
        return completed_episode_steps_.Get(aggregation_key->aggregation);
    }


    return std::nullopt;
}


// ======================================================
// EvalRunner
// ======================================================

EvalRunner::EvalRunner(
    std::shared_ptr<anet::rl::BatchEnv> env,
    std::shared_ptr<anet::rl::Agent> agent,
    std::shared_ptr<anet::rl::Notifier> notifier,
    const ActorRequest& request,
    std::string name)
    : RunnerBase(env, agent, notifier, request, std::move(name))
{
}

EvalRunner::EvalRunner(
    std::shared_ptr<anet::rl::EvalSessionEnv> env,
    std::shared_ptr<anet::rl::Agent> agent,
    std::shared_ptr<anet::rl::Notifier> notifier,
    const ActorRequest& request,
    std::string name)
    : RunnerBase(env, agent, notifier, request, std::move(name))
    , session_env_(std::move(env))
{
}

void EvalRunner::Sync(const StepCounts& source_counts)
{
    // 同期したnetworkと方策時計を同じセッションの学習座標に固定する。
    actor_->Sync();
    source_counts_ = source_counts;
}

StepCounts EvalRunner::DoStep(int64_t action, const StepCounts& event_counts)
{
    return DoStepInternal(action, event_counts, true);
}

StepCounts EvalRunner::DoStepInternal(
    int64_t action, const StepCounts& event_counts, bool notify_episode_end)
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard grad_guard;

    if (!env_initialized_) {
        // 環境初期化
        auto reset_result = env_->Reset();
        ValidateEpisodeStructure(env_->GetName(), env_->GetBatchSpec(), *reset_result);
        state_ = reset_result->state;
        env_initialized_ = true;
        ANET_LOG_DEBUG("env_->Reset() done. state=" << state_.ToString());
    }

    // ステップ前情報
    auto train_step = step_counts_.train_step;
    ANET_LOG_DEBUG("step=" << train_step << " state=" << state_.ToString());

    // 行動選択
    auto action_info_raw = actor_->MakeAction(source_counts_, state_);
    ANET_LOG_DEBUG("step=" << train_step << " action_info_raw=" << action_info_raw->ToString());

    // action_infoを生成
    std::shared_ptr<anet::rl::BatchActionInfo> action_info = action_info_raw;
    if (action >= 0) {
        action_info = action_info_raw->WithAction(torch::tensor(
            { action },
            torch::TensorOptions().dtype(torch::kInt64))); // 指定のactionがあれば強制
    }

    // 環境ステップ実行
    auto result = env_->Step(action_info);    // next_state, reward, done, truncated
    ANET_LOG_DEBUG("step=" << train_step << " action=" << action_info->ToString());
    ANET_LOG_DEBUG("step=" << train_step << " next_state=" << result->next_state.ToString());
    ANET_LOG_DEBUG("step=" << train_step << " continue_state=" << result->continue_state.ToString());
    ANET_LOG_DEBUG("step=" << train_step << " reward=" << anet::ToString(result->reward));
    ANET_ASSERT_DEVICE(result->next_state.obs, torch::kCPU);
    ANET_ASSERT_DEVICE(result->next_state.done, torch::kCPU);
    ANET_ASSERT_DEVICE(result->next_state.truncated, torch::kCPU);
    ANET_ASSERT_DEVICE(result->reward, torch::kCPU);
    ANET_ASSERT_DEVICE(result->continue_state.obs, torch::kCPU);
    ANET_ASSERT_DEVICE(result->continue_state.done, torch::kCPU);
    ANET_ASSERT_DEVICE(result->continue_state.truncated, torch::kCPU);

	// メトリクス更新
	//UpdateMetrics(result);

    // カウント更新
    step_counts_.train_step++;
    step_counts_.exp_step += result->n_transitions;
    step_counts_.episode_count += result->n_episode_end;

    anet::rl::BatchExperience exp({ state_, action_info, result->reward, result->next_state });

    // 更新後処理
    auto self = this->shared_from_this();
    anet::rl::TrainEvent event{ exp, self, step_counts_, agent_, BatchUpdateResultList(), env_, result, action_info };
    notifier_->Notify(event);
    if (notify_episode_end) {
        AccumulateAndNotifyEpisodeEnd(self, result, event_counts);
    } else {
        last_step_had_episode_end_ = false;
    }
    state_ = result->continue_state;

    return step_counts_;
}

StepCounts EvalRunner::DoStep(int64_t action)
{
    return DoStep(action, step_counts_);
}

StepCounts EvalRunner::DoStep(const StepCounts& event_counts)
{
    return DoStep(-1, event_counts);
}

StepCounts EvalRunner::DoStep()
{
    return DoStep(-1);
}

void EvalRunner::RunSession(const StepCounts& event_counts, std::stop_token stop)
{
    ANET_PROFILE_FUNC();
    ANET_CHECK_MSG(session_env_ != nullptr,
        "EvalRunner::RunSession is only available for configured Eval. runner='" << name_ << "'.");

    // 同期・Reset・通知まで含むセッション全体を、train 側の開始座標に対応づける。
    // 開始行は発火時点で scheduler 側が出すため、ここでは所要時間の起点だけを取る。
    const auto session_start = std::chrono::high_resolution_clock::now();

    Sync(event_counts);
    const auto reset_result = session_env_->Reset();
    ValidateEpisodeStructure(env_->GetName(), env_->GetBatchSpec(), *reset_result);
    state_ = reset_result->state;
    env_initialized_ = true;

    // 通常の完了通知を抑制し、採用 episode の確定値が有効な Step 直後にだけ通知する。
    size_t completed = 0;
    while (!stop.stop_requested() && !session_env_->GetSessionResult().has_value()) {
        DoStepInternal(-1, event_counts, false);
        for (const int64_t group : session_env_->LastAdoptedGroups()) {
            const int env_index = session_env_->GetBatchSpec().episode_scope == EpisodeScope::PER_LANE
                ? static_cast<int>(group) : -1;
            notifier_->Notify(EpisodeEndEvent{
                .runner = shared_from_this(), .counts = event_counts,
                .agent = agent_, .env = session_env_, .env_index = env_index });
            ++completed;
        }
    }

    // 協調停止では部分 episode trace だけを残し、session 確定通知と scalar は出さない。
    if (!session_env_->GetSessionResult().has_value()) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - session_start).count();
        LOG::info() << std::format(
            "eval.[{}]: session cancelled learn_step={} exp_step={} elapsed={:.2f}s completed={}",
            name_, event_counts.learn_step, event_counts.exp_step, elapsed, completed);
        if (const auto logger = anet::MetricsLogger::Instance()) {
            logger->Log("eval.[" + name_ + "].session_cancelled", anet::json{
                {"learn_step", event_counts.learn_step},
                {"exp_step", event_counts.exp_step},
                {"elapsed_sec", elapsed},
                {"completed", completed},
            });
        }
        return;
    }

    // Runner の return と steps は全採用 episode の完了後に確定し、セッションを一度だけ通知する。
    const auto session_result = session_env_->GetSessionResult();
    ANET_CHECK(session_result.has_value());
    SetCompletedEpisodes(session_result->episode_returns, session_result->episode_steps);
    last_step_had_episode_end_ = true;
    SessionEndEvent event{
    	.runner = shared_from_this(),
    	.counts = event_counts,
        .agent = agent_,
        .env = env_ };
    notifier_->Notify(event);

    // 正常な通知完了後に、metrics と同じ確定値で所要時間を記録する。
    const double elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - session_start).count();
    LOG::info() << std::format(
        "eval.[{}]: session end learn_step={} exp_step={} elapsed={:.2f}s"
        " mean.episode_return={} max.episode_return={} mean.episode_steps={} max.episode_steps={}",
        name_, event_counts.learn_step, event_counts.exp_step, elapsed,
        GetScalar("mean.episode_return").value(), GetScalar("max.episode_return").value(),
        GetScalar("mean.episode_steps").value(), GetScalar("max.episode_steps").value());
}


// ======================================================
// TrainRunner
// ======================================================

TrainRunner::TrainRunner(
    //const ConfigData& config_data,
    std::shared_ptr<anet::rl::BatchEnv> env, std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Notifier> notifier, const ActorRequest& request)
    : RunnerBase(env, agent, notifier, request, "train")
{
    this->learner_ = agent_->CreateLearner();
}

std::optional<float> TrainRunner::GetScalar(const std::string& key, int64_t index) const
{
    if (key == TRAIN_REWARD) return last_reward_;
    if (key == TRAIN_REWARD_EMA) return reward_ema_.Value();

    if (key == TRAIN_STEP_PER_SEC) {
        return train_step_per_sec_ema_.IsInitialized()
            ? train_step_per_sec_ema_.Value()
            : std::numeric_limits<float>::quiet_NaN();
    }
    if (key == EXP_STEP_PER_SEC) {
        return exp_step_per_sec_ema_.IsInitialized()
            ? exp_step_per_sec_ema_.Value()
            : std::numeric_limits<float>::quiet_NaN();
    }

    if (key == ELAPSE_HOUR) {
        std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        auto elapse_msec = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count());
        auto elapse_hour = elapse_msec / 1000.0f / 60.0f / 60.0f;
        return elapse_hour;
    }

    return RunnerBase::GetScalar(key, index);
}

void TrainRunner::Shutdown()
{
    env_->Shutdown();
    notifier_->Clear();
}

void TrainRunner::CalcPerformanceMetrics()
{
    // メトリクス算出（処理性能系）
    auto train_step = step_counts_.train_step;
    auto exp_step = step_counts_.exp_step;
    auto train_step_delta = train_step - last_train_step_;
    auto exp_step_delta = exp_step - last_exp_step_;
    auto now = std::chrono::high_resolution_clock::now();
    auto usec_diff = std::chrono::duration_cast<std::chrono::microseconds>(now - last_time_).count();

    // 閾値未満は last_time_ を進めず次回へ繰り越す（時間を捨てない）
    if (usec_diff < kPerfMinUsec) return;

    // 窓の長さで重み付けするため、レートと経過時間を組で EMA へ渡す
    const float dt = static_cast<float>(usec_diff) / 1000000.0f;
    train_step_per_sec_ema_.Update(static_cast<float>(train_step_delta) / dt, dt);
    exp_step_per_sec_ema_.Update(static_cast<float>(exp_step_delta) / dt, dt);

    last_time_ = now;
    last_train_step_ = train_step;
    last_exp_step_ = exp_step;
}


// ------------------------------------------------------
// SerialTrainRunner
// ------------------------------------------------------

SerialTrainRunner::SerialTrainRunner(
    std::shared_ptr<anet::rl::BatchEnv> env, std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Notifier> notifier, const ActorRequest& request)
    : TrainRunner(env, agent, notifier, request)
{
    ;
}

StepCounts SerialTrainRunner::DoStep()
{
    ANET_PROFILE_FUNC();

    ANET_ASSERT(status_ == anet::rl::RunnerStatus::RUNNING);

    // EnvSpec取得
    auto env_spec = env_->GetSpec();
    auto batch_env_spec = env_->GetBatchSpec();

    if (!env_initialized_) {
        ANET_PROFILE_SCOPE(initialize);

        // 環境初期化
        auto reset_result = env_->Reset();
        ValidateEpisodeStructure(env_->GetName(), env_->GetBatchSpec(), *reset_result);
        state_ = reset_result->state;
        env_initialized_ = true;
        ANET_LOG_DEBUG("env_->Reset() done. state=" << state_.ToString());
        ANET_ASSERT_DEVICE_CPU_MSG(state_.obs, "Initial state");
        ANET_ASSERT(env_spec.state_spec.ValidateObservation(state_.obs));

        // 時間計測開始
        start_time_ = std::chrono::high_resolution_clock::now();
        last_time_ = start_time_;
    }

    ANET_PROFILE_SCOPE(make_action);

    // --- 学習ステップを回す ---
    float frame_total_reward = 0.0f;
    int frame_step = 0;

    // ステップ前情報
    auto train_step = step_counts_.train_step;

    // Stateチェック
    ANET_LOG_DEBUG("step=" << train_step);// << " state=" << state_.ToString());
    ANET_ASSERT_MSG(state_.obs.IsValid(), "state_.obs is invalid.");
//    ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.obs));
    const int N = state_.obs.Size(0);

    // 行動選択
    auto action_info = actor_->MakeAction(step_counts_, state_);
    //ANET_LOG_DEBUG("step=" << train_step << " action=" << action_info->ToString());
    ANET_ASSERT_SHAPE(action_info->GetAction(), {N});

    ANET_PROFILE_SCOPE_NEXT(env_step);

    // 環境ステップ実行
    auto result = env_->Step(action_info);    // next_state, reward, done, truncated
    //ANET_LOG_DEBUG("step=" << train_step << " reward=" << anet::ToString(result->reward));
    //ANET_LOG_DEBUG("step=" << train_step << " next_state=" << result->next_state.ToString());
    ANET_ASSERT_DEVICE(result->next_state.obs, torch::kCPU);
    ANET_ASSERT_DEVICE(result->next_state.done, torch::kCPU);
    ANET_ASSERT_DEVICE(result->next_state.truncated, torch::kCPU);
    ANET_ASSERT_DEVICE(result->reward, torch::kCPU);
    ANET_ASSERT_DEVICE(result->continue_state.obs, torch::kCPU);
    ANET_ASSERT_DEVICE(result->continue_state.done, torch::kCPU);
    ANET_ASSERT_DEVICE(result->continue_state.truncated, torch::kCPU);
    ANET_ASSERT_SHAPE(result->next_state.done, { N });
    ANET_ASSERT_SHAPE(result->next_state.truncated, { N });
    ANET_ASSERT_SHAPE(result->reward, { N });
    ANET_ASSERT_SHAPE(result->continue_state.done, { N });
    ANET_ASSERT_SHAPE(result->continue_state.truncated, { N });

    auto self = this->shared_from_this();

    ANET_PROFILE_SCOPE_NEXT(env_step_post);

    //メトリクス更新
    UpdateMetrics(result);
    AccumulateAndNotifyEpisodeEnd(self, result, step_counts_);

    // Agent更新
    ANET_PROFILE_SCOPE_NEXT(update_agent);
    anet::rl::BatchExperience exp({
        state_.Clone(),
        action_info,                   // ActionはAgent内で新規アロケートされているため安全
        result->reward.clone(),
        result->next_state.Clone()
        });
    auto update_results = learner_->UpdateFromBatch(step_counts_, exp);

    // LearnEvent
    ANET_PROFILE_SCOPE_NEXT(learn_event);
    if (!update_results.empty()) {
        torch::NoGradGuard grad_guard;

        anet::rl::LearnEvent event{  exp, self, step_counts_, agent_, update_results };
        notifier_->Notify(event);
    }

    // TrainEvent
    ANET_PROFILE_SCOPE_NEXT(train_event);
    {
        torch::NoGradGuard grad_guard;

        anet::rl::TrainEvent train_event{ exp, self, step_counts_, agent_, update_results, env_, result, action_info };
        notifier_->Notify(train_event);
        state_ = result->continue_state;
    }

    // カウント更新(通知した後に次のSTEPに備えて更新する、の方針）
    step_counts_.train_step++;
    step_counts_.update_step++;
    step_counts_.learn_step += update_results.size();
    step_counts_.exp_step += result->n_transitions;
    step_counts_.episode_count += result->n_episode_end;

    // 性能関連メトリクスを更新
    CalcPerformanceMetrics();

    return step_counts_;
}


// ------------------------------------------------------
// PipelineTrainRunner
// ------------------------------------------------------

PipelineTrainRunner::PipelineTrainRunner(
    std::shared_ptr<anet::rl::BatchEnv> env, std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Notifier> notifier, const ActorRequest& request)
    : TrainRunner(env, agent, notifier, request)
{
    // 専用の学習用バックグラウンドスレッドを1つだけ生成
    learn_pool_ = std::make_unique<anet::PinnedThreadPool>(1, "LearnThread");
}

void PipelineTrainRunner::Shutdown()
{
    if (learn_pool_) {
        learn_pool_->WaitAll();
        learn_pool_->Stop();
    }
    TrainRunner::Shutdown();
}

StepCounts PipelineTrainRunner::DoStep()
{
    ANET_PROFILE_FUNC();

    // ==========================================================
    // 初期化処理
    // ==========================================================
    if (!env_initialized_) {
        ANET_PROFILE_SCOPE(initialize);

        // EnvSpec取得
        auto env_spec = env_->GetSpec();

        // 環境初期化
        auto reset_result = env_->Reset();
        ValidateEpisodeStructure(env_->GetName(), env_->GetBatchSpec(), *reset_result);
        state_ = reset_result->state;
        env_initialized_ = true;
        //ANET_LOG_DEBUG("env_->Reset() done. state=" << state_.ToString());
        ANET_ASSERT_DEVICE_CPU_MSG(state_.obs, "Initial state");
        ANET_ASSERT(env_spec.state_spec.ValidateObservation(state_.obs));

        // 時間計測開始
        start_time_ = std::chrono::high_resolution_clock::now();
        last_time_ = start_time_;
    }

    // ==========================================================
    // 待ち合わせ (前回の学習完了を確実に待つ)
    // ==========================================================
    if (learn_future_.valid()) {
    	// Learn完了待ち＆結果取得
        auto result_list = learn_future_.get();

        // LearnEvent
        if (!result_list.empty()) {
            torch::NoGradGuard grad_guard;
            anet::rl::LearnEvent learn_event{ prev_exp_, shared_from_this(), prev_counts_, agent_, result_list };
            notifier_->Notify(learn_event);
        }

        // TrainEvent
        {
            torch::NoGradGuard grad_guard;
            anet::rl::TrainEvent train_event{
                prev_exp_, shared_from_this(), prev_counts_, agent_, result_list, env_, prev_result_, prev_action_info_
            };
            notifier_->Notify(train_event);
        }

        step_counts_.update_step++;
        step_counts_.learn_step += result_list.size();
    }

    // ==========================================================
    // 推論 (GPUが空なので最速で終わる)
    // ==========================================================
    auto action_info = actor_->MakeAction(step_counts_, state_);

    // ==========================================================
    // 非同期学習の投入
    // ==========================================================
    if (has_prev_data_) {
        auto learner = learner_;
        auto exp = prev_exp_;
        auto counts = prev_counts_;

        learn_future_ = learn_pool_->EnqueueFuture(0, [learner, counts, exp]() {
            return learner->UpdateFromBatch(counts, exp);
            });
    }

    // ==========================================================
    // ENVを動かす(LearnerがGPUを全力で回している裏で重いEnv処理をCPUで回す)
    // ==========================================================
    auto result = env_->Step(action_info);
    UpdateMetrics(result);
    AccumulateAndNotifyEpisodeEnd(shared_from_this(), result, step_counts_);

    // ==========================================================
    // 後片付け
    // ==========================================================

    // 次ステップ向けのデータ保存
    prev_exp_ = anet::rl::BatchExperience({
        state_.Clone(),
        action_info,                   // ActionはAgent内で新規アロケートされているため安全
        result->reward.clone(),
        result->next_state.Clone()
        });
    prev_result_ = result;
    prev_action_info_ = action_info;
    prev_counts_ = step_counts_;
    has_prev_data_ = true;

	// 次ステップ向けにStepを更新
    state_ = result->continue_state;

    // カウンタ更新
    step_counts_.train_step++;
    step_counts_.exp_step += result->n_transitions;
    step_counts_.episode_count += result->n_episode_end;

    // 性能関連メトリクスを更新
    CalcPerformanceMetrics();

    return step_counts_;
}


// ------------------------------------------------------
// RunnerFactory
// ------------------------------------------------------
std::shared_ptr<TrainRunner> RunnerFactory::CreateMainRunner(
    const std::string& type, std::shared_ptr<anet::rl::BatchEnv> env,
    std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Notifier> notifier, const ActorRequest& request)
{
    if (anet::ToLower(type) == "pipeline") {
        return std::make_shared<PipelineTrainRunner>(env, agent, notifier, request);
    } else {
        return std::make_shared<SerialTrainRunner>(env, agent, notifier, request);
    }
}


// ======================================================
// RunManager
// ======================================================

struct RunManager::Config : public anet::Config
{
    uint64_t seed = 0;
    int num_envs = 1;
    std::string main_runner_type = "serial";
    std::string actor = "train";

    std::string eval_device = "auto";
    torch::Device effective_eval_device = torch::kCPU;

    Config(const anet::ConfigData& config_data, const std::string& config_prefix = "run")
        : anet::Config(config_data, config_prefix)
    {
        ANET_READ_CONFIG(config_data, seed);
        ReadConfig(config_data, "train.num_envs", num_envs);
        ReadConfig(config_data, "train.runner_type", main_runner_type);
        ReadConfig(config_data, "train.actor", actor);
        ANET_READ_CONFIG(config_data, eval_device);
        effective_eval_device = anet::ParseDevice(eval_device);
        my_config_json_["effective_eval_device"] = effective_eval_device.str();
    }

    torch::Device GetEvalDevice() const
    {
        return effective_eval_device;
    }
};

RunManager::RunManager(const ConfigData& config_data)
{
    // Config
    config_ = std::make_unique<Config>(config_data);
    LOG::info() << "RunManager effective_eval_device=" << config_->GetEvalDevice();

    // BatchEnvを1つも構築する前に、設定から決まるnameを一括検証する。
    auto eval_configs = config_data.MakeSubConfigData("run.eval");
    auto eval_schedule_configs = config_data.MakeSubConfigData("run.eval_schedule");
    struct EvalScheduleConfig {
        int interval;
        bool use_background;
        bool wait_on_exit;
    };
    std::unordered_map<std::string, EvalScheduleConfig> resolved_eval_schedules;
    std::unordered_map<std::string, std::string> planned_name_owners;
    const auto validate_planned_name = [&planned_name_owners](
        const std::string& name, const std::string& requested_owner) {
        if (name.empty()) {
            ANET_SYSTEM_ERROR("Env name must not be empty. requested_owner='" << requested_owner
                << "'. Env names must be unique within a Run.");
        }
        const auto [it, inserted] = planned_name_owners.emplace(name, requested_owner);
        if (!inserted) {
            ANET_SYSTEM_ERROR("Duplicate Env name '" << name << "' within Run: existing_owner='"
                << it->second << "', requested_owner='" << requested_owner
                << "'. Env names must be unique within a Run.");
        }
    };
    validate_planned_name("train", "main Train");
    validate_planned_name("EvalPanel", "EvalPanel");
    for (const auto& [tag, eval_config] : eval_configs) {
        (void)eval_config;
        validate_planned_name(tag, "configured Eval tag '" + tag + "'");
    }
    for (const auto& [tag, schedule_config] : eval_schedule_configs) {
        if (!eval_configs.contains(tag)) {
            ANET_SYSTEM_ERROR("Unknown run.eval_schedule tag '" << tag
                << "'. Define run.eval.[" << tag << "] before scheduling it.");
        }
        if (!schedule_config.Has("interval")) {
            ANET_SYSTEM_ERROR("Missing required run.eval_schedule.[" << tag
                << "].interval. Expected a non-negative integer (0 disables the schedule).");
        }
        int interval = 0;
        schedule_config.Read("interval", interval, interval);
        if (interval < 0) {
            ANET_SYSTEM_ERROR("Invalid run.eval_schedule.[" << tag << "].interval=" << interval
                << ". Expected a non-negative value.");
        }
        bool use_background = true;
        schedule_config.Read("use_background", use_background, use_background);
        bool wait_on_exit = true;
        schedule_config.Read("wait_on_exit", wait_on_exit, wait_on_exit);
        resolved_eval_schedules.emplace(tag, EvalScheduleConfig{
            .interval = interval,
            .use_background = use_background,
            .wait_on_exit = wait_on_exit,
        });
    }

    // Env種別を解決し、以後の生成は共通builder/factory契約だけで扱う。
    anet::rl::BatchEnvBuilderConfig env_config(config_data);

    // seed
    if (config_->seed == 0) {
        master_seed_ = std::make_unique<anet::MasterSeedManager>();
    } else {
        master_seed_ = std::make_unique<anet::MasterSeedManager>(config_->seed);
    }

    // seed値生成
    auto global_seed = master_seed_->GetMasterSeed();
    auto train_env_seed = master_seed_->GetGroupSeed("env");
    auto agent_seed = master_seed_->GetGroupSeed("agent");
    LOG::info() << "global_seed=" << global_seed << " train_env_seed="
        << train_env_seed << " agent_seed=" << agent_seed;

    // パラメータ記録
    anet::MetricsLogger::Instance()->Log("train/seed",
        { "global_seed", global_seed, "agent_seed", agent_seed, "train_env_seed", train_env_seed });
    anet::MetricsLogger::Instance()->Log(*config_);
    anet::MetricsLogger::Instance()->Flush();

    // Notifier生成
    notifier_ = std::make_shared<Notifier>();

    // BatchEnv生成
    LOG::info() << "env_config=" << env_config.ToString();
    env_factory_ = std::make_unique<anet::rl::BatchEnvBuilder>(env_config, config_data, config_->num_envs);
    env_class_id_ = env_config.class_id;
    auto env_device = env_factory_->GetDevice();
    EnsureEnvNameAvailable("train", "main Train");
    env_ = env_factory_->CreateBatchEnv("train", train_env_seed, -1, anet::rl::RunMode::Train);
    if (env_ == nullptr) {
        LOG::error() << "Failed to create env. class_id=\"" << env_class_id_
            << "\". Check that the selected workspace config includes an environment config.";
        return;
    }
    LogEnvConfig(*env_);

    // BatchEnvログ
    auto batch_env_spec = env_->GetBatchSpec();
    auto env_spec = env_->GetSpec();
    LOG::info() << "batch_env_spec=" << batch_env_spec.ToString();
    LOG::info() << "env_spec=" << env_spec.ToString();
    anet::MetricsLogger::Instance()->Log("env/batch_env_spec", batch_env_spec.ToJson());
    anet::MetricsLogger::Instance()->Log("env/env_spec", env_spec.ToJson());
    anet::MetricsLogger::Instance()->Flush();

    // Agent生成
    anet::rl::DefaultAgentFactoryConfig agent_factory_config(config_data);
    auto agent_factory = anet::rl::DefaultAgentFactory(
        agent_factory_config, env_spec, batch_env_spec, config_data, agent_seed);
    auto agent_device = agent_factory.GetDevice();
    agent_ = agent_factory.CreateAgent(notifier_);
    if (agent_ == nullptr) {
        LOG::error() << "Failed to create agent.";
        return;
    }

    // TrainRunner生成
    const ActorRequest train_request{.batch_env_spec = batch_env_spec, .env_spec = env_spec,
        .device = agent_device, .seed = master_seed_->GetGroupSeed("actor/train"), .actor_key = config_->actor};
    train_runner_ = anet::rl::RunnerFactory::CreateMainRunner(config_->main_runner_type, env_, agent_, notifier_, train_request);
    RegisterEnvName("train", "main Train");

    // 設定からObserverを生成して登録
    anet::rl::ObserverFactory factory(config_data);

    // 構築した eval の採用予定数を、後段で attach 済み metric 定義へ付与する。
    std::unordered_map<std::string, int> eval_episode_counts;
    // EpisodeEvalObserver
    for (const auto& kv : eval_configs) {
        // Eval設定取得
        const auto& tag = kv.first;
        const auto& eval_config_data = kv.second;
        std::string config_prefix = "run.eval.[" + tag + "].env";

        // Eval設定ログ
        anet::MetricsLogger::Instance()->Log(config_prefix, eval_config_data.ToJson());

        // RunMode取得
        std::string run_mode_str = "eval";
        eval_config_data.Read("run_mode", run_mode_str, run_mode_str);
        anet::rl::RunMode run_mode = anet::rl::RunModeFromString(run_mode_str);
        configured_eval_run_modes_.emplace(tag, run_mode);
        std::string actor_key = tag;
        eval_config_data.Read("actor", actor_key, actor_key);
        configured_actor_keys_.emplace(tag, actor_key);

        int eval_batch_size = 1;
        eval_config_data.Read("eval_batch_size", eval_batch_size, eval_batch_size);
        if (eval_batch_size <= 0) {
            ANET_SYSTEM_ERROR("Invalid run.eval.[" << tag << "].eval_batch_size="
                << eval_batch_size << ". Expected a positive value.");
        }

        int eval_episodes = 1;
        eval_config_data.Read("eval_episodes", eval_episodes, eval_episodes);
        if (eval_episodes <= 0) {
            ANET_SYSTEM_ERROR("Invalid run.eval.[" << tag << "].eval_episodes="
                << eval_episodes << ". Expected a positive value.");
        }


        // definition-only tagも宣言時schemaだけは検証し、Env/manifest/actorは生成しない。
        env_factory_->ValidateConfig(run_mode, config_prefix);
        const auto schedule_it = resolved_eval_schedules.find(tag);
        if (schedule_it == resolved_eval_schedules.end()) {
            LOG::info() << "eval.[" << tag << "]: definition-only";
            dormant_eval_tags_.insert(tag);
            RegisterEnvName(tag, "dormant configured Eval tag '" + tag + "'");
            continue;
        }

        const auto& schedule_config = schedule_it->second;
        const int interval = schedule_config.interval;
        const bool use_background = schedule_config.use_background;
        const bool wait_on_exit = schedule_config.wait_on_exit;

        if (interval == 0) {
            LOG::info() << "eval.[" << tag << "]: definition-only";
            dormant_eval_tags_.insert(tag);
            RegisterEnvName(tag, "dormant configured Eval tag '" + tag + "'");
            continue;
        }

        LOG::info() << "eval.[" << tag << "]: scheduled (interval=" << interval
            << ", background=" << (use_background ? "true" : "false")
            << ", wait_on_exit=" << (wait_on_exit ? "true" : "false")
            << ", episodes=" << eval_episodes << ", batch_size=" << eval_batch_size << ")";

        // この active Eval の session-end ENV metrics だけを decorator の購読対象にする。
        std::vector<std::string> subscribed_env_keys;
        std::unordered_set<std::string> subscribed_env_key_set;
        for (const auto& def : factory.GetScalarMetricDefs()) {
            const auto& subscription = def.subscription;
            if (def.scope != RunnerScope::EVAL || def.eval_name != tag
                || subscription.event != EventType::SESSION_END
                || subscription.target != EventField::ENV) {
                continue;
            }
            if (eval_episodes > 1 && !anet::ParseScalarAggregationKey(def.source_key).has_value()) {
                ANET_SYSTEM_ERROR("Multi-episode Eval ENV scalar requires an aggregation prefix. "
                    "eval_tag='" << tag << "' eval_episodes=" << eval_episodes
                    << " source_key='" << def.source_key << "' expected=mean|max|min|std.");
            }
            if (subscribed_env_key_set.insert(def.source_key).second) {
                subscribed_env_keys.push_back(def.source_key);
            }
        }

        // EvalRunner生成&登録
        auto actor_device = config_->GetEvalDevice();
        const auto owner = "configured Eval tag '" + tag + "'";
        EnsureEnvNameAvailable(tag, owner);
        const auto eval_seed_domain = "eval_env/" + tag;
        const auto eval_seed = master_seed_->GetGroupSeed(eval_seed_domain.c_str());
        auto eval_env = env_factory_->CreateBatchEnv(
            tag, eval_seed, eval_batch_size, run_mode, config_prefix);
        const auto batch_spec = eval_env->GetBatchSpec();
        const int64_t group_count = batch_spec.episode_scope == EpisodeScope::PER_LANE
            ? batch_spec.num_envs : 1;
        if (eval_episodes < group_count) {
            LOG::warn() << "Eval session has fewer adopted episodes than episode groups. "
                << "eval_tag='" << tag << "' eval_episodes=" << eval_episodes
                << " group_count=" << group_count << ".";
        }
        auto session_env = std::make_shared<EvalSessionEnv>(
            eval_env, eval_episodes, std::move(subscribed_env_keys));
        LogEnvConfig(*session_env);
        const ActorRequest request{.batch_env_spec = session_env->GetBatchSpec(), .env_spec = session_env->GetSpec(),
            .device = actor_device, .seed = master_seed_->GetGroupSeed(("actor/" + tag).c_str()), .actor_key = actor_key};
        auto eval_runner = std::make_shared<EvalRunner>(session_env, agent_, notifier_, request, tag);
        eval_runners.emplace(tag, eval_runner);
        eval_episode_counts.emplace(tag, eval_episodes);
        RegisterEnvName(tag, owner);

        // EvalObserver生成&登録
        notifier_->AttachScoped<anet::rl::EpisodeEvalObserver>(
            train_runner_,
            eval_runner,
            interval,
            use_background,
            wait_on_exit
        );
    };

    auto resolve_runner = [&](RunnerScope scope, const std::string& eval_name) -> std::shared_ptr<const Runner> {
        if (scope == RunnerScope::TRAIN) return train_runner_;
        auto it = eval_runners.find(eval_name);
        if (it == eval_runners.end()) {
            if (dormant_eval_tags_.contains(eval_name)) {
                if (warned_dormant_metric_tags_.insert(eval_name).second) {
                    LOG::warn() << "Skipping metrics for unscheduled eval tag. tag='" << eval_name
                        << "'. The tag is defined in run.eval but has no active "
                        << "run.eval_schedule entry.";
                }
                return nullptr;
            }
            ANET_SYSTEM_ERROR("Unknown eval name '" << eval_name << "' in metrics config.");
        }
        return it->second;
    };

    auto train_obs = factory.GetUpdateObservers();
    auto learn_obs = factory.GetLearnObservers();
    auto episode_end_obs = factory.GetEpisodeEndObservers();
    for (const auto& p : train_obs) {
        auto runner = resolve_runner(p.scope, p.eval_name);
        if (runner == nullptr) continue;
        auto scoped_obs = std::make_shared<RunnerScopedTrainObserver>(p.obs, runner);
        notifier_->Attach(scoped_obs);
    }
    for (const auto& p : learn_obs) {
        auto runner = resolve_runner(p.scope, p.eval_name);
        if (runner == nullptr) continue;
        auto scoped_obs = std::make_shared<RunnerScopedLearnObserver>(p.obs, runner);
        notifier_->Attach(scoped_obs);
    }
    for (const auto& p : episode_end_obs) {
        auto runner = resolve_runner(p.scope, p.eval_name);
        if (runner == nullptr) continue;
        auto scoped_obs = std::make_shared<RunnerScopedEpisodeEndObserver>(p.obs, runner);
        notifier_->Attach(scoped_obs);
    }
    for (const auto& p : factory.GetSessionEndObservers()) {
        auto runner = resolve_runner(p.scope, p.eval_name);
        if (runner == nullptr) continue;
        auto scoped_obs = std::make_shared<RunnerScopedSessionEndObserver>(p.obs, runner);
        notifier_->Attach(scoped_obs);
    }

    // 実際に attach した scalar metric の解決済み定義を 1 レコードだけ残す。
    // 解析側が config から step 座標系や source key を再導出しないための正本 (ADR 0029)。
    // 既存の type="json" record を使うため、Metrics Viewer の取り込みと cache 契約は変わらない。
    const auto complete_metric_definition = [&](auto& def) {
        // dormant eval は除外し、実際の購読先から eval 条件を確定する。
        const auto runner = resolve_runner(def.scope, def.eval_name);
        if (runner == nullptr) return false;
        if (def.scope == RunnerScope::EVAL) {
            def.eval_episodes = eval_episode_counts.at(def.eval_name);
            def.num_envs = runner->GetBatchEnv()->GetBatchSpec().num_envs;
        }
        return true;
    };
    std::vector<anet::rl::ObserverFactory::ScalarMetricDef> attached_defs;
    for (auto def : factory.GetScalarMetricDefs()) {
        if (!complete_metric_definition(def)) continue;
        attached_defs.push_back(std::move(def));
    }
    anet::json metric_defs = anet::rl::ScalarMetricDefsToJson(attached_defs);
    if (!metric_defs.empty()) {
        anet::MetricsLogger::Instance()->Log("metrics.scalar.defs", metric_defs);
    }

    // trace は別チャネルの定義とし、dormant eval と scalar 購読ヒントから分離する。
    std::vector<ObserverFactory::TraceMetricDef> attached_trace_defs;
    for (auto def : factory.GetTraceMetricDefs()) {
        if (!complete_metric_definition(def)) continue;
        attached_trace_defs.push_back(std::move(def));
    }
    const auto trace_defs = TraceMetricDefsToJson(attached_trace_defs);
    if (!trace_defs.empty()) {
        anet::MetricsLogger::Instance()->Log("metrics.trace.defs", trace_defs);
    }

    // 実際に attach された定義だけを、学習開始前の静的な購読情報として Agent へ渡す。
    std::vector<ScalarMetricSubscription> subscriptions;
    subscriptions.reserve(attached_defs.size());
    for (const auto& def : attached_defs) subscriptions.push_back(def.subscription);
    agent_->ConfigureScalarMetricSubscriptions(subscriptions);

    // 成功！
    status_ = anet::rl::RunnerStatus::RUNNING;
}

RunManager::~RunManager()
{
    this->env_.reset();
    this->agent_.reset();
}

void RunManager::EnsureEnvNameAvailable(const std::string& name, const std::string& requested_owner) const
{
    if (name.empty()) {
        ANET_SYSTEM_ERROR("Env name must not be empty. requested_owner='" << requested_owner
            << "'. Env names must be unique within a Run.");
    }
    const auto it = env_name_owners_.find(name);
    if (it != env_name_owners_.end()) {
        ANET_SYSTEM_ERROR("Duplicate Env name '" << name << "' within Run: existing_owner='"
            << it->second << "', requested_owner='" << requested_owner
            << "'. Env names must be unique within a Run.");
    }
}

void RunManager::RegisterEnvName(const std::string& name, const std::string& owner)
{
    EnsureEnvNameAvailable(name, owner);
    env_name_owners_.emplace(name, owner);
}

void RunManager::LogEnvConfig(const BatchEnv& env)
{
    const auto config_data = env.GetConfigData();
    if (!config_data.has_value()) {
        if (warned_unsupported_env_config_names_.insert(env.GetName()).second) {
            LOG::warn() << "Skipping Env config dump because GetConfigData is unsupported. env_name='"
                << env.GetName() << "'";
        }
        return;
    }

    // 実際のファイル名で所有者を管理し、sanitize 後に同じパスへ上書きされることを防ぐ。
    const auto filename = SanitizeEnvConfigFilename("env." + env.GetName()) + ".txt";
    const auto [it, inserted] = env_config_file_owners_.emplace(filename, env.GetName());
    ANET_CHECK_MSG(inserted || it->second == env.GetName(),
        "Env config filename collision. filename='" << filename
        << "' existing_env_name='" << it->second
        << "' requested_env_name='" << env.GetName() << "'");

    anet::MetricsLogger::Instance()->Log("env." + env.GetName(), *config_data);
}

std::shared_ptr<EvalRunner> RunManager::CreateEvalRunner(
    const std::string& name, const std::string& config_tag)
{
    ANET_ASSERT(status_ == anet::rl::RunnerStatus::RUNNING);
    const auto owner = "CreateEvalRunner '" + name + "'";
    EnsureEnvNameAvailable(name, owner);
    const auto selected_tag = config_tag.empty() ? name : config_tag;
    const auto it = configured_eval_run_modes_.find(selected_tag);
    if (it == configured_eval_run_modes_.end()) {
        ANET_SYSTEM_ERROR("Unknown Eval config tag='" << selected_tag << "'. Expected a declared run.eval.[tag].");
    }
    const auto seed = master_seed_->GetGroupSeed(("eval_panel/" + selected_tag).c_str());
    auto env = env_factory_->CreateBatchEnv(name, seed, 1, it->second, "run.eval.[" + selected_tag + "].env");
    LogEnvConfig(*env);
    const ActorRequest request{.batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(),
        .device = config_->GetEvalDevice(), .seed = master_seed_->GetGroupSeed(("actor/" + name).c_str()),
        .actor_key = configured_actor_keys_.at(selected_tag)};
    auto runner = std::make_shared<EvalRunner>(env, agent_, notifier_, request, name);
    eval_runners.emplace(name, runner);
    RegisterEnvName(name, owner);
    return runner;
}

// ======================================================
// RunnerThread
// ======================================================

RunnerThread::RunnerThread(
    const std::string& name,
    std::shared_ptr<anet::rl::Runner> runner,
    anet::rl::Runner::ControlFunction pre_func,
    anet::rl::Runner::ControlFunction post_func,
    ExceptionFunction exception_func)
    : anet::ThreadBase(name)
    , runner_(runner)
    , pre_func_(pre_func)
    , post_func_(post_func)
    , exception_func_(exception_func)
{
}

RunnerThread::~RunnerThread()
{
    Stop();
}

bool RunnerThread::ProcessStep()
{
    // 1フレーム実行 (既存のコールバック機構をそのまま利用)
    runner_->DoUpdateFrame(1, pre_func_, post_func_);

    // 完了ステータスになったら false を返してスレッドループを抜ける
    if (runner_->GetStatus() == anet::rl::RunnerStatus::COMPLETED) {
        return false;
    }
    return true;
}

void RunnerThread::OnException()
{
    if (exception_func_ != nullptr) {
        exception_func_();
    }
}
