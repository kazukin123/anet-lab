#include <limits>
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

static int16_t NormalizeSharedActorDeviceIndex(const torch::Device& device)
{
    if (device.type() != torch::kCUDA) {
        return -1;
    }
    return (device.has_index() && device.index() >= 0) ? device.index() : 0;
}

static bool IsSameSharedActorDevice(const torch::Device& lhs, const torch::Device& rhs)
{
    if (lhs.type() != rhs.type()) {
        return false;
    }
    if (lhs.type() == torch::kCUDA) {
        return NormalizeSharedActorDeviceIndex(lhs) == NormalizeSharedActorDeviceIndex(rhs);
    }
    return true;
}

static void ValidateSharedActorDevice(
    const std::string& runner_name,
    std::optional<bool> clone_model_override,
    const torch::Device& actor_device,
    const torch::Device& agent_device)
{
    if (!clone_model_override.has_value() || *clone_model_override) {
        return;
    }

    // shared actor は学習側 network を直接使うため、入力 device と network device を一致させる
    ANET_CHECK_MSG(IsSameSharedActorDevice(actor_device, agent_device),
        "RunnerBase actor device mismatch: runner_name=" << runner_name
        << " clone_model=false"
        << " actor_device=" << actor_device.str()
        << " agent_device=" << agent_device.str()
        << ". Set train.eval_device_type to the agent device, or set train.eval.["
        << runner_name << "].clone_model=true.");
}

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
    std::shared_ptr<anet::rl::BatchEnv> env, std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Notifier> notifier, RunMode run_mode, std::optional<bool> clone_model_override, std::optional<torch::Device> device, std::string name)
    : name_(std::move(name))
    , env_(env)
    , agent_(agent)
    , notifier_(notifier)
    , run_mode_(run_mode)
    , reward_ema_(0.001)
{
    InitializeMetrics();
    status_ = anet::rl::RunnerStatus::RUNNING;
    auto batch_env_spec = env_->GetBatchSpec();
    auto env_spec = env_->GetSpec();
    const auto agent_device = agent_->GetDevice();
    const auto actor_device = device.value_or(agent_device);
    ValidateSharedActorDevice(name_, clone_model_override, actor_device, agent_device);
    actor_ = agent_->CreateActor(
        batch_env_spec, env_spec, run_mode_, clone_model_override, actor_device);
}

void RunnerBase::InitializeMetrics()
{
    ANET_CHECK(env_ != nullptr);

    auto env_device = env_->GetDevice();
    auto batch_env_spec = env_->GetBatchSpec();

    // メトリクス初期化
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(env_device);
    episode_total_reward_cur_ = torch::zeros({ batch_env_spec.num_envs }, fopt);
    ANET_ASSERT_SHAPE(episode_total_reward_cur_, { batch_env_spec.num_envs });
    eps_total_reward_per_env_.assign(static_cast<size_t>(batch_env_spec.num_envs), 0.0f);
    last_episode_total_reward_ = std::numeric_limits<float>::quiet_NaN();
    last_step_had_episode_end_ = false;
}

void RunnerBase::UpdateMetrics(std::shared_ptr<const BatchStepResult> result)
{
    // 平均報酬更新
    float step_reward_mean = result->reward.mean().item<float>();
    last_reward_ = step_reward_mean;
    reward_ema_.Update(last_reward_);

    // エピソード合計報酬更新
    ANET_CHECK(episode_total_reward_cur_.defined());
    episode_total_reward_cur_ += result->reward; // 現エピソード報酬加算
    auto finished = result->next_state.done.to(torch::kBool) | result->next_state.truncated.to(torch::kBool);   // 終了マスク
    episode_total_reward_comp_ = episode_total_reward_cur_.masked_select(finished);  // 終了したエピソードの報酬を確定
    episode_total_reward_cur_.masked_fill_(finished, 0.0f);  // 終了したENVのエピソード総報酬をゼロクリア
    //ANET_LOG_DEBUG("finished=" << anet::ToString(finished));
    //ANET_LOG_DEBUG("episode_total_reward_comp_=" << anet::ToString(episode_total_reward_comp_));
    //ANET_LOG_DEBUG("episode_total_reward_cur_=" << anet::ToString(episode_total_reward_cur_));
}

bool RunnerBase::AccumulateAndNotifyEpisodeEnd(
    std::shared_ptr<const Runner> self, std::shared_ptr<const BatchStepResult> result, const StepCounts& event_counts)
{
    ANET_PROFILE_FUNC();

    ANET_CHECK(self != nullptr);
    ANET_CHECK(result != nullptr);

    const auto batch_env_spec = env_->GetBatchSpec();
    const int64_t num_envs = batch_env_spec.num_envs;
    if (static_cast<int64_t>(eps_total_reward_per_env_.size()) != num_envs) {
        eps_total_reward_per_env_.assign(static_cast<size_t>(num_envs), 0.0f);
    }

    // CPUテンソルに揃え、accessor取得の前提として contiguous 化しておく
    auto rewards_cpu = result->reward.to(torch::kCPU).contiguous();
    auto finished_cpu = (result->next_state.done.to(torch::kBool) | result->next_state.truncated.to(torch::kBool))
        .to(torch::kCPU).contiguous();
    ANET_ASSERT_SHAPE(rewards_cpu, { num_envs });
    ANET_ASSERT_SHAPE(finished_cpu, { num_envs });

    // ループ外で1度だけaccessorを取得し、per-elementの Tensor::operator[] + item() を回避
    auto rewards_acc = rewards_cpu.accessor<float, 1>();
    auto finished_acc = finished_cpu.accessor<bool, 1>();

    last_step_had_episode_end_ = false;
    float completed_total_reward_sum = 0.0f;

    // 終了したenvのindexを1パスで集める（2回目のループでfinishedを再評価しないため）
    std::vector<int> finished_indices;
    finished_indices.reserve(static_cast<size_t>(num_envs));

    // 第1パス: 報酬を各envに累積し、終了したenvを集計
    for (int64_t i = 0; i < num_envs; ++i) {
        eps_total_reward_per_env_[static_cast<size_t>(i)] += rewards_acc[i];
        if (!finished_acc[i]) continue;

        last_step_had_episode_end_ = true;
        completed_total_reward_sum += eps_total_reward_per_env_[static_cast<size_t>(i)];
        finished_indices.push_back(static_cast<int>(i));
    }

    // 完了エピソードの平均total_rewardを更新
    if (!finished_indices.empty()) {
        last_episode_total_reward_ = completed_total_reward_sum / static_cast<float>(finished_indices.size());
    }

    // 第2パス: 終了envについてEpisodeEndEvent通知＆累積リセット（昇順index順）
    for (int idx : finished_indices) {
        const float eps_total_reward = eps_total_reward_per_env_[static_cast<size_t>(idx)];
        EpisodeEndEvent event{ self, event_counts, agent_, env_, idx, eps_total_reward };
        notifier_->Notify(event);
        eps_total_reward_per_env_[static_cast<size_t>(idx)] = 0.0f;
    }

    return last_step_had_episode_end_;
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
    if (key == EPS_TOTAL_REWARD) return last_episode_total_reward_;


    return std::nullopt;
}


// ======================================================
// EvalRunner
// ======================================================

EvalRunner::EvalRunner(
    std::shared_ptr<anet::rl::BatchEnv> env,
    std::shared_ptr<anet::rl::Agent> agent,
    std::shared_ptr<anet::rl::Notifier> notifier,
    RunMode run_mode,
    bool clone_model,
    std::optional<torch::Device> device,
    std::string name)
    : RunnerBase(env, agent, notifier, run_mode, clone_model, device, std::move(name))
{
}

void EvalRunner::Sync()
{
    actor_->Sync();
}

StepCounts EvalRunner::DoStep(int64_t action, const StepCounts& event_counts)
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard grad_guard;

    if (!env_initialized_) {
        // 環境初期化
        auto reset_result = env_->Reset();
        state_ = reset_result->state;
        env_initialized_ = true;
        ANET_LOG_DEBUG("env_->Reset() done. state=" << state_.ToString());
    }

    // ステップ前情報
    auto train_step = step_counts_.train_step;
    ANET_LOG_DEBUG("step=" << train_step << " state=" << state_.ToString());

    // 行動選択
    auto action_info_raw = actor_->MakeAction(step_counts_, state_);
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
    AccumulateAndNotifyEpisodeEnd(self, result, event_counts);
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


// ======================================================
// TrainRunner
// ======================================================

TrainRunner::TrainRunner(
    //const ConfigData& config_data,
    std::shared_ptr<anet::rl::BatchEnv> env, std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Notifier> notifier)
    : RunnerBase(env, agent, notifier, anet::rl::RunMode::Train, std::nullopt, std::nullopt, "train")
{
    this->learner_ = agent_->CreateLearner();
}

std::optional<float> TrainRunner::GetScalar(const std::string& key, int64_t index) const
{
    if (key == TRAIN_REWARD) return last_reward_;
    if (key == TRAIN_REWARD_EMA) return reward_ema_.Value();

    if (key == TRAIN_EPISODE_REWARD) {
        if (episode_total_reward_comp_.defined()) {
			/// @todo 平均・最大選択可能にする
            /// @todo Train単位ではなくExpもしくはエピソード単位で取得可能とする
            //auto ret = anet::ToFloat(episode_total_reward_comp_.mean());   // エピソード総報酬の平均
            if (episode_total_reward_comp_.numel() == 0) {
                return std::numeric_limits<float>::quiet_NaN();
			}
            auto ret = anet::ToFloat(episode_total_reward_comp_.max());   // エピソード総報酬の最大
            ANET_LOG_DEBUG("key=" << key << " ret=" << ret << " episode_total_reward_comp_=" << anet::ToString(episode_total_reward_comp_));
            return ret;
        } else {
            return std::numeric_limits<float>::quiet_NaN();
        }
    }

    if (key == TRAIN_STEP_PER_SEC) return last_train_step_per_sec_;
    if (key == EXP_STEP_PER_SEC) return last_exp_step_per_sec_;

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
    if (usec_diff <= 0) usec_diff = 1;

    if (usec_diff >= 200000) { // 200msec 積算
        last_train_step_per_sec_ = static_cast<float>(train_step_delta) * 1000000.0f / usec_diff;
        last_exp_step_per_sec_ = static_cast<float>(exp_step_delta) * 1000000.0f / usec_diff;
        acc_train_steps_ = 0;
        acc_exp_steps_ = 0;
        last_time_ = now;
        last_train_step_ = train_step;
        last_exp_step_ = exp_step;
    }

}


// ------------------------------------------------------
// SerialTrainRunner
// ------------------------------------------------------

SerialTrainRunner::SerialTrainRunner(
    std::shared_ptr<anet::rl::BatchEnv> env, std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Notifier> notifier)
    : TrainRunner(env, agent, notifier)
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
    std::shared_ptr<anet::rl::BatchEnv> env, std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Notifier> notifier)
    : TrainRunner(env, agent, notifier)
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
    std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Notifier> notifier)
{
    if (anet::ToLower(type) == "pipeline") {
        return std::make_shared<PipelineTrainRunner>(env, agent, notifier);
    } else {
        return std::make_shared<SerialTrainRunner>(env, agent, notifier);
    }
}


// ======================================================
// RunManager
// ======================================================

struct RunManager::Config : public anet::Config
{
    uint64_t seed = 0;
    int num_envs = 1;
    int eval_interval = 50;
    std::string main_runner_type = "serial";

    std::string eval_device_type = "cpu";
    int eval_device_index = 0;

    Config(const anet::ConfigData& config_data, const std::string& config_prefix = "train") /// @todo config_prefixをrunに変更
        : anet::Config(config_data, config_prefix)
    {
        ANET_READ_CONFIG(config_data, seed);
        ANET_READ_CONFIG(config_data, num_envs);
        ANET_READ_CONFIG(config_data, eval_interval);
        ANET_READ_CONFIG(config_data, main_runner_type);
        ANET_READ_CONFIG(config_data, eval_device_type);
        ANET_READ_CONFIG(config_data, eval_device_index);
    }

    torch::Device GetEvalDevice() const
    {
        if (anet::ToLower(eval_device_type) == "cuda") {
            return torch::Device(torch::kCUDA, eval_device_index);
        }
        return torch::Device(torch::kCPU);
    }
};

RunManager::RunManager(const ConfigData& config_data)
{
    // Config
    config_ = std::make_unique<Config>(config_data);    ///< @todo config_prefixをtrainからrunに変更？

    // BatchEnvを1つも構築する前に、設定から決まるnameを一括検証する。
    auto eval_configs = config_data.MakeSubConfigData("train.eval");
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
        LOG::error() << "Failed to create env.";
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
    train_runner_ = anet::rl::RunnerFactory::CreateMainRunner(config_->main_runner_type, env_, agent_, notifier_);
    RegisterEnvName("train", "main Train");

    // 設定からObserverを生成して登録
    anet::rl::ObserverFactory factory(config_data);

    // EpisodeEvalObserver
    for (const auto& kv : eval_configs) {
        // Eval設定取得
        const auto& tag = kv.first;
        const auto& eval_config_data = kv.second;
        std::string config_prefix = "train.eval.[" + tag + "].env";

        // Eval設定ログ
        anet::MetricsLogger::Instance()->Log(config_prefix, eval_config_data.ToJson());

        // RunMode取得
        std::string run_mode_str = "eval1";
        eval_config_data.Read("run_mode", run_mode_str, run_mode_str);
        anet::rl::RunMode run_mode = anet::rl::RunModeFromString(run_mode_str);
        configured_eval_run_modes_.emplace(tag, run_mode);

        // Interval取得
        int interval = 100;
        eval_config_data.Read("interval", interval, interval);
        if (interval < 0) {
            ANET_SYSTEM_ERROR("Invalid train.eval.[" << tag << "].interval=" << interval
                << ". Expected a non-negative value.");
        }

        int eval_batch_size = 1;
        eval_config_data.Read("eval_batch_size", eval_batch_size, eval_batch_size);
        if (eval_batch_size <= 0) {
            ANET_SYSTEM_ERROR("Invalid train.eval.[" << tag << "].eval_batch_size="
                << eval_batch_size << ". Expected a positive value.");
        }

        // バックグラウンド実行の切り替え (デフォルト true)
        bool use_background = true;
        eval_config_data.Read("use_background", use_background, use_background);

        // Eval actor の network clone 有無を選ぶ (デフォルト true)
        bool clone_model = true;
        eval_config_data.Read("clone_model", clone_model, clone_model);

        // dormant tagも宣言時schemaだけは検証し、Env/manifest/actorは生成しない。
        env_factory_->ValidateConfig(run_mode, config_prefix);
        if (interval == 0) {
            dormant_eval_tags_.insert(tag);
            RegisterEnvName(tag, "dormant configured Eval tag '" + tag + "'");
            continue;
        }

        // EvalRunner生成&登録
        auto actor_device = config_->GetEvalDevice();
        const auto owner = "configured Eval tag '" + tag + "'";
        EnsureEnvNameAvailable(tag, owner);
        const auto eval_seed_domain = "eval_env/" + tag;
        const auto eval_seed = master_seed_->GetGroupSeed(eval_seed_domain.c_str());
        auto eval_env = env_factory_->CreateBatchEnv(
            tag, eval_seed, eval_batch_size, run_mode, config_prefix);
        LogEnvConfig(*eval_env);
        auto eval_runner = std::make_shared<EvalRunner>(
            eval_env, agent_, notifier_, run_mode, clone_model, actor_device, tag);
        eval_runners.emplace(tag, eval_runner);
        RegisterEnvName(tag, owner);

        // EvalObserver生成&登録
        notifier_->AttachScoped<anet::rl::EpisodeEvalObserver>(
            train_runner_,
            eval_runner,
            interval,
            use_background
        );
    };

    auto resolve_runner = [&](RunnerScope scope, const std::string& eval_name) -> std::shared_ptr<const Runner> {
        if (scope == RunnerScope::TRAIN) return train_runner_;
        auto it = eval_runners.find(eval_name);
        if (it == eval_runners.end()) {
            if (dormant_eval_tags_.contains(eval_name)) {
                if (warned_dormant_metric_tags_.insert(eval_name).second) {
                    LOG::warn() << "Skipping metrics for dormant eval tag. tag='" << eval_name
                        << "' interval=0";
                }
                return nullptr;
            }
            ANET_SYSTEM_ERROR("Unknown eval name '" << eval_name << "' in metrics.scalar config.");
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
    const std::string& name, RunMode run_mode, bool clone_model,
    std::optional<torch::Device> device, const std::string& config_tag)
{
    ANET_ASSERT(status_ == anet::rl::RunnerStatus::RUNNING);

    const auto owner = "CreateEvalRunner '" + name + "'";
    EnsureEnvNameAvailable(name, owner);
    const auto selected_tag = config_tag.empty() ? name : config_tag;
    if (!config_tag.empty()) {
        const auto mode_it = configured_eval_run_modes_.find(config_tag);
        if (mode_it == configured_eval_run_modes_.end()) {
            ANET_SYSTEM_ERROR("Unknown Eval config tag='" << config_tag << "'.");
        }
        run_mode = mode_it->second;
    }
    const auto eval_seed_domain = "eval_panel/" + selected_tag;
    const auto eval_seed = master_seed_->GetGroupSeed(eval_seed_domain.c_str());
    const auto config_prefix = config_tag.empty()
        ? std::string() : "train.eval.[" + config_tag + "].env";
    auto env = env_factory_->CreateBatchEnv(
        name, eval_seed, 1, run_mode, config_prefix);
    LogEnvConfig(*env);
    auto eval_runner = std::make_shared<EvalRunner>(env, agent_, notifier_, run_mode, clone_model, device, name);
    this->eval_runners.emplace(name, eval_runner);
    RegisterEnvName(name, owner);
    return eval_runner;
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
