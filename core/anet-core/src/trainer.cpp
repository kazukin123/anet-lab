#include "anet/trainer.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/profile.hpp"
#include "anet/observers.hpp"
#include "anet/tensor_util.hpp"
#include "anet/log.hpp"
#include "anet/env.hpp"
#include "anet/agent.hpp"
#include "anet/dqn_agent.hpp"


using namespace anet::rl;

namespace LOG = anet::log;

// =========================

RunnerBase::RunnerBase()
{
    // Notifier生成
    notifier_ = std::make_shared<anet::rl::Notifier>();
}

RunnerBase::RunnerBase(std::shared_ptr<BatchEnv> env) : env_(env)
{
    // Notifier生成
    notifier_ = std::make_shared<anet::rl::Notifier>();
}

StepCounts RunnerBase::DoUpdateFrame(int max_steps, ControlFunction pre_step_func, ControlFunction post_step_func)
{
    anet::ProfileRange r("DefaultTrainer::DoUpdateFrame");

    int frame_step = 0;
    //StepCounts step_counts_;

    // --- 学習ステップを複数回回す ---
    while (max_steps < 0 || frame_step < max_steps) {
        anet::ProfileRange r2("DefaultTrainer::DoUpdateFrame.step");

        // ステップ前制御
        auto control_signal_pre = pre_step_func(step_counts_);
        if (control_signal_pre == anet::rl::ControlSignal::STOP) {
            status_ = anet::rl::RunnerStatus::COMPLETED;
            break;
        }
        if (control_signal_pre == anet::rl::ControlSignal::BREAK) {
            break;
        }

        // Step実行
        step_counts_ = DoStep();

        // ステップ後処理
        auto control_signal_post = post_step_func(step_counts_);
        if (control_signal_post == anet::rl::ControlSignal::STOP) {
            status_ = anet::rl::RunnerStatus::COMPLETED;
            break;
        }
        if (control_signal_post == anet::rl::ControlSignal::BREAK) {
            break;
        }

        // Stepカウント
        frame_step++;
    }

    // ログflush
    //anet::MetricsLogger::Instance()->Flush();

    return step_counts_;
}

std::optional<float> RunnerBase::GetScalar(const std::string& key, int index) const
{
    if (key == TRAIN_STEP) return static_cast<float>(step_counts_.train_step);
    if (key == EXP_STEP) return static_cast<float>(step_counts_.exp_step);
    if (key == LEARN_STEP) return static_cast<float>(step_counts_.learn_step);
    if (key == EPISODE_COUNT) return static_cast<float>(step_counts_.episode_count);
    if (key == SIM_STEP) return static_cast<float>(step_counts_.sim_step);

    return std::nullopt;
}

// =========================

EvalRunner::EvalRunner(std::shared_ptr<BatchEnv> env, std::shared_ptr<const Agent> agent, RunMode runmode)
    : RunnerBase(env), agent_(agent), runmode_(runmode)
{
    ;
}

RunnerStatus EvalRunner::Initialize(const ConfigData& config_data)
{
    status_ = anet::rl::RunnerStatus::RUNNING;
    return status_;
}

StepCounts EvalRunner::DoStep(int64_t action)
{
    anet::ProfileRange r1("EvalRunner::DoStep");

    if (!env_initialized_) {
        // 環境初期化
        state_ = env_->Reset(runmode_);
        env_initialized_ = true;
        ANET_LOG_DEBUG("env_->Reset() done. state=" << state_.ToString());
    }

    // ステップ前Observer
    BeforeStepEvent before_step_event{ *this, step_counts_, agent_, env_ };
    notifier_->Notify(before_step_event);

    // ステップ前情報
    auto train_step = step_counts_.train_step;
    ANET_LOG_DEBUG("step=" << train_step << " state=" << state_.ToString());

    // 行動選択
    anet::rl::BatchActionInfo action_info = {
        torch::tensor({ action }),
        torch::tensor({ false })
    };

    // 環境ステップ実行
    auto result = env_->Step(action_info.action, runmode_);    // next_state, reward, done, truncated
    ANET_LOG_DEBUG("step=" << train_step << " action=" << action_info.ToString());
    ANET_LOG_DEBUG("step=" << train_step << " next_state=" << result->next_state.ToString());
    ANET_LOG_DEBUG("step=" << train_step << " continue_state=" << result->continue_state.ToString());
    ANET_LOG_DEBUG("step=" << train_step << " reward=" << anet::ToString(result->reward));
    ANET_CHECK_DEVICE(result->next_state.obs, torch::kCPU);
    ANET_CHECK_DEVICE(result->next_state.done, torch::kCPU);
    ANET_CHECK_DEVICE(result->next_state.truncated, torch::kCPU);
    ANET_CHECK_DEVICE(result->reward, torch::kCPU);
    ANET_CHECK_DEVICE(result->continue_state.obs, torch::kCPU);
    ANET_CHECK_DEVICE(result->continue_state.done, torch::kCPU);
    ANET_CHECK_DEVICE(result->continue_state.truncated, torch::kCPU);

    // カウント更新
    step_counts_.train_step++;
    step_counts_.exp_step += result->n_transitions;
    step_counts_.episode_count += result->n_done;

    anet::rl::BatchExperience exp({ state_, action_info, result->reward, result->next_state });

    // 更新後処理
    anet::rl::TrainEvent update_event{ exp, *this, step_counts_, agent_, nullptr, env_, result, action_info };
    notifier_->Notify(update_event);
    state_ = result->continue_state;

    return step_counts_;
}

StepCounts EvalRunner::DoStep()
{
    anet::ProfileRange r1("EvalRunner::DoStep");

    if (!env_initialized_) {
        // 環境初期化
        state_ = env_->Reset(runmode_);
        env_initialized_ = true;
        ANET_LOG_DEBUG("env_->Reset() done. state=" << state_.ToString());
    }

    //if (state_.episode_start)

    // ステップ前Observer
    BeforeStepEvent before_step_event{ *this, step_counts_, agent_, env_ };
    notifier_->Notify(before_step_event);

    // ステップ前情報
    auto train_step = step_counts_.train_step;

    // 行動選択
    auto action_info = agent_->MakeAction(step_counts_, state_, runmode_);
    ANET_LOG_DEBUG("step=" << train_step << " action=" << action_info.ToString());
    //ANET_CHECK_SHAPE(action_info.action, { N });

    // 環境ステップ実行
    auto result = env_->Step(action_info.action, runmode_);    // next_state, reward, done, truncated
    ANET_LOG_DEBUG("step=" << train_step << " reward=" << anet::ToString(result->reward));
    ANET_LOG_DEBUG("step=" << train_step << " next_state=" << result->next_state.ToString());
    ANET_CHECK_DEVICE(result->next_state.obs, torch::kCPU);
    ANET_CHECK_DEVICE(result->next_state.done, torch::kCPU);
    ANET_CHECK_DEVICE(result->next_state.truncated, torch::kCPU);
    ANET_CHECK_DEVICE(result->reward, torch::kCPU);
    ANET_CHECK_DEVICE(result->continue_state.obs, torch::kCPU);
    ANET_CHECK_DEVICE(result->continue_state.done, torch::kCPU);
    ANET_CHECK_DEVICE(result->continue_state.truncated, torch::kCPU);
    //ANET_CHECK_SHAPE(result->next_state.obs, { N, ANET_SHAPE_ENDANY });
    //ANET_CHECK_SHAPE(result->next_state.done, { N });
    //ANET_CHECK_SHAPE(result->next_state.truncated, { N });
    //ANET_CHECK_SHAPE(result->reward, { N });
    //ANET_CHECK_SHAPE(result->continue_state.obs, { N, ANET_SHAPE_ENDANY });
    //ANET_CHECK_SHAPE(result->continue_state.done, { N });
    //ANET_CHECK_SHAPE(result->continue_state.truncated, { N });
    //ANET_ASSERT(env_spec.state_spec.MatchesShape(state_.obs));
    //    ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.obs));

    // カウント更新
    step_counts_.train_step++;
    step_counts_.exp_step += result->n_transitions;
    step_counts_.episode_count += result->n_done;

    anet::rl::BatchExperience exp({ state_, action_info, result->reward, result->next_state });

    // 更新後処理
    anet::rl::TrainEvent update_event{ exp, *this, step_counts_, agent_, nullptr, env_, result, action_info };
    notifier_->Notify(update_event);
    state_ = result->continue_state;

    return step_counts_;
}

// =========================

struct DefaultTrainer::Config : public anet::Config
{
    uint64_t seed = 0;
    int batch_size = 1;
    int eval_interval = 50;

    DefaultTrainer::Config(const anet::ConfigData& config_data, const std::string& config_prefix = "train")
        : anet::Config(config_data, config_prefix)
    {
        ANET_READ_CONFIG(config_data, seed);
        ANET_READ_CONFIG(config_data, batch_size);
        ANET_READ_CONFIG(config_data, eval_interval);
    }
};

DefaultTrainer::DefaultTrainer(const ConfigData& config_data, const std::string& config_prefix)
    : RunnerBase()
    , config_(std::make_unique<Config>(config_data, config_prefix))
    , train_reward_ema_(0.001)
{
    //Initialize(config_data);
}

RunnerStatus DefaultTrainer::Initialize(const ConfigData& config_data)
{
    // seed
    if (config_->seed == 0) {
        master_seed_ = std::make_unique<anet::MasterSeedManager>();
    }else {
        master_seed_ = std::make_unique<anet::MasterSeedManager>(config_->seed);
    }

    // seed値生成
    auto global_seed = master_seed_->GetMasterSeed();
    auto train_env_seed = master_seed_->GetGroupSeed("env");
    auto eval_env_seed = master_seed_->GetGroupSeed("eval_env");
    auto agent_seed = master_seed_->GetGroupSeed("agent");
    auto eval_obs_seed = master_seed_->GetGroupSeed("eval_obs");
    LOG::info() << "global_seed=" << global_seed << " train_env_seed="
        << train_env_seed << " eval_env_seed" << eval_env_seed << " agent_seed=" << agent_seed;
    eval_env_seed_ = eval_env_seed;

    // パラメータ記録
    anet::MetricsLogger::Instance()->LogJson("train/seed",
        { "global_seed", global_seed, "agent_seed", agent_seed, "train_env_seed", train_env_seed });
    anet::MetricsLogger::Instance()->LogJson("train/config", config_->ToJson());
    anet::MetricsLogger::Instance()->Flush();

    // ENV生成
    anet::rl::DefaultBatchEnvFactoryConfig env_config(config_data);
    LOG::info() << "env_config=" << env_config.ToString();
    env_factory_ = std::make_unique<anet::rl::DefaultBatchEnvFactory>(env_config, config_data, config_->batch_size);
    auto env_device = env_factory_->GetDevice();
    auto single_env_factory = env_factory_->GetSingleFactory();
    env_ = env_factory_->CreateBatchEnv(train_env_seed, -1);
    if (env_ == nullptr) {
        LOG::error() << "Failed to create env.";
        status_ = anet::rl::RunnerStatus::COMPLETED;
        return status_;
    }

    // ログ
    auto batch_env_spec = env_->GetBatchSpec();
    auto env_spec = env_->GetSpec();
    LOG::info() << "batch_env_spec=" << batch_env_spec.ToString();
    LOG::info() << "env_spec=" << env_spec.ToString();
    anet::MetricsLogger::Instance()->LogJson("env/batch_env_spec", batch_env_spec.ToJson());
    anet::MetricsLogger::Instance()->LogJson("env/env_spec", env_spec.ToJson());
    anet::MetricsLogger::Instance()->Flush();

    // ランダム方策で環境難易度評価
    /// @todo EvaluateEnvironmentDifficultyを復活
    //auto eval_result = anet::rl::EvaluateEnvironmentDifficulty(*env_, 100);
    //anet::MetricsLogger::Instance()->LogJson("eval_env", eval_result.ToJson());

    // Agent生成
    anet::rl::DefaultAgentFactoryConfig agent_factory_config(config_data);
    auto agent_factory = anet::rl::DefaultAgentFactory(
        agent_factory_config, env_spec, batch_env_spec, config_data, agent_seed);
    auto agent_device = agent_factory.GetDevice();
    agent_ = agent_factory.CreateAgent(notifier_);
    if (agent_ == nullptr) {
        LOG::error() << "Failed to create agent." ;
        status_ = anet::rl::RunnerStatus::COMPLETED;
        return status_;
    }

    // EpisodeEvalObserver
    notifier_->Attach<anet::rl::EpisodeEvalObserver>(
        [this](float total_reward)
        {
            this->last_target_eval_reward_ = total_reward;
        },
        single_env_factory, config_data, env_device, anet::rl::RunMode::Eval1,
        config_->eval_interval, config_->eval_interval, eval_obs_seed);
    notifier_->Attach<anet::rl::EpisodeEvalObserver>(
        [this](float total_reward)
        {
            this->last_policy_eval_reward_ = total_reward;
        },
        single_env_factory, config_data, env_device, anet::rl::RunMode::Eval2,
        config_->eval_interval, config_->eval_interval, eval_obs_seed);

    // 設定からObserverを生成して登録
    anet::rl::ObserverFactory factory(config_data);
    auto train_obs = factory.GetUpdateObservers();
    auto learn_obs = factory.GetLearnObservers();
    for (auto obs : train_obs) notifier_->Attach(obs);
    for (auto obs : learn_obs) notifier_->Attach(obs);

    status_ = anet::rl::RunnerStatus::RUNNING;

    return status_;
}

StepCounts DefaultTrainer::DoStep()
{
    anet::ProfileRange r("DefaultTrainer::Step");

    ANET_ASSERT(status_ == anet::rl::RunnerStatus::RUNNING);

    // EnvSpec取得
    auto env_spec = env_->GetSpec();
    auto batch_env_spec = env_->GetBatchSpec();

    if (!env_initialized_) {
        // 環境初期化
        state_ = env_->Reset();  // ← reset() は 初期状態 を返す
        env_initialized_ = true;
        ANET_LOG_DEBUG("env_->Reset() done. state=" << state_.ToString());
        ANET_CHECK_DEVICE_CPU_MSG(state_.obs, "Initial state");
        ANET_ASSERT(env_spec.state_spec.MatchesShape(state_.obs));
        ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.obs));

        // 時間計測開始
        start_time_ = std::chrono::high_resolution_clock::now();
        last_time_ = start_time_;
    }

    // --- 学習ステップを回す ---
    float frame_total_reward = 0.0f;
    int frame_step = 0;

    anet::ProfileRange r2("DefaultTrainer::DoUpdateFrame.step");

    // ステップ前Observer
    BeforeStepEvent before_step_event{ *this, step_counts_, agent_, env_ };
    notifier_->Notify(before_step_event);

    // ステップ前情報
    auto train_step = step_counts_.train_step;

    // Stateチェック
    ANET_LOG_DEBUG("step=" << train_step << " state=" << state_.ToString());
    ANET_ASSERT(env_spec.state_spec.MatchesShape(state_.obs));
//    ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.obs));
    const int N = state_.obs.size(0);

    // 行動選択
    auto action_info = agent_->MakeAction(step_counts_, state_);
    ANET_LOG_DEBUG("step=" << train_step << " action=" << action_info.ToString());
    ANET_CHECK_SHAPE(action_info.action, { N });

    // 環境ステップ実行
    auto result = env_->Step(action_info.action);    // next_state, reward, done, truncated
    ANET_LOG_DEBUG("step=" << train_step << " reward=" << anet::ToString(result->reward));
    ANET_LOG_DEBUG("step=" << train_step << " next_state=" << result->next_state.ToString());
    ANET_CHECK_DEVICE(result->next_state.obs, torch::kCPU);
    ANET_CHECK_DEVICE(result->next_state.done, torch::kCPU);
    ANET_CHECK_DEVICE(result->next_state.truncated, torch::kCPU);
    ANET_CHECK_DEVICE(result->reward, torch::kCPU);
    ANET_CHECK_DEVICE(result->continue_state.obs, torch::kCPU);
    ANET_CHECK_DEVICE(result->continue_state.done, torch::kCPU);
    ANET_CHECK_DEVICE(result->continue_state.truncated, torch::kCPU);
    ANET_CHECK_SHAPE(result->next_state.obs, { N, ANET_SHAPE_ENDANY });
    ANET_CHECK_SHAPE(result->next_state.done, { N });
    ANET_CHECK_SHAPE(result->next_state.truncated, { N });
    ANET_CHECK_SHAPE(result->reward, { N });
    ANET_CHECK_SHAPE(result->continue_state.obs, { N, ANET_SHAPE_ENDANY });
    ANET_CHECK_SHAPE(result->continue_state.done, { N });
    ANET_CHECK_SHAPE(result->continue_state.truncated, { N });
    ANET_ASSERT(env_spec.state_spec.MatchesShape(state_.obs));
//    ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.obs));

    // 報酬更新
    float step_reward = result->reward.mean().item<float>();
    last_train_reward_ = step_reward;
	train_reward_ema_.Update(last_train_reward_);

    // カウント更新
    step_counts_.train_step++;
    step_counts_.exp_step += result->n_transitions;
    step_counts_.episode_count += result->n_done;

    // Agent更新
    anet::rl::BatchExperience exp({ state_, action_info, result->reward, result->next_state });
    auto update_result = agent_->UpdateFromBatch(step_counts_, exp, *this);

    // カウント更新
    step_counts_.update_step++;
    step_counts_.learn_step += update_result->GetLearnStepDiff();

    // メトリクス算出（処理性能系）
    auto exp_step = step_counts_.exp_step;
    auto exp_step_delta = exp_step - last_exp_step_;
    std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
    auto usec_diff = std::chrono::duration_cast<std::chrono::microseconds>(now - last_time_).count();
    if (usec_diff <= 0) usec_diff = 1;
    auto train_step_per_sec = 1.0f *1000000.0f / usec_diff;
    auto exp_step_per_sec = static_cast<float>(exp_step_delta) * 1000000.0f / usec_diff;
    last_train_step_per_sec_ = train_step_per_sec;
    last_exp_step_per_sec_ = exp_step_per_sec;

    // 更新後処理
    anet::rl::TrainEvent train_event{ exp, *this, step_counts_, agent_, update_result, env_, result, action_info };
    notifier_->Notify(train_event);
    state_ = result->continue_state;

    // 次準備
    last_time_ = now;
    last_exp_step_ = exp_step;

    return step_counts_;
}

std::shared_ptr<EvalRunner> DefaultTrainer::CreateEvalRunner(RunMode runmode) const
{
    /// @todo seed指定対応

    ANET_ASSERT(status_ == anet::rl::RunnerStatus::RUNNING);

    auto env = env_factory_->CreateBatchEnv(eval_env_seed_,1);
    auto eval_runner = std::make_shared<EvalRunner>(env, agent_, runmode);
    return eval_runner;
}

std::optional<float> DefaultTrainer::GetScalar(const std::string& key, int index) const
{
    if (key == TRAIN_REWARD) return last_train_reward_;
    if (key == TRAIN_REWARD_EMA) return train_reward_ema_.Value();

    if (key == TARGET_EVAL_REWARD) return last_target_eval_reward_;
    if (key == POLICY_EVAL_REWARD) return last_policy_eval_reward_;

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

// =========================

RunnerThread::RunnerThread(std::shared_ptr<anet::rl::Runner> runner,
        anet::rl::Runner::ControlFunction pre_func,
        anet::rl::Runner::ControlFunction post_func)
        : runner_(runner), pre_func_(pre_func), post_func_(post_func)
{
}

RunnerThread::~RunnerThread()
{
    Stop();  // ensure join
}

/// Start training thread (if not already running)
void RunnerThread::Start()
{
    // 既に実行中なら何もしない
    if (running_.load()) return;
    running_.store(true);

    // Trainerスレッドループ
    worker_ = std::thread([this]() { ThreadMain(); });
}

// Trainerスレッド停止＆停止待ち合わせ
void RunnerThread::Stop()
{
    ANET_LOG_DEBUG("BEGIN");

    // 実行中なら
    if (running_.load()) {
        // 実行中フラグを落とす
        running_.store(false);
    }

    // スレッド終了待ち合わせ
    if (worker_.joinable())
        worker_.join();

    ANET_LOG_DEBUG("END");
}

void RunnerThread::ThreadMain()
{
    ANET_LOG_DEBUG("BEGIN");

    // Trainループ
    while (running_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }

        // フレーム実行
        runner_->DoUpdateFrame(1, pre_func_, post_func_);

        // 学習終わってたらスレッド終了
        auto status = runner_->GetStatus();
        if (status == anet::rl::RunnerStatus::COMPLETED) break;
    }

    ANET_LOG_DEBUG("END");
}
