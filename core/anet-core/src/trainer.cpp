#include <limits>
#include "anet/trainer.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/profile.hpp"
#include "anet/observers.hpp"
#include "anet/tensor_util.hpp"
#include "anet/log.hpp"
#include "anet/env.hpp"
#include "anet/agent.hpp"
#include "test.hpp"


using namespace anet::rl;
namespace LOG = anet::log;


// ======================================================
// RunnerBase
// ======================================================

RunnerBase::RunnerBase(
    std::shared_ptr<anet::rl::BatchEnv> env, std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Notifier> notifier, RunMode runmode)
    : env_(env)
    , agent_(agent)
    , notifier_(notifier)
    , runmode_(runmode)
    , reward_ema_(0.001)
{
    InitializeMetrics();
    status_ = anet::rl::RunnerStatus::RUNNING;
    auto batch_env_spec = env_->GetBatchSpec();
    action_context_ = agent_->CreateActionContext(batch_env_spec, runmode_);
}

void RunnerBase::InitializeMetrics()
{
    ANET_CHECK(env_ != nullptr);

    auto env_device = env_->GetDevice();
    auto batch_env_spec = env_->GetBatchSpec();

    // メトリクス初期化
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(env_device);
    episode_total_reward_cur_ = torch::zeros({ batch_env_spec.batch_size }, fopt);
    ANET_ASSERT_SHAPE(episode_total_reward_cur_, { batch_env_spec.batch_size });
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

StepCounts RunnerBase::DoUpdateFrame(int max_steps, ControlFunction pre_step_func, ControlFunction post_step_func)
{
    anet::ProfileRange r("RunnerBase::DoUpdateFrame");

    int frame_step = 0;
    //StepCounts step_counts_;

    // --- 学習ステップを複数回回す ---
    while (max_steps < 0 || frame_step < max_steps) {
        anet::ProfileRange r1("RunnerBase::DoUpdateFrame.step");

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


    return std::nullopt;
}


// ======================================================
// EvalRunner
// ======================================================

EvalRunner::EvalRunner(
    std::shared_ptr<anet::rl::BatchEnv> env,
    std::shared_ptr<anet::rl::Agent> agent,
    std::shared_ptr<anet::rl::Notifier> notifier,
    RunMode runmode)
    : RunnerBase(env, agent, notifier, runmode)
{
}

StepCounts EvalRunner::DoStep(int64_t action)
{
    anet::ProfileRange r1("EvalRunner::DoStep");
    torch::NoGradGuard grad_guard;

    if (!env_initialized_) {
        // 環境初期化
        auto reset_result = env_->Reset(runmode_);
        state_ = reset_result->state;
        env_initialized_ = true;
        ANET_LOG_DEBUG("env_->Reset() done. state=" << state_.ToString());
    }

    // ステップ前情報
    auto train_step = step_counts_.train_step;
    ANET_LOG_DEBUG("step=" << train_step << " state=" << state_.ToString());

    // 行動選択
    auto action_info_raw = agent_->MakeAction(step_counts_, state_, action_context_);
    ANET_LOG_DEBUG("step=" << train_step << " action_info_raw=" << action_info_raw.ToString());

    // action_infoを生成
    anet::rl::BatchActionInfo action_info = {
        action < 0 ? action_info_raw.GetAction() : torch::tensor({ action }), // 指定のactionがあれば強制
        action_info_raw.GetAuxData()    // AuxはAgentが生成した内容
    };

    // 環境ステップ実行
    auto result = env_->Step(action_info, runmode_);    // next_state, reward, done, truncated
    ANET_LOG_DEBUG("step=" << train_step << " action=" << action_info.ToString());
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
    state_ = result->continue_state;

    return step_counts_;
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
    : RunnerBase(env, agent, notifier, anet::rl::RunMode::Train)
{

}

void TrainRunner::SetEvalLastReward(const std::string& name, float val)
{
    eval_last_rewards_[name] = val;
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

	//evalのlast_reward取得
    if (key.find("eval.[") == 0) {
        for (const auto& kv : eval_last_rewards_) {
            const auto& tag = kv.first;
			const auto& reward = kv.second;

            //auto prefix_ema = "eval.[" + kv.first + "].eps_total_reward_ema";
            //if (key.find(prefix_ema) == 0) {
            //    this->eval_last_rewards[tag];
            //}
            auto prefix = "eval.[" + kv.first + "].eps_total_reward";
            if (key.find(prefix) == 0) {
                return reward;
			}
        }
    }


    return RunnerBase::GetScalar(key, index);
}

void TrainRunner::Shutdown()
{
    env_->Shutdown();
    notifier_->Clear();
}


StepCounts TrainRunner::DoStep()
{
    anet::ProfileRange r("DefaultTrainer::DoStep");

    ANET_ASSERT(status_ == anet::rl::RunnerStatus::RUNNING);

    // EnvSpec取得
    auto env_spec = env_->GetSpec();
    auto batch_env_spec = env_->GetBatchSpec();

    if (!env_initialized_) {
        anet::ProfileRange r1("DefaultTrainer::DoUpdateFrame.initialize");

        // 環境初期化
        auto reset_result = env_->Reset();
        state_ = reset_result->state;
        env_initialized_ = true;
        ANET_LOG_DEBUG("env_->Reset() done. state=" << state_.ToString());
        ANET_ASSERT_DEVICE_CPU_MSG(state_.obs, "Initial state");
        ANET_ASSERT(env_spec.state_spec.MatchesShape(state_.obs));
        ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.obs));

        // 時間計測開始
        start_time_ = std::chrono::high_resolution_clock::now();
        last_time_ = start_time_;
    }

    anet::ProfileRange r2("DefaultTrainer::DoUpdateFrame.makeAction");

    // --- 学習ステップを回す ---
    float frame_total_reward = 0.0f;
    int frame_step = 0;

    // ステップ前情報
    auto train_step = step_counts_.train_step;

    // Stateチェック
    ANET_LOG_DEBUG("step=" << train_step);// << " state=" << state_.ToString());
    ANET_ASSERT(env_spec.state_spec.MatchesShape(state_.obs));
//    ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.obs));
    const int N = state_.obs.size(0);

    // 行動選択
    auto action_info = agent_->MakeAction(step_counts_, state_, action_context_);
    //ANET_LOG_DEBUG("step=" << train_step << " action=" << action_info.ToString());
    ANET_ASSERT_SHAPE(action_info.GetAction(), {N});

    anet::ProfileRange r3("DefaultTrainer::DoUpdateFrame.envStep", r2);

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
    ANET_ASSERT_SHAPE(result->next_state.obs, { N, ANET_SHAPE_ENDANY });
    ANET_ASSERT_SHAPE(result->next_state.done, { N });
    ANET_ASSERT_SHAPE(result->next_state.truncated, { N });
    ANET_ASSERT_SHAPE(result->reward, { N });
    ANET_ASSERT_SHAPE(result->continue_state.obs, { N, ANET_SHAPE_ENDANY });
    ANET_ASSERT_SHAPE(result->continue_state.done, { N });
    ANET_ASSERT_SHAPE(result->continue_state.truncated, { N });
    ANET_ASSERT(env_spec.state_spec.MatchesShape(state_.obs));
//    ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.obs));

    anet::ProfileRange r4("DefaultTrainer::DoUpdateFrame.envStepPost", r3);

    //メトリクス更新
    UpdateMetrics(result);

    // カウント更新
    step_counts_.train_step++;
    step_counts_.exp_step += result->n_transitions;
    step_counts_.episode_count += result->n_episode_end;


    // Agent更新
    anet::ProfileRange r5("DefaultTrainer::DoUpdateFrame.updateAgent", r4);
    anet::rl::BatchExperience exp({ state_, action_info, result->reward, result->next_state });
    auto self = this->shared_from_this();
    auto result_list = agent_->UpdateFromBatch(step_counts_, exp, self);

    // カウント更新
    anet::ProfileRange r6("DefaultTrainer::DoUpdateFrame.postUpdate", r5);
    step_counts_.update_step++;
    step_counts_.learn_step += result_list.size();

    // 更新後処理
    {
        anet::ProfileRange r7("DefaultTrainer::DoUpdateFrame.notify", r6);
        torch::NoGradGuard grad_guard;

        anet::rl::TrainEvent train_event{ exp, self, step_counts_, agent_, result_list, env_, result, action_info };
        notifier_->Notify(train_event);
        state_ = result->continue_state;
    }

    // メトリクス算出（処理性能系）
    auto trin_step = step_counts_.train_step;
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

    // カウント更新
    /// @todo カウント位置検討
    //step_counts_.train_step++;
    //step_counts_.exp_step += result->n_transitions;
    //step_counts_.episode_count += result->n_done;
    //step_counts_.update_step++;
    //step_counts_.learn_step += update_result->GetLearnStepDiff();

    return step_counts_;
}


// ======================================================
// RunManager
// ======================================================

struct RunManager::Config : public anet::Config
{
    uint64_t seed = 0;
    int batch_size = 1;
    int eval_interval = 50;

    RunManager::Config(const anet::ConfigData& config_data, const std::string& config_prefix = "train") /// @todo config_prefixをrunに変更
        : anet::Config(config_data, config_prefix)
    {
        ANET_READ_CONFIG(config_data, seed);
        ANET_READ_CONFIG(config_data, batch_size);
        ANET_READ_CONFIG(config_data, eval_interval);
    }
};

RunManager::RunManager(const ConfigData& config_data)
{
    // Config
    config_ = std::make_unique<Config>(config_data);    ///< @todo config_prefixをtrainからrunに変更？

    // seed
    if (config_->seed == 0) {
        master_seed_ = std::make_unique<anet::MasterSeedManager>();
    } else {
        master_seed_ = std::make_unique<anet::MasterSeedManager>(config_->seed);
    }

    // seed値生成
    auto global_seed = master_seed_->GetMasterSeed();
    auto train_env_seed = master_seed_->GetGroupSeed("env");
    auto eval_env_seed = master_seed_->GetGroupSeed("eval_env");
    auto agent_seed = master_seed_->GetGroupSeed("agent");
    auto eval_obs_seed = master_seed_->GetGroupSeed("eval_obs");
    LOG::info() << "global_seed=" << global_seed << " train_env_seed="
        << train_env_seed << " eval_env_seed=" << eval_env_seed << " agent_seed=" << agent_seed;
    eval_env_seed_ = eval_env_seed;

    // パラメータ記録
    anet::MetricsLogger::Instance()->Log("train/seed",
        { "global_seed", global_seed, "agent_seed", agent_seed, "train_env_seed", train_env_seed });
    anet::MetricsLogger::Instance()->Log(*config_);
    anet::MetricsLogger::Instance()->Flush();

    // Notifier生成
    notifier_ = std::make_shared<Notifier>();

    // BatchEnv生成
    anet::rl::DefaultBatchEnvFactoryConfig env_config(config_data);
    LOG::info() << "env_config=" << env_config.ToString();
    env_factory_ = std::make_unique<anet::rl::DefaultBatchEnvFactory>(env_config, config_data, config_->batch_size);
    env_class_id_ = env_config.class_id;
    auto env_device = env_factory_->GetDevice();
    auto single_env_factory = env_factory_->GetSingleFactory();
    env_ = env_factory_->CreateBatchEnv(train_env_seed, -1);
    if (env_ == nullptr) {
        LOG::error() << "Failed to create env.";
        return;
    }

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
    train_runner_ = std::make_shared<TrainRunner>(env_, agent_, notifier_);

    // 設定からObserverを生成して登録
    anet::rl::ObserverFactory factory(config_data);
    auto train_obs = factory.GetUpdateObservers();
    auto learn_obs = factory.GetLearnObservers();
    for (auto obs : train_obs) {
        auto scoped_obs = std::make_shared< RunnerScopedTrainObserver>(obs, train_runner_);
        notifier_->Attach(scoped_obs);
    }
    for (auto obs : learn_obs) {
        auto scoped_obs = std::make_shared< RunnerScopedLearnObserver>(obs, train_runner_);
        notifier_->Attach(scoped_obs);
    }

    // EpisodeEvalObserver
    auto eval_configs = config_data.MakeSubConfigData("train.eval");
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

        // Interval取得
        int interval = 100;
        eval_config_data.Read("interval", interval, interval);

        // メトリクス初期化
        //eval_last_rewards_[tag] = std::numeric_limits<float>::quiet_NaN();
        train_runner_->SetEvalLastReward(tag, std::numeric_limits<float>::quiet_NaN());

        // EvalObserver生成&登録
        //auto config_prefix = "train.eval." + tag;
        notifier_->AttachScoped<anet::rl::EpisodeEvalObserver>(
            train_runner_,
            [this, tag](float total_reward) {   // report_function
                ANET_LOG_DEBUG("EvalObserver: tag=" << tag << " total_reward=" << total_reward);
                train_runner_->SetEvalLastReward(tag, total_reward);
            },
            single_env_factory, config_data, env_device, run_mode,
            interval,   // log_interval
            interval,   // eval_interval
            eval_obs_seed,
            config_prefix
        );
    };

    // 成功！
    status_ = anet::rl::RunnerStatus::RUNNING;
}

RunManager::~RunManager()
{
    this->env_.reset();
    this->agent_.reset();
}

std::shared_ptr<EvalRunner> RunManager::CreateEvalRunner(const std::string& name, RunMode runmode)
{
    /// @todo seed指定対応

    ANET_ASSERT(status_ == anet::rl::RunnerStatus::RUNNING);

    auto env = env_factory_->CreateBatchEnv(eval_env_seed_, 1); // batch_size = 1
    auto eval_runner = std::make_shared<EvalRunner>(env, agent_, notifier_, runmode);
    this->eval_runners[name] = eval_runner;
    return eval_runner;
}

// =========================

RunnerThread::RunnerThread(std::shared_ptr<anet::rl::Runner> runner,
        anet::rl::Runner::ControlFunction pre_func,
        anet::rl::Runner::ControlFunction post_func,
        ExceptionFunction exception_func)
        : runner_(runner), pre_func_(pre_func), post_func_(post_func), exception_func_(exception_func)
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
    anet::ProfileThreadName th("RunnerThread");

    try {
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
    } catch (...) {
        if (exception_func_ != nullptr)
            exception_func_();
    }

    // 終わる
    runner_->Shutdown();

    // 終わったフラグ
    running_.store(false);

    ANET_LOG_DEBUG("END");
}
