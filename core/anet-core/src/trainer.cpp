#include "anet/trainer.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/profile.hpp"
#include "anet/observers.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/log.hpp"
#include "anet/env.hpp"
#include "anet/agent.hpp"
#include "anet/dqn_agent.hpp"


using namespace anet::rl;
using namespace anet::log;


struct DefaultTrainer::Config : public anet::Config
{
    uint64_t seed = 0;
    int batch_size = 1;
    int eval_interval = 50;

    DefaultTrainer::Config(const anet::ConfigData& config_data) : anet::Config(config_data, "train")
    {
        ANET_READ_CONFIG(config_data, seed);
        ANET_READ_CONFIG(config_data, batch_size);
        ANET_READ_CONFIG(config_data, eval_interval);
    }
};

std::optional<float> DefaultTrainer::GetScalar(const std::string& key) const
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

    if (key == TRAIN_STEP) return static_cast<float>(step_counts_.train_step);
    if (key == EXP_STEP) return static_cast<float>(step_counts_.exp_step);
    if (key == LEARN_STEP) return static_cast<float>(step_counts_.learn_step);
    if (key == EPISODE_COUNT) return static_cast<float>(step_counts_.episode_count);
    if (key == SIM_STEP) return static_cast<float>(step_counts_.sim_step);

    return std::nullopt;
}


DefaultTrainer::DefaultTrainer(const ConfigData& config_data)
    : config_(std::make_unique<Config>(config_data))
    , train_reward_ema_(0.001)
{
    //Initialize(config_data);
}

TrainerStatus DefaultTrainer::Initialize(const ConfigData& config_data)
{
    // seed
    if (config_->seed == 0) {
        master_seed_ = std::make_unique<anet::MasterSeedManager>();
    }else {
        master_seed_ = std::make_unique<anet::MasterSeedManager>(config_->seed);
    }

    // Notifier生成
    notifier_ = std::make_shared<anet::rl::Notifier>();

    // seed値生成
    auto global_seed = master_seed_->GetMasterSeed();
    auto env_seed = master_seed_->GetGroupSeed("env");
    auto agent_seed = master_seed_->GetGroupSeed("agent");
    log::info() << "global_seed=" << global_seed << " env_seed=" << env_seed << " agent_seed=" << agent_seed;

    // パラメータ記録
    //wxLogInfo("train.preset=%s confg=%s", wxGetApp().GetConfig("train").Get("preset"), config_->ToString());
    anet::MetricsLogger::Instance()->LogJson("train/seed",
        { "global_seed", global_seed, "agent_seed", agent_seed, "env_seed", env_seed });
    anet::MetricsLogger::Instance()->LogJson("train/config", config_->ToJson());
    anet::MetricsLogger::Instance()->Flush();

    using namespace anet::log;

    // ENV生成
    anet::rl::DefaultBatchEnvFactoryConfig env_config(config_data);

    log::info() << "env_config=" << env_config.ToString();
 
    auto env_factory = anet::rl::DefaultBatchEnvFactory(
        env_config, config_data, config_->batch_size, env_seed);
    auto env_device = env_factory.GetDevice();
    auto single_env_factory = env_factory.GetSingleFactory();
    env_ = env_factory.CreateBatchEnv();
    if (env_ == nullptr) {
        log::error() << "Failed to create env.";
        status_ = anet::rl::TrainerStatus::COMPLETED;
        return status_;
    }

    auto batch_env_spec = env_->GetBatchSpec();
    auto env_spec = env_->GetSpec();
    log::info() << "batch_env_spec=" << batch_env_spec.ToString();
    log::info() << "env_spec=" << env_spec.ToString();
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
    agent_ = agent_factory.CreateAgent(notifier_);
    if (agent_ == nullptr) {
        log::error() << "Failed to create agent." ;
        status_ = anet::rl::TrainerStatus::COMPLETED;
        return status_;
    }

    // EpisodeEvalObserver
    notifier_->Attach<anet::rl::EpisodeEvalObserver>(
        [this](float total_reward)
        {
            this->last_target_eval_reward_ = total_reward;
        },
        single_env_factory, config_data, env_device, anet::rl::RunMode::Eval1,
        config_->eval_interval, config_->eval_interval);
    notifier_->Attach<anet::rl::EpisodeEvalObserver>(
        [this](float total_reward)
        {
            this->last_policy_eval_reward_ = total_reward;
        },
        single_env_factory, config_data, env_device, anet::rl::RunMode::Eval2,
        config_->eval_interval, config_->eval_interval);

    // 設定からObserverを生成して登録
    anet::rl::ObserverFactory factory(config_data);
    auto train_obs = factory.GetUpdateObservers();
    auto learn_obs = factory.GetLearnObservers();
    for (auto obs : train_obs) notifier_->Attach(obs);
    for (auto obs : learn_obs) notifier_->Attach(obs);

    // 環境初期化
    state_ = env_->Reset();  // ← reset() は 初期状態 を返す
    ANET_CHECK_DEVICE_CPU_MSG(state_.obs, "Initial state");
    ANET_CHECK_SHAPE(state_.obs, { ANET_SHAPE_ANY, 4 });

    // 時間計測開始
    start_time_ = std::chrono::high_resolution_clock::now();
    last_time_ = start_time_;

    status_ = anet::rl::TrainerStatus::RUNNING;

    return status_;
}

StepCounts DefaultTrainer::DoUpdateFrame(int max_steps, ControlFunction pre_step_func, ControlFunction post_step_func)
{
    anet::ProfileRange r("DefaultTrainer::DoUpdateFrame");

    ANET_ASSERT(status_ == anet::rl::TrainerStatus::RUNNING);

    auto env_spec = env_->GetSpec();
    auto batch_env_spec = env_->GetBatchSpec();

    // フレーム開始時情報
    //auto frame_train_step_start = step_counts_.train_step;
    //auto frame_exp_step_start = step_counts_.exp_step;
    //auto frame_time_start = std::chrono::high_resolution_clock::now();

    // --- 学習ステップを複数回回す ---
    float frame_total_reward = 0.0f;
    int frame_step = 0;
    while (max_steps < 0 || frame_step < max_steps) {
        anet::ProfileRange r("DefaultTrainer::DoUpdateFrame.step");

        // ステップ前制御
        auto control_signal_pre = pre_step_func(step_counts_);
        if (control_signal_pre == anet::rl::ControlSignal::STOP) {
            status_ = anet::rl::TrainerStatus::COMPLETED;
            break;
        }
        if (control_signal_pre == anet::rl::ControlSignal::BREAK) {
            break;
        }

        // ステップ前Observer
        BeforeStepEvent before_step_event{ *this, step_counts_, agent_, env_ };
        notifier_->Notify(before_step_event);

        // ステップ前情報
        auto train_step = step_counts_.train_step;

        // Stateチェック
        ANET_LOG_DEBUG("step=" << train_step << " state=" << state_.ToString());
        ANET_ASSERT(env_spec.state_spec.MatchesShape(state_.obs));
        ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.Flatten().obs));
        const int N = state_.obs.size(0);

        // 行動選択
        auto action_info = agent_->MakeAction(step_counts_, state_);
        ANET_LOG_DEBUG("step=" << train_step << " action=" << action_info.ToString());
        ANET_CHECK_SHAPE(action_info.action, { N });

        // 環境ステップ実行
        anet::rl::BatchStepResult result = env_->Step(action_info.action);    // next_state, reward, done, truncated
        ANET_LOG_DEBUG("step=" << train_step << " reward=" << anet::ToString(result.reward));
        ANET_LOG_DEBUG("step=" << train_step << " next_state=" << result.next_state.ToString());
        ANET_CHECK_DEVICE(result.next_state.obs, torch::kCPU);
        ANET_CHECK_DEVICE(result.next_state.done, torch::kCPU);
        ANET_CHECK_DEVICE(result.next_state.truncated, torch::kCPU);
        ANET_CHECK_DEVICE(result.reward, torch::kCPU);
        ANET_CHECK_DEVICE(result.continue_state.obs, torch::kCPU);
        ANET_CHECK_DEVICE(result.continue_state.done, torch::kCPU);
        ANET_CHECK_DEVICE(result.continue_state.truncated, torch::kCPU);
        ANET_CHECK_SHAPE(result.next_state.obs, { N, ANET_SHAPE_ENDANY });
        ANET_CHECK_SHAPE(result.next_state.done, { N });
        ANET_CHECK_SHAPE(result.next_state.truncated, { N });
        ANET_CHECK_SHAPE(result.reward, { N });
        ANET_CHECK_SHAPE(result.continue_state.obs, { N, ANET_SHAPE_ENDANY });
        ANET_CHECK_SHAPE(result.continue_state.done, { N });
        ANET_CHECK_SHAPE(result.continue_state.truncated, { N });

        // 報酬更新
        float step_reward = result.reward.mean().item<float>();
        last_train_reward_ = step_reward;
		train_reward_ema_.Update(last_train_reward_);

        // カウント更新
        step_counts_.train_step++;
        step_counts_.exp_step += result.n_transitions;
        step_counts_.episode_count += result.n_done;

        // Agent更新
        anet::rl::BatchExperience exp({ state_, action_info, result.reward, result.next_state });
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
        anet::rl::TrainEvent update_event{ exp, *this, step_counts_, agent_, update_result, env_ };
        notifier_->Notify(update_event);
        state_ = result.continue_state;

        // 次準備
        last_time_ = now;
        last_exp_step_ = exp_step;

        // ステップ後処理
        auto control_signal_post = post_step_func(step_counts_);
        if (control_signal_post == anet::rl::ControlSignal::STOP) {
            status_ = anet::rl::TrainerStatus::COMPLETED;
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
