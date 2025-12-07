#include "anet/trainer.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/profile.hpp"
#include "anet/observers.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/env.hpp"
#include "anet/dqn_agent.hpp"


using namespace anet::rl;


struct DefaultTrainer::Config : public anet::Config
{
    uint64_t seed = 0;
    int batch_size = 1;
    int eval_interval = 50;
    int perf_log_interval = 100;

    DefaultTrainer::Config(const anet::ConfigData& config_data) : anet::Config(config_data, "train")
    {
        ANET_READ_CONFIG(config_data, seed);
        ANET_READ_CONFIG(config_data, batch_size);
        ANET_READ_CONFIG(config_data, eval_interval);
        ANET_READ_CONFIG(config_data, perf_log_interval);
    }
};

//const torch::Device ENV_DEVICE = torch::kCPU;
const torch::Device AGENT_DEVICE = torch::kCUDA;


DefaultTrainer::DefaultTrainer(const ConfigData& config_data)
    : config_(std::make_unique<Config>(config_data))
    , device_agent_(AGENT_DEVICE)
    , train_reward_ema_(0.001)
    , train_step_per_sec_ema_(0.001)
    , exp_step_per_sec_ema_(0.001)
    , target_eval_reward_ema_(0.1)
    , policy_eval_reward_ema_(0.1)
{
    Initialize(config_data);
}

void DefaultTrainer::Initialize(const ConfigData& config_data)
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
    wxLogInfo("global_seed=%lld env_seed=%lld agent_seed=%lld", global_seed, env_seed, agent_seed);

    // パラメータ記録
    //wxLogInfo("train.preset=%s confg=%s", wxGetApp().GetConfig("train").Get("preset"), config_->ToString());
    anet::MetricsLogger::Instance()->LogJson("train/seed",
        { "global_seed", global_seed, "agent_seed", agent_seed, "env_seed", env_seed });
    anet::MetricsLogger::Instance()->LogJson("train/config", config_->ToJson());
    anet::MetricsLogger::Instance()->Flush();

    // ENV生成
    anet::rl::DefaultBatchEnvFactoryConfig env_config(config_data);
    wxLogInfo("env_config=%s", env_config.ToString());
    auto env_factory = anet::rl::DefaultBatchEnvFactory(env_config, config_data, config_->batch_size, env_seed);
    auto env_device = env_factory.GetDevice();
    auto single_env_factory = env_factory.GetSingleFactory();
    env_ = env_factory.CreateBatchEnv();
    if (env_ == nullptr) {
        wxLogError("Failed to create env.");
        return;
    }

    auto batch_env_spec = env_->GetBatchSpec();
    auto env_spec = env_->GetSpec();
    wxLogInfo("batch_env_spec=" + batch_env_spec.ToString());
    wxLogInfo("env_spec=" + env_spec.ToString());
    anet::MetricsLogger::Instance()->LogJson("env/batch_env_spec", batch_env_spec.ToJson());
    anet::MetricsLogger::Instance()->LogJson("env/env_spec", env_spec.ToJson());
    anet::MetricsLogger::Instance()->Flush();

    // ランダム方策で環境難易度評価
    /// @todo EvaluateEnvironmentDifficultyを復活
    //auto eval_result = anet::rl::EvaluateEnvironmentDifficulty(*env_, 100);
    //anet::MetricsLogger::Instance()->LogJson("eval_env", eval_result.ToJson());

    // Agent生成
    anet::rl::DQNAgentConfig agent_config(config_data);
    agent_ = std::make_shared<anet::rl::DQNAgent>(
        agent_config, batch_env_spec, env_spec, device_agent_, notifier_, agent_seed);

    // EpisodeEvalObserver
    notifier_->Attach<anet::rl::EpisodeEvalObserver>(
        [this](float total_reward)
        {
            this->last_target_eval_reward_ = total_reward;
            this->target_eval_reward_ema_.Update(total_reward);
        },
        single_env_factory, config_data, env_device, anet::rl::RunMode::Eval1,
        config_->eval_interval, config_->eval_interval);
    notifier_->Attach<anet::rl::EpisodeEvalObserver>(
        [this](float total_reward)
        {
            this->last_policy_eval_reward_ = total_reward;
            this->policy_eval_reward_ema_.Update(total_reward);
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
}

std::optional<float> DefaultTrainer::GetScalar(const std::string& key) const
{
    if (key == TRAIN_REWARD_EMA)
        return train_reward_ema_.Value();

    if (key == TARGET_EVAL_REWARD)
        return last_target_eval_reward_;
    if (key == TARGET_EVAL_REWARD_EMA)
        return target_eval_reward_ema_.Value();

    if (key == POLICY_EVAL_REWARD)
        return last_policy_eval_reward_;
    if (key == POLICY_EVAL_REWARD_EMA)
        return policy_eval_reward_ema_.Value();

    return std::nullopt;
}

void DefaultTrainer::DoUpdateFrame(int max_step, ControlFunction pre_step_func, ControlFunction post_step_func)
{
    anet::ProfileRange r("DefaultTrainer::DoUpdateFrame");

    auto env_spec = env_->GetSpec();
    auto batch_env_spec = env_->GetBatchSpec();

    // --- 学習ステップを複数回回す ---
    float frame_total_reward = 0.0f;
    int frame_step = 0;
    while (max_step < 0 || frame_step < max_step) {
        anet::ProfileRange r("DefaultTrainer::DoUpdateFrame.step");

        // ステップ前処理
        BeforeStepEvent before_step_event{ *this, step_counts_, agent_, env_ };
        notifier_->Notify(before_step_event);
        bool do_break = pre_step_func();
        if (do_break) break;

        auto train_step = step_counts_.train_step;

        // Stateチェック
        wxLogDebug("CartPoleFrame::OnTimer() step=%llu state=%s", train_step, state_.ToString());
        ANET_ASSERT(env_spec.state_spec.MatchesShape(state_.obs));
        ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.Flatten().obs));
        const int N = state_.obs.size(0);

        // 行動選択
        auto action_info = agent_->MakeAction(step_counts_, state_);
        wxLogDebug("CartPoleFrame::OnTimer() step=%llu action=%s", train_step, action_info.ToString());
        ANET_CHECK_DEVICE(action_info.action, device_agent_);
        ANET_CHECK_SHAPE(action_info.action, { N });

        // 環境ステップ実行
        anet::rl::BatchStepResult result = env_->Step(action_info.action);    // next_state, reward, done, truncated
        wxLogDebug("CartPoleFrame::OnTimer() step=%llu reward=%s", train_step, anet::ToString(result.reward));
        wxLogDebug("CartPoleFrame::OnTimer() step=%llu next_state=%s", train_step, result.next_state.ToString());
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
        train_reward_ema_.Update(step_reward);

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

        // 更新後処理
        anet::rl::TrainEvent update_event{ exp, *this, step_counts_, agent_, update_result, env_ };
        notifier_->Notify(update_event);
        state_ = result.continue_state;

        // メトリクス算出（処理性能系）
        auto exp_step = step_counts_.exp_step;
        auto exp_step_delta = exp_step - last_exp_step_;
        std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        auto msec_diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time_).count();
        if (msec_diff <= 0) msec_diff = 1;
        auto train_step_per_sec = 1.0f / static_cast<float>(msec_diff) * 1000.0f;
        auto exp_step_per_sec = static_cast<float>(exp_step_delta) / static_cast<float>(msec_diff) * 1000.0f;
        auto elapse_msec = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count());
        auto elapse_hour = elapse_msec / 1000.0f / 60.0f / 60.0f;

        // メトリクス出力（処理性能系）
        if (train_step != 0) { // 0ステップ目は誤差が大きいので
            train_step_per_sec_ema_.Update(train_step_per_sec);
            exp_step_per_sec_ema_.Update(exp_step_per_sec);

            if (train_step % config_->perf_log_interval == 0) {
                auto train_ema = train_step_per_sec_ema_.Value();
                auto exp_ema = exp_step_per_sec_ema_.Value();

                //anet::MetricsLogger::Instance()->LogScalar("90_perf/11_train_step_per_sec", train_step, train_step_per_sec);
                //anet::MetricsLogger::Instance()->LogScalar("90_perf/12_exp_step_per_sec", train_step, exp_step_per_sec);
                anet::MetricsLogger::Instance()->LogScalar("90_perf/21_train_step_per_sec_ema", train_step, train_ema);
                anet::MetricsLogger::Instance()->LogScalar("90_perf/22_exp_step_per_sec_ema", train_step, exp_ema);
                anet::MetricsLogger::Instance()->LogScalar("90_perf/82_exp_step", train_step, exp_step);
                anet::MetricsLogger::Instance()->LogScalar("90_perf/90_elapse_hour", train_step, elapse_hour);
            }
        }
        last_time_ = now;
        last_exp_step_ = exp_step;

        // ステップ後処理
        bool do_break_post = post_step_func();
		if (do_break_post) break;

		// Stepカウント
        frame_step++;
    }

    auto train_step = step_counts_.train_step;
}