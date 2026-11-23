#include "anet/test/catch.hpp"
#include "anet/test/nn.hpp"
#include "dqn_fixture_test.hpp"

TEST_CASE("Quantile huber loss matches known QR-DQN inputs", "[dqn][quantile]")
{
    auto current_dist = torch::tensor({ { 1.0f, 3.0f } });
    auto target_dist = torch::tensor({ { 2.0f, 4.0f } });
    auto taus = torch::tensor({ 0.25f, 0.75f }).view({ 1, 2, 1 });

    auto loss = QuantileLearnerBaseAccess::ComputeQuantileHuberLoss(
        current_dist,
        target_dist,
        taus,
        1.0f);

    REQUIRE(ShapeOf(loss) == std::vector<int64_t>{ 1 });
    REQUIRE(loss.item<float>() == Catch::Approx(0.625f).margin(1.0e-6f));
}

TEST_CASE("IQN quantile huber loss sums current samples and averages target samples", "[dqn][iqn][loss]")
{
    auto current_dist = torch::tensor({ { 1.0f, 3.0f } });
    auto target_dist = torch::tensor({ { 2.0f, 3.25f, 4.0f } });
    auto taus = torch::tensor({ 0.25f, 0.75f }).view({ 1, 2, 1 });

    auto result = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        current_dist,
        target_dist,
        taus,
        0.5f);

    REQUIRE(ShapeOf(result.element_loss) == std::vector<int64_t>{ 1 });
    REQUIRE(result.element_loss.item<float>() == Catch::Approx(139.0f / 192.0f).margin(1.0e-6f));
    CHECK(result.pair_abs_td.item<float>() == Catch::Approx(17.0f / 12.0f).margin(1.0e-6f));
    CHECK(result.cancellation_ratio.item<float>() == Catch::Approx(4.0f / 17.0f).margin(1.0e-6f));
}

TEST_CASE("IQN and QR quantile huber losses agree when sample counts and kappa are one", "[dqn][iqn][loss]")
{
    auto current_dist = torch::tensor({ { 1.0f, 3.0f }, { -1.0f, 2.0f } });
    auto target_dist = torch::tensor({ { 2.0f, 4.0f }, { 0.5f, 3.0f } });
    auto taus = torch::tensor({ { 0.25f, 0.75f }, { 0.2f, 0.8f } }).unsqueeze(2);

    auto qr_loss = QuantileLearnerBaseAccess::ComputeQuantileHuberLoss(
        current_dist, target_dist, taus, 1.0f);
    auto iqn_loss = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        current_dist, target_dist, taus, 1.0f);

    REQUIRE(torch::allclose(iqn_loss.element_loss, qr_loss, 1.0e-6, 1.0e-6));
}

TEST_CASE("IQN quantile huber loss is finite for one current sample", "[dqn][iqn][loss]")
{
    auto current_dist = torch::tensor({ { 1.0f } });
    auto target_dist = torch::tensor({ { 0.5f, 2.0f, 4.0f } });
    auto taus = torch::tensor({ 0.5f }).view({ 1, 1, 1 });

    auto loss = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        current_dist, target_dist, taus, 0.5f);

    REQUIRE(ShapeOf(loss.element_loss) == std::vector<int64_t>{ 1 });
    REQUIRE(torch::isfinite(loss.element_loss).all().item<bool>());
}

TEST_CASE("IQN diagnostics preserve cancellation and N-normalized loss contracts", "[dqn][iqn][loss][metrics]")
{
    const auto target_dist = torch::tensor({ { -1.0f, 1.0f } });
    const auto one_current = torch::tensor({ { 0.0f } });
    const auto one_tau = torch::tensor({ 0.5f }).view({ 1, 1, 1 });
    const auto cancelling = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        one_current, target_dist, one_tau, 1.0f);
    CHECK(cancelling.pair_abs_td.item<float>() == Catch::Approx(1.0f));
    CHECK(cancelling.cancellation_ratio.item<float>() == Catch::Approx(1.0f));

    const auto same_sign_target = torch::tensor({ { 1.0f, 1.0f } });
    const auto same_sign = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        one_current, same_sign_target, one_tau, 1.0f);
    CHECK(same_sign.cancellation_ratio.item<float>() == Catch::Approx(0.0f).margin(1.0e-6f));

    // current quantileを同じ値で2本に増やすとsample lossは2倍になるが、/N後は一致する。
    const auto two_current = torch::tensor({ { 0.0f, 0.0f } });
    const auto two_taus = torch::tensor({ 0.5f, 0.5f }).view({ 1, 2, 1 });
    const auto two_samples = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        two_current, same_sign_target, two_taus, 1.0f);
    CHECK(two_samples.element_loss.item<float>() / 2.0f
        == Catch::Approx(same_sign.element_loss.item<float>()).margin(1.0e-6f));
}

TEST_CASE("DQN TBO scalar transforms match tensor transforms", "[dqn][tbo][per][actor_initial][math]")
{
    const std::array<float, 9> values{
        -1000.0f, -10.0f, -1.0f, -0.0f, 0.0f, 1.0e-6f, 1.0f, 10.0f, 1000.0f,
    };

    for (float epsilon : { 1.0e-2f, 1.0e-3f }) {
        INFO("epsilon=" << epsilon);
        for (float value : values) {
            INFO("value=" << value);
            const float tensor_h = dqn::TransformH(torch::tensor(value), epsilon).item<float>();
            const float tensor_h_inv = dqn::TransformHInv(torch::tensor(value), epsilon).item<float>();
            CHECK(dqn::TransformH(value, epsilon)
                == Catch::Approx(tensor_h).epsilon(1.0e-5f).margin(1.0e-6f));
            CHECK(dqn::TransformHInv(value, epsilon)
                == Catch::Approx(tensor_h_inv).epsilon(1.0e-5f).margin(1.0e-6f));
        }
    }
}

TEST_CASE("TBO real-space q scalars are exposed from batch update result", "[dqn][tbo][metrics]")
{
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.use_tbo = true;
    config.tbo_epsilon = 1.0e-2f;

    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };
    TestLearner learner(config, model, vars, batch_env_spec, env_spec, dqn::LearnerForwardResources{});

    auto raw_max_q = torch::tensor({ -10.0f, 0.0f, 100.0f });
    auto raw_q_sa = torch::tensor({ -1.0f, 10.0f, 1000.0f });
    auto max_q = learner.TransformH(raw_max_q);
    auto q_sa = learner.TransformH(raw_q_sa);

    dqn::OptimizerStepResult opt_result;
    dqn::PerPriorityUpdateInfo per_info;
    auto result = learner.MakeBatchUpdateResult(
        torch::tensor(0.0f),
        torch::zeros({ 3 }),
        opt_result,
        max_q,
        q_sa,
        per_info);

    auto max_mean = result->GetScalar("q_max_real_mean", -1);
    auto max_max = result->GetScalar("q_max_real_max", -1);
    auto max_std = result->GetScalar("q_max_real_std", -1);
    auto sa_mean = result->GetScalar("q_sa_real_mean", -1);
    REQUIRE(max_mean.has_value());
    REQUIRE(max_max.has_value());
    REQUIRE(max_std.has_value());
    REQUIRE(sa_mean.has_value());
    CHECK(*max_mean == Catch::Approx(raw_max_q.mean().item<float>()).margin(1.0e-4f));
    CHECK(*max_max == Catch::Approx(raw_max_q.max().item<float>()).margin(1.0e-4f));
    CHECK(*max_std == Catch::Approx(raw_max_q.std(false).item<float>()).margin(1.0e-4f));
    CHECK(*sa_mean == Catch::Approx(raw_q_sa.mean().item<float>()).margin(1.0e-4f));

    dqn::LearnerConfig off_config = WithDefaultLearnerForwardConfig();
    TestLearner off_learner(off_config, model, vars, batch_env_spec, env_spec, dqn::LearnerForwardResources{});
    auto off_result = off_learner.MakeBatchUpdateResult(
        torch::tensor(0.0f),
        torch::zeros({ 3 }),
        opt_result,
        raw_max_q,
        raw_q_sa,
        per_info);
    CHECK(off_result->GetScalar("q_max_real_mean", -1).has_value());
    CHECK(off_result->GetScalar("q_sa_real_mean", -1).has_value());
}

TEST_CASE("Learner stops polling replay buffer size after minibatch threshold is reached", "[dqn][replay_buffer][performance]")
{
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;

    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };
    TestLearner learner(config, model, vars, batch_env_spec, env_spec, dqn::LearnerForwardResources{});
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->SetSizeValues({ 0, config.replay_batch_size, 0 });
    learner.UseReplayBuffer(replay_buffer);

    auto make_counts = [&vars](rl::step_t step) {
        rl::StepCounts counts;
        counts.train_step = step;
        counts.exp_step = step;
        counts.learn_step = vars.learn_step;
        return counts;
    };

    auto cold_result = learner.UpdateFromBatch(make_counts(0), MakeDeterminismExperience(0, 1));
    REQUIRE(cold_result.empty());
    REQUIRE(replay_buffer->size_count == 1);
    REQUIRE(replay_buffer->sample_count == 0);

    auto first_update = learner.UpdateFromBatch(make_counts(1), MakeDeterminismExperience(1, 1));
    REQUIRE(first_update.size() == 1);
    REQUIRE(replay_buffer->size_count == 2);
    REQUIRE(replay_buffer->sample_count == 1);

    auto latched_update = learner.UpdateFromBatch(make_counts(2), MakeDeterminismExperience(2, 1));
    REQUIRE(latched_update.size() == 1);
    REQUIRE(replay_buffer->size_count == 2);
    REQUIRE(replay_buffer->sample_count == 2);
    REQUIRE(replay_buffer->push_count == 3);
}

TEST_CASE("Disabled learner leaves the replay buffer and the network untouched", "[dqn][learner][enabled]")
{
    // 学習停止時はReplayBuffer・forward・parameterのどれも動かさない。
    auto probe_state = std::make_shared<AutocastProbeState>();
    AutocastProbeNetworkModel model(probe_state, torch::Device(torch::kCPU));
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.enabled = false;
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;

    auto env_spec = MakeLearnerEnvSpec();
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    // 有効ならUpdateFromSamplesがforwardするので、probeが0のままならupdate自体が起きていない。
    learner.EnablePlasticityForward(false);
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->SetSizeValues({ 1024 });
    // 注入しても一切参照しないことを見る(本番ではRB自体を構築しない)。
    learner.UseReplayBuffer(replay_buffer);

    std::vector<torch::Tensor> before;
    for (const auto& parameter : model.GetOnlineParameters()) {
        before.push_back(parameter.detach().clone());
    }

    for (rl::step_t step = 0; step < 3; ++step) {
        rl::StepCounts counts;
        counts.train_step = step;
        counts.exp_step = step;
        auto results = learner.UpdateFromBatch(counts, rl::BatchExperience{});
        // 空を返すとLearnEventが発火せずevalが駆動されないので、1件返す契約。
        REQUIRE(results.size() == 1);
        REQUIRE(results.front() != nullptr);
    }

    CHECK(replay_buffer->push_count == 0);
    CHECK(replay_buffer->sample_count == 0);
    CHECK(replay_buffer->unique_sample_count == 0);
    CHECK(replay_buffer->size_count == 0);
    CHECK(replay_buffer->update_count == 0);
    CHECK(probe_state->forward_count == 0);
    CHECK(vars.learn_step == 0);

    const auto after = model.GetOnlineParameters();
    REQUIRE(after.size() == before.size());
    for (size_t i = 0; i < after.size(); ++i) {
        CHECK(after[i].equal(before[i]));
    }
}

TEST_CASE("Disabled learner reports known diagnostic keys as NaN", "[dqn][learner][enabled][nan]")
{
    // 未測定の既知keyはnullopt(未知扱い)でも 0(偽装)でもなくNaNで返す。
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.enabled = false;

    auto env_spec = MakeLearnerEnvSpec();
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    const auto results = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(results.size() == 1);
    const auto& result = results.front();

    for (const char* key : {
        "loss", "td_mean", "td_std", "grad_norm", "grad_clip_ratio",
        "q_std", "q_gap", "q_gap_rel",
        "q_max_mean", "q_max_max", "q_max_std", "q_sa_mean",
        "per_td_error_abs_max", "per_prio_clip_ratio" }) {
        CAPTURE(key);
        const auto value = result->GetScalar(key);
        REQUIRE(value.has_value());
        CHECK(std::isnan(*value));
    }
    CHECK_FALSE(result->GetScalar("no_such_metric_key").has_value());

    // 学習停止中も 6 群の既知キーは未知扱いにせず、未成立値として返す。
    for (const char* key : {"plasticity_dormant_ratio", "policy_churn_action_ratio",
        "replay_fit_unsampled_td_mean", kNoisyQAbsDiffKey, "attention_entropy.[0]", "grad_noise_scale_adam"}) {
        CAPTURE(key);
        const auto value = result->GetScalar(key);
        REQUIRE(value.has_value());
        CHECK(std::isnan(*value));
    }

    // $agent 経由の replaybuffer.* も Learner が所有する既知の名前空間なので、
    // RBを構築しないRunでもNaNで返す。nullopt だと購読側がupdateごとにWARNを出す。
    for (const char* key : {
        "replaybuffer.per.last_evicted_never_sampled_ratio",
        "replaybuffer.per.actor_initial_mass_ratio",
        "replaybuffer.size" }) {
        CAPTURE(key);
        const auto value = learner.GetScalar(key);
        REQUIRE(value.has_value());
        CHECK(std::isnan(*value));
    }
    CHECK_FALSE(learner.GetScalar("no_such_agent_key").has_value());
}

TEST_CASE("LearnerConfig keeps learning enabled by default", "[dqn][learner][enabled][config]")
{
    CHECK(dqn::LearnerConfig{}.enabled);
    CHECK(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(anet::ConfigData{})).learner.enabled);
}

TEST_CASE("DefaultDQNAgentConfig warns when learning is disabled without a checkpoint", "[dqn][learner][enabled][config]")
{
    // 初期重みの評価も意図的な構成としてあり得るので、止めずWARNだけ出す。
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.learner.enabled", "false");

    anet::test::LogCaptureGuard logs;
    const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));
    CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)));
    logs.Flush();

    CHECK_FALSE(config.learner.enabled);
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 1);
    CHECK(anet::test::HasRecordContaining(
        logs.Records(), wxLOG_Warning, { "learner.enabled=false", "auto_load_file" }));

    config_data.Set("DefaultDQNAgent.auto_load_file", "agent_close.anet");
    anet::test::LogCaptureGuard loaded_logs;
    const dqn::DefaultDQNAgentConfig loaded(WithDefaultForwardConfig(config_data));
    loaded_logs.Flush();

    CHECK_FALSE(loaded.learner.enabled);
    CHECK_FALSE(anet::test::HasRecordContaining(
        loaded_logs.Records(), wxLOG_Warning, { "learner.enabled=false" }));
}

TEST_CASE("PER priority prepare/apply counts only priorities changed by clipping", "[dqn][per][clip]")
{
    const bool with_diagnostics = GENERATE(false, true);
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.use_per = true;
    config.per_eps = 0.1f;
    config.use_per_prio_clip = true;
    config.per_prio_clip_value = 1.0f;

    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };
    TestLearner learner(config, model, vars, batch_env_spec, env_spec, dqn::LearnerForwardResources{});
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->priority_update_result = rl::ReplayPriorityUpdateResult{
        .applied_count = 2,
        .stale_count = 1,
        .actor_learner_pair_count = 4,
        .actor_learner_positive_pair_ratio = 0.75f,
        .actor_learner_ratio_median = 1.25f,
        .actor_learner_log_ratio_mean = -0.5f,
        .actor_learner_spearman = 0.6f,
    };
    learner.UseReplayBuffer(replay_buffer);

    rl::ExperienceSamples samples;
    samples.replay_item_keys = torch::tensor({ 3, 4, 5 }, torch::TensorOptions().dtype(torch::kInt64));
    samples.is_weights = torch::tensor({ 0.2f, 0.3f, 0.5f });
    samples.per_priority_sources = torch::tensor(
        { static_cast<int8_t>(rl::ReplayPrioritySource::FIXED_INITIAL),
          static_cast<int8_t>(rl::ReplayPrioritySource::LEARNER_UPDATED),
          static_cast<int8_t>(rl::ReplayPrioritySource::LEARNER_UPDATED) },
        torch::TensorOptions().dtype(torch::kInt8));

    // clip前priorityが上限未満・等値・超過となる3境界を同時に検証する。
    auto td_error = torch::tensor({ -0.2f, 0.9f, 2.0f });
    auto upper_tail_std = torch::tensor({ 1.0f, 2.0f, 3.0f });
    const auto iqn_diagnostics = with_diagnostics ? torch::arange(7, torch::kFloat32) + 10.0f : torch::Tensor();
    const auto munchausen_diagnostics = with_diagnostics ? torch::arange(5, torch::kFloat32) + 20.0f : torch::Tensor();
    auto pending = learner.PreparePerPriorityUpdate(
        samples, td_error, iqn_diagnostics, upper_tail_std, munchausen_diagnostics);

    REQUIRE(pending.enabled);
    const auto expected_indices = std::vector<int64_t>{ 3, 4, 5 };
    CHECK(pending.indices == expected_indices);
    REQUIRE(pending.per_sample_initial_count.defined());
    CHECK(pending.per_sample_initial_count.item<float>() == Catch::Approx(1.0f).margin(1.0e-6f));

    auto result = learner.ApplyPerPriorityUpdate(std::move(pending));
    if (with_diagnostics) {
        CHECK(torch::equal(result.iqn_diagnostics, iqn_diagnostics));
        CHECK(torch::equal(result.munchausen_diagnostics, munchausen_diagnostics));
    }

    CHECK(replay_buffer->update_count == 1);
    CHECK(replay_buffer->last_indices == expected_indices);
    REQUIRE(replay_buffer->last_priorities.size() == 3);
    CHECK(replay_buffer->last_priorities[0] == Catch::Approx(0.3f).margin(1.0e-6f));
    CHECK(replay_buffer->last_priorities[1] == Catch::Approx(1.0f).margin(1.0e-6f));
    CHECK(replay_buffer->last_priorities[2] == Catch::Approx(1.0f).margin(1.0e-6f));

    REQUIRE(result.per_priorities.defined());
    CHECK(result.per_priorities.device().is_cpu());
    CHECK(torch::allclose(result.per_priorities, torch::tensor({ 0.3f, 1.0f, 1.0f })));
    REQUIRE(result.per_clipped_count.defined());
    CHECK(result.per_clipped_count.device().is_cpu());
    CHECK(result.per_clipped_count.item<int64_t>() == 1);
    CHECK(result.per_minibatch_size == 3);
    REQUIRE(result.per_is_weights.defined());
    CHECK(torch::allclose(result.per_is_weights, samples.is_weights));
    REQUIRE(result.per_sample_initial_count.defined());
    CHECK(result.per_sample_initial_count.item<float>() == Catch::Approx(1.0f).margin(1.0e-6f));
    CHECK(result.per_update_result.applied_count == 2);
    CHECK(result.per_update_result.stale_count == 1);
    CHECK(result.per_update_result.actor_learner_pair_count == 4);
    CHECK(result.per_update_result.actor_learner_positive_pair_ratio == Catch::Approx(0.75f));
    CHECK(result.per_update_result.actor_learner_ratio_median == Catch::Approx(1.25f));
    CHECK(result.per_update_result.actor_learner_log_ratio_mean == Catch::Approx(-0.5f));
    CHECK(result.per_update_result.actor_learner_spearman == Catch::Approx(0.6f));

    dqn::OptimizerStepResult opt_result;
    auto batch_result = learner.MakeBatchUpdateResult(
        torch::tensor(0.0f),
        td_error,
        opt_result,
        torch::zeros({ 3 }),
        torch::zeros({ 3 }),
        result);
    auto sample_initial_ratio = batch_result->GetScalar("per_sample_initial_ratio", -1);
    REQUIRE(sample_initial_ratio.has_value());
    CHECK(*sample_initial_ratio == Catch::Approx(1.0f / 3.0f).margin(1.0e-6f));
    CHECK(batch_result->GetScalar("per_priority_update_stale_ratio", -1).value()
        == Catch::Approx(1.0f / 3.0f).margin(1.0e-6f));
    CHECK(batch_result->GetScalar("per_actor_learner_pair_count", -1).value() == Catch::Approx(4.0f));
    CHECK(batch_result->GetScalar("per_actor_learner_positive_pair_ratio", -1).value() == Catch::Approx(0.75f));
    CHECK(batch_result->GetScalar("per_actor_learner_ratio_median", -1).value() == Catch::Approx(1.25f));
    CHECK(batch_result->GetScalar("per_actor_learner_log_ratio_mean", -1).value() == Catch::Approx(-0.5f));
    CHECK(batch_result->GetScalar("per_actor_learner_spearman", -1).value() == Catch::Approx(0.6f));
    const auto upper_tail_spearman = batch_result->GetScalar("upper_tail_priority_spearman", -1);
    REQUIRE(upper_tail_spearman.has_value());
    CHECK(*upper_tail_spearman == Catch::Approx(std::sqrt(3.0f) / 2.0f).margin(1.0e-6f));
}

TEST_CASE("Upper-tail priority Spearman handles rank direction and undefined cases", "[dqn][per][quantile][metrics]")
{
    auto calculate = [](const torch::Tensor& td_error, const torch::Tensor& upper_tail_std, bool use_per) {
        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        config.use_per = use_per;
        config.per_eps = 0.0f;

        auto env_spec = MakeLearnerEnvSpec();
        TestNetworkModel model;
        dqn::RuntimeVars vars;
        TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
        auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
        learner.UseReplayBuffer(replay_buffer);

        const int64_t batch_size = td_error.size(0);
        rl::ExperienceSamples samples;
        samples.replay_item_keys = torch::arange(
            batch_size, torch::TensorOptions().dtype(torch::kInt64));
        auto pending = learner.PreparePerPriorityUpdate(
            samples, td_error, torch::Tensor(), upper_tail_std);
        auto per_result = learner.ApplyPerPriorityUpdate(std::move(pending));
        auto batch_result = learner.MakeBatchUpdateResult(
            torch::tensor(0.0f),
            td_error,
            dqn::OptimizerStepResult{},
            torch::zeros({ batch_size }),
            torch::zeros({ batch_size }),
            per_result);
        return batch_result->GetScalar("upper_tail_priority_spearman", -1).value();
    };

    // raw priorityと同順・逆順なら、それぞれ順位相関は+1・-1になる。
    const auto increasing_priority = torch::tensor({ 1.0f, 2.0f, 3.0f });
    CHECK(calculate(increasing_priority, torch::tensor({ 1.0f, 2.0f, 3.0f }), true)
        == Catch::Approx(1.0f));
    CHECK(calculate(increasing_priority, torch::tensor({ 3.0f, 2.0f, 1.0f }), true)
        == Catch::Approx(-1.0f));

    // 順位分散がない場合、batch不足、PER無効は相関を定義しない。
    CHECK(std::isnan(calculate(torch::tensor({ 1.0f }), torch::tensor({ 2.0f }), true)));
    CHECK(std::isnan(calculate(increasing_priority, torch::ones({ 3 }), true)));
    CHECK(std::isnan(calculate(torch::tensor({ 1.0f, -1.0f, 1.0f }),
        torch::tensor({ 1.0f, 2.0f, 3.0f }), true)));
    CHECK(std::isnan(calculate(increasing_priority, torch::tensor({ 1.0f, 2.0f, 3.0f }), false)));
}

TEST_CASE("PER IQN diagnostics classify only explicit initial priority sources", "[dqn][per][iqn][metrics]")
{
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.use_per = true;
    config.per_eps = 0.0f;

    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    learner.UseReplayBuffer(replay_buffer);

    rl::ExperienceSamples samples;
    samples.replay_item_keys = torch::arange(5, torch::TensorOptions().dtype(torch::kInt64));
    samples.is_weights = torch::ones({ 5 });
    samples.per_priority_sources = torch::tensor(
        { static_cast<int8_t>(rl::ReplayPrioritySource::FIXED_INITIAL),
          static_cast<int8_t>(rl::ReplayPrioritySource::MAX_INITIAL),
          static_cast<int8_t>(rl::ReplayPrioritySource::ACTOR_INITIAL),
          static_cast<int8_t>(rl::ReplayPrioritySource::NONE),
          static_cast<int8_t>(rl::ReplayPrioritySource::LEARNER_UPDATED) },
        torch::TensorOptions().dtype(torch::kInt8));
    const auto diagnostics = torch::tensor({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f });

    auto pending = learner.PreparePerPriorityUpdate(samples, torch::zeros({ 5 }), diagnostics);
    CHECK(pending.per_sample_initial_count.item<float>() == Catch::Approx(3.0f));
    CHECK(pending.per_sample_fixed_initial_count.item<float>() == Catch::Approx(1.0f));
    CHECK(pending.per_sample_max_initial_count.item<float>() == Catch::Approx(1.0f));
    CHECK(pending.per_sample_actor_initial_count.item<float>() == Catch::Approx(1.0f));

    const auto result = learner.ApplyPerPriorityUpdate(std::move(pending));
    CHECK(replay_buffer->update_count == 1);
    REQUIRE(result.iqn_diagnostics.defined());
    CHECK(torch::equal(result.iqn_diagnostics, diagnostics));
}

TEST_CASE("Optimizer helper keeps QR-DQN FP32 grad clip result contract", "[dqn][optimizer]")
{
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.use_amp = false;
    config.use_amp_bf16 = false;
    config.use_grad_clip = true;
    config.grad_clip_tau = 0.5f;

    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };
    TestLearner learner(config, model, vars, batch_env_spec, env_spec, dqn::LearnerForwardResources{});
    learner.UseSgd(0.1f);

    auto weight_before = model.GetOnlineParameters()[0].detach().clone().cpu();
    auto obs = anet::TensorDict{ { kVectorKey, torch::tensor({ { 3.0f, 4.0f } }) } };
    auto loss = model.GetOnlineNetwork()->Forward(obs).At("q").sum();
    auto result = learner.Optimize(loss, loss.reshape({1}));

    CHECK_FALSE(result.grad_norm.has_value());
    REQUIRE(result.grad_norm_tensor.defined());
    CHECK(result.grad_norm_tensor.item<float>() == Catch::Approx(5.0f).margin(1.0e-5f));
    CHECK(result.grad_clip_tau == Catch::Approx(0.5f).margin(1.0e-6f));
    CHECK(result.grad_clip_ratio == Catch::Approx(0.0f).margin(1.0e-6f));

    auto weight_delta = model.GetOnlineParameters()[0].detach().cpu() - weight_before;
    CHECK(weight_delta[0][0].item<float>() == Catch::Approx(-0.03f).margin(1.0e-5f));
    CHECK(weight_delta[0][1].item<float>() == Catch::Approx(-0.04f).margin(1.0e-5f));
}

TEST_CASE("DQN learner BF16 autocast follows learner device", "[dqn][learner][amp][bf16]")
{
    std::vector<torch::Device> devices{ torch::Device(torch::kCPU) };
    if (torch::cuda::is_available()) {
        devices.emplace_back(torch::Device(torch::kCUDA, 0));
    }

    for (const auto& device : devices) {
        INFO("device=" << device.str());

        auto probe_state = std::make_shared<AutocastProbeState>();
        AutocastProbeNetworkModel model(probe_state, device);
        dqn::RuntimeVars vars;
        auto env_spec = MakeLearnerEnvSpec();
        rl::BatchEnvSpec batch_env_spec{ 2, 1 };

        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        config.use_amp = true;
        config.use_amp_bf16 = true;
        config.use_fused_optimizer = false;
        config.use_grad_clip = false;
        config.use_per = false;
        config.replay_batch_size = 2;
        config.replay_capacity = 8;

        dqn::ActionPolicyConfig target_policy_config;
        target_policy_config.use_amp = true;
        target_policy_config.use_amp_bf16 = true;
        auto target_policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(
            target_policy_config,
            false,
            0,
            device);

        const auto device_type = device.type();
        const bool original_enabled = at::autocast::is_autocast_enabled(device_type);
        dqn::TDLearner learner(
            config,
            model,
            vars,
            nullptr,
            batch_env_spec,
            env_spec,
            device,
            123,
            kTestAgentSeed,
            target_policy,
            dqn::LearnerForwardResources{},
            std::nullopt,
            456);

        auto result = learner.UpdateFromSamples(MakeAutocastProbeSamples(device));

        REQUIRE(result != nullptr);
        REQUIRE(probe_state->forward_count > 0);
        CHECK(probe_state->enabled_count == probe_state->forward_count);
        CHECK(probe_state->last_device_type == device_type);
        CHECK(at::autocast::is_autocast_enabled(device_type) == original_enabled);
    }
}

TEST_CASE("Skip forward mode validates recognized values even when skip is disabled", "[dqn][rfd0106][rfd0106-mode]")
{
    const bool enabled = GENERATE(false, true);
    auto data = MakeRfd0104ConfigData();
    data.Set("DefaultDQNAgent.skip.enabled", enabled);
    data.Set("DefaultDQNAgent.actor.[eval].skip.nn_forward_mode", "unknown");
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring(
        "Invalid DefaultDQNAgent.actor.[eval].skip.nn_forward_mode='unknown'; expected shared or dedicated."));
}

TEST_CASE("Explicit skip forward references are validated in every mode", "[dqn][rfd0106][rfd0106-reference]")
{
    const bool enabled = GENERATE(false, true);
    const auto mode = GENERATE(std::string("shared"), std::string("dedicated"));
    auto data = MakeRfd0104ConfigData();
    data.Set("DefaultDQNAgent.skip.enabled", enabled);
    data.Set("DefaultDQNAgent.actor.[eval].skip.nn_forward_mode", mode);
    data.Set("DefaultDQNAgent.actor.[eval].skip.nn_forward_key", "undeclared");
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring(
        "Invalid DefaultDQNAgent.actor.[eval].skip.nn_forward_key='undeclared'; expected a declared nn_forward key."));
}

TEST_CASE("Dedicated skip requires keys for all declared Actors only while enabled", "[dqn][rfd0106][rfd0106-required]")
{
    const bool enabled = GENERATE(false, true);
    auto data = MakeRfd0104ConfigData();
    data.Set("DefaultDQNAgent.skip.enabled", enabled);
    data.Set("DefaultDQNAgent.actor.[train].skip.nn_forward_mode", "dedicated");
    data.Set("DefaultDQNAgent.actor.[eval].skip.nn_forward_mode", "dedicated");
    if (enabled) {
        CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data),
            Catch::Matchers::ContainsSubstring("Missing DefaultDQNAgent.actor.[train].skip.nn_forward_key; expected a declared nn_forward key when "
                "DefaultDQNAgent.actor.[train].skip.nn_forward_mode=dedicated and DefaultDQNAgent.skip.enabled=true.")
            && Catch::Matchers::ContainsSubstring("Missing DefaultDQNAgent.actor.[eval].skip.nn_forward_key; expected a declared nn_forward key when "
                "DefaultDQNAgent.actor.[eval].skip.nn_forward_mode=dedicated and DefaultDQNAgent.skip.enabled=true."));
    } else {
        CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(data));
    }
}

TEST_CASE("Shared skip warns once per explicit unused key only while enabled", "[dqn][rfd0106][rfd0106-warning]")
{
    const bool enabled = GENERATE(false, true);
    const auto mode = GENERATE(std::string("shared"), std::string("dedicated"));
    auto data = MakeRfd0104ConfigData();
    data.Set("DefaultDQNAgent.skip.enabled", enabled);
    for (const auto* actor : {"train", "eval"}) {
        const auto prefix = std::string("DefaultDQNAgent.actor.[") + actor + "].skip.";
        data.Set(prefix + "nn_forward_mode", mode);
        data.Set(prefix + "nn_forward_key", "test_default");
    }
    anet::test::LogCaptureGuard logs;
    CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(data));
    CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(data));
    logs.Flush();
    // 宣言ごとに一度だけ診断し、無効状態と dedicated の明示参照には出さない。
    for (const auto* actor : {"train", "eval"}) {
        const auto prefix = std::string("DefaultDQNAgent.actor.[") + actor + "].skip.";
        const auto expected = prefix + "nn_forward_key='test_default' is unused with " + prefix
            + "nn_forward_mode=shared; expected " + prefix + "nn_forward_mode=dedicated to use this value.";
        int count = 0;
        for (const auto& record : logs.Records()) {
            if (record.level == wxLOG_Warning && record.message == expected) ++count;
        }
        CHECK(count == (enabled && mode == "shared" ? 1 : 0));
    }
}

TEST_CASE("Skip length values are validated even when the Actor or Learner does not use them", "[dqn][rfd0104][config]")
{
    const bool enabled = GENERATE(false, true);
    const bool learning = GENERATE(false, true);
    const auto mode = GENERATE(std::string("full"), std::string("single"), std::string("variable"));
    const auto value = GENERATE(std::string("missing"), std::string("zero"), std::string("negative"),
        std::string("large"), std::string("valid"), std::string("unknown_type"));
    CAPTURE(enabled, learning, mode, value);
    auto data = MakeRfd0104ConfigData();
    data.Set("DefaultDQNAgent.skip.enabled", enabled);
    data.Set("DefaultDQNAgent.learner.enabled", learning);
    data.Set("DefaultDQNAgent.actor.[train].skip.max_length_mode", mode);
    const std::string key = "DefaultDQNAgent.actor.[train].skip.max_length";
    if (value != "missing") data.Set(key, value == "zero" ? 0 : value == "negative" ? -1 : value == "large" ? 4 : 1);
    if (value == "unknown_type") data.Set(key + ".type", "unknown");
    const bool invalid = value == "zero" || value == "negative" || value == "unknown_type"
        || (enabled && value == "large") || (enabled && mode == "variable" && value == "missing");
    anet::test::LogCaptureGuard logs;
    if (invalid) CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring("skip.max_length"));
    else CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(data));
}

TEST_CASE("Skip scalar readback timing", "[.][rfd0098-readback]")
{
    if (!torch::cuda::is_available()) return;
    // 代表のN=64/J=10で、初回取得の集約・同期転送とキャッシュ後の読取を合わせて測る。
    const auto options = torch::TensorOptions().device(torch::kCUDA);
    const auto actions = torch::zeros({64}, options.dtype(torch::kLong));
    const rl::AuxData aux{{"skip_decision", torch::ones({64}, options.dtype(torch::kBool))},
        {"skip_length", torch::ones({64}, options.dtype(torch::kLong))},
        {"skip_q_values", torch::ones({64, 10}, options.dtype(torch::kFloat32))}};
    const auto metrics = torch::ones({5}, options.dtype(torch::kFloat32));
    for (const bool actor : {true, false}) {
        double microseconds = 0;
        for (int index = 0; index < 220; ++index) {
            dqn::DQNActionInfo action(actions, {}, aux);
            dqn::BatchUpdateResult update;
            update.skip_metrics = metrics;
            const auto started = std::chrono::steady_clock::now();
            if (actor) {
                REQUIRE(*action.GetScalar("skip_decision_rate") == 1);
                for (int length = 1; length <= 10; ++length) action.GetScalar("skip_length_rate.[" + std::to_string(length) + "]");
            } else {
                REQUIRE(*update.GetScalar("skip_loss") == 1);
                update.GetScalar("skip_grad_norm");
            }
            if (index >= 20) microseconds += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
        }
        std::cout << "Skip readback " << (actor ? "Actor N=64 J=10" : "Learner 5 scalars")
            << " mean_microseconds=" << microseconds / 200 << std::endl;
    }
}

TEST_CASE("DQN checkpoint does not attribute fixed skip-primary decay to config", "[dqn][rfd0194]")
{
    anet::test::RegisterStandardNetworkModules();
    const auto env = MakeLearnerEnvSpec();
    anet::nn::NetworkConfig skip_net;
    skip_net.output_keys["features"] = "features";
    const anet::nn::WeightInitConfig init;
    dqn::SkipNetworkDefinition skip{.config = {.enabled = true, .max_length = 1}, .network_config = skip_net,
        .head_factory = std::make_shared<dqn::SkipHeadFactory>(1, 1, dqn::SkipNetworkMode::ActionOut, "none", 1, init), .seed = 99};
    dqn::NetworkModel model({}, torch::kCPU, MakeAgentForwardNetworkConfig(), env.state_spec.obs_spec, 1,
        std::make_shared<dqn::LinearHeadFactory>(1, init), false, 98, &skip);
    auto config = WithDefaultLearnerForwardConfig();
    config.replay_capacity = 16;
    config.replay_batch_size = 2;
    config.skip.feature_gradient = "split";
    dqn::RuntimeVars vars;
    TestLearner learner(config, model, vars, {1, 1}, env, dqn::LearnerForwardResources{});
    learner.SetupAdamW();
    auto& options = static_cast<torch::optim::AdamWOptions&>(learner.SkipPrimaryOptimizer().param_groups().front().options());
    options.weight_decay(0.25);

    std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(bytes);
    REQUIRE(learner.Save(output) > 0);
    bytes.seekg(0);
    anet::InputArchive input(bytes);
    anet::test::LogCaptureGuard logs;
    REQUIRE(learner.Load(input) > 0);
    logs.Flush();
    bool found_difference = false;
    for (const auto& record : logs.Records()) {
        if (record.message.find("role=primary_skip option=weight_decay") == std::string::npos) continue;
        found_difference = true;
        CHECK(record.message.find("checkpoint=0.25 fixed=0") != std::string::npos);
        CHECK(record.message.find("key=") == std::string::npos);
    }
    CHECK(found_difference);
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message, {"optimizer options applied", "fixed weight decay 0"}));
}

TEST_CASE("Shared skip baseline and feature seam preserve their gradient contracts", "[dqn][rfd0098]")
{
    for (const bool distributional : {false, true}) {
        const auto options = torch::TensorOptions().requires_grad(true);
        auto primary_q = torch::tensor({{3.0f, 5.0f}}, options);
        auto primary_dist = torch::tensor({{{2.0f, 4.0f}, {4.0f, 6.0f}}}, options);
        auto advantage = torch::arange(distributional ? 12 : 6, torch::kFloat32)
            .reshape(distributional ? std::vector<int64_t>{1, 2, 3, 2} : std::vector<int64_t>{1, 2, 3}).requires_grad_(true);
        anet::TensorDict primary{{"q", primary_q}, {"q_dist", primary_dist}};
        anet::TensorDict output{{distributional ? "skip_q_dist" : "skip_q", advantage}};
        dqn::SkipConfig config;
        config.max_length = 3;
        config.head.use_dueling = true;
        const auto values = dqn::ReadSkipValues(output, primary, torch::tensor({1}, torch::kInt64), config);
        const auto selected = values.At(distributional ? "q_dist" : "q");
        CHECK(torch::equal(selected.select(1, 0), distributional ? primary_dist.select(1, 1) : primary_q.select(1, 1)));
        selected.sum().backward();
        CHECK_FALSE(primary_q.grad().defined());
        CHECK_FALSE(primary_dist.grad().defined());
        REQUIRE(advantage.grad().defined());
        CHECK(advantage.grad().abs().sum().item<float>() > 0.0f);
        for (const bool stop : {false, true}) {
            const auto seam = dqn::MakeSkipInput({}, primary, torch::tensor({1}, torch::kInt64), stop);
            CHECK(seam.At("q").requires_grad() == !stop);
            CHECK(seam.At("q").is_same(primary_q) == !stop);
        }
    }
}

TEST_CASE("Skip updates reject nonfinite gradients before either optimizer applies them", "[dqn][rfd0098][rfd0098-review-nan]")
{
#if ANET_ENABLE_TENSOR_NAN_CHECK
    anet::test::RegisterStandardNetworkModules();
    const bool poison_skip = GENERATE(false, true);
    const bool bf16 = GENERATE(false, true);
    const auto poison = GENERATE(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity());
    const auto env = MakeLearnerEnvSpec();
    anet::nn::NetworkConfig skip_net;
    skip_net.output_keys["features"] = "features";
    const anet::nn::WeightInitConfig init;
    dqn::SkipNetworkDefinition skip{.config = {.enabled = true, .max_length = 1}, .network_config = skip_net,
        .head_factory = std::make_shared<dqn::SkipHeadFactory>(1, 1, anet::rl::dqn::SkipNetworkMode::ActionOut, "none", 1, init), .seed = 99};
    dqn::NetworkModel model({}, torch::kCPU, MakeAgentForwardNetworkConfig(), env.state_spec.obs_spec, 1,
        std::make_shared<dqn::LinearHeadFactory>(1, init), false, 98, &skip);
    auto config = WithDefaultLearnerForwardConfig();
    config.replay_capacity = 16;
    config.replay_batch_size = 2;
    config.use_per = false;
    config.use_tbo = false;
    config.use_td_clip = false;
    config.use_amp = bf16;
    config.use_amp_bf16 = bf16;
    config.use_fused_optimizer = false;
    dqn::RuntimeVars vars;
    dqn::TDLearner learner(config, model, vars, nullptr, {.num_envs = 2, .num_threads = 1}, env, torch::kCPU, 98, kTestAgentSeed,
        std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), {});
    const auto network = poison_skip ? model.GetSkipOnlineNetwork() : model.GetOnlineNetwork();
    const auto parameter = network->parameters().front();
    const auto before = parameter.detach().clone();
    // forward値は有限のまま、backwardで選んだNetworkだけを壊す。
    parameter.register_hook([poison](torch::Tensor grad) { return torch::full_like(grad, poison); });
    rl::ExperienceSamples samples;
    samples.obs = anet::TensorDict{{kVectorKey, torch::ones({2, 2})}};
    samples.actions = torch::zeros({2}, torch::kInt64);
    samples.target_returns = torch::ones({2});
    samples.n_steps = torch::ones({2}, torch::kInt64);
    samples.next_state.next_obs = samples.obs;
    samples.next_state.terminals = torch::ones({2}, torch::kBool);
    samples.window.lengths = samples.n_steps;
    samples.window.max_lengths = samples.n_steps;
    samples.window.returns = samples.target_returns;
    samples.window.next_obs = samples.obs;
    samples.window.terminals = samples.next_state.terminals;
    samples.window.truncated = torch::zeros({2}, torch::kBool);
    CHECK_THROWS_WITH(learner.UpdateFromSamples(samples), Catch::Matchers::ContainsSubstring("Nonfinite gradient norms"));
    CHECK(torch::equal(parameter, before));
#else
    SUCCEED("Gradient NaN checks are disabled in this build");
#endif
}

TEST_CASE("Split skip update adds independent Adam updates from the same parameters", "[dqn][rfd0103][split_numeric]")
{
    const bool fused = GENERATE(false, true);
    const bool clipped = GENERATE(false, true);
    anet::test::RegisterStandardNetworkModules();
    const auto env = MakeLearnerEnvSpec();
    anet::ConfigData data;
    data.Set("net.block.[SharedLinear].type", "Linear");
    data.Set("net.block.[SharedLinear].linear.out_features", 2);
    data.Set("net.branch.[shared].bind", "vector");
    data.Set("net.branch.[shared].structure", "SharedLinear");
    data.Set("net.body.output.[features]", "shared");
    anet::nn::NetworkConfig skip_net;
    skip_net.output_keys["features"] = "features";
    const anet::nn::WeightInitConfig init;
    dqn::SkipNetworkDefinition skip{.config = {.enabled = true, .max_length = 1}, .network_config = skip_net,
        .head_factory = std::make_shared<dqn::SkipHeadFactory>(1, 1, dqn::SkipNetworkMode::ActionOut, "none", 1, init), .seed = 99};
    dqn::NetworkModel model({}, torch::kCPU, anet::nn::NetworkConfig(data, "net"), env.state_spec.obs_spec, 1,
        std::make_shared<dqn::LinearHeadFactory>(1, init), false, 98, &skip);
    auto config = WithDefaultLearnerForwardConfig();
    config.skip.feature_gradient = "split";
    config.replay_capacity = 16;
    config.replay_batch_size = 2;
    config.learning_rate.value = 0.01f;
    config.weight_decay.value = 0.2f;
    config.use_per = false;
    config.use_tbo = false;
    config.use_td_clip = false;
    config.use_amp = false;
    config.use_grad_clip = clipped;
    config.grad_clip_tau = 0.001f;
    config.use_fused_optimizer = fused;
    dqn::RuntimeVars vars;
    dqn::TDLearner learner(config, model, vars, nullptr, {.num_envs = 2, .num_threads = 1}, env, torch::kCPU, 98, kTestAgentSeed,
        std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), {});
    auto primary_parameters = model.GetOnlineParameters();
    auto skip_parameters = model.GetSkipOnlineNetwork()->parameters();
    {
        torch::NoGradGuard no_grad;
        for (auto& parameter : primary_parameters) parameter.fill_(0.1f);
        for (auto& parameter : skip_parameters) parameter.fill_(0.1f);
    }
    rl::ExperienceSamples samples;
    samples.obs = anet::TensorDict{{kVectorKey, torch::ones({2, 2})}};
    samples.actions = torch::zeros({2}, torch::kInt64);
    samples.target_returns = torch::full({2}, 3.0f);
    samples.n_steps = torch::ones({2}, torch::kInt64);
    samples.next_state.next_obs = samples.obs;
    samples.next_state.terminals = torch::ones({2}, torch::kBool);
    samples.window.lengths = samples.n_steps;
    samples.window.max_lengths = samples.n_steps;
    samples.window.returns = torch::full({2}, 4.0f);
    samples.window.next_obs = samples.obs;
    samples.window.terminals = samples.next_state.terminals;
    samples.window.truncated = torch::zeros({2}, torch::kBool);

    // 同じ初期値から二つのlossを微分し、初回Adamの式で期待する更新量を作る。
    const auto primary = model.GetOnlineNetwork()->Forward(samples.obs);
    const auto auxiliary = model.GetSkipOnlineNetwork()->Forward(dqn::MakeSkipInput(samples.obs, primary, samples.actions, false));
    const auto primary_loss = torch::nn::functional::smooth_l1_loss(primary.At("q").squeeze(1), samples.target_returns);
    const auto skip_loss = torch::nn::functional::smooth_l1_loss(auxiliary.At("skip_q").reshape({2}), samples.window.returns);
    const auto norm = [](const std::vector<torch::Tensor>& parameters) {
        float squared = 0;
        for (const auto& parameter : parameters) {
            if (parameter.grad().defined()) squared += parameter.grad().square().sum().item<float>();
        }
        return std::sqrt(squared);
    };
    const auto delta = [&](const torch::Tensor& gradient, float global_norm) {
        const float scale = clipped ? std::min(1.0f, config.grad_clip_tau / (global_norm + 1e-6f)) : 1.0f;
        const auto grad = gradient * scale;
        return config.learning_rate.value * grad / (grad.abs() + config.adam_eps);
    };
    primary_loss.backward({}, true);
    const float primary_norm = norm(primary_parameters);
    std::vector<torch::Tensor> expected;
    for (auto& parameter : primary_parameters) {
        const auto grad = parameter.grad();
        expected.push_back(parameter.detach().clone() * (1 - config.learning_rate.value * config.weight_decay.value)
            - delta(grad, primary_norm));
        parameter.mutable_grad() = torch::Tensor();
    }
    skip_loss.backward();
    const float skip_primary_norm = norm(primary_parameters);
    const float skip_network_norm = norm(skip_parameters);
    for (size_t i = 0; i < primary_parameters.size(); ++i) {
        const auto grad = primary_parameters[i].grad();
        if (grad.defined()) expected[i] -= delta(grad, skip_primary_norm);
    }
    for (auto& parameter : skip_parameters) {
        const auto grad = parameter.grad();
        expected.push_back(parameter.detach().clone() * (1 - config.learning_rate.value * config.weight_decay.value)
            - delta(grad, skip_network_norm));
    }

    // 公開の学習更新を1回通し、共有部は二つ、各Headは一つの更新になることを確認する。
    const auto result = learner.UpdateFromSamples(samples);
    REQUIRE(result->GetScalar("skip_primary_grad_norm").has_value());
    CHECK(*result->GetScalar("skip_primary_grad_norm") == Catch::Approx(skip_primary_norm));
    REQUIRE(result->GetScalar("skip_primary_clip_ratio").has_value());
    CHECK(*result->GetScalar("skip_primary_clip_ratio") == (clipped ? 1.0f : 0.0f));
    auto actual = primary_parameters;
    actual.insert(actual.end(), skip_parameters.begin(), skip_parameters.end());
    for (size_t i = 0; i < actual.size(); ++i) {
        INFO("parameter=" << i);
        CHECK(torch::allclose(actual[i], expected[i], 1e-5, 1e-6));
    }

    // 両方のmomentを復元すると、同じparameterからの次の更新が完全一致する。
    std::stringstream checkpoint(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(checkpoint);
    learner.Save(output);
    checkpoint.seekg(0);
    anet::InputArchive header(checkpoint);
    int64_t role_count = 0;
    header.Read(role_count);
    REQUIRE(role_count == 5);
    std::vector<torch::Tensor> saved_parameters;
    for (const auto& parameter : actual) saved_parameters.push_back(parameter.detach().clone());
    learner.UpdateFromSamples(samples);
    std::vector<torch::Tensor> continued;
    for (const auto& parameter : actual) continued.push_back(parameter.detach().clone());
    {
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < actual.size(); ++i) actual[i].copy_(saved_parameters[i]);
    }
    checkpoint.seekg(0);
    anet::InputArchive input(checkpoint);
    learner.Load(input);
    learner.UpdateFromSamples(samples);
    for (size_t i = 0; i < actual.size(); ++i) CHECK(torch::equal(actual[i], continued[i]));
}

TEST_CASE("Skip kmax weights multiply IS weights without changing TD", "[dqn][rfd0103][skip_kmax]")
{
    const bool per = GENERATE(false, true);
    const std::string weighting = GENERATE(std::string("uniform"), std::string("kmax"));
    anet::test::RegisterStandardNetworkModules();
    const auto env = MakeLearnerEnvSpec();
    anet::nn::NetworkConfig skip_net;
    skip_net.output_keys["features"] = "features";
    const anet::nn::WeightInitConfig init;
    dqn::SkipNetworkDefinition skip{.config = {.enabled = true, .max_length = 3}, .network_config = skip_net,
        .head_factory = std::make_shared<dqn::SkipHeadFactory>(1, 3, dqn::SkipNetworkMode::ActionOut, "none", 1, init), .seed = 99};
    dqn::NetworkModel model({}, torch::kCPU, MakeAgentForwardNetworkConfig(), env.state_spec.obs_spec, 1,
        std::make_shared<dqn::LinearHeadFactory>(1, init), false, 98, &skip);
    {
        torch::NoGradGuard no_grad;
        for (const auto& net : {model.GetOnlineNetwork(), model.GetSkipOnlineNetwork()}) {
            for (auto& parameter : net->parameters()) parameter.zero_();
        }
    }
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.learner.skip.sample_weight", weighting);
    auto config = WithDefaultLearnerForwardConfig();
    config.skip = dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(data)).learner.skip;
    config.replay_capacity = 18;
    config.replay_batch_size = 3;
    config.update_warmup_steps = 1000;
    config.learning_rate.value = 0;
    config.use_per = per;
    config.use_tbo = false;
    config.use_td_clip = false;
    config.use_amp = false;
    config.use_grad_clip = false;
    config.use_fused_optimizer = false;
    dqn::RuntimeVars vars;
    dqn::TDLearner learner(config, model, vars, nullptr, {.num_envs = 3, .num_threads = 1}, env, torch::kCPU, 98, kTestAgentSeed,
        std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), {});
    // PERの更新先を、warmup中の公開Push経路で実在する世代にする。
    for (rl::step_t step = 0; step < 6; ++step) {
        learner.UpdateFromBatch({.exp_step = step * 3}, MakeDeterminismExperience(step, 3));
    }
    rl::ExperienceSamples samples;
    samples.obs = anet::TensorDict{{kVectorKey, torch::zeros({3, 2})}};
    samples.actions = torch::zeros({3}, torch::kInt64);
    samples.target_returns = torch::tensor({1.0f, 2.0f, 3.0f});
    samples.n_steps = torch::ones({3}, torch::kInt64);
    samples.is_weights = torch::tensor({1.0f, 0.5f, 0.25f});
    samples.replay_item_keys = torch::arange(3, torch::kInt64) + config.replay_capacity;
    samples.next_state.next_obs = samples.obs;
    samples.next_state.terminals = torch::ones({3}, torch::kBool);
    samples.window.lengths = samples.n_steps;
    samples.window.max_lengths = torch::tensor({1, 2, 3}, torch::kInt64);
    samples.window.returns = samples.target_returns;
    samples.window.next_obs = samples.obs;
    samples.window.terminals = samples.next_state.terminals;
    samples.window.truncated = torch::zeros({3}, torch::kBool);
    const auto result = learner.UpdateFromSamples(samples);
    auto expected = torch::tensor({0.5f, 1.5f, 2.5f});
    if (per) expected *= samples.is_weights;
    if (weighting == "kmax") expected *= torch::tensor({0.5f, 1.0f, 1.5f});
    CHECK(*result->GetScalar("skip_loss") == Catch::Approx(expected.mean().item<float>()));
    CHECK(*result->GetScalar("skip_td_abs_mean") == 2.0f);
    CHECK(torch::equal(result->td_error, -samples.target_returns));
}

TEST_CASE("Skip target uses window return and duration and masks only terminal bootstrap", "[dqn][rfd0098]")
{
    anet::test::RegisterStandardNetworkModules();
    const bool hard_sync = GENERATE(false, true);
    const auto env = MakeLearnerEnvSpec();
    anet::nn::NetworkConfig skip_net;
    skip_net.output_keys["features"] = "features";
    const anet::nn::WeightInitConfig init;
    dqn::SkipNetworkDefinition skip{.config = {.enabled = true, .max_length = 3}, .network_config = skip_net,
        .head_factory = std::make_shared<dqn::SkipHeadFactory>(1, 3, anet::rl::dqn::SkipNetworkMode::ActionOut, "none", 1, init), .seed = 99};
    dqn::NetworkModel model({.soft_update_tau = {.value = 0.25f}, .hard_update_interval = hard_sync ? 2 : -1}, torch::kCPU, MakeAgentForwardNetworkConfig(), env.state_spec.obs_spec, 1,
        std::make_shared<dqn::LinearHeadFactory>(1, init), false, 98, &skip);
    {
        torch::NoGradGuard no_grad;
        for (const auto& net : {model.GetOnlineNetwork(), model.GetTargetNetwork(), model.GetSkipOnlineNetwork(), model.GetSkipTargetNetwork()}) {
            for (auto& parameter : net->parameters()) parameter.zero_();
        }
        model.GetTargetNetwork()->named_parameters()["head.linear.bias"].fill_(4.0f);
        model.GetSkipOnlineNetwork()->named_parameters()["head.linear.bias"].fill_(1.0f);
    }
    auto config = WithDefaultLearnerForwardConfig();
    config.replay_capacity = 16;
    config.replay_batch_size = 2;
    config.learning_rate.value = 0;
    config.gamma.value = 0.5f;
    config.use_fused_optimizer = false;
    config.use_per = false;
    config.use_tbo = false;
    config.use_td_clip = false;
    config.use_amp = false;
    config.use_grad_clip = true;
    config.grad_clip_tau = 0.1f;
    dqn::RuntimeVars vars;
    dqn::TDLearner learner(config, model, vars, nullptr, {.num_envs = 2, .num_threads = 1}, env, torch::kCPU, 98, kTestAgentSeed,
        std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), {});
    rl::ExperienceSamples samples;
    samples.obs = anet::TensorDict{{kVectorKey, torch::zeros({2, 2})}};
    samples.actions = torch::zeros({2}, torch::kInt64);
    samples.target_returns = torch::tensor({2.0f, 2.0f});
    samples.n_steps = torch::full({2}, 2, torch::kInt64);
    samples.next_state.next_obs = samples.obs;
    samples.next_state.terminals = torch::tensor({false, true});
    samples.window.lengths = torch::tensor({3, 1}, torch::kInt64);
    samples.window.max_lengths = torch::full({2}, 3, torch::kInt64);
    samples.window.returns = torch::tensor({3.0f, 5.0f});
    samples.window.next_obs = samples.obs;
    samples.window.terminals = samples.next_state.terminals;
    samples.window.truncated = torch::tensor({true, false});
    const auto result = learner.UpdateFromSamples(samples);
    CHECK(torch::equal(result->td_error, torch::tensor({-3.0f, -2.0f})));
    CHECK(*result->GetScalar("skip_td_abs_mean") == 3.25f);
    CHECK(*result->GetScalar("skip_loss") == 2.75f);
    // 別々にclipされるため、primaryとskipのいずれも閾値へ到達する。
    // 一括clipなら両方が閾値未満になり、この観測で検出できる。
    for (const auto& network : {model.GetOnlineNetwork(), model.GetSkipOnlineNetwork()}) {
        float squared_norm = 0;
        for (const auto& parameter : network->parameters()) {
            if (parameter.grad().defined()) squared_norm += parameter.grad().square().sum().item<float>();
        }
        CHECK(std::sqrt(squared_norm) == Catch::Approx(0.1f).margin(1e-5f));
    }
    // targetを使うActorの2 Networkは、hard/softとも同じ更新境界で揃う。
    model.UpdateTarget(1, 0.25f);
    CHECK(model.GetTargetNetwork()->named_parameters()["head.linear.bias"].item<float>() == Catch::Approx(hard_sync ? 4.0f : 3.0f));
    CHECK(model.GetSkipTargetNetwork()->named_parameters()["head.linear.bias"][0].item<float>() == Catch::Approx(hard_sync ? 0.0f : 0.25f));
    model.UpdateTarget(2, 0.25f);
    CHECK(model.GetTargetNetwork()->named_parameters()["head.linear.bias"].item<float>() == Catch::Approx(hard_sync ? 0.0f : 2.25f));
    CHECK(model.GetSkipTargetNetwork()->named_parameters()["head.linear.bias"][0].item<float>() == Catch::Approx(hard_sync ? 1.0f : 0.4375f));
}

TEST_CASE("Skip Munchausen targets match numeric returns for TD QR IQN and TBO", "[dqn][rfd0098][skip_numeric]")
{
    const auto mode = GENERATE(std::string("none"), std::string("qr"), std::string("iqn"));
    const auto bonus_mode = GENERATE(std::string("target"), std::string("online"), std::string("online_reuse"));
    const bool tbo = GENERATE(false, true);
    anet::test::RegisterStandardNetworkModules();
    const auto env = MakeIqnTracerEnvSpec();
    const auto network_config = mode == "iqn"
        ? anet::nn::NetworkConfig(MakeIqnTracerConfigData(), "DefaultDQNAgent.net") : MakeAgentForwardNetworkConfig();
    const anet::nn::WeightInitConfig init;
    std::shared_ptr<anet::nn::NetworkHeadFactory> head;
    if (mode == "none") head = std::make_shared<dqn::LinearHeadFactory>(2, init);
    else if (mode == "qr") head = std::make_shared<dqn::QuantileHeadFactory>(2, 4, init);
    else head = std::make_shared<dqn::IQNHeadFactory>(2, init);
    auto skip_network_config = network_config;
    dqn::SkipNetworkDefinition skip{.config = {.enabled = true, .max_length = 3}, .network_config = skip_network_config,
        .head_factory = std::make_shared<dqn::SkipHeadFactory>(2, 3, anet::rl::dqn::SkipNetworkMode::ActionOut, mode, 4, init), .seed = 99};
    auto input_spec = env.state_spec.obs_spec;
    if (mode == "iqn") input_spec[anet::rl::dqn::kKey_Taus] = anet::TensorSpec{.shape = {64}, .dtype = torch::kFloat32};
    dqn::NetworkModel model({}, torch::kCPU, network_config, input_spec, 2, head, mode != "none", 98, &skip);
    const auto transform = [tbo](float value) { return tbo ? std::sqrt(value + 1.0f) - 1.0f + 0.01f * value : value; };
    {
        torch::NoGradGuard no_grad;
        for (const auto& [network, value] : std::vector<std::pair<std::shared_ptr<anet::nn::Network>, float>>{
            {model.GetOnlineNetwork(), 2}, {model.GetTargetNetwork(), 4}, {model.GetSkipOnlineNetwork(), 1}}) {
            for (const auto& parameter : network->named_parameters()) {
                parameter.value().zero_();
                if (parameter.key().starts_with("head.") && parameter.key().ends_with("bias")) parameter.value().fill_(transform(value));
            }
        }
    }
    auto config = WithDefaultLearnerForwardConfig();
    config.quantile_mode = mode;
    config.num_quantiles = 4;
    config.replay_capacity = 16;
    config.replay_batch_size = 2;
    config.learning_rate.value = 0;
    config.gamma.value = 0.5f;
    config.use_fused_optimizer = false;
    config.use_per = true;
    config.use_double_dqn = false;
    config.use_tbo = tbo;
    config.tbo_epsilon = 0.01f;
    config.use_td_clip = false;
    config.use_amp = false;
    config.munchausen.enabled = true;
    config.munchausen.log_policy_mode = bonus_mode;
    config.munchausen.alpha.value = 0.9f;
    config.munchausen.entropy_tau.value = 0.03f;
    config.munchausen.clip_value_min = -1;
    config.skip.priority_lambda = 0.4f;
    config.update_warmup_steps = 1000;
    dqn::RuntimeVars vars;
    dqn::ActionPolicyConfig policy_config;
    policy_config.quantile_mode = mode;
    const auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(policy_config);
    std::unique_ptr<dqn::Learner> learner;
    const rl::BatchEnvSpec batch_spec{.num_envs = 2, .num_threads = 1};
    if (mode == "none") learner = std::make_unique<dqn::TDLearner>(config, model, vars, nullptr, batch_spec, env, torch::kCPU, 98, kTestAgentSeed, policy, dqn::LearnerForwardResources{});
    else if (mode == "qr") learner = std::make_unique<dqn::QRLearner>(config, model, vars, nullptr, batch_spec, env, torch::kCPU, 98, kTestAgentSeed, policy, dqn::LearnerForwardResources{});
    else learner = std::make_unique<dqn::IQNLearner>(config, model, vars, nullptr, batch_spec, env, torch::kCPU, 98, kTestAgentSeed, policy, dqn::LearnerForwardResources{});
    // PERの更新先は実在する世代のslotを使う。warmup中の公開Push経路で用意する。
    const auto flags = torch::zeros({2}, torch::kBool);
    for (int index = 0; index < 6; ++index) {
        const rl::BatchState state(anet::TensorDict{{kVectorKey, torch::zeros({2, 4})}}, flags, flags,
            torch::full({2}, index == 0, torch::kBool));
        const rl::BatchState next(anet::TensorDict{{kVectorKey, torch::zeros({2, 4})}}, flags, flags, flags);
        learner->UpdateFromBatch({.exp_step = static_cast<rl::step_t>(index * 2)}, rl::BatchExperience(state,
            std::make_shared<rl::BatchActionInfo>(torch::zeros({2}, torch::kLong)), torch::ones({2}), next));
    }
    rl::ExperienceSamples samples;
    samples.obs = anet::TensorDict{{kVectorKey, torch::zeros({2, 4})}};
    samples.actions = torch::zeros({2}, torch::kLong);
    samples.target_returns = torch::full({2}, 2.0f);
    samples.n_steps = torch::full({2}, 2, torch::kLong);
    samples.is_weights = torch::tensor({0.25f, 1.0f});
    samples.replay_item_keys = torch::arange(2, torch::kLong) + 16;
    samples.next_state.next_obs = samples.obs;
    samples.next_state.terminals = torch::tensor({false, true});
    samples.window.lengths = torch::tensor({3, 1}, torch::kLong);
    samples.window.max_lengths = torch::full({2}, 3, torch::kLong);
    samples.window.returns = torch::tensor({3.0f, 5.0f});
    samples.window.next_obs = samples.obs;
    samples.window.terminals = samples.next_state.terminals;
    samples.window.truncated = torch::tensor({true, false});
    const auto result = mode == "none" ? static_cast<dqn::TDLearner&>(*learner).UpdateFromSamples(samples)
        : mode == "qr" ? static_cast<dqn::QRLearner&>(*learner).UpdateFromSamples(samples)
        : static_cast<dqn::IQNLearner&>(*learner).UpdateFromSamples(samples);
    // 同じ2 actionのsoft value=4+tau*log2、先頭bonus=-alpha*tau*log2を一度だけ足す。
    const float entropy = 0.03f * std::log(2.0f);
    const float bonus = -0.9f * entropy;
    const float first = transform(3.0f + bonus + 0.125f * (4.0f + entropy));
    const float terminal = transform(5.0f + bonus);
    const float current = transform(1.0f);
    CHECK(*result->GetScalar("skip_td_abs_mean") == Catch::Approx((first + terminal) * 0.5f - current).margin(2e-4f));
    const auto expected_priority = torch::tensor({
        std::abs(transform(2.0f) - transform(2.0f + bonus + 0.25f * (4.0f + entropy))) + 0.4f * (first - current),
        std::abs(transform(2.0f) - transform(2.0f + bonus)) + 0.4f * (terminal - current)});
    CHECK(torch::allclose(result->per_priorities, expected_priority + config.per_eps, 1e-4, 2e-4));
    if (mode == "none") {
        const auto huber = [](float error) { return error < 1 ? 0.5f * error * error : error - 0.5f; };
        CHECK(*result->GetScalar("skip_loss") == Catch::Approx((0.25f * huber(first - current) + huber(terminal - current)) * 0.5f).margin(2e-4f));
    }
}

TEST_CASE("Primary checkpoint group mismatch reports saved and expected counts when enabling skip", "[dqn][rfd0098][serialize][rfd0098-review-checkpoint]")
{
    anet::test::RegisterStandardNetworkModules();
    const bool saved_noisy = GENERATE(false, true);
    const auto env = MakeLearnerEnvSpec();
    const anet::nn::WeightInitConfig init;
    anet::nn::NetworkConfig skip_net;
    skip_net.output_keys["features"] = "features";
    dqn::SkipNetworkDefinition skip{.config = {.enabled = true, .max_length = 1}, .network_config = skip_net,
        .head_factory = std::make_shared<dqn::SkipHeadFactory>(1, 1, anet::rl::dqn::SkipNetworkMode::ActionOut, "none", 1, init), .seed = 99};
    const auto make_model = [&](bool noisy, bool enabled) {
        anet::ConfigData projection;
        projection.Set("noisy.enabled", noisy);
        return std::make_unique<dqn::NetworkModel>(dqn::NetworkModelConfig{}, torch::kCPU,
            MakeAgentForwardNetworkConfig(), env.state_spec.obs_spec, 1,
            std::make_shared<dqn::LinearHeadFactory>(1, init, projection), false, 98, enabled ? &skip : nullptr);
    };
    auto source = make_model(saved_noisy, false);
    auto destination = make_model(!saved_noisy, true);
    auto config = WithDefaultLearnerForwardConfig();
    config.enabled = false;
    config.use_fused_optimizer = false;
    dqn::RuntimeVars source_vars, destination_vars;
    TestLearner first(config, *source, source_vars, {.num_envs = 1, .num_threads = 1}, env, {});
    TestLearner second(config, *destination, destination_vars, {.num_envs = 1, .num_threads = 1}, env, {});
    first.SetupAdamW();
    second.SetupAdamW();
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(stream);
    first.Save(output);
    stream.seekg(0);
    anet::InputArchive input(stream);
    // architectureが異なるcheckpointを受け入れず、optimizer境界で両方の数を示す。
    CHECK_THROWS_WITH(second.LoadCheckpoint(input, false),
        Catch::Matchers::ContainsSubstring("checkpoint_groups=" + std::to_string(saved_noisy ? 2 : 1))
        && Catch::Matchers::ContainsSubstring("expected_primary_groups=" + std::to_string(saved_noisy ? 1 : 2)));
}

TEST_CASE("RFD0061 captures fixed checkpoint inference", "[.][rfd0061-inference]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    torch::NoGradGuard no_grad;
    wxString manifest_name;
    REQUIRE(wxGetEnv("ANET_RFD0061_INFERENCE", &manifest_name));
    std::ifstream manifest_file(std::filesystem::path(manifest_name.ToStdWstring()));
    REQUIRE(manifest_file.good());
    const auto manifest = anet::json::parse(manifest_file);
    for (const auto& item : manifest) {
        INFO(item.at("id"));
        const auto root = std::filesystem::path(item.at("run").get<std::string>());
        const auto output = std::filesystem::path(item.at("output").get<std::string>());
        REQUIRE_FALSE(std::filesystem::exists(output));
        std::filesystem::create_directories(output);
        std::ifstream config_file(root / "json/config_data.json");
        std::ifstream spec_file(root / "json/env-env_spec.json");
        REQUIRE(config_file.good());
        REQUIRE(spec_file.good());
        auto config_json = anet::json::parse(config_file).at("data");
        if (item.contains("config_capture")) {
            std::ifstream capture(item.at("config_capture").get<std::string>());
            REQUIRE(capture.good());
            config_json = anet::json::parse(capture).at("values");
        }
        const auto spec_json = anet::json::parse(spec_file).at("data");
        anet::ConfigData config;
        for (const auto& [key, value] : config_json.items()) config.Set(key, value.get<std::string>());
        // 学習資源の容量だけを抑え、保存済みnetworkと決定的な方策で同じ入力を測る。
        const auto checkpoint = item.value("checkpoint", (root / "agent_close.anet").generic_string());
        const bool create_checkpoint = item.value("create_checkpoint", false);
        config.Set("DefaultDQNAgent.auto_load_file", create_checkpoint ? std::string() : checkpoint);
        config.Set("DefaultDQNAgent.learner.enabled", false);
        config.Set("DefaultDQNAgent.learner.replay_capacity", 32);
        config.Set("DefaultDQNAgent.learner.replay_batch_size", 2);
        config.Set("DefaultDQNAgent.learner.use_rb_prefetch", false);
        for (const auto key : {"eval", "eval_target"}) {
            const auto prefix = std::string("DefaultDQNAgent.actor.[") + key + "]";
            config.Set(prefix + ".policy.policy_type", "Greedy");
            config.Set(prefix + ".policy.tau_rule.sample_mode", "fixed");
            config.Set(prefix + ".policy.full_distribution_query.enabled", false);
        }
        rl::EnvSpec env_spec;
        env_spec.action_spec.is_discrete = true;
        env_spec.action_spec.value_labels = spec_json.at("action_spec").at("value_labels").get<std::vector<std::string>>();
        anet::TensorDict observations;
        for (const auto& [key, value] : spec_json.at("state_spec").at("obs_spec").items()) {
            anet::TensorSpec spec;
            spec.type = value.at("type") == "Grid" ? anet::SpaceType::Grid : anet::SpaceType::Vector;
            spec.shape = value.at("shape").get<std::vector<int64_t>>();
            spec.num_classes = value.at("num_classes").get<int64_t>();
            const auto dtype = value.at("dtype").get<std::string>();
            REQUIRE((dtype == "Float" || dtype == "Byte" || dtype == "Char" || dtype == "Long"));
            spec.dtype = dtype == "Byte" ? torch::kUInt8 : dtype == "Char" ? torch::kInt8
                : dtype == "Long" ? torch::kInt64 : torch::kFloat32;
            env_spec.state_spec.obs_spec[key] = spec;
            auto shape = spec.shape;
            shape.insert(shape.begin(), 2);
            auto tensor = torch::zeros(shape, torch::TensorOptions().dtype(spec.dtype));
            observations.Set(key, tensor);
            torch::save(tensor, (output / ("obs-" + key + ".pt")).string());
        }
        dqn::DefaultDQNAgentFactory factory;
        const rl::BatchEnvSpec batch_spec{ 2, 1 };
        // 保存Tensorのdeviceを維持する既存Load契約に合わせてCUDA上で比較する。
        auto agent = factory.CreateAgent(env_spec, batch_spec, torch::kCUDA, WithDefaultForwardConfig(config), nullptr, 123);
        if (create_checkpoint) {
            REQUIRE_FALSE(std::filesystem::exists(checkpoint));
            std::ofstream checkpoint_file(checkpoint, std::ios::binary);
            REQUIRE(checkpoint_file.good());
            anet::OutputArchive archive(checkpoint_file);
            agent->Save(archive);
        }
        const auto flags = torch::zeros({2}, torch::TensorOptions().dtype(torch::kBool));
        const rl::BatchState state(observations, flags, flags, flags);
        for (const auto& [key, name] : std::map<std::string, std::string>{{"eval_target", "target"}, {"eval", "online"}}) {
            const auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch_spec, .env_spec = env_spec, .device = torch::Device(torch::kCUDA), .seed = 123, .actor_key = key});
            const auto result = std::dynamic_pointer_cast<dqn::DQNActionInfo>(actor->MakeAction(rl::StepCounts{}, state));
            REQUIRE(result != nullptr);
            torch::save(result->GetAction(), (output / (std::string(name) + "-action.pt")).string());
            torch::save(result->GetAuxData().at("q_values"), (output / (std::string(name) + "-q.pt")).string());
            if (item.contains("baseline")) {
                const auto baseline = std::filesystem::path(item.at("baseline").get<std::string>());
                torch::Tensor expected_action, expected_q;
                torch::load(expected_action, (baseline / (std::string(name) + "-action.pt")).string());
                torch::load(expected_q, (baseline / (std::string(name) + "-q.pt")).string());
                CHECK(torch::equal(result->GetAction(), expected_action));
                CHECK(torch::allclose(result->GetAuxData().at("q_values"), expected_q, 1e-6, 1e-6));
            }
        }
    }
}

TEST_CASE("DefaultDQN selects six Head shapes and three Learner kinds", "[rfd0128][dqn][head]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const std::string mode = GENERATE("none", "qr", "iqn");
    const bool dueling = GENERATE(false, true);
    auto data = mode == "iqn" ? MakeIqnTracerConfigData() : anet::ConfigData{};
    if (mode != "iqn") {
        data.Set("DefaultDQNAgent.net.block.[Flatten].type", "Flatten");
        data.Set("DefaultDQNAgent.net.branch.[main].bind", kVectorKey);
        data.Set("DefaultDQNAgent.net.branch.[main].structure", "Flatten");
        data.Set("DefaultDQNAgent.net.body.output.[features]", "main");
    }
    data.Set("DefaultDQNAgent.quantile_mode", mode);
    data.Set("DefaultDQNAgent.head.use_dueling", dueling);
    data.Set("DefaultDQNAgent.learner.replay_capacity", 16);
    data.Set("DefaultDQNAgent.learner.replay_batch_size", 2);

    anet::test::LogCaptureGuard logs;
    dqn::DefaultDQNAgentFactory factory;
    const auto agent = factory.CreateAgent(MakeIqnTracerEnvSpec(), rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1},
        torch::kCPU, WithDefaultForwardConfig(data), nullptr, 128);
    REQUIRE(agent != nullptr);
    const auto forward = agent->GetTensorDictFunction(mode == "none" ? kPolicyForwardKey : kPolicyForwardDistributionKey);
    REQUIRE(forward.has_value());
    anet::TensorDict input{{kVectorKey, torch::ones({2, 4})}};
    if (mode == "iqn") input.Set(anet::rl::dqn::kKey_Taus, torch::full({2, 3}, 0.5f));
    const auto output = (*forward)(input);
    const auto values = output.At(mode == "none" ? kQValueKey : kQDistributionKey);
    CHECK(values.size(0) == 2);
    CHECK(values.size(1) == 2);
    if (mode != "none") CHECK(values.size(2) == (mode == "iqn" ? 3 : 51));
    logs.Flush();
    const std::string head = mode == "none" ? (dueling ? "Dueling" : "Plain Linear")
        : mode == "qr" ? (dueling ? "Quantile Dueling" : "Quantile Plain")
        : (dueling ? "IQN Dueling" : "IQN Plain");
    const std::string learner = mode == "none" ? "TDLearner" : mode == "qr" ? "QRLearner" : "IQNLearner";
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message, {"Network Head:", head}));
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message, {"Initialized", learner}));
}

TEST_CASE("QR selected crossing depth aggregates BF16 quantiles in float32", "[dqn][qr][action_policy][metrics][bf16]")
{
    const auto q_quantiles = torch::tensor({ { { 0.0f, 2.0f, 1.0f, 4.0f, 2.0f } } })
        .to(torch::kBFloat16);
    const auto obs = anet::TensorDict{
        { "q", q_quantiles.mean(2) },
        { "q_dist", q_quantiles },
    };

    dqn::ActionPolicyConfig config;
    config.quantile_mode = "qr";
    dqn::EpsilonGreedyActionPolicy policy(config);
    const anet::nn::NetworkForwardCallInfo call_info_4({});
    auto action_info = policy.SelectAction(
        obs, /*greedy_only=*/true, *MakeDefaultExecutor(MakePassthroughNetwork(1, 5)),
        std::make_shared<anet::RandomGenerator>(123), call_info_4, {});

    CHECK(*action_info->GetScalar("policy_selected_crossing_depth_p90_ratio")
        == Catch::Approx(0.5f));
}

TEST_CASE("Decomposed combination preserves tau pairs and exact zero beta values", "[rfd0121][dqn][combine]")
{
    const bool tbo = GENERATE(false, true);
    const float beta = GENERATE(0.0f, 0.3f);
    constexpr double epsilon = 0.01;
    const auto external = torch::tensor({{{0.1f, 0.5f, 1.0f}, {-0.25f, -1.5f, 2.0f}}});
    const auto internal = torch::tensor({{{-0.5f, 0.25f, 1.2f}, {0.2f, 1.0f, -0.8f}}});
    anet::TensorDict output{{kRfd0121ExtrinsicQDist, external}, {kRfd0121IntrinsicQDist, internal}};
    CHECK_FALSE(output.Contains(kRfd0121Q));
    CHECK_FALSE(output.Contains(kRfd0121QDist));
    dqn::DecomposedValueCombiner::Combine(output, beta, tbo, epsilon);
    if (beta == 0) CHECK(torch::equal(output.At(kRfd0121QDist), external));
    const auto h = [](double value) { return std::copysign(std::sqrt(std::abs(value) + 1) - 1, value) + epsilon * value; };
    const auto inverse = [](double value) {
        const double root = (std::sqrt(1 + 4 * epsilon * (std::abs(value) + 1 + epsilon)) - 1) / (2 * epsilon);
        return std::copysign(root * root - 1, value);
    };
    for (int action = 0; action < 2; ++action) {
        for (int tau = 0; tau < 3; ++tau) {
            const double ext = external[0][action][tau].item<float>(), intr = internal[0][action][tau].item<float>();
            const double expected = tbo ? h(inverse(ext) + beta * inverse(intr)) : ext + beta * intr;
            CHECK(output.At(kRfd0121QDist)[0][action][tau].item<float>() == Catch::Approx(expected).epsilon(1e-5).margin(1e-6));
        }
    }
    CHECK(torch::equal(output.At(kRfd0121Q), output.At(kRfd0121QDist).mean(2)));
}

TEST_CASE("Decomposed Heads validate shared and dedicated feature shapes", "[rfd0121][dqn][decomposed_head]")
{
    const auto mode = GENERATE(std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone), std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeQr), std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeIqn));
    const bool dueling = GENERATE(false, true);
    const bool dedicated = GENERATE(false, true);
    const bool iqn = mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeIqn;
    anet::TensorDict features;
    const auto external = torch::ones(iqn ? std::vector<int64_t>{2, 3, 4} : std::vector<int64_t>{2, 4});
    const auto internal = torch::ones(iqn ? std::vector<int64_t>{2, 3, 5} : std::vector<int64_t>{2, 5});
    features.Set(kRfd0121Features, external);
    features.Set(kRfd0121ValueFeature, external);
    features.Set(kRfd0121AdvFeature, external);
    features.Set(kRfd0121IntrinsicFeatures, internal);
    features.Set(kRfd0121IntrinsicValueFeature, internal);
    features.Set(kRfd0121IntrinsicAdvFeature, internal);
    anet::ConfigData projection;
    projection.Set("noisy.enabled", true);
    const anet::nn::WeightInitConfig init;
    dqn::DecomposedHeadFactory factory(
        dqn::DqnComponentAssembler::CreateHeadFactory(mode, dueling, 2, 3, init, {}),
        dqn::DqnComponentAssembler::CreateHeadFactory(mode, dueling, 2, 3, init, projection), dedicated, dueling,
        anet::ConfigFullKeyTable::FormatKeyNotReadFromConfig("input_mode"));
    const auto forward = [&](const anet::TensorDict& input) {
        const auto head = factory.CreateHead(input);
        REQUIRE_FALSE(head->GetNoisyParameterEntries().empty());
        return head->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    };
    const auto output = forward(features);
    CHECK_FALSE(output.Contains(kRfd0121Q));
    CHECK_FALSE(output.Contains(kRfd0121QDist));
    CHECK(output.At(kRfd0121ExtrinsicQ).sizes() == torch::IntArrayRef({2, 2}));
    CHECK(output.At(kRfd0121IntrinsicQ).sizes() == torch::IntArrayRef({2, 2}));
    if (!dedicated) return;
    anet::TensorDict missing;
    for (const auto& [key, value] : features) if (!key.starts_with(kRfd0121IntrinsicPrefix)) missing.Set(key, value);
    CHECK_THROWS_WITH(forward(missing), Catch::Matchers::ContainsSubstring(
        std::string("net.body.output.[") + kRfd0121IntrinsicFeatures + "]"));
    if (!dueling) {
        anet::TensorDict missing_default;
        for (const auto& [key, value] : features) if (key != kRfd0121IntrinsicFeatures) missing_default.Set(key, value);
        CHECK_THROWS_WITH(forward(missing_default), Catch::Matchers::ContainsSubstring(
            std::string("net.body.output.[") + kRfd0121IntrinsicFeatures + "]"));
    }
    if (dueling) {
        for (const auto* missing_key : {kRfd0121IntrinsicValueFeature, kRfd0121IntrinsicAdvFeature}) {
            anet::TensorDict partial;
            for (const auto& [key, value] : features) if (key != missing_key) partial.Set(key, value);
            CHECK_THROWS_WITH(forward(partial), Catch::Matchers::ContainsSubstring(
                std::string("net.body.output.[") + missing_key + "]"));
        }
    }
    auto wrong_rank = features;
    const auto wrong = torch::ones(iqn ? std::vector<int64_t>{2, 5} : std::vector<int64_t>{2, 3, 5});
    for (const auto* key : {kRfd0121IntrinsicFeatures, kRfd0121IntrinsicValueFeature, kRfd0121IntrinsicAdvFeature}) wrong_rank.Set(key, wrong);
    CHECK_THROWS(forward(wrong_rank));
    if (iqn) {
        auto wrong_taus = features;
        for (const auto* key : {kRfd0121IntrinsicFeatures, kRfd0121IntrinsicValueFeature, kRfd0121IntrinsicAdvFeature})
            wrong_taus.Set(key, torch::ones({2, 4, 5}));
        CHECK_THROWS(forward(wrong_taus));
    }
}

TEST_CASE("Decomposed learners match component Bellman targets and losses", "[rfd0121][dqn][decomposed_numeric]")
{
    ScopedNoopMetricsLogger logger;
    anet::test::RegisterStandardNetworkModules();
    const auto mode = GENERATE(std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone), std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeQr), std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeIqn));
    const bool tbo = GENERATE(false, true);
    const auto munch = GENERATE(std::string(kRfd0121Off), std::string(kRfd0121TargetLogPolicyMode), std::string(kRfd0121OnlineLogPolicyMode), std::string(kRfd0121OnlineReuse));
    const bool separate = GENERATE(false, true);
    const bool continuing = GENERATE(false, true);
    const bool external_policy = GENERATE(false, true);
    const bool double_dqn = GENERATE(false, true);
    const float weight = GENERATE(0.2f, 0.5f, 1.0f);
    CAPTURE(mode, tbo, munch, separate, continuing, external_policy, double_dqn, weight);
    const auto env = MakeIqnTracerEnvSpec();
    auto specs = env.state_spec.obs_spec;
    anet::nn::NetworkConfig net;
    net.output_keys[kRfd0121Features] = kVectorKey;
    if (mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeIqn) {
        net = anet::nn::NetworkConfig(MakeIqnTracerConfigData(), "DefaultDQNAgent.net");
        specs[anet::rl::dqn::kKey_Taus] = {.shape = {4}, .dtype = torch::kFloat32};
    }
    const anet::nn::WeightInitConfig init;
    auto head = std::make_shared<dqn::DecomposedHeadFactory>(
        dqn::DqnComponentAssembler::CreateHeadFactory(mode, false, 2, 3, init, {}),
        dqn::DqnComponentAssembler::CreateHeadFactory(mode, false, 2, 3, init, {}), false, false,
        anet::ConfigFullKeyTable::FormatKeyNotReadFromConfig("input_mode"));
    dqn::NetworkModel model({}, torch::kCPU, net, specs, 2, head, mode != anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone, 121);
    const auto h = [tbo](double value) {
        return tbo ? std::copysign(std::sqrt(std::abs(value) + 1) - 1, value) + 0.01 * value : value;
    };
    // Headの重みを0にして各行動のbiasを既知の値に固定し、独立したdoubleの式で参照値を作る。
    const std::array<double, 2> online_external{1, 3}, online_internal{8, 1}, target_external{4, 6}, target_internal{1, 2};
    const auto set_values = [&](const std::shared_ptr<anet::nn::Network>& network,
        const auto& external, const auto& internal) {
        torch::NoGradGuard guard;
        for (auto& entry : network->named_parameters()) {
            entry.value().zero_();
            const bool ext = entry.key() == kRfd0121HeadExtrinsicLinearBias;
            const bool intr = entry.key() == kRfd0121HeadIntrinsicLinearBias;
            if (!ext && !intr) continue;
            const auto& values = ext ? external : internal;
            const int64_t width = entry.value().numel() / 2;
            for (int action = 0; action < 2; ++action) entry.value().narrow(0, action * width, width).fill_(h(values[action]));
        }
    };
    set_values(model.GetOnlineNetwork(), online_external, online_internal);
    set_values(model.GetTargetNetwork(), target_external, target_internal);
    auto config = WithDefaultLearnerForwardConfig();
    config.quantile_mode = mode;
    config.num_quantiles = 3;
    config.iqn.current_taus = {.sample_mode = kRfd0121FixedTauMode, .num_taus = 4};
    config.iqn.target_taus = {.sample_mode = kRfd0121FixedTauMode, .num_taus = 3};
    config.intrinsic.decomposed = true;
    config.intrinsic.loss_weight = weight;
    config.intrinsic.priority_lambda = 0.3f;
    config.intrinsic.discount_mode = separate ? anet::rl::dqn::LearnerConfig::kDiscountSeparate : anet::rl::dqn::LearnerConfig::kDiscountShared;
    config.intrinsic.gamma.value = 0.25f;
    config.intrinsic.terminal_mode = continuing ? anet::rl::dqn::LearnerConfig::kTerminalNonEpisodic : anet::rl::dqn::LearnerConfig::kTerminalEpisodic;
    config.intrinsic.extrinsic_bootstrap_mode = external_policy ? anet::rl::dqn::LearnerConfig::kBootstrapExtrinsic : anet::rl::dqn::LearnerConfig::kBootstrapCombined;
    config.gamma.value = 0.5f;
    config.use_tbo = tbo;
    config.tbo_epsilon = 0.01f;
    config.use_double_dqn = double_dqn;
    config.use_per = false;
    config.use_td_clip = false;
    config.use_amp = false;
    config.learning_rate.value = 0;
    config.replay_batch_size = 4;
    config.replay_capacity = 64;
    config.use_n_step = false;
    config.update_warmup_steps = 100000;
    config.munchausen.enabled = munch != kRfd0121Off;
    config.munchausen.log_policy_mode = munch == kRfd0121Off ? kRfd0121TargetLogPolicyMode : munch;
    config.munchausen.alpha.value = 0.9f;
    config.munchausen.entropy_tau.value = 0.7f;
    dqn::ActionPolicyConfig policy_config;
    policy_config.policy_type = anet::rl::dqn::kActionPolicyTypeStr_Greedy;
    policy_config.quantile_mode = mode;
    auto policy = dqn::DqnComponentAssembler::CreateActionPolicy(policy_config, mode, false, 4, torch::kCPU);
    dqn::RuntimeVars vars;
    std::shared_ptr<dqn::Learner> learner;
    const auto create = [&]<typename T>() {
        return std::make_shared<T>(config, model, vars, nullptr, rl::BatchEnvSpec{4, 1}, env, torch::kCPU, 121, kTestAgentSeed, policy, dqn::LearnerForwardResources{});
    };
    if (mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone) learner = create.operator()<dqn::TDLearner>();
    else if (mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeQr) learner = create.operator()<dqn::QRLearner>();
    else learner = create.operator()<dqn::IQNLearner>();
    rl::ExperienceSamples samples;
    samples.obs = anet::TensorDict{{kVectorKey, torch::ones({4, 4})}};
    samples.next_state.next_obs = samples.obs;
    samples.actions = torch::tensor({0, 1, 0, 1}, torch::kInt64);
    samples.target_returns = torch::tensor({{0.2f, 1.0f}, {0.4f, 2.0f}, {0.6f, 3.0f}, {0.8f, 4.0f}});
    samples.n_steps = torch::tensor({1, 2, 3, 4}, torch::kInt64);
    samples.next_state.terminals = torch::tensor({false, true, false, true});
    samples.next_state.truncated = torch::tensor({false, false, true, true});
    const auto flags = torch::zeros({4}, torch::kBool);
    learner->UpdateFromBatch({}, {rl::BatchState(samples.obs, flags, flags, ~flags),
        std::make_shared<rl::BatchActionInfo>(samples.actions), torch::zeros({4, 2}), rl::BatchState(samples.obs, flags, flags, flags)}, 0.5f);
    const auto result = learner->UpdateFromSamples(samples);
    // Headの表示用Qと区別し、割合はTBOを戻した報酬の単位で求める。
    constexpr char kIntrinsicQShareMetric[] = "intrinsic_q_share";
    const auto share = learner->GetScalar(kIntrinsicQShareMetric);
    REQUIRE(share);
    CHECK(*share == Catch::Approx(0.5 * (4.0 / 5.0 + 0.5 / 3.5)).epsilon(1e-5).margin(1e-6));
    const auto log_policy = [](const std::array<double, 2>& q) {
        const double maximum = std::max(q[0], q[1]);
        const double log_z = std::log(std::exp((q[0] - maximum) / 0.7) + std::exp((q[1] - maximum) / 0.7));
        return std::array<double, 2>{q[0] - maximum - 0.7 * log_z, q[1] - maximum - 0.7 * log_z};
    };
    const auto combine = [](const auto& ext, const auto& intr) { return std::array<double, 2>{ext[0] + 0.5 * intr[0], ext[1] + 0.5 * intr[1]}; };
    const auto combined_next_log = log_policy(combine(target_external, target_internal));
    const auto external_next_log = log_policy(external_policy ? target_external : combine(target_external, target_internal));
    const auto& current_ext = munch == kRfd0121TargetLogPolicyMode ? target_external : online_external;
    const auto& current_int = munch == kRfd0121TargetLogPolicyMode ? target_internal : online_internal;
    const auto current_log = log_policy(external_policy ? current_ext : combine(current_ext, current_int));
    const auto& selection_ext = double_dqn ? online_external : target_external;
    const auto& selection_int = double_dqn ? online_internal : target_internal;
    const auto combined_selection = combine(selection_ext, selection_int);
    const int combined_action = combined_selection[0] >= combined_selection[1] ? 0 : 1;
    double expected_external_loss = 0, expected_internal_loss = 0;
    const auto huber = [](double error) { const double value = std::abs(error); return value < 1 ? 0.5 * value * value : value - 0.5; };
    for (int lane = 0; lane < 4; ++lane) {
        const int action = lane % 2;
        double external_bootstrap = target_external[external_policy ? 1 : combined_action];
        double internal_bootstrap = target_internal[combined_action];
        double bonus = 0;
        if (munch != kRfd0121Off) {
            bonus = 0.9 * std::clamp(current_log[action], static_cast<double>(config.munchausen.clip_value_min), 0.0);
            external_bootstrap = internal_bootstrap = 0;
            for (int next_action = 0; next_action < 2; ++next_action) {
                external_bootstrap += std::exp(external_next_log[next_action] / 0.7) * (target_external[next_action] - external_next_log[next_action]);
                internal_bootstrap += std::exp(combined_next_log[next_action] / 0.7) * target_internal[next_action];
            }
        }
        const bool done = lane == 1 || lane == 3;
        const bool internal_done = done && (!continuing || lane == 3);
        const double external_target = 0.2 * (lane + 1) + bonus + (done ? 0 : std::pow(0.5, lane + 1) * external_bootstrap);
        const double internal_target = lane + 1 + (internal_done ? 0 : std::pow(separate ? 0.25 : 0.5, lane + 1) * internal_bootstrap);
        const double external_error = h(online_external[action]) - h(external_target);
        const double internal_error = h(online_internal[action]) - h(internal_target);
        CHECK(result->td_error[lane].item<float>() == Catch::Approx(std::abs(external_error) + 0.3 * std::abs(internal_error)).epsilon(1e-5).margin(1e-6));
        const double loss_factor = mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone ? 1 : mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeQr ? 1.5 : 2.0;
        expected_external_loss += loss_factor * huber(external_error) / 4;
        expected_internal_loss += loss_factor * huber(internal_error) / 4;
    }
    CHECK(*learner->GetScalar(kRfd0121ExtrinsicLoss) == Catch::Approx(expected_external_loss).epsilon(1e-5).margin(1e-6));
    CHECK(*learner->GetScalar(kRfd0121IntrinsicLoss) == Catch::Approx(expected_internal_loss).epsilon(1e-5).margin(1e-6));
    CHECK(result->loss.item<float>() == Catch::Approx(expected_external_loss + weight * expected_internal_loss).epsilon(1e-5).margin(1e-6));
}

TEST_CASE("Shared episodic component targets reproduce a combined learner target", "[rfd0121][dqn][combined_target_equivalence]")
{
    ScopedNoopMetricsLogger logger;
    anet::test::RegisterStandardNetworkModules();
    const auto mode = GENERATE(std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone), std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeQr), std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeIqn));
    const bool tbo = GENERATE(false, true);
    const bool munchausen = GENERATE(false, true);
    const auto env = MakeIqnTracerEnvSpec();
    auto specs = env.state_spec.obs_spec;
    anet::nn::NetworkConfig net;
    net.output_keys[kRfd0121Features] = kVectorKey;
    if (mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeIqn) {
        net = anet::nn::NetworkConfig(MakeIqnTracerConfigData(), "DefaultDQNAgent.net");
        specs[anet::rl::dqn::kKey_Taus] = {.shape = {4}, .dtype = torch::kFloat32};
    }
    const auto h = [tbo](double value) { return tbo ? std::sqrt(value + 1) - 1 + 0.01 * value : value; };
    const auto real = [tbo](double value) {
        const double z = 2 * value / (std::sqrt(1.02 * 1.02 + 0.04 * value) + 1.02);
        return tbo ? z * (z + 2) : value;
    };
    const double loss_factor = mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone ? 1 : mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeQr ? 1.5 : 2;
    // 現在値0・正の定数分布なので、公開lossからHuberを逆に解いて目標値を観測できる。
    const auto target_from_loss = [&](float value) {
        const double loss = value / loss_factor;
        return real(loss >= 0.5 ? loss + 0.5 : std::sqrt(2 * loss));
    };
    double combined_target = 0, component_target = 0;
    for (const bool decomposed : {false, true}) {
        const anet::nn::WeightInitConfig init;
        const auto factory = [&] { return dqn::DqnComponentAssembler::CreateHeadFactory(mode, false, 2, 3, init, {}); };
        std::shared_ptr<anet::nn::NetworkHeadFactory> head = factory();
        if (decomposed) {
            head = std::make_shared<dqn::DecomposedHeadFactory>(factory(), factory(), false, false,
                anet::ConfigFullKeyTable::FormatKeyNotReadFromConfig("input_mode"));
        }
        dqn::NetworkModel model({}, torch::kCPU, net, specs, 2, head, mode != anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone, 121);
        {
            torch::NoGradGuard guard;
            for (auto& parameter : model.GetOnlineNetwork()->parameters()) parameter.zero_();
            for (auto& parameter : model.GetTargetNetwork()->parameters()) parameter.zero_();
            for (auto& parameter : model.GetTargetNetwork()->named_parameters()) {
                if (parameter.key() == "head.linear.bias") parameter.value().fill_(h(5));
                if (parameter.key() == kRfd0121HeadExtrinsicLinearBias) parameter.value().fill_(h(4));
                if (parameter.key() == kRfd0121HeadIntrinsicLinearBias) parameter.value().fill_(h(2));
            }
        }
        auto config = WithDefaultLearnerForwardConfig();
        config.quantile_mode = mode;
        config.num_quantiles = 3;
        config.iqn.current_taus = {.sample_mode = kRfd0121FixedTauMode, .num_taus = 4};
        config.iqn.target_taus = {.sample_mode = kRfd0121FixedTauMode, .num_taus = 3};
        config.intrinsic.decomposed = decomposed;
        config.gamma.value = 0.5f;
        config.use_tbo = tbo;
        config.tbo_epsilon = 0.01f;
        config.use_double_dqn = false;
        config.use_per = false;
        config.use_td_clip = false;
        config.use_amp = false;
        config.learning_rate.value = 0;
        config.replay_batch_size = 2;
        config.replay_capacity = 16;
        config.use_n_step = false;
        config.update_warmup_steps = 100000;
        config.munchausen.enabled = munchausen;
        config.munchausen.entropy_tau.value = 0.7f;
        dqn::ActionPolicyConfig policy_config;
        policy_config.policy_type = anet::rl::dqn::kActionPolicyTypeStr_Greedy;
        policy_config.quantile_mode = mode;
        auto policy = dqn::DqnComponentAssembler::CreateActionPolicy(policy_config, mode, false, 2, torch::kCPU);
        dqn::RuntimeVars vars;
        const auto create = [&]<typename T>() { return std::make_shared<T>(config, model, vars, nullptr,
            rl::BatchEnvSpec{2, 1}, env, torch::kCPU, 121, kTestAgentSeed, policy, dqn::LearnerForwardResources{}); };
        std::shared_ptr<dqn::Learner> learner;
        if (mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone) learner = create.operator()<dqn::TDLearner>();
        else if (mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeQr) learner = create.operator()<dqn::QRLearner>();
        else learner = create.operator()<dqn::IQNLearner>();
        rl::ExperienceSamples samples;
        samples.obs = anet::TensorDict{{kVectorKey, torch::ones({2, 4})}};
        samples.next_state.next_obs = samples.obs;
        samples.actions = torch::zeros({2}, torch::kInt64);
        samples.target_returns = decomposed ? torch::tensor({{1.0f, 2.0f}, {1.0f, 2.0f}}) : torch::full({2}, 2.0f);
        samples.n_steps = torch::ones({2}, torch::kInt64);
        const auto flags = torch::zeros({2}, torch::kBool);
        samples.next_state.terminals = flags;
        samples.next_state.truncated = flags;
        learner->UpdateFromBatch({}, {rl::BatchState(samples.obs, flags, flags, ~flags),
            std::make_shared<rl::BatchActionInfo>(samples.actions), torch::zeros_like(samples.target_returns),
            rl::BatchState(samples.obs, flags, flags, flags)}, 0.5f);
        const auto result = learner->UpdateFromSamples(samples);
        if (decomposed) component_target = target_from_loss(*learner->GetScalar(kRfd0121ExtrinsicLoss))
            + 0.5 * target_from_loss(*learner->GetScalar(kRfd0121IntrinsicLoss));
        else combined_target = target_from_loss(result->loss.item<float>());
    }
    CAPTURE(mode, tbo, munchausen);
    CHECK(component_target == Catch::Approx(combined_target).epsilon(1e-5).margin(1e-6));
    CHECK(component_target == Catch::Approx(4.5 + (munchausen ? -0.4 * 0.7 * std::log(2.0) : 0)).epsilon(1e-5).margin(1e-6));
}

TEST_CASE("Decomposed updates reject nonfinite gradients before both optimizer steps", "[rfd0121][dqn][rfd0121-review-nan]")
{
#if ANET_ENABLE_TENSOR_NAN_CHECK
    anet::test::RegisterStandardNetworkModules();
    const bool split = GENERATE(false, true);
    const bool poison_intrinsic = GENERATE(false, true);
    const float poison = GENERATE(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity());
    const auto env = MakeLearnerEnvSpec();
    const anet::nn::WeightInitConfig init;
    auto head = std::make_shared<dqn::DecomposedHeadFactory>(std::make_shared<dqn::LinearHeadFactory>(1, init),
        std::make_shared<dqn::LinearHeadFactory>(1, init), false, false,
        anet::ConfigFullKeyTable::FormatKeyNotReadFromConfig("input_mode"));
    dqn::NetworkModel model({}, torch::kCPU, MakeAgentForwardNetworkConfig(), env.state_spec.obs_spec, 1, head, false, 121);
    auto config = WithDefaultLearnerForwardConfig();
    config.intrinsic.decomposed = true;
    config.intrinsic.feature_gradient = split ? dqn::LearnerConfig::kFeatureGradientSplit : dqn::LearnerConfig::kFeatureGradientMerge;
    config.replay_capacity = 16;
    config.replay_batch_size = 2;
    config.use_per = false;
    config.use_tbo = false;
    config.use_td_clip = false;
    config.use_amp = false;
    config.use_fused_optimizer = false;
    dqn::RuntimeVars vars;
    dqn::TDLearner learner(config, model, vars, nullptr, {.num_envs = 2, .num_threads = 1}, env, torch::kCPU, 121, kTestAgentSeed,
        std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), {});
    const auto parameters = model.GetOnlineParameters();
    std::vector<torch::Tensor> before;
    for (const auto& parameter : parameters) before.push_back(parameter.detach().clone());
    bool hooked = false;
    for (const auto& entry : model.GetOnlineNetwork()->named_parameters()) {
        if (entry.key().find(poison_intrinsic ? kRfd0121Intrinsic : kRfd0121ExtrinsicHead) == std::string::npos) continue;
        // forwardとlossは有限のまま、選んだHeadのbackwardだけを壊す。
        entry.value().register_hook([poison](torch::Tensor grad) { return torch::full_like(grad, poison); });
        hooked = true;
        break;
    }
    REQUIRE(hooked);
    rl::ExperienceSamples samples;
    samples.obs = anet::TensorDict{{kVectorKey, torch::ones({2, 2})}};
    samples.actions = torch::zeros({2}, torch::kInt64);
    samples.target_returns = torch::ones({2, 2});
    samples.n_steps = torch::ones({2}, torch::kInt64);
    samples.next_state.next_obs = samples.obs;
    samples.next_state.terminals = torch::ones({2}, torch::kBool);
    samples.next_state.truncated = torch::zeros({2}, torch::kBool);
    CHECK_THROWS_WITH(learner.UpdateFromSamples(samples), Catch::Matchers::ContainsSubstring("Nonfinite gradient norms"));
    for (size_t i = 0; i < parameters.size(); ++i) CHECK(torch::equal(parameters[i], before[i]));
#else
    SUCCEED("Gradient NaN checks are disabled in this build");
#endif
}

TEST_CASE("Decomposed split adds independently clipped Adam updates", "[rfd0121][dqn][split_numeric]")
{
    const bool fused = GENERATE(false, true);
    const bool clipped = GENERATE(false, true);
    const float decay = GENERATE(0.0f, 0.1f);
    const bool scheduled = GENERATE(false, true);
    anet::test::RegisterStandardNetworkModules();
    const auto env = MakeLearnerEnvSpec();
    anet::ConfigData data;
    data.Set("net.block.[SharedLinear].type", "Linear");
    data.Set("net.block.[SharedLinear].linear.out_features", 2);
    data.Set("net.branch.[shared].bind", kVectorKey);
    data.Set("net.branch.[shared].structure", "SharedLinear");
    data.Set("net.body.output.[features]", kRfd0121SharedBranchKey);
    const anet::nn::WeightInitConfig init;
    auto head = std::make_shared<dqn::DecomposedHeadFactory>(std::make_shared<dqn::LinearHeadFactory>(1, init),
        std::make_shared<dqn::LinearHeadFactory>(1, init), false, false,
        anet::ConfigFullKeyTable::FormatKeyNotReadFromConfig("input_mode"));
    dqn::NetworkModel model({}, torch::kCPU, anet::nn::NetworkConfig(data, "net"), env.state_spec.obs_spec, 1, head, false, 98);
    auto config = WithDefaultLearnerForwardConfig();
    config.intrinsic.decomposed = true;
    config.intrinsic.feature_gradient = "split";
    config.intrinsic.loss_weight = 0.2f;
    config.replay_capacity = 16;
    config.replay_batch_size = 2;
    config.learning_rate.value = 0.01f;
    config.weight_decay.value = decay;
    config.update_warmup_steps = 100000;
    if (scheduled) {
        config.learning_rate.type = kRfd0121Linear;
        config.learning_rate.start = 0.01f;
        config.learning_rate.end = 0.02f;
        config.learning_rate.steps = 4;
        config.weight_decay.type = kRfd0121Linear;
        config.weight_decay.start = decay;
        config.weight_decay.end = decay + 0.1f;
        config.weight_decay.steps = 4;
    }
    const float effective_lr = scheduled ? 0.02f : 0.01f;
    const float effective_decay = scheduled ? decay + 0.1f : decay;
    config.use_per = false;
    config.use_tbo = false;
    config.use_td_clip = false;
    config.use_amp = false;
    config.use_grad_clip = clipped;
    config.grad_clip_tau = 0.5f;
    config.use_fused_optimizer = fused;
    dqn::RuntimeVars vars;
    dqn::TDLearner learner(config, model, vars, nullptr, {.num_envs = 2, .num_threads = 1}, env, torch::kCPU, 98, kTestAgentSeed,
        std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), {});
    const auto advance_schedule = [&](bool first) {
        const auto flags = torch::zeros({2}, torch::kBool);
        const anet::TensorDict obs{{kVectorKey, torch::ones({2, 2})}};
        learner.UpdateFromBatch({.exp_step = 4}, {rl::BatchState(obs, flags, flags, torch::full({2}, first, torch::kBool)),
            std::make_shared<rl::BatchActionInfo>(torch::zeros({2}, torch::kInt64)),
            torch::zeros({2, 2}), rl::BatchState(obs, flags, flags, flags)}, 0.1f);
    };
    if (scheduled) advance_schedule(true);
    auto parameters = model.GetOnlineParameters();
    {
        torch::NoGradGuard guard;
        for (auto& parameter : parameters) parameter.fill_(0.1f);
    }
    rl::ExperienceSamples samples;
    samples.obs = anet::TensorDict{{kVectorKey, torch::ones({2, 2})}};
    samples.actions = torch::zeros({2}, torch::kInt64);
    samples.target_returns = torch::tensor({{3.0f, -4.0f}, {3.0f, -4.0f}});
    samples.n_steps = torch::ones({2}, torch::kInt64);
    samples.next_state.next_obs = samples.obs;
    samples.next_state.terminals = torch::ones({2}, torch::kBool);
    samples.next_state.truncated = torch::zeros({2}, torch::kBool);
    const auto output = model.GetOnlineNetwork()->Forward(samples.obs);
    const auto external_loss = torch::nn::functional::smooth_l1_loss(output.At(kRfd0121ExtrinsicQ).squeeze(1), samples.target_returns.select(1, 0));
    const auto internal_loss = config.intrinsic.loss_weight * torch::nn::functional::smooth_l1_loss(output.At(kRfd0121IntrinsicQ).squeeze(1), samples.target_returns.select(1, 1));
    const auto norm = [&] {
        float squared = 0;
        for (const auto& parameter : parameters) if (parameter.grad().defined()) squared += parameter.grad().square().sum().item<float>();
        return std::sqrt(squared);
    };
    const auto delta = [&](const torch::Tensor& gradient, float total) {
        const auto grad = gradient * (clipped ? std::min(1.0f, config.grad_clip_tau / (total + 1e-6f)) : 1.0f);
        return effective_lr * grad / (grad.abs() + config.adam_eps);
    };
    std::vector<torch::Tensor> expected;
    std::vector<bool> external_used;
    external_loss.backward({}, true);
    const auto external_norm = norm();
    for (auto& parameter : parameters) {
        auto value = parameter.detach().clone();
        external_used.push_back(parameter.grad().defined());
        if (parameter.grad().defined()) value = value * (1 - effective_lr * effective_decay) - delta(parameter.grad(), external_norm);
        expected.push_back(value);
        parameter.mutable_grad() = torch::Tensor();
    }
    internal_loss.backward();
    const auto internal_norm = norm();
    if (clipped) {
        CHECK(external_norm > config.grad_clip_tau);
        CHECK(internal_norm < config.grad_clip_tau);
    }
    for (size_t i = 0; i < parameters.size(); ++i) {
        if (parameters[i].grad().defined()) {
            if (!external_used[i]) expected[i] *= 1 - effective_lr * effective_decay;
            expected[i] -= delta(parameters[i].grad(), internal_norm);
        }
        parameters[i].mutable_grad() = torch::Tensor();
    }
    learner.UpdateFromSamples(samples);
    for (size_t i = 0; i < parameters.size(); ++i) {
        CAPTURE(i, fused, clipped);
        CHECK(torch::allclose(parameters[i], expected[i], 1e-5, 1e-6));
    }
    REQUIRE(learner.GetScalar(kRfd0121ExtrinsicGradNorm));
    REQUIRE(learner.GetScalar(kRfd0121IntrinsicGradNorm));
    CHECK(*learner.GetScalar(kRfd0121ExtrinsicGradNorm) == Catch::Approx(external_norm));
    CHECK(*learner.GetScalar(kRfd0121IntrinsicGradNorm) == Catch::Approx(internal_norm));
    std::stringstream checkpoint(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive archive(checkpoint);
    learner.Save(archive);
    std::vector<torch::Tensor> saved, continued;
    for (const auto& parameter : parameters) saved.push_back(parameter.detach().clone());
    learner.UpdateFromSamples(samples);
    for (const auto& parameter : parameters) continued.push_back(parameter.detach().clone());
    {
        torch::NoGradGuard guard;
        for (size_t i = 0; i < parameters.size(); ++i) parameters[i].copy_(saved[i]);
    }
    checkpoint.seekg(0);
    anet::InputArchive input(checkpoint);
    learner.Load(input);
    if (scheduled) advance_schedule(false);
    learner.UpdateFromSamples(samples);
    for (size_t i = 0; i < parameters.size(); ++i) CHECK(torch::equal(parameters[i], continued[i]));
}

TEST_CASE("DQN initial priorities sum independent head errors", "[rfd0121][dqn][per][actor_initial]")
{
    const float weight = GENERATE(1.0f, 0.2f, 0.0f);
    const bool tbo = GENERATE(false, true);
    dqn::LearnerConfig config;
    config.intrinsic.decomposed = true;
    config.intrinsic.priority_lambda = weight;
    config.use_tbo = tbo;
    config.per_eps = 0.01f;
    config.use_per_prio_clip = false;
    auto estimator = dqn::CreateInitialPriorityEstimator(config);
    const auto value = [&](float input) { return tbo ? dqn::TransformH(input, config.tbo_epsilon) : input; };
    const std::array<float, 5> start{value(5), 0, 0, value(2), 0};
    const std::array<float, 5> bootstrap{0, value(2), 0, 0, value(4)};
    const std::array<float, 1> returns{7};
    const std::array<float, 1> discounts{0.25f};
    const auto priority = estimator->Estimate({.start_hint = start, .bootstrap_hint = bootstrap,
        .target_return = 1, .discount = 0.5f, .extra_component_returns = returns, .extra_component_discounts = discounts});
    REQUIRE(priority);
    CHECK(*priority == Catch::Approx(std::abs(value(2) - value(5)) + weight * std::abs(value(8) - value(2)) + config.per_eps));
}

TEST_CASE("ActionPolicy spatial tensor generation handles supported scale types", "[dqn][action_policy][spatial]")
{
    auto device = torch::Device(torch::kCPU);

    auto linear = ActionPolicyAccess::BuildTensor(3, 1.0f, 0.0f, "linear", device);
    CHECK(ShapeOf(linear) == std::vector<int64_t>{ 3 });
    CHECK(torch::allclose(linear, torch::tensor({ 1.0f, 0.5f, 0.0f })));

    auto log = ActionPolicyAccess::BuildTensor(3, 1.0f, 0.01f, "log", device);
    CHECK(torch::allclose(log, torch::tensor({ 1.0f, 0.1f, 0.01f }), 1.0e-5, 1.0e-5));

    auto clamped = ActionPolicyAccess::BuildTensor(2, 0.0f, 0.0f, "log", device);
    CHECK(torch::allclose(clamped, torch::tensor({ 0.0f, 0.0f })));

    auto single = ActionPolicyAccess::BuildTensor(1, 0.25f, 0.75f, "linear", device);
    CHECK(ShapeOf(single) == std::vector<int64_t>{ 1 });
    CHECK(single[0].item<float>() == Catch::Approx(0.25f).margin(1.0e-6f));

    CHECK_THROWS(ActionPolicyAccess::BuildTensor(2, 1.0f, 0.0f, "invalid", device));
}

TEST_CASE("ActionPolicy reverses spatial parameter tuples for env lane assignment", "[dqn][action_policy][spatial]")
{
    auto device = torch::Device(torch::kCPU);

    // 補間結果を厳密に反転し、値の集合を変えずにend側をenv[0]へ割り当てる。
    auto linear = ActionPolicyAccess::BuildTensor(3, 1.0f, 0.0f, "linear", device);
    auto lane_linear = ActionPolicyAccess::BuildLaneTensor(3, 1.0f, 0.0f, "linear", device);
    CHECK(torch::equal(lane_linear, linear.flip({ 0 })));
    CHECK(torch::equal(lane_linear, torch::tensor({ 0.0f, 0.5f, 1.0f })));

    auto log = ActionPolicyAccess::BuildTensor(3, 1.0f, 0.01f, "log", device);
    auto lane_log = ActionPolicyAccess::BuildLaneTensor(3, 1.0f, 0.01f, "log", device);
    CHECK(torch::equal(lane_log, log.flip({ 0 })));

    // epsilonとtauを同じ向きで反転し、パラメータの組の集合を維持する。
    auto original_pairs = torch::stack({
        ActionPolicyAccess::BuildTensor(3, 0.6f, 0.0f, "log", device),
        ActionPolicyAccess::BuildTensor(3, 0.95f, 0.85f, "log", device),
    }, 1);
    auto lane_pairs = torch::stack({
        ActionPolicyAccess::BuildLaneTensor(3, 0.6f, 0.0f, "log", device),
        ActionPolicyAccess::BuildLaneTensor(3, 0.95f, 0.85f, "log", device),
    }, 1);
    CHECK(torch::equal(lane_pairs, original_pairs.flip({ 0 })));
    CHECK(lane_pairs[0][0].item<float>() == Catch::Approx(0.0f).margin(1.0e-7f));
    CHECK(lane_pairs[2][0].item<float>() == Catch::Approx(0.6f).margin(1.0e-6f));

    // laneが1つなら反転はno-opとなり、従来どおりstart値を使う。
    auto single = ActionPolicyAccess::BuildLaneTensor(1, 0.25f, 0.75f, "linear", device);
    CHECK(single[0].item<float>() == Catch::Approx(0.25f).margin(1.0e-6f));
}

TEST_CASE("Learner allows sample prefetch on CPU device", "[dqn][prefetch]")
{
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.use_rb_prefetch = true;
    config.replay_batch_size = 1;

    TestNetworkModel model;
    dqn::RuntimeVars vars;
    auto env_spec = MakeLearnerEnvSpec();
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };

    CHECK_NOTHROW(TestLearner(config, model, vars, batch_env_spec, env_spec, dqn::LearnerForwardResources{}));
}

TEST_CASE("DQN learner prefetch path is deterministic under fixed jitter", "[dqn][prefetch][determinism]")
{
    constexpr int kTrialPairs = 48;
    constexpr int64_t kNumSteps = 256;
    constexpr int kMaxSleepUs = 300;
    constexpr size_t kSleepStride = 64;
    RequireLearnerDeterminismPairs(kTrialPairs, kNumSteps, kMaxSleepUs, kSleepStride);
}

TEST_CASE("PipelineTrainRunner prefetch path is deterministic under fixed jitter", "[dqn][prefetch][determinism][runner]")
{
    constexpr int kTrialPairs = 12;
    constexpr int64_t kNumSteps = 256;
    constexpr int kMaxSleepUs = 300;
    constexpr size_t kSleepStride = 64;

    for (int trial = 0; trial < kTrialPairs; ++trial) {
        const auto replay_seed = static_cast<anet::seed_t>(30000 + trial * 43);
        const auto jitter_seed = static_cast<uint64_t>(40000 + trial * 109);
        INFO("trial=" << trial << " replay_seed=" << replay_seed << " jitter_seed=" << jitter_seed);

        auto first = RunPipelineDeterminismTrial(replay_seed, jitter_seed, kNumSteps, kMaxSleepUs, kSleepStride);
        auto second = RunPipelineDeterminismTrial(replay_seed, jitter_seed, kNumSteps, kMaxSleepUs, kSleepStride);

        REQUIRE_FALSE(first.learner_trace.empty());
        REQUIRE(first.counts.train_step == second.counts.train_step);
        REQUIRE(first.counts.exp_step == second.counts.exp_step);
        REQUIRE(first.counts.learn_step == second.counts.learn_step);
        REQUIRE(first.learner_trace.size() == second.learner_trace.size());
        REQUIRE(DeterminismTracesEqual(first.learner_trace, second.learner_trace));
    }
}

TEST_CASE("DQN learner prefetch path is deterministic under extended jitter stress", "[.][dqn][prefetch][determinism][stress]")
{
    if (!CurrentCatchFilterMentionsStress()) {
        SUCCEED("Skipping extended stress unless [stress] is explicitly selected.");
        return;
    }

    constexpr int kTrialPairs = 200;
    constexpr int64_t kNumSteps = 512;
    constexpr int kMaxSleepUs = 1000;
    constexpr size_t kSleepStride = 64;
    RequireLearnerDeterminismPairs(kTrialPairs, kNumSteps, kMaxSleepUs, kSleepStride);
}

TEST_CASE("DQN update result exposes captured plasticity metrics lazily", "[dqn][plasticity]")
{
    dqn::BatchUpdateResult result;
    result.plasticity_features = torch::eye(2, torch::TensorOptions().dtype(torch::kFloat32));
    result.plasticity_request = anet::nn::PlasticityMetricRequest::All();
    result.plasticity_weight_norms = torch::tensor({ 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f, 0.0f });

    REQUIRE(result.GetScalar("plasticity_srank").has_value());
    CHECK(*result.GetScalar("plasticity_srank") == 2.0f);
    CHECK(*result.GetScalar("plasticity_srank_ratio") == Catch::Approx(1.0f));
    CHECK(*result.GetScalar("plasticity_srank_delta_005") == 2.0f);
    CHECK(*result.GetScalar("plasticity_srank_ratio_delta_020") == Catch::Approx(1.0f));
    CHECK(std::isnan(*result.GetScalar("plasticity_target_srank")));
    CHECK_FALSE(result.plasticity_weight_norms_cpu.defined());
    CHECK(*result.GetScalar("plasticity_weight_norm_feature") == Catch::Approx(3.0f));
    REQUIRE(result.plasticity_weight_norms_cpu.defined());
    const auto* cpu_data = result.plasticity_weight_norms_cpu.data_ptr<float>();
    CHECK(*result.GetScalar("plasticity_weight_norm_readout") == Catch::Approx(4.0f));
    CHECK(*result.GetScalar("plasticity_weight_norm_feature_effective") == Catch::Approx(5.0f));
    CHECK(*result.GetScalar("plasticity_weight_norm_readout_effective") == Catch::Approx(6.0f));
    CHECK(*result.GetScalar("plasticity_spectral_sigma_feature") == Catch::Approx(7.0f));
    CHECK(*result.GetScalar("plasticity_spectral_sigma_readout") == Catch::Approx(8.0f));
    CHECK(result.plasticity_weight_norms_cpu.data_ptr<float>() == cpu_data);
    CHECK_FALSE(result.GetScalar("plasticity_unknown").has_value());

    dqn::BatchUpdateResult partial_result;
    partial_result.plasticity_features = result.plasticity_features;
    partial_result.plasticity_request.Add(anet::nn::PlasticityMetric::FEATURE_NORM);
    CHECK(std::isfinite(*partial_result.GetScalar("plasticity_feature_norm")));
    CHECK(std::isnan(*partial_result.GetScalar("plasticity_srank")));
    CHECK(std::isnan(*partial_result.GetScalar("plasticity_srank_delta_005")));
}

TEST_CASE("DQN spectral normalization sentinel reports the invalid network and layer", "[dqn][spectral_norm][plasticity]")
{
    auto online = MakeSpectralLinearNetwork();
    auto target = online->Clone(torch::Device(torch::kCPU));
    {
        torch::NoGradGuard no_grad;
        online->GetSpectralNormEntries().front().u.fill_(
            std::numeric_limits<float>::quiet_NaN());
    }

    dqn::BatchUpdateResult result;
    result.plasticity_weight_norms = torch::tensor({
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f });
    result.plasticity_online_network = online;
    result.plasticity_target_network = target;

    CHECK_THROWS_WITH(
        result.GetScalar("plasticity_weight_norm_feature"),
        Catch::Matchers::ContainsSubstring("network=online")
            && Catch::Matchers::ContainsSubstring("feature.Linear_0.linear"));
}

TEST_CASE("DQN learner reports greedy action churn for one optimizer update", "[dqn][policy_churn]")
{
    // 1 update で action 0 から action 1 へ反転する合成 network と probe を準備する。
    TestNetworkModel model(/*distributional=*/false, /*n_actions=*/2);
    {
        torch::NoGradGuard no_grad;
        model.GetOnlineParameters().front().copy_(torch::tensor({
            { 1.0f, 0.0f },
            { 0.0f, 0.0f },
        }));
        model.GetOnlineNetwork()->CopyTo(*model.GetTargetNetwork());
    }
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.replay_batch_size = 1;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;
    config.use_per = false;
    auto env_spec = MakeLearnerEnvSpec();
    env_spec.action_spec.value_labels = { "a0", "a1" };
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    learner.UseSgd(1.0f);
    learner.EnablePolicyChurnOptimizerUpdate();
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->SetSizeValues({ 1 });
    replay_buffer->unique_sample_enabled = true;
    replay_buffer->vector_value = 1.0f;
    learner.UseReplayBuffer(replay_buffer);
    std::vector<rl::ScalarMetricSubscription> subscriptions;
    for (const auto* key : std::span(dqn::kPolicyChurnMetricKeys).first(dqn::PolicyChurnMetric::TargetPolicyDisagreement)) {
        subscriptions.push_back(rl::ScalarMetricSubscription{
            .source_key = key,
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN,
        });
    }
    learner.ConfigureScalarMetricSubscriptions(subscriptions);

    // public update result から、probe 1 件の greedy action が反転した割合を読む。
    const auto results = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(results.size() == 1);
    const auto churn = results.front()->GetScalar("policy_churn_action_ratio");
    REQUIRE(churn.has_value());
    CHECK(*churn == Catch::Approx(1.0f));
    CHECK(*results.front()->GetScalar("policy_churn_q_delta_abs_mean") == Catch::Approx(1.0f));
    CHECK(*results.front()->GetScalar("policy_churn_q_delta_signed_max") == Catch::Approx(2.0f));
    CHECK(*results.front()->GetScalar("policy_churn_q_delta_signed_min") == Catch::Approx(0.0f));
}

TEST_CASE("DQN replay fit evaluates the received PER batch before optimization", "[dqn][replay_fit][per_probe]")
{
    // sampleとnetworkを既存の公開境界で固定し、学習後ではなく更新直前の値を確認する。
    const auto probe = std::make_shared<AutocastProbeState>();
    AutocastProbeNetworkModel model(probe, torch::kCPU);
    {
        torch::NoGradGuard no_grad;
        for (auto& parameter : model.GetOnlineParameters()) parameter.fill_(1.0f);
        model.GetOnlineNetwork()->CopyTo(*model.GetTargetNetwork());
    }
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;
    config.use_per = true;
    config.gamma.value = 0.5f;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), dqn::LearnerForwardResources{});
    const auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({ 2 });
    replay->vector_value = 1.0f;
    learner.UseReplayBuffer(replay);
    learner.UseSgd(1.0f);
    learner.EnablePolicyChurnOptimizerUpdate();
    learner.ConfigureScalarMetricSubscriptions({
        rl::ScalarMetricSubscription{ .source_key = "replay_fit_per_td_mean", .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT, .interval = 1, .scope = rl::RunnerScope::TRAIN },
    });
    const auto results = learner.UpdateFromBatch(rl::StepCounts{ .exp_step = 5 }, MakeDeterminismExperience(5, 1));
    REQUIRE(results.size() == 1);
    const auto value = results.front()->GetScalar("replay_fit_per_td_mean");
    REQUIRE(value.has_value());
    CHECK(*value == Catch::Approx(1.0f));
    CHECK(replay->history_probe_count == 0);
    CHECK(replay->sample_count == 1);
    CHECK(probe->forward_count == 4);
}

TEST_CASE("DQN replay fit populations reach the update result without another forward", "[dqn][replay_fit][tracer]")
{
    // 実ReplayBufferに未抽選遷移を用意し、通常Sample直後の二群を公開結果で観測する。
    const auto env_spec = MakeLearnerEnvSpec();
    rl::ReplayBufferConfig replay_config;
    replay_config.capacity = 32;
    const auto replay = rl::CreateReplayBuffer(replay_config, env_spec, 1, torch::kCPU, false, 73001);
    for (int64_t step = 0; step < 5; ++step) replay->Push(MakeDeterminismExperience(step, 1), std::array{0.99f});
    auto probe = std::make_shared<AutocastProbeState>();
    AutocastProbeNetworkModel model(probe, torch::kCPU);
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.replay_batch_size = 1;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;
    config.use_per = false;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    learner.UseReplayBuffer(replay);
    learner.UseSgd(0.01f);
    learner.EnablePolicyChurnOptimizerUpdate();
    learner.ConfigureScalarMetricSubscriptions({
        rl::ScalarMetricSubscription{ .source_key = "replay_fit_unsampled_count", .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT, .interval = 1, .scope = rl::RunnerScope::TRAIN },
        rl::ScalarMetricSubscription{ .source_key = "replay_fit_sampled_count", .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT, .interval = 1, .scope = rl::RunnerScope::TRAIN },
    });
    const auto before = probe->forward_count;
    const auto results = learner.UpdateFromBatch(rl::StepCounts{ .exp_step = 5 }, MakeDeterminismExperience(5, 1));
    REQUIRE(results.size() == 1);
    const auto unsampled = results.front()->GetScalar("replay_fit_unsampled_count");
    const auto sampled = results.front()->GetScalar("replay_fit_sampled_count");
    REQUIRE(unsampled.has_value());
    REQUIRE(sampled.has_value());
    CHECK(*sampled == 1.0f);
    CHECK(*unsampled == static_cast<float>(replay->Size() - 1));
    CHECK(probe->forward_count == before + 1);
}

TEST_CASE("DQN replay fit routes each subscription and retains update-specific values", "[dqn][replay_fit][demand]")
{
    // 各出力を単独購読し、依存する群・forward数・公開する値を同じ数値oracleで確認する。
    const int metric = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
    const bool per = GENERATE(false, true);
    CAPTURE(metric, per);
    const auto probe = std::make_shared<AutocastProbeState>();
    AutocastProbeNetworkModel model(probe, torch::kCPU);
    {
        torch::NoGradGuard no_grad;
        for (auto& parameter : model.GetOnlineParameters()) parameter.fill_(1.0f);
        model.GetOnlineNetwork()->CopyTo(*model.GetTargetNetwork());
    }
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;
    config.use_per = per;
    config.gamma.value = 0.5f;
    config.use_td_clip = false;
    config.replay_fit.probe.batch_size = 2;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), dqn::LearnerForwardResources{});
    const auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({ 2 });
    rl::ExperienceSamples u, s;
    replay->vector_value = 2.0f;
    replay->Sample(u, 2, 0.0f, std::array{0.99f});
    replay->vector_value = 1.0f;
    replay->Sample(s, 2, 0.0f, std::array{0.99f});
    replay->history_result = rl::SamplingHistoryProbeResult{
        .unsampled_count = 6, .sampled_count = 2, .unsampled_age_mean = 3.0f, .sampled_age_mean = 5.0f,
        .unsampled = u, .sampled = s,
    };
    replay->vector_value = 3.0f;
    replay->sample_count = 0;
    learner.UseReplayBuffer(replay);
    learner.EnablePlasticityForward(false);
    std::vector<rl::ScalarMetricSubscription> subscriptions;
    if (metric < 13) subscriptions.push_back(rl::ScalarMetricSubscription{
        .source_key = dqn::kReplayFitMetricKeys[metric], .event = rl::EventType::LEARN,
        .target = rl::EventField::UPDATE_RESULT, .interval = 2, .scope = rl::RunnerScope::TRAIN });
    learner.ConfigureScalarMetricSubscriptions(subscriptions);
    const auto first = learner.UpdateFromBatch(rl::StepCounts{ .exp_step = 5 }, MakeDeterminismExperience(5, 1));
    REQUIRE(first.size() == 1);
    const bool uniform = metric == 10 || (per && metric == 12);
    const bool need_u = metric == 0 || metric == 2 || metric == 3 || metric == 5 || uniform;
    const bool need_s = metric == 1 || metric == 2 || metric == 4 || metric == 5 || uniform;
    const bool need_per = per && (metric == 11 || metric == 12);
    const bool counts = (metric >= 6 && metric <= 9) || uniform;
    CHECK(replay->history_probe_count == (need_u || need_s || counts ? 1 : 0));
    CHECK(replay->last_history_request.unsampled_batch_size.has_value() == need_u);
    CHECK(replay->last_history_request.sampled_batch_size.has_value() == need_s);
    CHECK(replay->last_history_request.counts == counts);
    CHECK(probe->forward_count == 1 + 3 * (int(need_u) + int(need_s) + int(need_per)));
    CHECK(replay->sample_count == 1);
    const std::array<float, 13> expected{ 2.0f, 1.0f, 2.0f, 1.5f, 0.5f, 3.0f,
        6.0f, 2.0f, 3.0f, 5.0f, 1.75f, 3.0f, 3.0f / 1.75f };
    for (int i = 0; i < 13; ++i) {
        const auto value = first.front()->GetScalar(dqn::kReplayFitMetricKeys[i]);
        REQUIRE(value.has_value());
        if (i == metric && (per || i < 11)) CHECK(*value == Catch::Approx(expected[i]));
        else CHECK(std::isnan(*value));
    }
    CHECK_FALSE(first.front()->GetScalar("replay_fit_unknown").has_value());
    const int forwards = probe->forward_count;
    const int probes = replay->history_probe_count;
    const auto second = learner.UpdateFromBatch(rl::StepCounts{ .exp_step = 6 }, MakeDeterminismExperience(6, 1));
    REQUIRE(second.size() == 1);
    CHECK(probe->forward_count == forwards + 1);
    CHECK(replay->history_probe_count == probes);
    if (metric < 13) {
        CHECK(std::isnan(*second.front()->GetScalar(dqn::kReplayFitMetricKeys[metric])));
        if (per || metric < 11) CHECK(*first.front()->GetScalar(dqn::kReplayFitMetricKeys[metric]) == Catch::Approx(expected[metric]));
    }
}

TEST_CASE("DQN learner reports subscribed parameter norm without activating feature metrics", "[dqn][plasticity][weight_norm]")
{
    auto probe_state = std::make_shared<AutocastProbeState>();
    AutocastProbeNetworkModel model(probe_state, torch::Device(torch::kCPU));
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = 2.0f;
    config.final_feature_branch_key = kFeatureKey;
    auto env_spec = MakeLearnerEnvSpec();
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    learner.EnablePlasticityForward(false);
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->SetSizeValues({ 2 });
    learner.UseReplayBuffer(replay_buffer);
    learner.ConfigureScalarMetricSubscriptions({
        rl::ScalarMetricSubscription{
            .source_key = "plasticity_weight_norm_feature",
            .event = rl::EventType::LEARN,
            .interval = 2,
            .scope = rl::RunnerScope::TRAIN,
        },
        rl::ScalarMetricSubscription{
            .source_key = "plasticity_weight_norm_readout",
            .event = rl::EventType::LEARN,
            .interval = 3,
            .scope = rl::RunnerScope::TRAIN,
        },
        rl::ScalarMetricSubscription{
            .source_key = "plasticity_weight_norm_feature_effective",
            .event = rl::EventType::LEARN,
            .interval = 4,
            .scope = rl::RunnerScope::TRAIN,
        },
        rl::ScalarMetricSubscription{
            .source_key = "plasticity_spectral_sigma_readout",
            .event = rl::EventType::LEARN,
            .interval = 5,
            .scope = rl::RunnerScope::TRAIN,
        },
    });

    const auto results = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});

    REQUIRE(results.size() == 1);
    const auto feature_norm = results.front()->GetScalar("plasticity_weight_norm_feature");
    const auto readout_norm = results.front()->GetScalar("plasticity_weight_norm_readout");
    REQUIRE(feature_norm.has_value());
    REQUIRE(readout_norm.has_value());
    CHECK(std::isfinite(*feature_norm));
    CHECK(std::isfinite(*readout_norm));
    CHECK(std::isfinite(*results.front()->GetScalar("plasticity_weight_norm_feature_effective")));
    CHECK(std::isnan(*results.front()->GetScalar("plasticity_spectral_sigma_readout")));
    CHECK(std::isnan(*results.front()->GetScalar("plasticity_srank")));
    CHECK(replay_buffer->unique_sample_count == 0);

    const auto skipped = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(skipped.size() == 1);
    CHECK(std::isnan(*skipped.front()->GetScalar("plasticity_weight_norm_feature")));
    CHECK(std::isnan(*skipped.front()->GetScalar("plasticity_weight_norm_readout")));

    const auto measured_again = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(measured_again.size() == 1);
    CHECK(std::isfinite(*measured_again.front()->GetScalar("plasticity_weight_norm_feature")));
    CHECK(std::isfinite(*measured_again.front()->GetScalar("plasticity_weight_norm_readout")));
}

TEST_CASE("DQN parameter norm records the value before the learner update", "[dqn][plasticity][weight_norm]")
{
    auto probe_state = std::make_shared<AutocastProbeState>();
    AutocastProbeNetworkModel model(probe_state, torch::Device(torch::kCPU));
    const float expected_readout = model.GetOnlineNetwork()
        ->ComputeParameterNormSplit(kFeatureKey).readout.item<float>();
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = 2.0f;
    config.final_feature_branch_key = kFeatureKey;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), dqn::LearnerForwardResources{});
    learner.EnablePlasticityForward(false);
    learner.MutateOnlineParametersDuringUpdate(1.0f);
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->SetSizeValues({ 2 });
    learner.UseReplayBuffer(replay_buffer);
    learner.ConfigureScalarMetricSubscriptions({ rl::ScalarMetricSubscription{
        .source_key = "plasticity_weight_norm_readout",
        .event = rl::EventType::LEARN,
        .interval = 1,
        .scope = rl::RunnerScope::TRAIN,
    } });

    const auto results = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});

    REQUIRE(results.size() == 1);
    CHECK(*results.front()->GetScalar("plasticity_weight_norm_readout")
        == Catch::Approx(expected_readout));
    const float updated_readout = model.GetOnlineNetwork()
        ->ComputeParameterNormSplit(kFeatureKey).readout.item<float>();
    CHECK(updated_readout != Catch::Approx(expected_readout));
}

TEST_CASE("DQN plasticity subscriptions gate actual target and probe work independently", "[dqn][plasticity]")
{
    auto probe_state = std::make_shared<AutocastProbeState>();
    AutocastProbeNetworkModel model(probe_state, torch::Device(torch::kCPU));
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = 2.0f;
    config.final_feature_branch_key = kFeatureKey;
    config.plasticity.probe.batch_size = 2;
    auto env_spec = MakeLearnerEnvSpec();
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    learner.EnablePlasticityForward(true);
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->SetSizeValues({ 2 });
    replay_buffer->unique_sample_enabled = true;
    learner.UseReplayBuffer(replay_buffer);

    learner.ConfigureScalarMetricSubscriptions({
        rl::ScalarMetricSubscription{
            .source_key = "plasticity_srank",
            .event = rl::EventType::LEARN,
            .interval = 2,
            .scope = rl::RunnerScope::TRAIN,
        },
        rl::ScalarMetricSubscription{
            .source_key = "plasticity_feature_norm",
            .event = rl::EventType::LEARN,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN,
        },
        rl::ScalarMetricSubscription{
            .source_key = "plasticity_target_srank",
            .event = rl::EventType::LEARN,
            .interval = 3,
            .scope = rl::RunnerScope::TRAIN,
        },
        rl::ScalarMetricSubscription{
            .source_key = "plasticity_probe_srank",
            .event = rl::EventType::LEARN,
            .target = rl::EventField::AGENT,
            .interval = 2,
            .scope = rl::RunnerScope::TRAIN,
        },
    });

    const auto first = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(first.size() == 1);
    CHECK(std::isfinite(*first.front()->GetScalar("plasticity_srank")));
    CHECK(std::isfinite(*first.front()->GetScalar("plasticity_target_srank")));
    CHECK(std::isfinite(*learner.GetScalar("plasticity_probe_srank")));
    CHECK(replay_buffer->unique_sample_count == 1);

    const auto second = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(second.size() == 1);
    CHECK(std::isnan(*second.front()->GetScalar("plasticity_srank")));
    CHECK(std::isfinite(*second.front()->GetScalar("plasticity_feature_norm")));
    CHECK(std::isnan(*second.front()->GetScalar("plasticity_target_srank")));
    CHECK(std::isnan(*learner.GetScalar("plasticity_probe_srank")));
    CHECK(replay_buffer->unique_sample_count == 1);

    const auto third = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(third.size() == 1);
    CHECK(std::isfinite(*third.front()->GetScalar("plasticity_srank")));
    CHECK(std::isnan(*third.front()->GetScalar("plasticity_target_srank")));
    CHECK(std::isfinite(*learner.GetScalar("plasticity_probe_srank")));
    CHECK(replay_buffer->unique_sample_count == 2);

    replay_buffer->unique_sample_enabled = false;
    const auto fourth = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(fourth.size() == 1);
    CHECK(std::isnan(*learner.GetScalar("plasticity_probe_srank")));
    const auto fifth = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(fifth.size() == 1);
    CHECK(std::isnan(*learner.GetScalar("plasticity_probe_srank")));
    CHECK(replay_buffer->unique_sample_count == 3);
}

TEST_CASE("DQN plasticity probe survives later updates in the same LearnEvent",
    "[dqn][plasticity][demand]")
{
    auto probe_state = std::make_shared<AutocastProbeState>();
    AutocastProbeNetworkModel model(probe_state, torch::Device(torch::kCPU));
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = 8.0f;
    config.final_feature_branch_key = kFeatureKey;
    config.plasticity.probe.batch_size = 2;
    TestLearner learner(
        config, model, vars, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), dqn::LearnerForwardResources{});
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->SetSizeValues({ 2 });
    replay_buffer->unique_sample_enabled = true;
    learner.UseReplayBuffer(replay_buffer);
    learner.ConfigureScalarMetricSubscriptions({ rl::ScalarMetricSubscription{
        .source_key = "plasticity_probe_srank",
        .event = rl::EventType::LEARN,
        .target = rl::EventField::AGENT,
        .interval = 8,
        .scope = rl::RunnerScope::TRAIN,
    } });

    // 1 event 内の4 updateの先頭だけで測定しても、後続3 update後まで値を保持する。
    const auto first = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(first.size() == 4);
    CHECK(std::isfinite(*learner.GetScalar("plasticity_probe_srank")));
    CHECK(replay_buffer->unique_sample_count == 1);

    // 次の event に測定stepが無ければ、前eventの値を再利用しない。
    const auto second = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(second.size() == 4);
    CHECK(std::isnan(*learner.GetScalar("plasticity_probe_srank")));
    CHECK(replay_buffer->unique_sample_count == 1);

    const auto third = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
    REQUIRE(third.size() == 4);
    CHECK(std::isfinite(*learner.GetScalar("plasticity_probe_srank")));
    CHECK(replay_buffer->unique_sample_count == 2);
}

TEST_CASE("DQN plasticity srank interval reduces SVD executions independently of capture cadence",
    "[dqn][plasticity][demand]")
{
    const auto count_svd = [](const int srank_interval) {
        auto probe_state = std::make_shared<AutocastProbeState>();
        AutocastProbeNetworkModel model(probe_state, torch::Device(torch::kCPU));
        dqn::RuntimeVars vars;
        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        config.replay_batch_size = 2;
        config.update_warmup_steps = 0;
        config.update_interval = 1;
        config.replay_ratio = 2.0f;
        config.final_feature_branch_key = kFeatureKey;
        TestLearner learner(
            config, model, vars, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), dqn::LearnerForwardResources{});
        learner.EnablePlasticityForward(true);
        auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
        replay_buffer->SetSizeValues({ 2 });
        learner.UseReplayBuffer(replay_buffer);
        learner.ConfigureScalarMetricSubscriptions({
            rl::ScalarMetricSubscription{
                .source_key = "plasticity_feature_norm",
                .event = rl::EventType::LEARN,
                .interval = 1,
                .scope = rl::RunnerScope::TRAIN,
            },
            rl::ScalarMetricSubscription{
                .source_key = "plasticity_srank",
                .event = rl::EventType::LEARN,
                .interval = srank_interval,
                .scope = rl::RunnerScope::TRAIN,
            },
        });

        return CountProfileOperator([&] {
            for (int step = 0; step < 10; ++step) {
                const auto results = learner.UpdateFromBatch(
                    rl::StepCounts{}, rl::BatchExperience{});
                REQUIRE(results.size() == 1);
                CHECK(std::isfinite(*results.front()->GetScalar("plasticity_feature_norm")));
                const auto srank = results.front()->GetScalar("plasticity_srank");
                REQUIRE(srank.has_value());
                if (step % srank_interval == 0) {
                    CHECK(std::isfinite(*srank));
                } else {
                    CHECK(std::isnan(*srank));
                }
            }
        }, "aten::linalg_svdvals");
    };

    CHECK(count_svd(1) == 10);
    CHECK(count_svd(10) == 1);
}

TEST_CASE("TD Munchausen target adds one bonus and masks only bootstrap", "[dqn][learner][munchausen][tracer]")
{
    const std::string mode = GENERATE("target", "online", "online_reuse");
    CAPTURE(mode);
    // 設定読取から実Learner更新までを通し、未知キーを無視する旧実装では数値で失敗させる。
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.learner.munchausen.enabled", true);
    data.Set("DefaultDQNAgent.learner.munchausen.entropy_tau", 0.5f);
    data.Set("DefaultDQNAgent.learner.munchausen.log_policy_mode", mode);
    data.Set("DefaultDQNAgent.learner.munchausen.clip_value_min", -10.0f);
    data.Set("DefaultDQNAgent.learner.use_double_dqn", false);
    auto config = dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(data)).learner;
    config.current.nn_forward_key.clear();
    config.target.nn_forward_key.clear();
    config.learning_rate.value = 0.0f;
    config.use_fused_optimizer = false;
    config.use_grad_clip = false;
    config.use_td_clip = false;
    config.use_per = false;
    config.replay_capacity = 8;
    config.replay_batch_size = 2;
    config.gamma.value = 0.9f;
    TestNetworkModel model(false, 2);
    {
        torch::NoGradGuard guard;
        model.GetOnlineNetwork()->parameters().at(0).copy_(torch::eye(2));
        model.GetTargetNetwork()->parameters().at(0).copy_(2.0f * torch::eye(2));
    }
    auto spec = MakeLearnerEnvSpec();
    spec.action_spec.value_labels = { "a0", "a1" };
    dqn::RuntimeVars vars;
    auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{});
    dqn::TDLearner learner(config, model, vars, nullptr, rl::BatchEnvSpec{ 2, 2 }, spec,
        torch::kCPU, 67002, kTestAgentSeed, policy, dqn::LearnerForwardResources{},
        std::nullopt, 67003);
    auto samples = MakeAutocastProbeSamples(torch::kCPU);
    samples.n_steps = torch::tensor({ 3, 2 }, torch::kInt64);
    samples.next_state.terminals = torch::tensor({ false, true });
    const auto result = learner.UpdateFromSamples(samples);
    const double score_gap = mode == "target" ? 2.0 : 1.0;
    const double bonus = 0.9 * (-score_gap - 0.5 * std::log1p(std::exp(-score_gap / 0.5)));
    const double next_soft = 2.0 + 0.5 * std::log1p(std::exp(-2.0));
    CHECK(result->td_error[0].item<float>() == Catch::Approx(1.0 - (0.1 + bonus + std::pow(0.9, 3) * next_soft)).margin(1e-5));
    CHECK(result->td_error[1].item<float>() == Catch::Approx(3.0 - (0.2 + bonus)).margin(1e-5));
    const auto diagnostic = result->GetScalar("munchausen_bonus_mean");
    REQUIRE(diagnostic.has_value());
    CHECK(*diagnostic == Catch::Approx(bonus).margin(1e-5));
}

TEST_CASE("DQN Learner uses selected contexts and separate boxes for current and target", "[dqn][noisynet][learner]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const bool sample_current = GENERATE(false, true);
    const bool sample_target = GENERATE(false, true);
    const bool munchausen = GENERATE(false, true);
    CAPTURE(sample_current, sample_target, munchausen);
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.mode", "sample");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.hold", "call");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.share", "sample");
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", sample_current ? "sample" : "test_default");
    data.Set("DefaultDQNAgent.learner.target.nn_forward_key", sample_target ? "sample" : "test_default");
    data.Set("net.block.[Linear].type", "Linear");
    data.Set("net.block.[Linear].linear.out_features", 2);
    data.Set("net.block.[Linear].noisy.enabled", true);
    data.Set("net.branch.[feature].bind", kVectorKey);
    data.Set("net.branch.[feature].structure", "Linear");
    data.Set("net.body.output.[features]", "feature");
    const auto env_spec = MakeLearnerEnvSpec();
    const auto records = std::make_shared<std::vector<ForwardContextRecord>>();
    const auto factory = std::make_shared<ForwardContextRecordingHeadFactory>(records);
    dqn::NetworkModel model(dqn::NetworkModelConfig{}, torch::kCPU, anet::nn::NetworkConfig(data),
        env_spec.state_spec.obs_spec, 1, factory, false, 301);
    const auto online = model.GetOnlineNetwork();
    const auto target = model.GetTargetNetwork();
    auto config = dqn::DefaultDQNAgentConfig(data).learner;
    config.replay_batch_size = 2;
    config.replay_capacity = 16;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.use_rb_prefetch = false;
    config.use_n_step = false;
    config.use_per = false;
    config.use_fused_optimizer = false;
    config.use_amp = false;
    config.use_td_clip = false;
    config.munchausen.enabled = munchausen;
    if (munchausen) config.use_double_dqn = false;
    dqn::RuntimeVars vars;
    dqn::TDLearner learner(config, model, vars, nullptr, rl::BatchEnvSpec{2, 1}, env_spec,
        torch::kCPU, 302, kTestAgentSeed, std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), SelectLearnerForwardResources(data),
        std::nullopt, 303);
    std::map<std::string, torch::Tensor> before;
    for (const auto& entry : online->named_parameters()) {
        if (entry.key().find("sigma_") != std::string::npos) before.emplace(entry.key(), entry.value().clone());
    }
    REQUIRE(before.size() == 2);
    size_t updates = 0;
    records->clear();
    for (rl::step_t step = 0; step < 3; ++step) {
        auto experience = MakeDeterminismExperience(step, 2);
        // current/next が同じ観測なら、同じ遷移への epsilon 共有を特徴出力でも確認できる。
        experience.state.obs = anet::TensorDict{{kVectorKey, torch::ones({2, 2})}};
        experience.next_state.obs = experience.state.obs.Clone();
        updates += learner.UpdateFromBatch(rl::StepCounts{.exp_step = step}, experience).size();
    }
    REQUIRE(updates > 0);
    CHECK(vars.learn_step == updates);
    for (const auto& entry : online->named_parameters()) {
        if (!before.contains(entry.key())) continue;
        CAPTURE(entry.key());
        CHECK(entry.value().grad().defined() == sample_current);
        CHECK(torch::equal(entry.value(), before.at(entry.key())) == !sample_current);
    }
    std::set<const anet::nn::NetworkForwardState*> current_states;
    std::set<const anet::nn::NetworkForwardState*> target_states;
    std::set<uint64_t> current_clocks;
    std::set<uint64_t> target_clocks;
    for (const auto& record : *records) {
        const auto selected = record.training ? config.current.nn_forward_key : config.target.nn_forward_key;
        CHECK(record.key == "DefaultDQNAgent.nn_forward.[" + selected + "]");
        CHECK(record.state != nullptr);
        CHECK(record.clock < updates);
        if (munchausen && !record.training) {
            CHECK(record.row_mapping.defined());
            if (record.row_mapping.defined()) CHECK(torch::equal(record.row_mapping, torch::tensor({0, 1, 0, 1}, torch::kInt64)));
            REQUIRE(record.features.size(0) == 4);
            CHECK(torch::allclose(record.features.narrow(0, 0, 2), record.features.narrow(0, 2, 2)));
        }
        (record.training ? current_states : target_states).insert(record.state);
        (record.training ? current_clocks : target_clocks).insert(record.clock);
    }
    REQUIRE(current_states.size() == 1);
    REQUIRE(target_states.size() == (munchausen ? 1 : 2));
    for (const auto* state : target_states) CHECK(*current_states.begin() != state);
    CHECK(current_clocks.size() == updates);
    CHECK(target_clocks.size() == updates);

    // target に sample を指定しても、NoGrad の target 構築では sigma の勾配を作らない。
    for (const auto& entry : target->named_parameters()) {
        if (before.contains(entry.key())) CHECK_FALSE(entry.value().grad().defined());
    }
}

TEST_CASE("Decomposed target plasticity captures only bootstrap rows", "[rfd0121][dqn][plasticity]")
{
    ScopedNoopMetricsLogger logger;
    anet::test::RegisterStandardNetworkModules();
    auto data = MakeNoisyOptimizerTestData();
    data.Set("net.block.[Linear].noisy.enabled", false);
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.intrinsic.decomposed = true;
    config.quantile_mode = anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone;
    config.munchausen.enabled = true;
    config.munchausen.log_policy_mode = kRfd0121TargetLogPolicyMode;
    config.final_feature_branch_key = "feature";
    config.current.nn_forward_key.clear();
    config.target.nn_forward_key.clear();
    const auto env = MakeLearnerEnvSpec();
    const auto factory = std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{});
    auto head = std::make_shared<dqn::DecomposedHeadFactory>(factory, factory, false, false,
        anet::ConfigFullKeyTable::FormatKeyNotReadFromConfig("input_mode"));
    dqn::NetworkModel model({}, torch::kCPU, anet::nn::NetworkConfig(data), env.state_spec.obs_spec, 1, head, false, 121);
    dqn::RuntimeVars vars;
    auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{});
    dqn::TDLearner learner(config, model, vars, nullptr, rl::BatchEnvSpec{1, 1}, env,
        torch::kCPU, 121, kTestAgentSeed, policy, dqn::LearnerForwardResources{});
    learner.ConfigureScalarMetricSubscriptions({{.source_key = kRfd0121PlasticityTargetSrank,
        .event = rl::EventType::LEARN, .interval = 1, .scope = rl::RunnerScope::TRAIN}});
    bool updated = false;
    for (rl::step_t step = 0; step < 4; ++step) {
        auto experience = MakeNoisyOptimizerExperience(step, 1);
        experience.reward = torch::ones({1, 2});
        for (const auto& base : learner.UpdateFromBatch({.exp_step = step}, experience, 0.5f)) {
            const auto result = std::dynamic_pointer_cast<const dqn::BatchUpdateResult>(base);
            REQUIRE(result);
            REQUIRE(result->plasticity_target_features.defined());
            CHECK(result->plasticity_target_features.size(0) == config.replay_batch_size);
            updated = true;
        }
    }
    CHECK(updated);
}

TEST_CASE("Learner checkpoint applies current optimizer options and preserves state", "[dqn][learner][serialize][rfd0091]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const bool noisy = GENERATE(false, true);
    const bool source_fused = GENERATE(false, true);
    const bool destination_fused = GENERATE(false, true);
    const bool disabled = GENERATE(false, true);
    auto data = MakeNoisyOptimizerTestData();
    data.Set("net.block.[Linear].noisy.enabled", noisy);
    const auto env_spec = MakeLearnerEnvSpec();
    const auto factory = std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{});
    dqn::NetworkModel source({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec,
        1, factory, false, 491);
    dqn::NetworkModel restored({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec,
        1, factory, false, 492);
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.learning_rate.value = 1e-4f;
    config.weight_decay.value = 0.01f;
    config.adam_eps = 1e-5f;
    config.use_fused_optimizer = source_fused;
    const bool same_options = GENERATE(false, true);
    const bool change_defaults = GENERATE(false, true);
    auto current = config;
    current.learning_rate.value = 5e-5f;
    current.weight_decay.value = 0.02f;
    current.adam_eps = 1e-4f;
    current.noisy_weight_decay = 0.03f;
    if (same_options) current = config;
    const auto expected_learning_rate = current.learning_rate.value;
    const auto expected_weight_decay = current.weight_decay.value;
    std::string lr_key = "DefaultDQNAgent.learner.learning_rate";
    std::string decay_key = "DefaultDQNAgent.learner.weight_decay";
    if (change_defaults) {
        // 引継ぎ先の時間の値は、optimizerに残っていた値と独立したstep 0から始める。
        auto scheduled = data;
        scheduled.Set(lr_key + ".type", "linear");
        scheduled.Set(lr_key + ".start", expected_learning_rate);
        scheduled.Set(lr_key + ".end", 0.0f);
        scheduled.Set(lr_key + ".steps", 8);
        scheduled.Set(decay_key + ".type", "phased");
        scheduled.Set(decay_key + ".phases", "initial final");
        scheduled.Set(decay_key + ".phase.[initial].type", "linear");
        scheduled.Set(decay_key + ".phase.[initial].start", expected_weight_decay);
        scheduled.Set(decay_key + ".phase.[initial].end", 0.0f);
        scheduled.Set(decay_key + ".phase.[initial].steps", 8);
        scheduled.Set(decay_key + ".phase.[final].type", "constant");
        scheduled.Set(decay_key + ".phase.[final].value", 0.0f);
        scheduled.Set(decay_key + ".phase.[final].steps", 8);
        const auto schedule_config = MakeNoisyOptimizerLearnerConfig(scheduled);
        current.learning_rate = schedule_config.learning_rate;
        current.weight_decay = schedule_config.weight_decay;
        lr_key += ".start";
        decay_key += ".phase.[initial].start";
    }
    current.use_fused_optimizer = destination_fused;
    current.enabled = !disabled;
    dqn::RuntimeVars source_vars, restored_vars;
    TestLearner first(config, source, source_vars, rl::BatchEnvSpec{1, 1}, env_spec, SelectLearnerForwardResources(data));
    TestLearner second(current, restored, restored_vars, rl::BatchEnvSpec{1, 1}, env_spec, SelectLearnerForwardResources(data));
    first.SetupAdamW();
    second.SetupAdamW();

    // 全 parameter に非ゼロ勾配を与え、checkpoint が引き継ぐ moment と step を作る。
    for (auto& group : first.Optimizer().param_groups()) {
        for (auto& parameter : group.params()) parameter.mutable_grad() = torch::ones_like(parameter);
    }
    if (change_defaults) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(first.Optimizer().param_groups()[0].options());
        options.betas(std::make_tuple(0.8, 0.99));
        options.amsgrad(true);
    }
    first.Optimizer().step();
    std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(bytes);
    REQUIRE(first.Save(output) > 0);
    bytes.seekg(0);
    anet::InputArchive input(bytes);
    anet::test::LogCaptureGuard logs;
    CHECK(second.Load(input) > 0);
    logs.Flush();

    // 差のある項目だけを通知し、適用後の値を必ず1行で示す。
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Message)
        == (noisy ? 2 : 1) + (same_options ? 0 : (noisy ? 6 : 3)) + (change_defaults ? 2 : 0));
    if (!same_options) {
        CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
            {"Learner::Load:", "group=0", "option=lr", "key=" + lr_key + " ", "checkpoint=0.0001", "config=5e-05"}));
        CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
            {"group=0", "option=weight_decay", "key=" + decay_key + " ", "checkpoint=0.01", "config=0.02"}));
        CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
            {"group=0", "option=eps", "key=DefaultDQNAgent.learner.adam_eps", "checkpoint=1e-05", "config=0.0001"}));
    }
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
        {"Learner::Load: optimizer options applied from config.", "group=0",
            lr_key + (same_options ? "=0.0001" : "=5e-05"),
            decay_key + (same_options ? "=0.01" : "=0.02"),
            same_options ? "DefaultDQNAgent.learner.adam_eps=1e-05" : "DefaultDQNAgent.learner.adam_eps=0.0001"}));
    if (noisy) {
        CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
            {"optimizer options applied from config.", "group=1",
                same_options ? "DefaultDQNAgent.learner.noisy_weight_decay=0" : "DefaultDQNAgent.learner.noisy_weight_decay=0.03"}));
        if (!same_options) {
            CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
                {"group=1", "option=lr", "key=" + lr_key + " ", "checkpoint=0.0001", "config=5e-05"}));
            CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
                {"group=1", "option=eps", "key=DefaultDQNAgent.learner.adam_eps", "checkpoint=1e-05", "config=0.0001"}));
            CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
                {"group=1", "option=weight_decay", "key=DefaultDQNAgent.learner.noisy_weight_decay", "checkpoint=0", "config=0.03"}));
        }
    }
    if (change_defaults) {
        CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
            {"group=0", "option=betas", "checkpoint=(0.8,0.99)", "config=(0.9,0.999)"}));
        CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
            {"group=0", "option=amsgrad", "checkpoint=1", "config=0"}));
        for (const auto& record : logs.Records()) {
            if (record.message.find("option=betas") != std::string::npos
                || record.message.find("option=amsgrad") != std::string::npos) {
                CHECK(record.message.find("key=") == std::string::npos);
            }
        }
    }
    CHECK(bytes.tellg() == static_cast<std::streamoff>(bytes.str().size()));

    // options は現在の設定、state は保存時の値であり、別 parameter へ正しく対応付けられる。
    const auto& saved_groups = first.Optimizer().param_groups();
    const auto& loaded_groups = second.Optimizer().param_groups();
    REQUIRE(loaded_groups.size() == (noisy ? 2 : 1));
    for (size_t group_index = 0; group_index < loaded_groups.size(); ++group_index) {
        const auto& options = static_cast<const torch::optim::AdamWOptions&>(loaded_groups[group_index].options());
        CHECK(options.lr() == static_cast<double>(expected_learning_rate));
        CHECK(options.weight_decay() == static_cast<double>(group_index == 0 ? expected_weight_decay : current.noisy_weight_decay));
        CHECK(options.eps() == static_cast<double>(current.adam_eps));
        CHECK(options.betas() == torch::optim::AdamWOptions{}.betas());
        CHECK_FALSE(options.amsgrad());
        REQUIRE(loaded_groups[group_index].params().size() == saved_groups[group_index].params().size());
        for (size_t i = 0; i < loaded_groups[group_index].params().size(); ++i) {
            const auto& saved = static_cast<const torch::optim::AdamWParamState&>(
                *first.Optimizer().state().at(saved_groups[group_index].params()[i].unsafeGetTensorImpl()));
            const auto& loaded = static_cast<const torch::optim::AdamWParamState&>(
                *second.Optimizer().state().at(loaded_groups[group_index].params()[i].unsafeGetTensorImpl()));
            CHECK(saved.step() == 1);
            CHECK(loaded.step() == saved.step());
            CHECK(saved.exp_avg().abs().sum().item<float>() > 0.0f);
            CHECK(torch::equal(loaded.exp_avg(), saved.exp_avg()));
            CHECK(torch::equal(loaded.exp_avg_sq(), saved.exp_avg_sq()));
            if (change_defaults && group_index == 0) CHECK(torch::equal(loaded.max_exp_avg_sq(), saved.max_exp_avg_sq()));
        }
    }
}

TEST_CASE("Noisy sigma has independent AdamW decay and freezes without a gradient", "[dqn][noisynet][optimizer]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const bool fused = GENERATE(false, true);
    const float sigma_decay = GENERATE(0.0f, 0.4f);
    auto data = MakeNoisyOptimizerTestData();
    data.Set("DefaultDQNAgent.learner.noisy_weight_decay", sigma_decay);
    const auto env_spec = MakeLearnerEnvSpec();
    dqn::NetworkModel model({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec,
        1, std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{}), false, 411);
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    CHECK(config.noisy_weight_decay == Catch::Approx(sigma_decay));
    config.learning_rate.value = 0.1f;
    config.weight_decay.value = 0.8f;
    config.use_fused_optimizer = fused;
    const auto entries = model.GetOnlineNetwork()->GetNoisyParameterEntries();
    REQUIRE(entries.size() == 2);
    // HeadをゼロにしてQとTD誤差をゼロにする。Bodyのμは非ゼロなので独立decayを観測できる。
    {
        torch::NoGradGuard no_grad;
        for (auto& parameter : model.GetOnlineParameters()) parameter.zero_();
        for (const auto& entry : entries) {
            entry.mean.fill_(0.25f);
            entry.sigma.fill_(0.5f);
        }
    }
    std::vector<torch::Tensor> mean_before, sigma_before;
    for (const auto& entry : entries) {
        mean_before.push_back(entry.mean.detach().clone());
        sigma_before.push_back(entry.sigma.detach().clone());
    }
    dqn::RuntimeVars vars;
    dqn::TDLearner learner(config, model, vars, nullptr, rl::BatchEnvSpec{1, 1}, env_spec,
        torch::kCPU, 412, kTestAgentSeed, std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), SelectLearnerForwardResources(data),
        std::nullopt, 413);
    // ReplayBufferの先頭状態を公開経路で投入してから、最初の学習可能な遷移を更新する。
    REQUIRE(learner.UpdateFromBatch(rl::StepCounts{}, MakeNoisyOptimizerExperience(0, 0.0f)).empty());
    const auto results = learner.UpdateFromBatch(rl::StepCounts{.exp_step=1}, MakeNoisyOptimizerExperience(1, 0.0f));
    REQUIRE(results.size() == 1);
    REQUIRE(results.front()->GetScalar("loss").has_value());
    CHECK(*results.front()->GetScalar("loss") == 0.0f);
    for (size_t i = 0; i < entries.size(); ++i) {
        REQUIRE(entries[i].sigma.grad().defined());
        CHECK(entries[i].sigma.grad().abs().sum().item<float>() == 0.0f);
        CHECK(torch::allclose(entries[i].mean, mean_before[i] * (1.0f - config.learning_rate.value * config.weight_decay.value), 1e-6, 1e-7));
        CHECK(torch::allclose(entries[i].sigma, sigma_before[i] * (1.0f - config.learning_rate.value * sigma_decay), 1e-6, 1e-7));
        sigma_before[i] = entries[i].sigma.detach().clone();
    }
    // 公開Save/Loadで直前のoptimizer状態を引き継ぎ、μ-only更新がσの勾配とdecayを止めることを確認する。
    std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(bytes);
    const auto saved_size = learner.Save(output);
    auto mean_data = data;
    mean_data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "test_default");
    auto mean_config = MakeNoisyOptimizerLearnerConfig(mean_data);
    mean_config.learning_rate.value = config.learning_rate.value;
    mean_config.weight_decay.value = config.weight_decay.value;
    mean_config.use_fused_optimizer = fused;
    dqn::RuntimeVars mean_vars;
    dqn::TDLearner mean_learner(mean_config, model, mean_vars, nullptr, rl::BatchEnvSpec{1, 1}, env_spec,
        torch::kCPU, 414, kTestAgentSeed, std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), SelectLearnerForwardResources(mean_data),
        std::nullopt, 415);
    bytes.seekg(0);
    anet::InputArchive input(bytes);
    REQUIRE(saved_size > 0);
    CHECK(mean_learner.Load(input) > 0);
    CHECK(bytes.tellg() == static_cast<std::streamoff>(bytes.str().size()));
    REQUIRE(mean_learner.UpdateFromBatch(rl::StepCounts{}, MakeNoisyOptimizerExperience(0, 1.0f)).empty());
    REQUIRE(mean_learner.UpdateFromBatch(rl::StepCounts{.exp_step=1}, MakeNoisyOptimizerExperience(1, 1.0f)).size() == 1);
    for (size_t i = 0; i < entries.size(); ++i) {
        CHECK_FALSE(entries[i].sigma.grad().defined());
        CHECK(torch::equal(entries[i].sigma, sigma_before[i]));
    }
}

TEST_CASE("Noisy statistics subscriptions capture pre-update values without running a probe", "[dqn][noisynet][statistics][demand]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const bool enabled = GENERATE(false, true);
    auto data = WithDefaultForwardConfig();
    data.Set("net.block.[Linear].type", "Linear");
    data.Set("net.block.[Linear].linear.out_features", 2);
    data.Set("net.block.[Linear].noisy.enabled", enabled);
    data.Set("net.branch.[feature].bind", kVectorKey);
    data.Set("net.branch.[feature].structure", "Linear");
    data.Set("net.body.output.[features]", "feature");
    anet::ConfigData projection;
    projection.Set("noisy.enabled", enabled);
    const auto env_spec = MakeLearnerEnvSpec();
    dqn::NetworkModel model({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec,
        1, std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{}, projection), false, 421);
    auto config = dqn::DefaultDQNAgentConfig(data).learner;
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.replay_ratio = 2;
    config.final_feature_branch_key = "feature";
    dqn::RuntimeVars vars;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{1, 1}, env_spec, SelectLearnerForwardResources(data));
    learner.EnablePlasticityForward(false);
    learner.MutateOnlineParametersDuringUpdate(1.0f);
    auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({2});
    learner.UseReplayBuffer(replay);
    const std::array<const char*, 8> keys{
        "noisy_sigma_mean_abs_feature_weight", "noisy_sigma_mean_abs_feature_bias",
        "noisy_sigma_mean_abs_readout_weight", "noisy_sigma_mean_abs_readout_bias",
        "noisy_sigma_l2_ratio_feature_weight", "noisy_sigma_l2_ratio_feature_bias",
        "noisy_sigma_l2_ratio_readout_weight", "noisy_sigma_l2_ratio_readout_bias"};
    std::vector<rl::ScalarMetricSubscription> subscriptions;
    for (const auto key : keys) {
        const dqn::BatchUpdateResult empty{};
        REQUIRE(empty.GetScalar(key).has_value());
        CHECK(std::isnan(*empty.GetScalar(key)));
        subscriptions.push_back({.source_key=key, .event=rl::EventType::LEARN,
            .target=rl::EventField::UPDATE_RESULT, .interval=2, .scope=rl::RunnerScope::TRAIN});
    }
    learner.ConfigureScalarMetricSubscriptions(subscriptions);
    for (int step = 0; step < 3; ++step) {
        const auto expected = model.GetOnlineNetwork()->ComputeNoisyParameterStatistics("feature").cpu();
        const auto results = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
        REQUIRE(results.size() == 1);
        for (int64_t metric = 0; metric < 8; ++metric) {
            const auto value = results.front()->GetScalar(keys[metric]);
            REQUIRE(value.has_value());
            if (step % 2 || std::isnan(expected[metric].item<float>())) CHECK(std::isnan(*value));
            else CHECK(*value == Catch::Approx(expected[metric].item<float>()));
        }
        CHECK(std::isnan(*results.front()->GetScalar("noisy_q_abs_diff")));
        CHECK_FALSE(results.front()->GetScalar("noisy_unknown").has_value());
    }
    CHECK(replay->unique_sample_count == 0);
}

TEST_CASE("Noisy probe uses dedicated contexts with batch sharing and subscription gating", "[dqn][noisynet][probe][demand]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const bool enabled = GENERATE(false, true);
    const bool sufficient = GENERATE(false, true);
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.learner.noisy_probe.probe.batch_size", 2);
    data.Set("DefaultDQNAgent.learner.noisy_probe.num_draws", 4);
    data.Set("DefaultDQNAgent.learner.noisy_probe.iqn.num_taus", 5);
    data.Set("net.block.[Linear].type", "Linear");
    data.Set("net.block.[Linear].linear.out_features", 2);
    data.Set("net.block.[Linear].noisy.enabled", enabled);
    data.Set("net.branch.[feature].bind", kVectorKey);
    data.Set("net.branch.[feature].structure", "Linear");
    data.Set("net.body.output.[features]", "feature");
    const auto env_spec = MakeLearnerEnvSpec();
    auto records = std::make_shared<std::vector<ForwardContextRecord>>();
    dqn::NetworkModel model({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec,
        1, std::make_shared<ForwardContextRecordingHeadFactory>(records), false, 431);
    auto config = dqn::DefaultDQNAgentConfig(data).learner;
    CHECK(config.noisy_probe.probe.batch_size == 2);
    CHECK(config.noisy_probe.num_draws == 4);
    CHECK(config.noisy_probe.iqn.num_taus == 5);
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.replay_ratio = 2;
    dqn::RuntimeVars vars;
    anet::ConfigData probe_data;
    probe_data.Set("nn_forward.[probe].noisy.mode", "sample");
    probe_data.Set("nn_forward.[probe].noisy.hold", "call");
    probe_data.Set("nn_forward.[probe].noisy.share", "batch");
    const anet::nn::NetworkForwardConfig probe_config("nn_forward.[probe]", SelectForwardConfigData(probe_data, "nn_forward.[probe]"));

    TestLearner learner(config, model, vars, rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, env_spec, SelectLearnerForwardResources(data), std::make_shared<const anet::nn::NetworkForwardConfig>(probe_config));
    learner.EnablePlasticityForward(false);
    auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({2});
    replay->vector_value = 1.0f;
    replay->unique_sample_enabled = sufficient;
    learner.UseReplayBuffer(replay);
    std::vector<rl::ScalarMetricSubscription> subscriptions;
    for (const auto key : {"noisy_q_abs_diff", "noisy_action_disagreement_ratio", "noisy_action_entropy"}) {
        subscriptions.push_back({.source_key=key, .event=rl::EventType::LEARN,
            .target=rl::EventField::UPDATE_RESULT, .interval=2, .scope=rl::RunnerScope::TRAIN});
    }
    learner.ConfigureScalarMetricSubscriptions(subscriptions);
    for (int step = 0; step < 3; ++step) {
        records->clear();
        const auto result = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
        const bool measured = enabled && sufficient && step % 2 == 0;
        REQUIRE(records->size() == (measured ? 6 : 1));
        for (const auto key : {"noisy_q_abs_diff", "noisy_action_disagreement_ratio", "noisy_action_entropy"}) {
            REQUIRE(result->GetScalar(key).has_value());
            CHECK(std::isfinite(*result->GetScalar(key)) == measured);
        }
        if (measured) {
            CHECK(*result->GetScalar("noisy_q_abs_diff") > 0.0f);
            CHECK(*result->GetScalar("noisy_action_disagreement_ratio") == 0.0f);
            CHECK(*result->GetScalar("noisy_action_entropy") == 0.0f);
            CHECK(records->front().state == nullptr);
            CHECK(records->front().key == "default");
            const auto box = (*records)[1].state;
            REQUIRE(box != nullptr);
            for (size_t draw = 1; draw <= 4; ++draw) {
                const auto& record = (*records)[draw];
                CHECK(record.key == probe_config.key);
                CHECK(record.state == box);
                CHECK_FALSE(record.training);
                CHECK(record.features.scalar_type() == torch::kFloat32);
                CHECK(torch::equal(record.features[0], record.features[1]));
                if (draw > 1) CHECK(record.clock == (*records)[draw-1].clock + 1);
            }
        }
    }
    CHECK(replay->unique_sample_count == (enabled ? 2 : 0));
}

TEST_CASE("Noisy diagnostics preserve TD QR IQN updates and spectral buffers across target modes", "[dqn][noisynet][probe][determinism][target]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const auto mode = GENERATE(std::string("none"), std::string("qr"), std::string("iqn"));
    const bool dueling = GENERATE(false, true);
    const auto munch = GENERATE(std::string("disabled"), std::string("target"), std::string("online"), std::string("online_reuse"));
    CAPTURE(mode, dueling, munch);
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.quantile_mode", mode);
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.mode", "sample");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.hold", "call");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.share", "sample");
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "sample");
    data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "sample");
    data.Set("net.block.[Feature].type", "Linear");
    data.Set("net.block.[Feature].linear.out_features", 4);
    data.Set("net.block.[Feature].weight_norm.mode", "spectral");
    data.Set("net.block.[Feature].noisy.enabled", true);
    data.Set("net.branch.[feature].bind", kVectorKey);
    data.Set("net.branch.[feature].structure", "Feature");
    data.Set("net.body.output.[features]", "feature");
    auto specs = MakeLearnerEnvSpec().state_spec.obs_spec;
    if (mode == "iqn") {
        data.Set("net.block.[Cos].type", "CosineEmbedding");
        data.Set("net.block.[Cos].cos.num_basis", 4);
        data.Set("net.block.[Tau].type", "Linear");
        data.Set("net.block.[Tau].linear.out_features", 4);
        data.Set("net.branch.[tau].bind", anet::rl::dqn::kKey_Taus);
        data.Set("net.branch.[tau].structure", "Cos > Tau");
        data.Set("net.branch.[fusion].bind", "feature * tau");
        data.Set("net.body.output.[features]", "fusion");
        specs[anet::rl::dqn::kKey_Taus] = anet::TensorSpec{.type=anet::SpaceType::Vector, .shape={4}, .dtype=torch::kFloat32};
    }
    anet::ConfigData projection;
    projection.Set("noisy.enabled", true);
    std::shared_ptr<anet::nn::NetworkHeadFactory> factory;
    const anet::nn::WeightInitConfig init{};
    if (mode == "none") {
        if (dueling) factory = std::make_shared<dqn::DuelingHeadFactory>(2, init, projection);
        else factory = std::make_shared<dqn::LinearHeadFactory>(2, init, projection);
    } else if (mode == "qr") {
        if (dueling) factory = std::make_shared<dqn::QuantileDuelingHeadFactory>(2, 5, init, projection);
        else factory = std::make_shared<dqn::QuantileHeadFactory>(2, 5, init, projection);
    } else {
        if (dueling) factory = std::make_shared<dqn::IQNDuelingHeadFactory>(2, init, projection);
        else factory = std::make_shared<dqn::IQNHeadFactory>(2, init, projection);
    }
    dqn::NetworkModelConfig model_config{};
    model_config.hard_update_interval = 2;
    torch::manual_seed(441);
    dqn::NetworkModel first(model_config, torch::kCPU, anet::nn::NetworkConfig(data), specs, 2, factory, mode != "none", 441);
    torch::manual_seed(441);
    dqn::NetworkModel second(model_config, torch::kCPU, anet::nn::NetworkConfig(data), specs, 2, factory, mode != "none", 441);
    auto config = dqn::DefaultDQNAgentConfig(data).learner;
    config.quantile_mode = mode;
    config.num_quantiles = 5;
    config.iqn.current_taus.num_taus = 4;
    config.iqn.target_taus.num_taus = 4;
    config.final_feature_branch_key = "feature";
    config.noisy_probe.probe.batch_size = 2;
    config.noisy_probe.num_draws = 3;
    config.noisy_probe.iqn.num_taus = 4;
    config.replay_batch_size = 2;
    config.replay_capacity = 32;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.use_per = false;
    config.use_n_step = false;
    config.use_rb_prefetch = false;
    config.use_fused_optimizer = false;
    config.use_amp = false;
    config.use_grad_clip = true;
    config.grad_clip_tau = 0.7f;
    config.noisy_weight_decay = 0.1f;
    config.munchausen.enabled = munch != "disabled";
    config.munchausen.log_policy_mode = munch == "disabled" ? "target" : munch;
    config.use_double_dqn = !config.munchausen.enabled;
    auto env_spec = MakeLearnerEnvSpec();
    env_spec.action_spec.value_labels = {"a0", "a1"};
    dqn::RuntimeVars first_vars, second_vars;
    anet::ConfigData probe_data;
    probe_data.Set("probe.noisy.mode", "sample");
    probe_data.Set("probe.noisy.hold", "call");
    probe_data.Set("probe.noisy.share", "batch");
    const anet::nn::NetworkForwardConfig probe_config("probe", SelectForwardConfigData(probe_data, "probe"));

    auto make_learner = [&](dqn::NetworkModel& model, dqn::RuntimeVars& vars) -> std::unique_ptr<dqn::Learner> {
        dqn::ActionPolicyConfig policy_config;
        policy_config.quantile_mode = mode;
        policy_config.tau_rule.num_taus = 4;
        auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(policy_config);
        if (mode == "none") return std::make_unique<dqn::TDLearner>(config, model, vars, nullptr,
            rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env_spec, torch::kCPU, 442, kTestAgentSeed, policy, SelectLearnerForwardResources(data), std::nullopt, 443, std::make_shared<const anet::nn::NetworkForwardConfig>(probe_config));
        if (mode == "qr") return std::make_unique<dqn::QRLearner>(config, model, vars, nullptr,
            rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env_spec, torch::kCPU, 442, kTestAgentSeed, policy, SelectLearnerForwardResources(data), std::nullopt, 443, std::make_shared<const anet::nn::NetworkForwardConfig>(probe_config));
        return std::make_unique<dqn::IQNLearner>(config, model, vars, nullptr,
            rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env_spec, torch::kCPU, 442, kTestAgentSeed, policy, SelectLearnerForwardResources(data), std::nullopt, 443, std::make_shared<const anet::nn::NetworkForwardConfig>(probe_config));
    };
    auto measured = make_learner(first, first_vars);
    auto ordinary = make_learner(second, second_vars);
    std::vector<rl::ScalarMetricSubscription> subscriptions;
    for (const auto key : dqn::kNoisyMetricKeys) {
        subscriptions.push_back({.source_key=key, .event=rl::EventType::LEARN,
            .target=rl::EventField::UPDATE_RESULT, .interval=1, .scope=rl::RunnerScope::TRAIN});
    }
    measured->ConfigureScalarMetricSubscriptions(subscriptions);
    ordinary->ConfigureScalarMetricSubscriptions({});
    auto compare_networks = [](const auto& lhs, const auto& rhs) {
        const auto other_parameters = rhs->named_parameters();
        for (const auto& entry : lhs->named_parameters()) {
            CAPTURE(entry.key());
            CHECK(torch::equal(entry.value(), other_parameters[entry.key()]));
            if (entry.value().grad().defined()) CHECK(torch::equal(entry.value().grad(), other_parameters[entry.key()].grad()));
        }
        const auto other_buffers = rhs->named_buffers();
        for (const auto& entry : lhs->named_buffers()) CHECK(torch::equal(entry.value(), other_buffers[entry.key()]));
    };
    compare_networks(first.GetOnlineNetwork(), second.GetOnlineNetwork());
    compare_networks(first.GetTargetNetwork(), second.GetTargetNetwork());
    size_t probes = 0;
    for (rl::step_t step = 0; step < 7; ++step) {
        const auto experience = MakeDeterminismExperience(step, 2);
        const auto left = measured->UpdateFromBatch(rl::StepCounts{.exp_step=step}, experience);
        const auto right = ordinary->UpdateFromBatch(rl::StepCounts{.exp_step=step}, experience);
        REQUIRE(left.size() == right.size());
        for (size_t update = 0; update < left.size(); ++update) {
            for (const auto key : {"loss", "grad_norm", "q_sa_mean", "td_mean"}) {
                const auto a = left[update]->GetScalar(key);
                const auto b = right[update]->GetScalar(key);
                REQUIRE(a.has_value() == b.has_value());
                if (a && b) CHECK((*a == *b || (std::isnan(*a) && std::isnan(*b))));
            }
            const auto value = left[update]->GetScalar("noisy_q_abs_diff");
            REQUIRE(value.has_value());
            if (std::isfinite(*value)) ++probes;
            CHECK(std::isnan(*right[update]->GetScalar("noisy_q_abs_diff")));
        }
        CHECK(first_vars.learn_step == second_vars.learn_step);
        compare_networks(first.GetOnlineNetwork(), second.GetOnlineNetwork());
        compare_networks(first.GetTargetNetwork(), second.GetTargetNetwork());
    }
    CHECK(probes >= 2);
}

TEST_CASE("Global gradient norm and clipping include every defined Noisy sigma gradient", "[dqn][noisynet][optimizer][clip]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const bool fused = GENERATE(false, true);
    const auto data = MakeNoisyOptimizerTestData();
    const auto env_spec = MakeLearnerEnvSpec();
    const auto factory = std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{});
    dqn::NetworkModel reference({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec,
        1, factory, false, 451);
    dqn::NetworkModel clipped({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec,
        1, factory, false, 452);
    // 両更新へ同じ重み・入力・抽選列を渡し、clipの有無だけを比較する。
    {
        torch::NoGradGuard no_grad;
        for (auto& parameter : reference.GetOnlineParameters()) parameter.fill_(1.0f);
    }
    reference.GetOnlineNetwork()->CopyTo(*clipped.GetOnlineNetwork());
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.learning_rate.value = 0.1f;
    config.weight_decay.value = 0.0f;
    config.noisy_weight_decay = 0.0f;
    config.use_fused_optimizer = fused;
    dqn::RuntimeVars reference_vars, clipped_vars;
    dqn::TDLearner reference_learner(config, reference, reference_vars, nullptr, rl::BatchEnvSpec{1, 1}, env_spec,
        torch::kCPU, 453, kTestAgentSeed, std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), SelectLearnerForwardResources(data),
        std::nullopt, 454);
    config.use_grad_clip = true;
    config.grad_clip_tau = 0.5f;
    dqn::TDLearner clipped_learner(config, clipped, clipped_vars, nullptr, rl::BatchEnvSpec{1, 1}, env_spec,
        torch::kCPU, 453, kTestAgentSeed, std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), SelectLearnerForwardResources(data),
        std::nullopt, 454);
    REQUIRE(reference_learner.UpdateFromBatch(rl::StepCounts{}, MakeNoisyOptimizerExperience(0, 100.0f)).empty());
    REQUIRE(clipped_learner.UpdateFromBatch(rl::StepCounts{}, MakeNoisyOptimizerExperience(0, 100.0f)).empty());
    const auto reference_results = reference_learner.UpdateFromBatch(rl::StepCounts{.exp_step=1},
        MakeNoisyOptimizerExperience(1, 100.0f));
    const auto clipped_results = clipped_learner.UpdateFromBatch(rl::StepCounts{.exp_step=1},
        MakeNoisyOptimizerExperience(1, 100.0f));
    REQUIRE(reference_results.size() == 1);
    REQUIRE(clipped_results.size() == 1);
    // 公開parameterの勾配を独立に集計し、更新結果の全体normがσを含むことを確認する。
    float squared = 0.0f;
    for (const auto& parameter : reference.GetOnlineParameters()) {
        REQUIRE(parameter.grad().defined());
        squared += parameter.grad().square().sum().item<float>();
    }
    float sigma_squared = 0.0f;
    const auto entries = reference.GetOnlineNetwork()->GetNoisyParameterEntries();
    REQUIRE(entries.size() == 2);
    for (const auto& entry : entries) {
        REQUIRE(entry.sigma.grad().defined());
        sigma_squared += entry.sigma.grad().square().sum().item<float>();
    }
    REQUIRE(sigma_squared > 0.0f);
    const float norm = std::sqrt(squared);
    REQUIRE(norm > config.grad_clip_tau);
    for (const auto& results : {reference_results, clipped_results}) {
        const auto measured_norm = results.front()->GetScalar("grad_norm");
        REQUIRE(measured_norm.has_value());
        CHECK(*measured_norm == Catch::Approx(norm).epsilon(1e-5));
    }
    // 全勾配へ同じ係数が掛かり、σを含めた実際のnormが上限へ収まることを確認する。
    const auto clipped_parameters = clipped.GetOnlineNetwork()->named_parameters();
    const float coefficient = config.grad_clip_tau / (norm + 1e-6f);
    float clipped_squared = 0.0f;
    for (const auto& entry : reference.GetOnlineNetwork()->named_parameters()) {
        REQUIRE(clipped_parameters.contains(entry.key()));
        const auto gradient = clipped_parameters[entry.key()].grad();
        REQUIRE(gradient.defined());
        CHECK(torch::allclose(gradient, entry.value().grad() * coefficient, 1e-5, 1e-6));
        clipped_squared += gradient.square().sum().item<float>();
    }
    CHECK(std::sqrt(clipped_squared) == Catch::Approx(config.grad_clip_tau).epsilon(1e-5));
}

TEST_CASE("Noisy probe compares a prepared sample against the default Head execution", "[dqn][noisynet][probe][context]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    auto data = WithDefaultForwardConfig();
    data.Set("net.block.[Linear].type", "Linear");
    data.Set("net.block.[Linear].linear.out_features", 2);
    data.Set("net.block.[Linear].noisy.enabled", true);
    data.Set("net.block.[Linear].noisy.sigma0", 0.0f);
    data.Set("net.branch.[feature].bind", kVectorKey);
    data.Set("net.branch.[feature].structure", "Linear");
    data.Set("net.body.output.[features]", "feature");
    auto records = std::make_shared<std::vector<ForwardContextRecord>>();
    const auto env_spec = MakeLearnerEnvSpec();
    dqn::NetworkModel model({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec,
        1, std::make_shared<ForwardContextRecordingHeadFactory>(records), false, 461);
    auto config = dqn::DefaultDQNAgentConfig(data).learner;
    config.noisy_probe.probe.batch_size = 2;
    config.noisy_probe.num_draws = 3;
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.replay_ratio = 2;
    dqn::RuntimeVars vars;
    anet::ConfigData probe_data;
    probe_data.Set("probe.noisy.mode", "sample");
    probe_data.Set("probe.noisy.hold", "call");
    probe_data.Set("probe.noisy.share", "batch");
    probe_data.Set("probe.probe_test.scale", 7.0f);
    const anet::nn::NetworkForwardConfig probe_config("probe", SelectForwardConfigData(probe_data, "probe"));

    TestLearner learner(config, model, vars, rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, env_spec, SelectLearnerForwardResources(data), std::make_shared<const anet::nn::NetworkForwardConfig>(probe_config));
    learner.EnablePlasticityForward(false);
    auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({2});
    replay->unique_sample_enabled = true;
    replay->vector_value = 1.0f;
    learner.UseReplayBuffer(replay);
    learner.ConfigureScalarMetricSubscriptions({{.source_key="noisy_q_abs_diff", .event=rl::EventType::LEARN,
        .target=rl::EventField::UPDATE_RESULT, .interval=1, .scope=rl::RunnerScope::TRAIN}});
    const auto result = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
    REQUIRE(result->GetScalar("noisy_q_abs_diff").has_value());
    // 平均側は既定のHeadを使い、sample側だけが選択された独自設定を読む。
    CHECK(*result->GetScalar("noisy_q_abs_diff") > 0.0f);
    CHECK(records->front().key == "default");
    CHECK(records->front().state == nullptr);
    CHECK((*records)[1].key == probe_config.key);
}

TEST_CASE("Noisy optimizer serialization preserves the next update with identical settings", "[dqn][noisynet][optimizer][serialize]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const bool fused = GENERATE(false, true);
    const auto data = MakeNoisyOptimizerTestData();
    const auto env_spec = MakeLearnerEnvSpec();
    const auto factory = std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{});
    dqn::NetworkModel source({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec,
        1, factory, false, 471);
    dqn::NetworkModel restored({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec,
        1, factory, false, 472);
    source.GetOnlineNetwork()->CopyTo(*restored.GetOnlineNetwork());
    source.GetTargetNetwork()->CopyTo(*restored.GetTargetNetwork());
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.learning_rate.value = 0.1f;
    config.weight_decay.value = 0.8f;
    config.noisy_weight_decay = 0.4f;
    config.use_fused_optimizer = fused;
    dqn::RuntimeVars source_vars, restored_vars;
    dqn::TDLearner first(config, source, source_vars, nullptr, rl::BatchEnvSpec{1, 1}, env_spec,
        torch::kCPU, 473, kTestAgentSeed, std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), SelectLearnerForwardResources(data),
        std::nullopt, 474);
    dqn::TDLearner second(config, restored, restored_vars, nullptr, rl::BatchEnvSpec{1, 1}, env_spec,
        torch::kCPU, 473, kTestAgentSeed, std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), SelectLearnerForwardResources(data),
        std::nullopt, 474);
    // 両方を同じ公開更新回数と抽選列へ進め、両groupの非ゼロmomentを作る。
    for (rl::step_t step = 0; step < 3; ++step) {
        const auto experience = MakeNoisyOptimizerExperience(step, step == 2 ? -100.0f : 100.0f);
        const auto expected = step == 0 ? 0 : 1;
        REQUIRE(first.UpdateFromBatch(rl::StepCounts{.exp_step=step}, experience).size() == expected);
        REQUIRE(second.UpdateFromBatch(rl::StepCounts{.exp_step=step}, experience).size() == expected);
    }
    REQUIRE(source_vars.learn_step == 2);
    REQUIRE(restored_vars.learn_step == source_vars.learn_step);
    for (const auto& entry : source.GetOnlineNetwork()->GetNoisyParameterEntries()) {
        REQUIRE(entry.mean.grad().defined());
        REQUIRE(entry.sigma.grad().defined());
        CHECK(entry.mean.grad().abs().sum().item<float>() > 0.0f);
        CHECK(entry.sigma.grad().abs().sum().item<float>() > 0.0f);
    }
    // NetworkModelとLearnerの公開archive経路でparameterとoptimizerをまとめて復元する。
    std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(bytes);
    REQUIRE(source.Save(output) > 0);
    REQUIRE(first.Save(output) > 0);
    bytes.seekg(0);
    anet::InputArchive input(bytes);
    CHECK(restored.Load(input) > 0);
    CHECK(second.Load(input) > 0);
    CHECK(bytes.tellg() == static_cast<std::streamoff>(bytes.str().size()));
    // 完全なRun再開は検証しない。同一設定の次の遷移で、引き継いだstep・momentの効果を観測する。
    const auto experience = MakeNoisyOptimizerExperience(3, 100.0f);
    REQUIRE(first.UpdateFromBatch(rl::StepCounts{.exp_step=3}, experience).size() == 1);
    REQUIRE(second.UpdateFromBatch(rl::StepCounts{.exp_step=3}, experience).size() == 1);
    const auto parameters = restored.GetOnlineNetwork()->named_parameters();
    for (const auto& entry : source.GetOnlineNetwork()->named_parameters()) {
        REQUIRE(parameters.contains(entry.key()));
        CHECK(torch::equal(entry.value(), parameters[entry.key()]));
    }
}

TEST_CASE("Learner rejects missing forward resources", "[dqn][noisynet][forward-resource]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    auto config = WithDefaultLearnerForwardConfig();
    const auto purpose = GENERATE(std::string("current"), std::string("target"));
    (purpose == "current" ? config.current : config.target).nn_forward_key = "sample";
    config.replay_capacity = 16;
    config.replay_batch_size = 2;
    config.use_rb_prefetch = false;
    config.use_fused_optimizer = false;
    const auto construct = [&] {
        return std::make_unique<dqn::TDLearner>(config, model, vars, nullptr,
            rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, MakeLearnerEnvSpec(), torch::kCPU, 801, kTestAgentSeed,
            std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}), dqn::LearnerForwardResources{},
            std::nullopt, 802);
    };
    CHECK_THROWS_WITH(construct(), Catch::Matchers::ContainsSubstring(purpose + ".nn_forward_key (not read from config)"));
}

TEST_CASE("Learner forward resources match selections before preparation",
    "[dqn][noisynet][forward-resource]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    auto config = WithDefaultLearnerForwardConfig();
    config.replay_capacity = 16;
    config.replay_batch_size = 2;
    config.use_rb_prefetch = false;
    config.use_fused_optimizer = false;
    const auto scenario = GENERATE(std::string("named"), std::string("default"),
        std::string("swapped"), std::string("wrong_current"), std::string("wrong_target"),
        std::string("empty_current"), std::string("empty_target"), std::string("disabled"));
    INFO(scenario);
    config.current.nn_forward_key = "current";
    config.target.nn_forward_key = "target";
    std::string current_key = "current";
    std::string target_key = "target";
    std::string mismatch;
    if (scenario == "default") {
        config.current.nn_forward_key.clear();
        config.target.nn_forward_key.clear();
        current_key.clear();
        target_key.clear();
    } else if (scenario == "swapped") {
        std::swap(current_key, target_key);
        mismatch = "current";
    } else if (scenario == "wrong_current") {
        current_key = "other";
        mismatch = "current";
    } else if (scenario == "wrong_target") {
        target_key = "other";
        mismatch = "target";
    } else if (scenario == "empty_current") {
        config.current.nn_forward_key.clear();
        mismatch = "current";
    } else if (scenario == "empty_target") {
        config.target.nn_forward_key.clear();
        mismatch = "target";
    } else if (scenario == "disabled") {
        config.enabled = false;
        current_key = "wrong";
        target_key.clear();
    }
    // 設定内容には触れず、選択キーとResourceの同一用途への対応だけを検証する。
    const auto resource = [](const std::string& key) {
        return key.empty() ? anet::nn::NetworkForwardConfig::GetDefault()
            : anet::nn::NetworkForwardConfig("catalog." + dqn::MakeNetworkForwardConfigKey(key), anet::ConfigData{});
    };
    const dqn::LearnerForwardResources resources{
        .current = resource(current_key), .target = resource(target_key)};
    const auto construct = [&] {
        return std::make_unique<dqn::TDLearner>(config, model, vars, nullptr,
            rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, MakeLearnerEnvSpec(), torch::kCPU, 811, kTestAgentSeed,
            std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}),
            resources, std::nullopt, 812);
    };
    if (mismatch.empty()) {
        CHECK_NOTHROW(construct());
    } else {
        const auto& selection = mismatch == "current" ? config.current : config.target;
        const auto& supplied = mismatch == "current" ? resources.current : resources.target;
        CHECK_THROWS_WITH(construct(),
            Catch::Matchers::ContainsSubstring(mismatch + ".nn_forward_key (not read from config)=" + selection.nn_forward_key)
            && Catch::Matchers::ContainsSubstring("supplied resource=" + supplied.key));
    }
}

TEST_CASE("Gradient noise raw values belong only to subscribed update results", "[dqn][gradnoise]")
{
    REQUIRE(dqn::BatchUpdateResult{}.GetScalar("grad_noise_signal_raw").has_value());
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const auto data = MakeNoisyOptimizerTestData();
    const auto env_spec = MakeLearnerEnvSpec();
    dqn::NetworkModel model({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec, 1,
        std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{}), false, 920);
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.replay_batch_size = 8;
    config.replay_ratio = 8;
    dqn::RuntimeVars vars;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, SelectLearnerForwardResources(data));
    learner.SetupAdamW();
    learner.EnablePolicyChurnOptimizerUpdate();
    auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({ 8 });
    learner.UseReplayBuffer(replay);
    learner.ConfigureScalarMetricSubscriptions({ { .source_key = "grad_noise_signal_raw",
        .event = rl::EventType::LEARN,
        .target = rl::EventField::UPDATE_RESULT,
        .interval = 2,
        .scope = rl::RunnerScope::TRAIN } });
    for (int step = 0; step < 3; ++step) {
        const auto results = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
        REQUIRE(results.size() == 1);
        const auto value = results.front()->GetScalar("grad_noise_signal_raw");
        REQUIRE(value.has_value());
        if (step % 2)
            CHECK(std::isnan(*value));
        else
            CHECK(std::isfinite(*value));
        CHECK_FALSE(results.front()->GetScalar("grad_noise_unknown").has_value());
    }
}

TEST_CASE("Gradient noise shares measurement points and publishes all fifteen keys at their own cadence", "[dqn][gradnoise]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const auto data = MakeNoisyOptimizerTestData();
    const auto env_spec = MakeLearnerEnvSpec();
    dqn::NetworkModel model({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec, 1,
        std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{}), false, 921);
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.replay_batch_size = 8;
    config.replay_ratio = 8;
    config.final_feature_branch_key = "feature";
    config.grad_noise.num_micro_batches = 2;
    config.grad_noise.ema_alpha = 1;
    dqn::RuntimeVars vars;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, SelectLearnerForwardResources(data));
    learner.SetupAdamW();
    learner.EnablePolicyChurnOptimizerUpdate();
    auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({ 8 });
    learner.UseReplayBuffer(replay);
    const std::vector<std::string> keys{ "grad_noise_scale_adam", "grad_noise_scale_adam_feature", "grad_noise_scale_adam_readout",
        "grad_noise_ratio_adam", "grad_noise_scale", "grad_noise_signal_adam", "grad_noise_variance_adam", "grad_noise_signal_adam_raw",
        "grad_noise_variance_adam_raw", "grad_noise_signal_raw", "grad_noise_variance_raw", "grad_noise_signal_adam_feature_raw",
        "grad_noise_variance_adam_feature_raw", "grad_noise_signal_adam_readout_raw", "grad_noise_variance_adam_readout_raw" };
    std::vector<rl::ScalarMetricSubscription> subscriptions;
    for (const auto& key : keys)
        subscriptions.push_back({ .source_key = key,
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = key == "grad_noise_signal_raw" ? 2 : 1,
            .scope = rl::RunnerScope::TRAIN });
    learner.ConfigureScalarMetricSubscriptions(subscriptions);
    for (int step = 0; step < 4; ++step) {
        const auto results = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
        REQUIRE(results.size() == 1);
        for (size_t i = 0; i < keys.size(); ++i) {
            CAPTURE(step, keys[i]);
            const auto value = results.front()->GetScalar(keys[i]);
            REQUIRE(value.has_value());
            if ((step == 0 && i != 4 && i != 9 && i != 10) || (step % 2 && i == 9))
                CHECK(std::isnan(*value));
            else
                CHECK(std::isfinite(*value));
        }
        if (step > 0) {
            const auto value = [&](const char* key) { return *results.front()->GetScalar(key); };
            CHECK(value("grad_noise_signal_adam_raw") ==
                  Catch::Approx(value("grad_noise_signal_adam_feature_raw") + value("grad_noise_signal_adam_readout_raw")).margin(1e-4));
            CHECK(value("grad_noise_ratio_adam") == Catch::Approx(value("grad_noise_scale_adam") / 8));
        }
    }
}

TEST_CASE("Gradient noise is noninterfering through TD QR IQN FP32 and CUDA BF16 updates", "[dqn][gradnoise][noninterference]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const auto mode = GENERATE(std::string("none"), std::string("qr"), std::string("iqn"));
    const bool dueling = true;
    const bool fused = GENERATE(false, true);
    const bool bf16 = GENERATE(false, true);
    if (bf16 && !torch::cuda::is_available()) SKIP("CUDA BF16 unavailable");
    const torch::Device device = bf16 ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    const auto munch = GENERATE(std::string("disabled"), std::string("online_reuse"));
    CAPTURE(mode, dueling, munch, bf16, fused);
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.quantile_mode", mode);
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.mode", "sample");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.hold", "call");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.share", "batch");
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "sample");
    data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "sample");
    data.Set("net.block.[Feature].type", "Linear");
    data.Set("net.block.[Feature].linear.out_features", 4);
    data.Set("net.block.[Feature].weight_norm.mode", "spectral");
    data.Set("net.block.[Feature].noisy.enabled", true);
    data.Set("net.branch.[feature].bind", kVectorKey);
    data.Set("net.branch.[feature].structure", "Feature");
    data.Set("net.body.output.[features]", "feature");
    auto specs = MakeLearnerEnvSpec().state_spec.obs_spec;
    if (mode == "iqn") {
        data.Set("net.block.[Cos].type", "CosineEmbedding");
        data.Set("net.block.[Cos].cos.num_basis", 4);
        data.Set("net.block.[Tau].type", "Linear");
        data.Set("net.block.[Tau].linear.out_features", 4);
        data.Set("net.branch.[tau].bind", anet::rl::dqn::kKey_Taus);
        data.Set("net.branch.[tau].structure", "Cos > Tau");
        data.Set("net.branch.[fusion].bind", "feature * tau");
        data.Set("net.body.output.[features]", "fusion");
        specs[anet::rl::dqn::kKey_Taus] = anet::TensorSpec{ .type = anet::SpaceType::Vector, .shape = { 4 }, .dtype = torch::kFloat32 };
    }
    anet::ConfigData projection;
    projection.Set("noisy.enabled", true);
    std::shared_ptr<anet::nn::NetworkHeadFactory> factory;
    const anet::nn::WeightInitConfig init{};
    if (mode == "none") {
        if (dueling)
            factory = std::make_shared<dqn::DuelingHeadFactory>(2, init, projection);
        else
            factory = std::make_shared<dqn::LinearHeadFactory>(2, init, projection);
    } else if (mode == "qr") {
        if (dueling)
            factory = std::make_shared<dqn::QuantileDuelingHeadFactory>(2, 5, init, projection);
        else
            factory = std::make_shared<dqn::QuantileHeadFactory>(2, 5, init, projection);
    } else {
        if (dueling)
            factory = std::make_shared<dqn::IQNDuelingHeadFactory>(2, init, projection);
        else
            factory = std::make_shared<dqn::IQNHeadFactory>(2, init, projection);
    }
    dqn::NetworkModelConfig model_config{};
    model_config.hard_update_interval = 2;
    torch::manual_seed(441);
    dqn::NetworkModel first(model_config, device, anet::nn::NetworkConfig(data), specs, 2, factory, mode != "none", 441);
    torch::manual_seed(441);
    dqn::NetworkModel second(model_config, device, anet::nn::NetworkConfig(data), specs, 2, factory, mode != "none", 441);
    auto config = dqn::DefaultDQNAgentConfig(data).learner;
    config.quantile_mode = mode;
    config.num_quantiles = 5;
    config.iqn.current_taus.num_taus = 4;
    config.iqn.target_taus.num_taus = 4;
    config.final_feature_branch_key = "feature";
    config.noisy_probe.probe.batch_size = 2;
    config.noisy_probe.num_draws = 3;
    config.noisy_probe.iqn.num_taus = 4;
    config.replay_batch_size = 2;
    config.replay_capacity = 32;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.use_per = true;
    config.use_fused_optimizer = fused;
    config.use_amp = bf16;
    config.use_amp_bf16 = bf16;
    config.grad_noise.num_micro_batches = 2;
    config.use_n_step = false;
    config.use_rb_prefetch = false;
    config.use_grad_clip = true;
    config.grad_clip_tau = 0.7f;
    config.noisy_weight_decay = 0.1f;
    config.munchausen.enabled = munch != "disabled";
    config.munchausen.log_policy_mode = munch == "disabled" ? "target" : munch;
    config.use_double_dqn = !config.munchausen.enabled;
    auto env_spec = MakeLearnerEnvSpec();
    env_spec.action_spec.value_labels = { "a0", "a1" };
    dqn::RuntimeVars first_vars, second_vars;
    anet::ConfigData probe_data;
    probe_data.Set("probe.noisy.mode", "sample");
    probe_data.Set("probe.noisy.hold", "call");
    probe_data.Set("probe.noisy.share", "batch");
    const anet::nn::NetworkForwardConfig probe_config("probe", SelectForwardConfigData(probe_data, "probe"));

    auto make_learner = [&](dqn::NetworkModel& model, dqn::RuntimeVars& vars) -> std::unique_ptr<dqn::Learner> {
        dqn::ActionPolicyConfig policy_config;
        policy_config.quantile_mode = mode;
        policy_config.tau_rule.num_taus = 4;
        auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(policy_config);
        if (mode == "none")
            return std::make_unique<dqn::TDLearner>(config, model, vars, nullptr, rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env_spec, device, 442, kTestAgentSeed, policy,
                SelectLearnerForwardResources(data), std::nullopt, 443, std::make_shared<const anet::nn::NetworkForwardConfig>(probe_config));
        if (mode == "qr")
            return std::make_unique<dqn::QRLearner>(config, model, vars, nullptr, rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env_spec, device, 442, kTestAgentSeed, policy,
                SelectLearnerForwardResources(data), std::nullopt, 443, std::make_shared<const anet::nn::NetworkForwardConfig>(probe_config));
        return std::make_unique<dqn::IQNLearner>(config, model, vars, nullptr, rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env_spec, device, 442, kTestAgentSeed, policy,
            SelectLearnerForwardResources(data), std::nullopt, 443, std::make_shared<const anet::nn::NetworkForwardConfig>(probe_config));
    };
    auto measured = make_learner(first, first_vars);
    auto ordinary = make_learner(second, second_vars);
    std::vector<rl::ScalarMetricSubscription> subscriptions;
    for (const auto key : dqn::kGradientNoiseMetricKeys) {
        subscriptions.push_back({ .source_key = key,
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN });
    }
    measured->ConfigureScalarMetricSubscriptions(subscriptions);
    ordinary->ConfigureScalarMetricSubscriptions({});
    auto compare_networks = [](const auto& lhs, const auto& rhs) {
        const auto other_parameters = rhs->named_parameters();
        for (const auto& entry : lhs->named_parameters()) {
            CAPTURE(entry.key());
            CHECK(torch::equal(entry.value(), other_parameters[entry.key()]));
            if (entry.value().grad().defined()) CHECK(torch::equal(entry.value().grad(), other_parameters[entry.key()].grad()));
        }
        const auto other_buffers = rhs->named_buffers();
        for (const auto& entry : lhs->named_buffers())
            CHECK(torch::equal(entry.value(), other_buffers[entry.key()]));
    };
    const auto snapshot_optimizer = [](dqn::Learner& learner, dqn::NetworkModel& model) {
        std::unordered_set<const c10::TensorImpl*> sigma;
        for (const auto& entry : model.GetOnlineNetwork()->GetNoisyParameterEntries())
            sigma.insert(entry.sigma.unsafeGetTensorImpl());
        std::vector<torch::Tensor> ordinary, noisy;
        for (const auto& parameter : model.GetOnlineParameters())
            (sigma.contains(parameter.unsafeGetTensorImpl()) ? noisy : ordinary).push_back(parameter);
        std::vector<torch::optim::OptimizerParamGroup> groups;
        groups.emplace_back(ordinary);
        groups.emplace_back(noisy);
        auto optimizer = std::make_unique<torch::optim::AdamW>(groups, torch::optim::AdamWOptions(0.001));
        std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
        anet::OutputArchive output(stream);
        learner.Save(output);
        stream.seekg(0);
        anet::InputArchive input(stream);
        input.ReadTorchObject(*optimizer);
        return optimizer;
    };
    compare_networks(first.GetOnlineNetwork(), second.GetOnlineNetwork());
    compare_networks(first.GetTargetNetwork(), second.GetTargetNetwork());
    size_t probes = 0;
    for (rl::step_t step = 0; step < 7; ++step) {
        const auto experience = MakeDeterminismExperience(step, 2);
        const auto left = measured->UpdateFromBatch(rl::StepCounts{ .exp_step = step }, experience);
        const auto right = ordinary->UpdateFromBatch(rl::StepCounts{ .exp_step = step }, experience);
        REQUIRE(left.size() == right.size());
        for (size_t update = 0; update < left.size(); ++update) {
            for (const auto key : { "loss", "grad_norm", "q_sa_mean", "td_mean" }) {
                const auto a = left[update]->GetScalar(key);
                const auto b = right[update]->GetScalar(key);
                REQUIRE(a.has_value() == b.has_value());
                if (a && b) CHECK((*a == *b || (std::isnan(*a) && std::isnan(*b))));
            }
            const auto value = left[update]->GetScalar("grad_noise_signal_raw");
            REQUIRE(value.has_value());
            if (std::isfinite(*value)) ++probes;
            CHECK(std::isnan(*right[update]->GetScalar("grad_noise_signal_raw")));
        }
        CHECK(torch::equal(measured->GetRandomGenerator()->GetTorchGenerator(device).get_state(),
            ordinary->GetRandomGenerator()->GetTorchGenerator(device).get_state()));
        CHECK(measured->GetRandomGenerator()->RandUint64() == ordinary->GetRandomGenerator()->RandUint64());
        const auto left_optimizer = snapshot_optimizer(*measured, first);
        const auto right_optimizer = snapshot_optimizer(*ordinary, second);
        REQUIRE(left_optimizer->state().size() == right_optimizer->state().size());
        for (size_t group = 0; group < left_optimizer->param_groups().size(); ++group) {
            const auto& left_parameters = left_optimizer->param_groups()[group].params();
            const auto& right_parameters = right_optimizer->param_groups()[group].params();
            for (size_t i = 0; i < left_parameters.size(); ++i) {
                const auto left_state = left_optimizer->state().find(left_parameters[i].unsafeGetTensorImpl());
                const auto right_state = right_optimizer->state().find(right_parameters[i].unsafeGetTensorImpl());
                REQUIRE((left_state == left_optimizer->state().end()) == (right_state == right_optimizer->state().end()));
                if (left_state == left_optimizer->state().end()) continue;
                const auto& a = static_cast<const torch::optim::AdamWParamState&>(*left_state->second);
                const auto& b = static_cast<const torch::optim::AdamWParamState&>(*right_state->second);
                CHECK(a.step() == b.step());
                CHECK(torch::equal(a.exp_avg(), b.exp_avg()));
                CHECK(torch::equal(a.exp_avg_sq(), b.exp_avg_sq()));
            }
        }
        CHECK(first_vars.learn_step == second_vars.learn_step);
        compare_networks(first.GetOnlineNetwork(), second.GetOnlineNetwork());
        compare_networks(first.GetTargetNetwork(), second.GetTargetNetwork());
    }
    CHECK(probes >= 2);
}

TEST_CASE("Gradient noise EMA follows the union once including simultaneous firing", "[dqn][gradnoise][cadence]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const auto data = MakeNoisyOptimizerTestData();
    const auto env_spec = MakeLearnerEnvSpec();
    const auto factory = std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{});
    dqn::NetworkModel measured_model({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec, 1, factory, false, 930);
    dqn::NetworkModel reference_model({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec, 1, factory, false, 930);
    measured_model.GetOnlineNetwork()->CopyTo(*reference_model.GetOnlineNetwork());
    measured_model.GetTargetNetwork()->CopyTo(*reference_model.GetTargetNetwork());
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.replay_batch_size = 8;
    config.replay_ratio = 8;
    config.grad_noise.num_micro_batches = 2;
    config.grad_noise.ema_alpha = 0.5;
    dqn::RuntimeVars measured_vars, reference_vars;
    TestLearner measured(config, measured_model, measured_vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, SelectLearnerForwardResources(data));
    TestLearner reference(config, reference_model, reference_vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, SelectLearnerForwardResources(data));
    auto measured_replay = std::make_shared<RecordingReplayBuffer>();
    auto reference_replay = std::make_shared<RecordingReplayBuffer>();
    for (const auto& replay : { measured_replay, reference_replay })
        replay->SetSizeValues({ 8 });
    measured.UseReplayBuffer(measured_replay);
    reference.UseReplayBuffer(reference_replay);
    measured.SetupAdamW();
    reference.SetupAdamW();
    measured.EnablePolicyChurnOptimizerUpdate();
    reference.EnablePolicyChurnOptimizerUpdate();
    const auto subscription = [](const char* key, int interval) {
        return rl::ScalarMetricSubscription{ .source_key = key,
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = interval,
            .scope = rl::RunnerScope::TRAIN };
    };
    measured.ConfigureScalarMetricSubscriptions(
        { subscription("grad_noise_signal_adam", 3), subscription("grad_noise_signal_adam_raw", 2) });
    reference.ConfigureScalarMetricSubscriptions(
        { subscription("grad_noise_signal_adam_raw", 1), subscription("grad_noise_variance_adam_raw", 1) });
    anet::EmaFilter<double> expected(0.5);
    for (int step = 0; step < 8; ++step) {
        measured_replay->vector_value = reference_replay->vector_value = static_cast<float>(step + 1);
        const auto actual = measured.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
        const auto dense = reference.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
        const auto raw = *dense->GetScalar("grad_noise_signal_adam_raw");
        if ((step % 2 == 0 || step % 3 == 0) && std::isfinite(raw)) expected.Update(raw);
        const auto smooth = *actual->GetScalar("grad_noise_signal_adam");
        if (step % 3 == 0 && expected.IsInitialized())
            CHECK(smooth == Catch::Approx(expected.Value()).epsilon(1e-5));
        else
            CHECK(std::isnan(smooth));
        CHECK(std::isnan(*actual->GetScalar("grad_noise_signal_adam_raw")) == (step == 0 || step % 2 != 0));
    }
}

TEST_CASE("Gradient noise publishes NaN scales for a nonpositive smoothed signal", "[dqn][gradnoise]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const bool cancel_micro_batches = GENERATE(true, false);
    CAPTURE(cancel_micro_batches);
    const auto data = MakeNoisyOptimizerTestData();
    const auto env_spec = MakeLearnerEnvSpec();
    dqn::NetworkModel model({}, torch::kCPU, anet::nn::NetworkConfig(data), env_spec.state_spec.obs_spec, 1,
        std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{}), false, 931);
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.replay_batch_size = 8;
    config.replay_ratio = 8;
    config.grad_noise.num_micro_batches = 2;
    config.grad_noise.ema_alpha = 1;
    dqn::RuntimeVars vars;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, SelectLearnerForwardResources(data));
    learner.SetupAdamW();
    learner.EnablePolicyChurnOptimizerUpdate();
    auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({ 8 });
    learner.UseReplayBuffer(replay);
    // 同一観測の勾配を小batch間で相殺し、全行同符号の対照と区別する。
    auto scale = torch::ones({ 8 });
    if (cancel_micro_batches) scale.narrow(0, 4, 4).fill_(-1);
    learner.SetOptimizerElementLossScale(scale);
    std::vector<rl::ScalarMetricSubscription> subscriptions;
    for (const auto* key : { "grad_noise_scale", "grad_noise_signal_raw", "grad_noise_variance_raw", "grad_noise_scale_adam",
             "grad_noise_ratio_adam", "grad_noise_signal_adam", "grad_noise_variance_adam" }) {
        subscriptions.push_back({ .source_key = key,
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN });
    }
    learner.ConfigureScalarMetricSubscriptions(subscriptions);
    // 初回更新でAdam stateを形成し、以後は公開されたraw・EMA・scaleを一緒に確認する。
    for (int step = 0; step < 3; ++step) {
        const auto results = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{});
        REQUIRE(results.size() == 1);
        const auto value = [&](const char* key) {
            const auto scalar = results.front()->GetScalar(key);
            REQUIRE(scalar.has_value());
            return *scalar;
        };
        const auto signal = value("grad_noise_signal_raw");
        const auto variance = value("grad_noise_variance_raw");
        REQUIRE(std::isfinite(signal));
        REQUIRE(std::isfinite(variance));
        if (cancel_micro_batches) {
            CHECK(signal < 0);
            CHECK(variance > 0);
            CHECK(variance == Catch::Approx(-8 * signal));
            CHECK(std::isnan(value("grad_noise_scale")));
        } else {
            CHECK(signal > 0);
            CHECK(std::isfinite(value("grad_noise_scale")));
        }
        if (step == 0) continue;
        REQUIRE(std::isfinite(value("grad_noise_signal_adam")));
        REQUIRE(std::isfinite(value("grad_noise_variance_adam")));
        if (cancel_micro_batches) {
            CHECK(value("grad_noise_signal_adam") <= 0);
            CHECK(std::isnan(value("grad_noise_scale_adam")));
            CHECK(std::isnan(value("grad_noise_ratio_adam")));
        } else {
            CHECK(value("grad_noise_signal_adam") > 0);
            CHECK(std::isfinite(value("grad_noise_scale_adam")));
            CHECK(std::isfinite(value("grad_noise_ratio_adam")));
        }
    }
}

TEST_CASE("Attention update results expose indexed sparse values", "[dqn][attention][rfd0093]")
{
    dqn::BatchUpdateResult result;
    REQUIRE(result.GetScalar("attention_entropy.[0]").has_value());
    CHECK(std::isnan(*result.GetScalar("attention_entropy.[0]")));
    result.attention_entropy = torch::tensor({ 0.25f, 0.75f });
    CHECK(*result.GetScalar("attention_entropy.[0]") == 0.25f);
    CHECK(*result.GetScalar("attention_entropy.[1]") == 0.75f);
    CHECK_FALSE(result.GetScalar("unknown_attention").has_value());
    for (const auto& key : { "attention_entropy.[-1]", "attention_entropy.[x]", "attention_entropy.[1.5]", "attention_entropy.[]", "attention_entropy.[9223372036854775808]" }) {
        CHECK_THROWS_WITH(result.GetScalar(key), Catch::Matchers::ContainsSubstring(key));
    }
}

TEST_CASE("Attention probe uses the union of subscription intervals and returns pre-update values", "[dqn][attention][rfd0093]")
{
    auto online = MakeAttentionProbeTestNetwork(), target = MakeAttentionProbeTestNetwork();
    online->CopyTo(*target);
    AttentionProbeTestNetworkModel model(online, target);
    dqn::RuntimeVars vars;
    auto config = WithDefaultLearnerForwardConfig();
    config.final_feature_branch_key = "feature"; config.attention_entropy.probe.batch_size = 2;
    config.replay_batch_size = 2; config.update_warmup_steps = 0; config.update_interval = 1; config.replay_ratio = 2;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), dqn::LearnerForwardResources{});
    learner.EnablePlasticityForward(false);
    learner.MutateOnlineParametersDuringUpdate(0.1f);
    auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({ 2 }); replay->unique_sample_enabled = true; replay->vector_value = 1;
    learner.UseReplayBuffer(replay);
    learner.ConfigureScalarMetricSubscriptions({ AttentionSubscription(0, 3), AttentionSubscription(3, 2) });
    int due_count = 0;
    std::shared_ptr<const rl::BatchUpdateResult> first;
    float first_value = 0;
    for (int step = 0; step < 6; ++step) {
        // 同じ観測と更新前の重みで独立にprobeの期待値を採る。
        torch::Tensor expected;
        {
            torch::NoGradGuard no_grad;
            anet::TrainingModeGuard guard(*online, false);
            const auto prefix = std::string(anet::nn::kTransformerEncoderAttentionEntropyCaptureKey) + "/";
            anet::ForwardCapture capture({ prefix });
            online->ForwardUpTo(anet::TensorDict{ { kVectorKey, torch::ones({ 2, 2 }) } }, "feature", &capture);
            const auto entries = capture.GetAll(prefix);
            expected = torch::cat({ entries[0].second, entries[1].second });
        }
        const auto result = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
        const bool due = step % 3 == 0 || step % 2 == 0;
        if (due) {
            ++due_count;
            CHECK(*result->GetScalar("attention_entropy.[0]") == Catch::Approx(expected[0].item<float>()).margin(1e-6));
            CHECK(*result->GetScalar("attention_entropy.[3]") == Catch::Approx(expected[3].item<float>()).margin(1e-6));
        } else CHECK(std::isnan(*result->GetScalar("attention_entropy.[0]")));
        CHECK(replay->unique_sample_count == due_count);
        if (step == 0) { first = result; first_value = *result->GetScalar("attention_entropy.[0]"); }
    }
    CHECK(*first->GetScalar("attention_entropy.[0]") == first_value);
    replay->unique_sample_enabled = false;
    const auto missing = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
    CHECK(std::isnan(*missing->GetScalar("attention_entropy.[0]")));
    const auto samples = replay->unique_sample_count;
    learner.ConfigureScalarMetricSubscriptions({});
    const auto inactive = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
    CHECK(std::isnan(*inactive->GetScalar("attention_entropy.[0]")));
    CHECK(replay->unique_sample_count == samples);
}

TEST_CASE("Attention subscriptions reject malformed indices and unavailable layers at the intended boundary", "[dqn][attention][rfd0093][validation]")
{
    const bool transformer = GENERATE(false, true);
    auto online = MakeAttentionProbeTestNetwork(transformer), target = MakeAttentionProbeTestNetwork(transformer);
    AttentionProbeTestNetworkModel model(online, target);
    dqn::RuntimeVars vars;
    auto config = WithDefaultLearnerForwardConfig();
    config.final_feature_branch_key = "feature"; config.attention_entropy.probe.batch_size = 2;
    config.replay_batch_size = 2; config.update_warmup_steps = 0; config.update_interval = 1; config.replay_ratio = 2;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), dqn::LearnerForwardResources{});
    learner.EnablePlasticityForward(false);
    auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({ 2 }); replay->unique_sample_enabled = true;
    learner.UseReplayBuffer(replay);
    auto invalid = AttentionSubscription(0); invalid.source_key = "attention_entropy.[-1]";
    CHECK_THROWS_WITH(learner.ConfigureScalarMetricSubscriptions({ invalid }), Catch::Matchers::ContainsSubstring(invalid.source_key));
    CHECK_NOTHROW(learner.ConfigureScalarMetricSubscriptions({ AttentionSubscription(4) }));
    CHECK(replay->unique_sample_count == 0);
    const bool training_before = online->is_training();
    std::map<std::string, torch::Tensor> buffers_before;
    for (const auto& entry : online->named_buffers()) buffers_before.emplace(entry.key(), entry.value().clone());
    CHECK_THROWS_WITH(learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}),
        Catch::Matchers::ContainsSubstring(transformer ? "layer_count=4" : "unfulfilled"));
    CHECK(online->is_training() == training_before);
    for (const auto& entry : online->named_buffers()) CHECK(torch::equal(entry.value(), buffers_before.at(entry.key())));
}

TEST_CASE("Attention probe preserves TD QR IQN learning RNG and spectral buffers", "[dqn][attention][rfd0093][determinism]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const auto mode = GENERATE(std::string("none"), std::string("qr"), std::string("iqn"));
    const bool dueling = true;
    const bool cuda = GENERATE(false, true);
    if (cuda && !torch::cuda::is_available()) SKIP("CUDA unavailable; attention BF16 acceptance remains unverified");
    const torch::Device device(cuda ? torch::kCUDA : torch::kCPU);
    const auto munch = GENERATE(std::string("disabled"), std::string("target"), std::string("online"), std::string("online_reuse"));
    CAPTURE(mode, dueling, munch);
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.quantile_mode", mode);
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.mode", "sample");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.hold", "call");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.share", "sample");
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "sample");
    data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "sample");
    data.Set("net.block.[Feature].type", "Linear");
    data.Set("net.block.[Feature].linear.out_features", 4);
    data.Set("net.block.[Feature].weight_norm.mode", "spectral");
    data.Set("net.block.[Feature].noisy.enabled", true);
    data.Set("net.branch.[feature].bind", kVectorKey);
    data.Set("net.branch.[feature].structure", "Feature > Tokens > TF > Flatten");
    data.Set("net.block.[Tokens].type", "Reshape"); data.Set("net.block.[Tokens].reshape.dims", "2 2");
    data.Set("net.block.[TF].type", "TransformerEncoder");
    data.Set("net.block.[TF].tf.d_model", 2); data.Set("net.block.[TF].tf.nhead", 1);
    data.Set("net.block.[TF].tf.num_layers", 2); data.Set("net.block.[TF].tf.dim_feedforward", 4);
    data.Set("net.block.[TF].tf.hidden_dropout_rate", 0.1);
    data.Set("net.block.[TF].weight_norm.mode", "spectral");
    data.Set("net.block.[Flatten].type", "Flatten");
    data.Set("net.body.output.[features]", "feature");
    auto specs = MakeLearnerEnvSpec().state_spec.obs_spec;
    if (mode == "iqn") {
        data.Set("net.block.[Cos].type", "CosineEmbedding");
        data.Set("net.block.[Cos].cos.num_basis", 4);
        data.Set("net.block.[Tau].type", "Linear");
        data.Set("net.block.[Tau].linear.out_features", 4);
        data.Set("net.branch.[tau].bind", anet::rl::dqn::kKey_Taus);
        data.Set("net.branch.[tau].structure", "Cos > Tau");
        data.Set("net.branch.[fusion].bind", "feature * tau");
        data.Set("net.body.output.[features]", "fusion");
        specs[anet::rl::dqn::kKey_Taus] = anet::TensorSpec{.type=anet::SpaceType::Vector, .shape={4}, .dtype=torch::kFloat32};
    }
    anet::ConfigData projection;
    projection.Set("noisy.enabled", true);
    std::shared_ptr<anet::nn::NetworkHeadFactory> factory;
    const anet::nn::WeightInitConfig init{};
    if (mode == "none") {
        if (dueling) factory = std::make_shared<dqn::DuelingHeadFactory>(2, init, projection);
        else factory = std::make_shared<dqn::LinearHeadFactory>(2, init, projection);
    } else if (mode == "qr") {
        if (dueling) factory = std::make_shared<dqn::QuantileDuelingHeadFactory>(2, 5, init, projection);
        else factory = std::make_shared<dqn::QuantileHeadFactory>(2, 5, init, projection);
    } else {
        if (dueling) factory = std::make_shared<dqn::IQNDuelingHeadFactory>(2, init, projection);
        else factory = std::make_shared<dqn::IQNHeadFactory>(2, init, projection);
    }
    dqn::NetworkModelConfig model_config{};
    model_config.hard_update_interval = 2;
    torch::manual_seed(441);
    dqn::NetworkModel first(model_config, device, anet::nn::NetworkConfig(data), specs, 2, factory, mode != "none", 441);
    torch::manual_seed(441);
    dqn::NetworkModel second(model_config, device, anet::nn::NetworkConfig(data), specs, 2, factory, mode != "none", 441);
    auto config = dqn::DefaultDQNAgentConfig(data).learner;
    config.quantile_mode = mode;
    config.num_quantiles = 5;
    config.iqn.current_taus.num_taus = 4;
    config.iqn.target_taus.num_taus = 4;
    config.final_feature_branch_key = "feature";
    config.attention_entropy.probe.batch_size = 2;
    config.replay_batch_size = 2;
    config.replay_capacity = 32;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.use_per = true;
    config.use_n_step = false;
    config.use_rb_prefetch = false;
    config.use_fused_optimizer = false;
    config.use_amp = cuda;
    config.use_amp_bf16 = true;
    config.use_grad_clip = true;
    config.grad_clip_tau = 0.7f;
    config.noisy_weight_decay = 0.1f;
    config.munchausen.enabled = munch != "disabled";
    config.munchausen.log_policy_mode = munch == "disabled" ? "target" : munch;
    config.use_double_dqn = !config.munchausen.enabled;
    auto env_spec = MakeLearnerEnvSpec();
    env_spec.action_spec.value_labels = {"a0", "a1"};
    rl::ObservationNormalizerConfig normalization_config;
    normalization_config.use_robust_update = false;
    auto normalizer = rl::ObservationNormalizerFactory(normalization_config).CreateObservationNormalizer(env_spec.state_spec);
    normalizer->NormalizeAndUpdateStats(anet::TensorDict{ { kVectorKey, torch::ones({ 2, 2 }, torch::TensorOptions().device(device)) } });
    class ReferenceNormalizerForwarder final : public anet::Forwarder {
    public:
        explicit ReferenceNormalizerForwarder(std::shared_ptr<rl::ObservationNormalizer> normalizer)
            : normalizer_(std::move(normalizer)) {}
    public:
        anet::TensorDict Forward(const anet::TensorDict& input, anet::ForwardCapture*) override
        {
            return normalizer_->Normalize(input);
        }
    private:
        const std::shared_ptr<rl::ObservationNormalizer> normalizer_;
    };
    const auto sample_perception = std::make_shared<ReferenceNormalizerForwarder>(normalizer);
    const auto count_before = normalizer->GetScalar(rl::ObservationNormalizer::kKeyCount);
    REQUIRE(count_before.has_value());
    dqn::RuntimeVars first_vars, second_vars;
    auto make_learner = [&](dqn::NetworkModel& model, dqn::RuntimeVars& vars) -> std::unique_ptr<dqn::Learner> {
        dqn::ActionPolicyConfig policy_config;
        policy_config.quantile_mode = mode;
        policy_config.tau_rule.num_taus = 4;
        auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(policy_config);
        if (mode == "none") return std::make_unique<dqn::TDLearner>(config, model, vars, sample_perception,
            rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env_spec, device, 442, kTestAgentSeed, policy, SelectLearnerForwardResources(data), std::nullopt, 443);
        if (mode == "qr") return std::make_unique<dqn::QRLearner>(config, model, vars, sample_perception,
            rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env_spec, device, 442, kTestAgentSeed, policy, SelectLearnerForwardResources(data), std::nullopt, 443);
        return std::make_unique<dqn::IQNLearner>(config, model, vars, sample_perception,
            rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env_spec, device, 442, kTestAgentSeed, policy, SelectLearnerForwardResources(data), std::nullopt, 443);
    };
    auto measured = make_learner(first, first_vars);
    auto ordinary = make_learner(second, second_vars);
    measured->ConfigureScalarMetricSubscriptions({ AttentionSubscription(0), AttentionSubscription(1) });
    ordinary->ConfigureScalarMetricSubscriptions({});
    auto compare_networks = [](const auto& lhs, const auto& rhs) {
        const auto other_parameters = rhs->named_parameters();
        for (const auto& entry : lhs->named_parameters()) {
            CAPTURE(entry.key());
            CHECK(torch::equal(entry.value(), other_parameters[entry.key()]));
            if (entry.value().grad().defined()) CHECK(torch::equal(entry.value().grad(), other_parameters[entry.key()].grad()));
        }
        const auto other_buffers = rhs->named_buffers();
        for (const auto& entry : lhs->named_buffers()) CHECK(torch::equal(entry.value(), other_buffers[entry.key()]));
    };
    compare_networks(first.GetOnlineNetwork(), second.GetOnlineNetwork());
    compare_networks(first.GetTargetNetwork(), second.GetTargetNetwork());
    size_t probes = 0;
    for (rl::step_t step = 0; step < 20; ++step) {
        const auto experience = MakeDeterminismExperience(step, 2);
        torch::manual_seed(930500 + step);
        const auto left = measured->UpdateFromBatch(rl::StepCounts{.exp_step=step}, experience);
        torch::manual_seed(930500 + step);
        const auto right = ordinary->UpdateFromBatch(rl::StepCounts{.exp_step=step}, experience);
        REQUIRE(left.size() == right.size());
        for (size_t update = 0; update < left.size(); ++update) {
            for (const auto key : {"loss", "grad_norm", "q_sa_mean", "td_mean"}) {
                const auto a = left[update]->GetScalar(key);
                const auto b = right[update]->GetScalar(key);
                REQUIRE(a.has_value() == b.has_value());
                if (a && b) CHECK((*a == *b || (std::isnan(*a) && std::isnan(*b))));
            }
            const auto value = left[update]->GetScalar("attention_entropy.[0]");
            REQUIRE(value.has_value());
            if (std::isfinite(*value)) ++probes;
            CHECK(std::isnan(*right[update]->GetScalar("attention_entropy.[0]")));
        }
        CHECK(first_vars.learn_step == second_vars.learn_step);
        CHECK(normalizer->GetScalar(rl::ObservationNormalizer::kKeyCount) == count_before);
        CHECK(torch::equal(measured->GetRandomGenerator()->GetTorchGenerator(device).get_state(),
            ordinary->GetRandomGenerator()->GetTorchGenerator(device).get_state()));
        CHECK(measured->GetRandomGenerator()->RandUint64() == ordinary->GetRandomGenerator()->RandUint64());
        CHECK(first.GetOnlineNetwork()->is_training() == second.GetOnlineNetwork()->is_training());
        const auto left_eviction = measured->GetScalar(rl::ReplayBuffer::PER_LAST_EVICTED_NEVER_SAMPLED_RATIO);
        const auto right_eviction = ordinary->GetScalar(rl::ReplayBuffer::PER_LAST_EVICTED_NEVER_SAMPLED_RATIO);
        REQUIRE(left_eviction.has_value() == right_eviction.has_value());
        if (left_eviction) CHECK((*left_eviction == *right_eviction || (std::isnan(*left_eviction) && std::isnan(*right_eviction))));
        compare_networks(first.GetOnlineNetwork(), second.GetOnlineNetwork());
        compare_networks(first.GetTargetNetwork(), second.GetTargetNetwork());
    }
    CHECK(probes >= 2);
}

TEST_CASE("RFD0101 learner evaluates learning rate before warmup and switches phases", "[rfd0101][dqn]")
{
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.nn_forward.[mean].noisy.mode", "mu_only");
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "mean");
    data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "mean");
    data.Set("DefaultDQNAgent.learner.enabled", false);
    data.Set("DefaultDQNAgent.learner.learning_rate.type", "phased");
    data.Set("DefaultDQNAgent.learner.learning_rate.phases", "high low");
    data.Set("DefaultDQNAgent.learner.learning_rate.phase.[high].value", 0.002f);
    data.Set("DefaultDQNAgent.learner.learning_rate.phase.[high].steps", 8);
    data.Set("DefaultDQNAgent.learner.learning_rate.phase.[low].value", 0.0002f);
    data.Set("DefaultDQNAgent.learner.learning_rate.phase.[low].steps", 16);
    auto config = dqn::DefaultDQNAgentConfig(data).learner;
    config.enabled = true;
    config.current = WithDefaultLearnerForwardConfig().current;
    config.target = WithDefaultLearnerForwardConfig().target;
    config.update_warmup_steps = 100;
    config.use_rb_prefetch = false;
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    CapturingLearner learner(config, model, vars, {4, 1}, MakeLearnerEnvSpec(), 101, nullptr);
    REQUIRE(learner.GetScalar("learning_rate").has_value());
    CHECK(learner.GetScalar("learning_rate").value() == Catch::Approx(0.002f));
    rl::StepCounts counts;
    counts.exp_step = 8;
    learner.UpdateFromBatch(counts, MakeDeterminismExperience(0, 4));
    CHECK(learner.GetScalar("learning_rate").value() == Catch::Approx(0.0002f));
}

TEST_CASE("RFD0101 optimizer schedules update every group and retain sigma decay", "[rfd0101][dqn]")
{
    const bool skip = GENERATE(false, true);
    auto data = MakeNoisyOptimizerTestData();
    data.Set("DefaultDQNAgent.learner.weight_decay.type", "linear");
    data.Set("DefaultDQNAgent.learner.weight_decay.start", 0.02f);
    data.Set("DefaultDQNAgent.learner.weight_decay.end", 0.04f);
    data.Set("DefaultDQNAgent.learner.weight_decay.steps", 8);
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.update_warmup_steps = 100;
    config.noisy_weight_decay = 0.07f;
    config.learning_rate.type = "linear";
    config.learning_rate.start = 0.002f;
    config.learning_rate.end = 0.001f;
    config.learning_rate.steps = 8;
    const auto spec = MakeLearnerEnvSpec();
    auto factory = std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{});
    anet::nn::NetworkConfig skip_net;
    skip_net.output_keys["features"] = "features";
    dqn::SkipNetworkDefinition skip_definition{.config = {.enabled = true, .max_length = 3}, .network_config = skip_net,
        .head_factory = std::make_shared<dqn::SkipHeadFactory>(1, 3, dqn::SkipNetworkMode::ActionOut, "none", 1, anet::nn::WeightInitConfig{}), .seed = 102};
    dqn::NetworkModel model({}, torch::kCPU, anet::nn::NetworkConfig(data), spec.state_spec.obs_spec,
        1, factory, false, 101, skip ? &skip_definition : nullptr);
    dqn::RuntimeVars vars;
    TestLearner learner(config, model, vars, {1, 1}, spec, SelectLearnerForwardResources(data));
    learner.SetupAdamW();
    learner.UseReplayBuffer(std::make_shared<RecordingReplayBuffer>());
    rl::StepCounts counts;
    counts.exp_step = 8;
    learner.UpdateFromBatch(counts, MakeNoisyOptimizerExperience(0, 1.0f));
    auto& groups = learner.Optimizer().param_groups();
    REQUIRE(groups.size() == (skip ? 4 : 2));
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto& options = static_cast<const torch::optim::AdamWOptions&>(groups[i].options());
        CHECK(options.lr() == Catch::Approx(0.001f));
        CHECK(options.weight_decay() == Catch::Approx(i % 2 == 1 ? 0.07f : 0.04f));
    }
}

TEST_CASE("RFD0101 beta begins at zero and schedules the next sampling request", "[rfd0101][dqn]")
{
    auto data = MakeNoisyOptimizerTestData();
    data.Set("DefaultDQNAgent.learner.per_beta.type", "phased");
    data.Set("DefaultDQNAgent.learner.per_beta.phases", "first next");
    data.Set("DefaultDQNAgent.learner.per_beta.phase.[first].value", 0.2f);
    data.Set("DefaultDQNAgent.learner.per_beta.phase.[first].steps", 8);
    data.Set("DefaultDQNAgent.learner.per_beta.phase.[next].value", 0.8f);
    data.Set("DefaultDQNAgent.learner.per_beta.phase.[next].steps", 100);
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.current = {};
    config.target = {};
    config.use_per = true;
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    CapturingLearner learner(config, model, vars, {1, 1}, MakeLearnerEnvSpec(), 101, nullptr);
    CHECK(vars.per_beta == Catch::Approx(0.2f));
    auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({1});
    learner.UseReplayBuffer(replay);
    rl::StepCounts counts;
    counts.exp_step = 8;
    learner.UpdateFromBatch(counts, MakeDeterminismExperience(0, 1));
    counts.exp_step = 16;
    learner.UpdateFromBatch(counts, MakeDeterminismExperience(1, 1));
    REQUIRE(replay->sampled_betas.size() == 2);
    CHECK(replay->sampled_betas[0] == Catch::Approx(0.2f));
    CHECK(replay->sampled_betas[1] == Catch::Approx(0.8f));
}

TEST_CASE("RFD0101 soft target update uses the current scheduled tau", "[rfd0101][dqn]")
{
    auto data = MakeNoisyOptimizerTestData();
    data.Set("DefaultDQNAgent.model.soft_update_tau.type", "linear");
    data.Set("DefaultDQNAgent.model.soft_update_tau.start", 0.0f);
    data.Set("DefaultDQNAgent.model.soft_update_tau.end", 0.1f);
    data.Set("DefaultDQNAgent.model.soft_update_tau.steps", 8);
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.current = {};
    config.target = {};
    TestNetworkModel model(false, 1, dqn::DefaultDQNAgentConfig(data).model);
    {
        torch::NoGradGuard guard;
        for (auto& p : model.GetOnlineNetwork()->parameters()) p.fill_(1.0f);
        for (auto& p : model.GetTargetNetwork()->parameters()) p.zero_();
    }
    dqn::RuntimeVars vars;
    CapturingLearner learner(config, model, vars, {1, 1}, MakeLearnerEnvSpec(), 101, nullptr);
    auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({1});
    learner.UseReplayBuffer(replay);
    rl::StepCounts counts;
    counts.exp_step = 8;
    learner.UpdateFromBatch(counts, MakeDeterminismExperience(0, 1));
    for (const auto& p : model.GetTargetNetwork()->parameters()) {
        CHECK(torch::allclose(p, torch::full_like(p, 0.1f)));
    }
}

TEST_CASE("RFD0101 learner passes its current gamma to push and every sample", "[rfd0101][dqn]")
{
    auto data = MakeNoisyOptimizerTestData();
    data.Set("DefaultDQNAgent.learner.gamma.type", "linear");
    data.Set("DefaultDQNAgent.learner.gamma.start", 0.5f);
    data.Set("DefaultDQNAgent.learner.gamma.end", 0.9f);
    data.Set("DefaultDQNAgent.learner.gamma.steps", 8);
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.current = {};
    config.target = {};
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    CapturingLearner learner(config, model, vars, {1, 1}, MakeLearnerEnvSpec(), 101, nullptr);
    auto replay = std::make_shared<RecordingReplayBuffer>();
    replay->SetSizeValues({1});
    learner.UseReplayBuffer(replay);
    rl::StepCounts counts;
    counts.exp_step = 8;
    learner.UpdateFromBatch(counts, MakeDeterminismExperience(0, 1));
    REQUIRE(replay->pushed_gammas.size() == 1);
    REQUIRE(replay->sampled_gammas.size() == 1);
    CHECK(replay->pushed_gammas[0] == Catch::Approx(0.9f));
    CHECK(replay->sampled_gammas[0] == Catch::Approx(0.9f));
    REQUIRE(learner.GetScalar("gamma").has_value());
    CHECK(*learner.GetScalar("gamma") == Catch::Approx(0.9f));
}

TEST_CASE("RFD0101 learner exposes the current Munchausen values", "[rfd0101][dqn]")
{
    auto data = MakeNoisyOptimizerTestData();
    for (const auto* name : {"alpha", "entropy_tau"}) {
        const auto root = std::string("DefaultDQNAgent.learner.munchausen.") + name;
        data.Set(root + ".type", "linear");
        data.Set(root + ".start", 0.1f);
        data.Set(root + ".end", name == std::string("alpha") ? 0.4f : 0.7f);
        data.Set(root + ".steps", 8);
    }
    auto config = MakeNoisyOptimizerLearnerConfig(data);
    config.current = {};
    config.target = {};
    config.update_warmup_steps = 100;
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    CapturingLearner learner(config, model, vars, {1, 1}, MakeLearnerEnvSpec(), 101, nullptr);
    rl::StepCounts counts;
    counts.exp_step = 8;
    learner.UpdateFromBatch(counts, MakeDeterminismExperience(0, 1));
    REQUIRE(learner.GetScalar("munchausen_alpha").has_value());
    REQUIRE(learner.GetScalar("munchausen_entropy_tau").has_value());
    CHECK(*learner.GetScalar("munchausen_alpha") == Catch::Approx(0.4f));
    CHECK(*learner.GetScalar("munchausen_entropy_tau") == Catch::Approx(0.7f));
}

TEST_CASE("Gradient noise local settings are validated without subscriptions", "[rfd0117][dqn][grad_noise_config]")
{
    anet::test::LogCaptureGuard logs;
    const auto entry = GENERATE(std::pair{"num_micro_batches", "1"},
        std::pair{"ema_alpha", "0"}, std::pair{"ema_alpha", "1.01"}, std::pair{"ema_alpha", "nan"});
    auto data = WithDefaultForwardConfig(anet::ConfigData{});
    const auto key = std::string("DefaultDQNAgent.learner.grad_noise.") + entry.first;
    data.Set(key, entry.second);
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring(key));
}

TEST_CASE("Observation post process modes are rejected at construction", "[rfd0117][dqn][obs_config]")
{
    const int mode = GENERATE(-1, 4, 99);
    const bool pass_through = GENERATE(false, true);
    CAPTURE(mode, pass_through);
    anet::ConfigData data;
    const std::string key = "perception.block.[norm].running.post_process_type";
    data.Set("perception.structure", "norm");
    data.Set("perception.block.[norm].type", "Normalize");
    data.Set(key, mode);
    data.Set("perception.block.[norm].mode", pass_through ? "none" : "running");
    const auto specs = MakeLearnerEnvSpec().state_spec.obs_spec;
    CHECK_THROWS_WITH(anet::ForwarderChain(data, "perception", specs, 42),
        Catch::Matchers::ContainsSubstring(key) && Catch::Matchers::ContainsSubstring("0..3"));
    rl::ObservationNormalizerConfig config;
    config.post_process_type = mode;
    config.pass_through = pass_through;
    CHECK_THROWS_WITH(rl::ObservationNormalizerFactory(config),
        Catch::Matchers::ContainsSubstring("post_process_type") && Catch::Matchers::ContainsSubstring("0..3"));
}
