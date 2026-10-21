#include "anet/catch_test.hpp"
#include "anet/step_trace.hpp"
#include "anet/env.hpp"
#include "anet/observers.hpp"
#include <cmath>

using namespace anet::rl;

TEST_CASE("Step trace columns parse ordered declarations and reject invalid contracts", "[prd094][columns]")
{
    EnvSpec spec;
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = { "NOOP", "UP" };
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[20]", "$env_step reward label:r display:pct");
    config.Set("app.eval_trace_panel.columns.[2]", "$action_info action display:name");
    auto columns = StepTraceColumnSpec::Parse(config, spec);
    REQUIRE(columns.size() == 2);
    CHECK(columns[0].number == 2);
    CHECK(columns[0].label == "action");
    CHECK(columns[0].display == "name");
    CHECK(columns[1].label == "r");
    CHECK(columns[1].display == "pct");

    // 閉じた列DSLはmetricsの制御トークンや時点の違うtargetを受け付けない。
    const std::string invalid = GENERATE(
        "$env_step reward extra", "reward", "$env_step $runner reward", "$unknown reward",
        "$agent loss", "$env lives", "$exp reward", "$update_result loss", "$agent_step unknown",
        "$env_step mean.reward", "$runner mean.reward", "$env_step reward label:",
        "$env_step reward label:a label:b", "$env_step reward display:value display:pct",
        "$env_step reward interval:2", "$env_step reward @train", "$env_step reward $exp_step",
        "$env_step reward display:name", "$action_info action display:pct", "$env_step reward bogus:1");
    anet::ConfigData bad;
    bad.Set("app.eval_trace_panel.columns.[1]", invalid);
    CHECK_THROWS(StepTraceColumnSpec::Parse(bad, spec));
    const std::string number = GENERATE("0", "-1", "1.5", "x", "1x", "2147483648");
    anet::ConfigData bad_number;
    bad_number.Set("app.eval_trace_panel.columns.[" + number + "]", "$env_step reward");
    CHECK_THROWS(StepTraceColumnSpec::Parse(bad_number, spec));
    spec.action_spec.is_discrete = false;
    anet::ConfigData continuous;
    continuous.Set("app.eval_trace_panel.columns.[1]", "$action_info action");
    CHECK_THROWS(StepTraceColumnSpec::Parse(continuous, spec));
}

TEST_CASE("Trace player obeys Live entry playback boundaries and row eviction", "[prd094][trace_player]")
{
    TracePlayer player;
    CHECK_FALSE(player.GetSelected());
    player.SetRows({10, 11, 12});
    CHECK(player.GetSpeed() == 1.0);
    CHECK(player.Play() == "Trace: play");
    CHECK(player.GetSelected() == 10);
    CHECK(player.IsPlaying());
    CHECK(player.Tick(1.0, 1.0).empty());
    CHECK(player.GetSelected() == 11);
    CHECK(player.Tick(1.0, 1.0) == "Trace: end");
    CHECK(player.GetSelected() == 12);
    CHECK_FALSE(player.IsPlaying());
    CHECK(player.Play() == "Trace: play");
    CHECK(player.GetSelected() == 10);
    player.Pause();
    CHECK(player.StepBack() == "Trace: start");
    CHECK(player.StepForward() == "Trace: step");
    CHECK(player.GetSelected() == 11);
    player.SetRows({12, 13});
    CHECK(player.GetSelected() == 12);
    CHECK(player.Stop() == "Trace: stop (live)");
    CHECK_FALSE(player.GetSelected());
    CHECK(player.Pause() == "Trace: pause");
    CHECK(player.GetSelected() == 13);
    CHECK(player.StepForward() == "Trace: end");
    player.Stop();
    CHECK(player.StepBack() == "Trace: step back");
    CHECK(player.GetSelected() == 13);
    player.Stop();
    CHECK(player.StepForward() == "Trace: step");
    CHECK(player.GetSelected() == 13);
    for (int i = 0; i < 8; ++i) player.Slower();
    CHECK(player.GetSpeed() == 0.125);
    CHECK(player.Slower() == "Trace: speed x1/8 (min)");
    for (int i = 0; i < 8; ++i) player.Faster();
    CHECK(player.GetSpeed() == 8.0);
    CHECK(player.Faster() == "Trace: speed x8 (max)");
    player.SetRows({});
    CHECK_FALSE(player.GetSelected());
    CHECK_FALSE(player.IsPlaying());
    CHECK(player.Play().empty());
}

TEST_CASE("Trace display uses rolling midrank and preserves integer formatting", "[prd094][trace_display]")
{
    std::deque<float> window;
    CHECK(AppendTracePercentile(window, 1.0f, 3) == 50.0f);
    CHECK(AppendTracePercentile(window, 1.0f, 3) == 50.0f);
    CHECK(AppendTracePercentile(window, 3.0f, 3) == Catch::Approx(100.0f * 2.5f / 3.0f));
    CHECK(AppendTracePercentile(window, 2.0f, 3) == 50.0f);
    CHECK(std::isnan(AppendTracePercentile(window, std::numeric_limits<float>::quiet_NaN(), 3)));
    CHECK(std::isnan(AppendTracePercentile(window, std::numeric_limits<float>::infinity(), 3)));
    CHECK(window.size() == 3);
    CHECK(AppendTracePercentile(window, 1.0f, 3) == Catch::Approx(100.0f * 0.5f / 3.0f));
    StepTraceColumnSpec number;
    ActionSpec actions;
    actions.is_discrete = true;
    actions.value_labels = {"NOOP", "UP"};
    CHECK(number.Format(1000000.0f, actions) == "1000000");
    CHECK(number.Format(0.123456f, actions) == "0.1235");
    CHECK(number.Format(std::numeric_limits<float>::quiet_NaN(), actions).empty());
    CHECK(number.Format(std::numeric_limits<float>::infinity(), actions).empty());
    StepTraceColumnSpec action;
    action.target = "$action_info";
    action.key = "action";
    action.display = "value_name";
    CHECK(action.Format(1.0f, actions) == "1:UP");
    action.display = "name";
    CHECK(action.Format(1.0f, actions) == "UP");
    action.display = "value";
    CHECK(action.Format(1.0f, actions) == "1");
}

TEST_CASE("Step trace rejects malformed column locations before recording", "[prd094][column_bounds]")
{
    EnvSpec spec;
    const std::string key = GENERATE("app.eval_trace_panel.columns.foo", "app.eval_trace_panel.columns.[1].label");
    anet::ConfigData config;
    config.Set(key, "$env_step reward");
    CHECK_THROWS(StepTraceColumnSpec::Parse(config, spec));
    anet::ConfigData empty;
    CHECK_THROWS(StepTraceBuilder(empty, spec, nullptr, -1));
}

TEST_CASE("Step trace reads continuous actions and reserves transition fields", "[prd094][trace_reader]")
{
    EnvSpec spec;
    spec.action_spec.is_discrete = false;
    spec.action_spec.dims.resize(2);
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[1]", "$action_info action.[1]");
    config.Set("app.eval_trace_panel.columns.[2]", "$env_step reward");
    config.Set("app.eval_trace_panel.columns.[3]", "$agent_step q_sa");
    const BatchState state{anet::TensorDict{}, torch::zeros({1}, torch::kBool),
        torch::zeros({1}, torch::kBool), torch::zeros({1}, torch::kBool)};
    auto action = std::make_shared<BatchActionInfo>(torch::tensor({{0.25f, -0.5f}}));
    const BatchExperience experience{state, action, torch::tensor({2.0f}), state};
    auto result = std::make_shared<PlainBatchStepResult>(experience.reward, state, state, 1, 0,
        std::vector<AuxData>{AuxData{{"reward", torch::tensor(999.0f)}}});
    const TrainEvent event{{experience, nullptr, {.exp_step=1}, nullptr, {}}, nullptr, result, action};
    StepTraceBuilder builder(config, spec, nullptr, 1);
    std::shared_ptr<StepTrace> row;
    builder.SetHandlers([&](const auto& value, const TrainEvent&) { row = value; }, {});
    builder.OnTrain(event);
    REQUIRE(row);
    CHECK(row->values[0] == -0.5f);
    CHECK(row->values[1] == 2.0f);
    CHECK(std::isnan(row->values[2]));
    config.Set("app.eval_trace_panel.columns.[4]", "$env_step missing");
    StepTraceBuilder unknown(config, spec, nullptr, 1);
    CHECK_THROWS(unknown.OnTrain(event));
}

TEST_CASE("Lane action reads reject incomplete dimension syntax", "[prd094][trace_action_dimension]")
{
    const auto info = std::make_shared<BatchActionInfo>(torch::tensor({{0.25f, -0.5f}}));
    const MetricValueContext context{.action_info=info.get()};
    CHECK(MetricValueReader::Read(context, EventField::ACTION_INFO, "action.[1]", 0) == -0.5f);
    const std::string key = GENERATE("action.[0x]", "action.[x]", "action.[-1]", "action.[0",
        "action.[2147483648]", "action.[]", "action.[0]junk");
    CHECK_THROWS_WITH(MetricValueReader::Read(context, EventField::ACTION_INFO, key, 0),
        Catch::Matchers::ContainsSubstring("expected=action.[integer dimension]"));
}

TEST_CASE("Trace playback availability follows the current base rate", "[prd094][trace_play_available]")
{
    TracePlayer player;
    CHECK_FALSE(player.CanPlay(30.0));
    player.SetRows({1, 2, 3});
    CHECK_FALSE(player.CanPlay(0.0));
    CHECK(player.CanPlay(30.0));
    player.Play();
    CHECK(player.Tick(60.0, 0.0).empty());
    CHECK(player.IsPlaying());
    CHECK(player.GetSelected() == 1);
    CHECK(player.Tick(1.0 / 30.0, 30.0).empty());
    CHECK(player.GetSelected() == 2);
    player.Pause();
    CHECK(player.StepBack() == "Trace: step back");
    CHECK(player.GetSelected() == 1);
    CHECK(player.StepForward() == "Trace: step");
    CHECK(player.Stop() == "Trace: stop (live)");
    CHECK_FALSE(player.CanPlay(0.0));
    CHECK(player.CanPlay(15.0));
}
