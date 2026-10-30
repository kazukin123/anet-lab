#include "anet/catch_test.hpp"
#include "anet/gui.hpp"
#include "anet/env.hpp"
#include "anet/metrics.hpp"
#include <array>
#include <cmath>

using namespace anet::rl;
using namespace anet::rl::gui;

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
    CHECK(player.GetSelected() == 12);
    player.Stop();
    CHECK(player.StepForward() == "Trace: end");
    CHECK(player.GetSelected() == 13);
    for (int i = 0; i < 8; ++i) player.Slower();
    CHECK(player.GetSpeed() == 0.125);
    CHECK(player.Slower() == "Trace: speed x0.125 (min)");
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
    CHECK(FormatTraceValue(number, 1000000.0f, actions) == "1000000");
    CHECK(FormatTraceValue(number, 0.123456f, actions) == "0.1235");
    CHECK(FormatTraceValue(number, std::numeric_limits<float>::quiet_NaN(), actions).empty());
    CHECK(FormatTraceValue(number, std::numeric_limits<float>::infinity(), actions).empty());
    StepTraceColumnSpec action;
    action.target = "$action_info";
    action.key = "action";
    action.display = "value_name";
    CHECK(FormatTraceValue(action, 1.0f, actions) == "1:UP");
    action.display = "name";
    CHECK(FormatTraceValue(action, 1.0f, actions) == "UP");
    action.display = "value";
    CHECK(FormatTraceValue(action, 1.0f, actions) == "1");
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

TEST_CASE("Trace Live stepping selects the adjacent frame and reports boundaries", "[prd094][trace_live_step]")
{
    TracePlayer player;
    player.SetRows({10, 11, 12});
    CHECK(player.StepBack() == "Trace: step back");
    CHECK(player.GetSelected() == 11);
    CHECK_FALSE(player.IsPlaying());
    player.Stop();
    CHECK(player.StepForward() == "Trace: end");
    CHECK(player.GetSelected() == 12);
    CHECK_FALSE(player.IsPlaying());
}

TEST_CASE("Trace Live stepping handles empty and single-row histories", "[prd094][trace_live_boundary]")
{
    TracePlayer player;
    CHECK(player.StepBack().empty());
    CHECK(player.StepForward().empty());
    CHECK_FALSE(player.GetSelected());
    player.SetRows({42});
    CHECK(player.StepBack() == "Trace: start");
    CHECK(player.GetSelected() == 42);
    player.Stop();
    CHECK(player.StepForward() == "Trace: end");
    CHECK(player.GetSelected() == 42);
}

TEST_CASE("Trace row sorting uses numeric steps and toggles the selected column", "[prd105][trace_sort]")
{
    // 表示文字列ではなく数値で並べ、step列の再クリックで向きを変える。
    TraceRowSorter sorter;
    sorter.Reset(0);
    sorter.Upsert({.id=9, .step=9, .values={9.0f, 4.0f}});
    sorter.Upsert({.id=10, .step=10, .values={10.0f, 2.0f}});
    sorter.Upsert({.id=100, .step=100, .values={100.0f, 3.0f}});
    CHECK(sorter.GetDisplayOrder() == std::vector<int64_t>{100, 10, 9});
    sorter.ToggleColumn(0);
    CHECK(sorter.GetDisplayOrder() == std::vector<int64_t>{9, 10, 100});
    sorter.ToggleColumn(1);
    CHECK_FALSE(sorter.IsAscending());
    CHECK(sorter.GetDisplayOrder() == std::vector<int64_t>{9, 100, 10});
    sorter.ToggleColumn(1);
    CHECK(sorter.IsAscending());
    CHECK(sorter.GetDisplayOrder() == std::vector<int64_t>{10, 100, 9});
    sorter.ToggleColumn(1);
    CHECK(sorter.GetDisplayOrder() == std::vector<int64_t>{9, 100, 10});
}

TEST_CASE("Trace sorting keeps blanks last and evicts by step after delayed updates", "[prd105][trace_sort_updates]")
{
    // 表示順を変えても、保持する時間の範囲は変えない。
    TraceRowSorter sorter;
    sorter.Upsert({.id=1, .step=9, .values={3.0f}});
    sorter.Upsert({.id=2, .step=10, .values={std::numeric_limits<float>::quiet_NaN()}});
    sorter.Upsert({.id=3, .step=100, .values={3.0f}});
    sorter.Upsert({.id=4, .step=101, .values={1.0f}});
    sorter.ToggleColumn(0);
    CHECK(sorter.GetDisplayOrder() == std::vector<int64_t>{3, 1, 4, 2});
    sorter.ToggleColumn(0);
    CHECK(sorter.GetDisplayOrder() == std::vector<int64_t>{4, 3, 1, 2});
    CHECK(sorter.Upsert({.id=5, .step=102, .values={2.0f}}) == 1);
    CHECK(sorter.GetDisplayOrder() == std::vector<int64_t>{4, 5, 3, 1, 2});
    CHECK(sorter.Upsert({.id=2, .step=10, .values={0.0f}}) == 0);
    CHECK(sorter.GetDisplayOrder() == std::vector<int64_t>{2, 4, 5, 3, 1});
    CHECK(sorter.Trim(4) == 1);
    CHECK(sorter.GetDisplayOrder() == std::vector<int64_t>{2, 4, 5, 3});
    CHECK_FALSE(sorter.Trim(4));
    sorter.Upsert({.id=6, .step=103, .values={std::numeric_limits<float>::infinity()}});
    sorter.ToggleColumn(0);
    CHECK(sorter.GetDisplayOrder() == std::vector<int64_t>{3, 5, 4, 2, 6});
}

TEST_CASE("Trace playback follows display order even when row IDs are not chronological", "[prd105][trace_sorted_playback]")
{
    TracePlayer player;
    player.SetRows({100, 9, 10}); // 表の下端から上端。
    player.Play();
    CHECK(player.GetSelected() == 100);
    player.Tick(1.0, 1.0);
    CHECK(player.GetSelected() == 9);
    CHECK(player.Tick(1.0, 1.0) == "Trace: end");
    CHECK(player.GetSelected() == 10);
    player.Play();
    CHECK(player.GetSelected() == 100);
    player.StepForward();
    CHECK(player.GetSelected() == 9);
    player.StepBack();
    CHECK(player.GetSelected() == 100);
    player.SetRows({9, 10});
    CHECK(player.GetSelected() == 9);
}

TEST_CASE("Trace dump separates scalar summaries from the complete notification text", "[prd105][trace_dump]")
{
    // 通知から保持する部品だけで文字列を作り、GUIやAgent固有の値へ問い合わせない。
    const BatchState state{anet::TensorDict{{"obs", torch::tensor({{11.0f, 22.0f}})}},
        torch::zeros({1}, torch::kBool), torch::zeros({1}, torch::kBool), torch::ones({1}, torch::kBool)};
    const anet::TensorDict info{{"decision", torch::ones({1}, torch::kInt64)}};
    const AuxData aux{{"q_values", torch::tensor({{123.25f, 987.5f}})},
        {"missing", torch::Tensor()}, {"empty", torch::empty({0})}};
    const auto hint = torch::tensor({{7.0f, 8.0f, 9.0f}});
    auto action = std::make_shared<BatchActionInfo>(torch::ones({1}, torch::kInt64), info, aux,
        ReplayInitialPriorityHint(hint));
    auto result = std::make_shared<PlainBatchStepResult>(torch::tensor({1.25f}), state, state,
        1, 0, std::vector<AuxData>{{{"image", torch::tensor({33.0f, 44.0f})}}});
    const TraceDumpFormatter::Data data{.state=state, .action_info=action, .step_result=result,
        .counts={.train_step=3, .exp_step=42, .update_step=4, .learn_step=5, .episode_count=6, .sim_step=7},
        .runner_name="eval", .env_name="env", .agent_name="agent", .forced=1.0f};
    const StepTrace trace{.id=1, .step=42, .lane=0, .reward=1.25f,
        .values={1.0f, NAN, NAN}, .value_ready={true, false, true}};
    const std::vector<StepTraceColumnSpec> columns{
        {.label="forced"}, {.label="prio"}, {.label="blank"}};
    ActionSpec actions;
    actions.is_discrete = true;
    actions.value_labels = {"NOOP", "UP"};
    const auto summary = TraceDumpFormatter::FormatInfo(data, trace, columns, trace.values, actions);
    CHECK(summary.starts_with("Trace dump step=42 lane=0:"));
    CHECK(summary.find_first_of("\r\n") == std::string::npos);
    CHECK(summary.find("forced=1 prio=pending blank=") != std::string::npos);
    CHECK(summary.find("counts.update_step=4") != std::string::npos);
    CHECK(summary.find("action_info.action=1:UP") != std::string::npos);
    CHECK(summary.find("runner.forced=1") != std::string::npos);
    CHECK(summary.find("runner.forced=") == summary.rfind("runner.forced="));
    CHECK(summary.find("step_result.reward=1.25") != std::string::npos);
    CHECK(summary.find("action_info.info.decision=1") != std::string::npos);
    CHECK(summary.find("action_info.aux.q_values=[1, 2] Float cpu") != std::string::npos);
    CHECK(summary.find("action_info.aux.missing=undefined") != std::string::npos);
    CHECK(summary.find("action_info.aux.empty=[0] Float cpu") != std::string::npos);
    CHECK(summary.find("123.2") == std::string::npos);
    CHECK(summary.find("987.5") == std::string::npos);
    const auto without_columns = TraceDumpFormatter::FormatInfo(data, trace, {}, {}, actions);
    CHECK(without_columns.find("runner.forced=1") != std::string::npos);
    CHECK(without_columns.find("step_result.reward=1.25") != std::string::npos);
    const auto verbose = TraceDumpFormatter::FormatVerbose(data, trace);
    CHECK(verbose.starts_with("Trace dump step=42 lane=0:"));
    CHECK(verbose.find(state.ToString()) != std::string::npos);
    CHECK(verbose.find(action->ToString()) != std::string::npos);
    CHECK(verbose.find(info.ToString()) != std::string::npos);
    CHECK(verbose.find(anet::ToString(hint)) != std::string::npos);
    CHECK(verbose.find(result->ToString()) != std::string::npos);
}

TEST_CASE("Trace speed labels use decimal multipliers", "[prd094][trace_speed_label]")
{
    TracePlayer player;
    const std::array<const char*, 7> labels{"x0.125", "x0.25", "x0.5", "x1", "x2", "x4", "x8"};
    for (int i = 0; i < 3; ++i) player.Slower();
    for (const auto* label : labels) {
        CHECK(player.GetSpeedLabel() == label);
        player.Faster();
    }
}
