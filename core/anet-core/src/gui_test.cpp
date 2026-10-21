#include "anet/catch_test.hpp"
#include "anet/gui.hpp"
#include "anet/metrics.hpp"
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
