#include "anet/catch_test.hpp"

#include "anet/schedule.hpp"

TEST_CASE("ProfiledValue evaluates constant and caches current value", "[profiled_value]")
{
    anet::ProfiledValueConfig<double> config;
    config.type = "constant";
    config.value = 0.25;

    anet::ProfiledValue<double> value(config);
    CHECK(value.Value() == Catch::Approx(0.25));
    CHECK(value.Evaluate(100) == Catch::Approx(0.25));

    value.Update(100);
    CHECK(value.Value() == Catch::Approx(0.25));
}

TEST_CASE("ProfiledValue evaluates linear schedule and clamps overrun", "[profiled_value]")
{
    anet::ProfiledValueConfig<double> config;
    config.type = "linear";
    config.start = 1.0;
    config.end = 0.0;
    config.steps = 10;

    anet::ProfiledValue<double> value(config);
    CHECK(value.Evaluate(0) == Catch::Approx(1.0));
    CHECK(value.Evaluate(5) == Catch::Approx(0.5));
    CHECK(value.Evaluate(10) == Catch::Approx(0.0));
    CHECK(value.Evaluate(15) == Catch::Approx(0.0));
}

TEST_CASE("ProfiledValue truncates fractional unsigned schedule values", "[profiled_value][step_t]")
{
    anet::ProfiledValueConfig<uint64_t> config;
    config.type = "linear";
    config.start = 400;
    config.end = 399;
    config.steps = 5;
    anet::ProfiledValue<uint64_t> value(config);

    CHECK(value.Evaluate(0) == 400);
    CHECK(value.Evaluate(1) == 399);
    CHECK(value.Evaluate(5) == 399);
}

TEST_CASE("ProfiledValue evaluates cosine schedule and clamps overrun", "[profiled_value]")
{
    anet::ProfiledValueConfig<double> config;
    config.type = "cosine";
    config.start = 1.0;
    config.end = 0.0;
    config.steps = 10;

    anet::ProfiledValue<double> value(config);
    CHECK(value.Evaluate(0) == Catch::Approx(1.0));
    CHECK(value.Evaluate(5) == Catch::Approx(0.5));
    CHECK(value.Evaluate(10) == Catch::Approx(0.0));
    CHECK(value.Evaluate(15) == Catch::Approx(0.0));
}

TEST_CASE("ProfiledValue evaluates cosine restart schedule", "[profiled_value]")
{
    anet::ProfiledValueConfig<double> config;
    config.type = "cosine_restart";
    config.start = 1.0;
    config.end = 0.0;
    config.steps = 4;
    config.cycle_mult = 2.0;

    anet::ProfiledValue<double> value(config);
    CHECK(value.Evaluate(0) == Catch::Approx(1.0));
    CHECK(value.Evaluate(2) == Catch::Approx(0.5));
    CHECK(value.Evaluate(4) == Catch::Approx(1.0));
    CHECK(value.Evaluate(8) == Catch::Approx(0.5));
    CHECK(value.Evaluate(12) == Catch::Approx(1.0));
}

TEST_CASE("ProfiledValue evaluates shrinking cosine restart schedule", "[profiled_value]")
{
    anet::ProfiledValueConfig<double> config;
    config.type = "cosine_restart";
    config.start = 1.0;
    config.end = 0.0;
    config.steps = 8;
    config.cycle_mult = 0.5;

    anet::ProfiledValue<double> value(config);
    CHECK(value.Evaluate(0) == Catch::Approx(1.0));
    CHECK(value.Evaluate(4) == Catch::Approx(0.5));
    CHECK(value.Evaluate(8) == Catch::Approx(1.0));
    CHECK(value.Evaluate(10) == Catch::Approx(0.5));
    CHECK(value.Evaluate(12) == Catch::Approx(1.0));
    CHECK(value.Evaluate(13) == Catch::Approx(0.5));
    CHECK(value.Evaluate(14) == Catch::Approx(1.0));
    CHECK(value.Evaluate(20) == Catch::Approx(1.0));
}

TEST_CASE("ProfiledValue evaluates phased schedule", "[profiled_value]")
{
    anet::ProfiledValueConfig<double> config;
    config.type = "phased";
    config.phases = { "warmup", "main" };

    anet::ProfiledValuePhaseConfig<double> warmup;
    warmup.type = "linear";
    warmup.start = 0.0;
    warmup.end = 1.0;
    warmup.steps = 10;
    config.phase.Set("warmup", warmup);

    anet::ProfiledValuePhaseConfig<double> main;
    main.type = "cosine";
    main.start = 1.0;
    main.end = 0.0;
    main.steps = 10;
    config.phase.Set("main", main);

    anet::ProfiledValue<double> value(config);
    CHECK(value.Evaluate(0) == Catch::Approx(0.0));
    CHECK(value.Evaluate(5) == Catch::Approx(0.5));
    CHECK(value.Evaluate(10) == Catch::Approx(1.0));
    CHECK(value.Evaluate(15) == Catch::Approx(0.5));
    CHECK(value.Evaluate(20) == Catch::Approx(0.0));
    CHECK(value.Evaluate(25) == Catch::Approx(0.0));
}

TEST_CASE("ProfiledValue evaluates supported profile index types", "[profiled_value]")
{
    SECTION("constant uses value")
    {
        anet::ProfiledValueConfig<double> config;
        config.type = "constant";
        config.value = -1.0;
        config.start = 2.0;
        config.end = 4.0;

        anet::ProfiledValue<double> value(config);
        CHECK(value.EvaluateByIndex(0, 1) == Catch::Approx(-1.0));
        CHECK(value.EvaluateByIndex(2, 5) == Catch::Approx(-1.0));
    }

    SECTION("linear interpolates from start to end")
    {
        anet::ProfiledValueConfig<double> config;
        config.type = "linear";
        config.start = 2.0;
        config.end = 4.0;
        config.steps = 10;

        anet::ProfiledValue<double> value(config);
        CHECK(value.EvaluateByIndex(0, 1) == Catch::Approx(2.0));
        CHECK(value.EvaluateByIndex(0, 5) == Catch::Approx(2.0));
        CHECK(value.EvaluateByIndex(2, 5) == Catch::Approx(3.0));
        CHECK(value.EvaluateByIndex(4, 5) == Catch::Approx(4.0));
        CHECK(value.EvaluateByIndex(99, 5) == Catch::Approx(4.0));
    }

    SECTION("cosine interpolates from start to end")
    {
        anet::ProfiledValueConfig<double> config;
        config.type = "cosine";
        config.start = 1.0;
        config.end = 0.0;
        config.steps = 10;

        anet::ProfiledValue<double> value(config);
        CHECK(value.EvaluateByIndex(0, 1) == Catch::Approx(1.0));
        CHECK(value.EvaluateByIndex(0, 5) == Catch::Approx(1.0));
        CHECK(value.EvaluateByIndex(2, 5) == Catch::Approx(0.5));
        CHECK(value.EvaluateByIndex(4, 5) == Catch::Approx(0.0));
        CHECK(value.EvaluateByIndex(99, 5) == Catch::Approx(0.0));
    }
}

TEST_CASE("ProfiledValue rejects unsupported profile index types", "[profiled_value]")
{
    SECTION("cosine_restart")
    {
        anet::ProfiledValueConfig<double> config;
        config.type = "cosine_restart";
        config.start = 1.0;
        config.end = 0.0;
        config.steps = 10;

        anet::ProfiledValue<double> value(config);
        CHECK_THROWS_AS(value.EvaluateByIndex(0, 5), std::runtime_error);
    }

    SECTION("phased")
    {
        anet::ProfiledValueConfig<double> config;
        config.type = "phased";
        config.phases = { "main" };

        anet::ProfiledValuePhaseConfig<double> main;
        main.type = "constant";
        main.value = 1.0;
        main.steps = 10;
        config.phase.Set("main", main);

        anet::ProfiledValue<double> value(config);
        CHECK_THROWS_AS(value.EvaluateByIndex(0, 5), std::runtime_error);
    }
}

TEST_CASE("ProfiledValue rejects invalid config", "[profiled_value]")
{
    SECTION("unknown root type")
    {
        anet::ProfiledValueConfig<double> config;
        config.type = "unknown";
        CHECK_THROWS_AS(anet::ProfiledValue<double>(config), std::runtime_error);
    }

    SECTION("time-based type without steps")
    {
        anet::ProfiledValueConfig<double> config;
        config.type = "linear";
        config.start = 1.0;
        config.end = 0.0;
        CHECK_THROWS_AS(anet::ProfiledValue<double>(config), std::runtime_error);
    }

    SECTION("cosine restart without positive cycle multiplier")
    {
        anet::ProfiledValueConfig<double> config;
        config.type = "cosine_restart";
        config.start = 1.0;
        config.end = 0.0;
        config.steps = 10;
        config.cycle_mult = 0.0;
        CHECK_THROWS_AS(anet::ProfiledValue<double>(config), std::runtime_error);
    }

    SECTION("phased type without phases")
    {
        anet::ProfiledValueConfig<double> config;
        config.type = "phased";
        CHECK_THROWS_AS(anet::ProfiledValue<double>(config), std::runtime_error);
    }

    SECTION("phased type with undefined phase")
    {
        anet::ProfiledValueConfig<double> config;
        config.type = "phased";
        config.phases = { "warmup" };
        CHECK_THROWS_AS(anet::ProfiledValue<double>(config), std::runtime_error);
    }

    SECTION("phase without positive steps")
    {
        anet::ProfiledValueConfig<double> config;
        config.type = "phased";
        config.phases = { "warmup" };

        anet::ProfiledValuePhaseConfig<double> warmup;
        warmup.type = "constant";
        warmup.value = 1.0;
        warmup.steps = 0;
        config.phase.Set("warmup", warmup);

        CHECK_THROWS_AS(anet::ProfiledValue<double>(config), std::runtime_error);
    }
}
