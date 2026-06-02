#include "catch.hpp"

#include "anet/config.hpp"
#include "anet/test_util.hpp"

#include <vector>

using namespace anet::test;

TEST_CASE("ConfigData Read warns for present invalid values only", "[config]")
{
    anet::ConfigData config_data;
    config_data.Set("invalid_int", "not-int");
    config_data.Set("invalid_bool", "maybe");
    config_data.Set("invalid_float_vector", "1.0 x");
    config_data.Set("valid_int", "42");

    SECTION("invalid int warns and returns the default")
    {
        LogCaptureGuard logs;

        int value = 5;
        CHECK_FALSE(config_data.Read("invalid_int", value, 9));
        CHECK(value == 9);

        logs.Flush();
        CHECK(CountRecords(logs.Records(), wxLOG_Warning) == 1);
        CHECK(HasRecordContaining(
            logs.Records(),
            wxLOG_Warning,
            { "ConfigData::Read failed", "key=invalid_int", "value=\"not-int\"", "expected=int" }));
    }

    SECTION("invalid bool warns and returns the default")
    {
        LogCaptureGuard logs;

        bool value = true;
        CHECK_FALSE(config_data.Read("invalid_bool", value, false));
        CHECK_FALSE(value);

        logs.Flush();
        CHECK(CountRecords(logs.Records(), wxLOG_Warning) == 1);
        CHECK(HasRecordContaining(
            logs.Records(),
            wxLOG_Warning,
            { "ConfigData::Read failed", "key=invalid_bool", "value=\"maybe\"", "expected=bool" }));
    }

    SECTION("invalid vector warns and returns the default")
    {
        LogCaptureGuard logs;

        std::vector<float> value = { 9.0f };
        const std::vector<float> default_value = { 1.0f, 2.0f };
        CHECK_FALSE(config_data.Read("invalid_float_vector", value, default_value));
        CHECK(value == default_value);

        logs.Flush();
        CHECK(CountRecords(logs.Records(), wxLOG_Warning) == 1);
        CHECK(HasRecordContaining(
            logs.Records(),
            wxLOG_Warning,
            { "ConfigData::Read failed", "key=invalid_float_vector", "value=\"1.0 x\"", "expected=float vector" }));
    }

    SECTION("missing key keeps the existing no-warning behavior")
    {
        LogCaptureGuard logs;

        int value = 5;
        CHECK_FALSE(config_data.Read("missing_int", value, 9));
        CHECK(value == 9);

        logs.Flush();
        CHECK(CountRecords(logs.Records(), wxLOG_Warning) == 0);
    }

    SECTION("valid value does not warn")
    {
        LogCaptureGuard logs;

        int value = 5;
        CHECK(config_data.Read("valid_int", value, 9));
        CHECK(value == 42);

        logs.Flush();
        CHECK(CountRecords(logs.Records(), wxLOG_Warning) == 0);
    }
}
