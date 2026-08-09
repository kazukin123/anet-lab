// test_util_test.cpp

#include "anet/test_util.hpp"
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>
#include "anet/catch_test.hpp"

namespace {

anet::test::PreparedTestArgs Prepare(
    std::vector<std::string> args,
    const char* failure_dialog_env = nullptr)
{
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& arg : args) {
        argv.push_back(arg.data());
    }
    return anet::test::PrepareTestArgs(static_cast<int>(argv.size()), argv.data(), failure_dialog_env);
}

std::vector<std::string> ToStrings(const anet::test::PreparedTestArgs& args)
{
    std::vector<std::string> result;
    result.reserve(args.argv.size());
    for (const auto* arg : args.argv) {
        result.emplace_back(arg);
    }
    return result;
}

bool Contains(std::string_view text, std::string_view pattern)
{
    return text.find(pattern) != std::string_view::npos;
}

void RequireInvalidArgs(
    const std::vector<std::string>& args,
    const char* failure_dialog_env,
    std::string_view expected_message)
{
    try {
        auto prepared = Prepare(args, failure_dialog_env);
        static_cast<void>(prepared);
        FAIL("PrepareTestArgs should reject invalid arguments.");
    } catch (const std::invalid_argument& e) {
        CHECK(Contains(e.what(), expected_message));
    }
}

} // namespace

TEST_CASE("PrepareTestArgs defaults failure dialog off", "[test_util]")
{
    const auto prepared = Prepare({ "anet-core-test.exe", "[test_util]" });
    const std::vector<std::string> expected{ "anet-core-test.exe", "[test_util]" };

    CHECK(prepared.failure_dialog_enabled == false);
    CHECK(ToStrings(prepared) == expected);
}

TEST_CASE("PrepareTestArgs strips failure dialog CLI option", "[test_util]")
{
    const auto prepared = Prepare({
        "anet-core-test.exe",
        "--anet-test-failure-dialog=on",
        "[test_util]",
        "--success",
    });
    const std::vector<std::string> expected{
        "anet-core-test.exe",
        "[test_util]",
        "--success",
    };

    CHECK(prepared.failure_dialog_enabled == true);
    CHECK(ToStrings(prepared) == expected);
}

TEST_CASE("PrepareTestArgs accepts CLI off value", "[test_util]")
{
    const auto prepared = Prepare({
        "anet-core-test.exe",
        "--anet-test-failure-dialog=off",
        "[test_util]",
    }, "on");
    const std::vector<std::string> expected{ "anet-core-test.exe", "[test_util]" };

    CHECK(prepared.failure_dialog_enabled == false);
    CHECK(ToStrings(prepared) == expected);
}

TEST_CASE("PrepareTestArgs accepts environment value", "[test_util]")
{
    const auto on = Prepare({ "anet-core-test.exe" }, "on");
    const auto off = Prepare({ "anet-core-test.exe" }, "off");

    CHECK(on.failure_dialog_enabled == true);
    CHECK(off.failure_dialog_enabled == false);
}

TEST_CASE("PrepareTestArgs rejects invalid failure dialog values", "[test_util]")
{
    RequireInvalidArgs(
        { "anet-core-test.exe", "--anet-test-failure-dialog" },
        nullptr,
        "Missing value");
    RequireInvalidArgs(
        { "anet-core-test.exe", "--anet-test-failure-dialog=yes" },
        nullptr,
        "Invalid --anet-test-failure-dialog value");
    RequireInvalidArgs(
        { "anet-core-test.exe", "--anet-test-failure-dialog:on" },
        nullptr,
        "Invalid --anet-test-failure-dialog format");
    RequireInvalidArgs(
        { "anet-core-test.exe" },
        "yes",
        "Invalid ANET_TEST_FAILURE_DIALOG value");
}
