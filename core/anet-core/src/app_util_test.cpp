#include "catch.hpp"

#include "anet/app_util.hpp"

TEST_CASE("app_util exposes executable-based config directory", "[app_util]")
{
    const auto exe_path = anet::GetExecutablePath();

    CHECK_FALSE(exe_path.empty());
    CHECK(anet::GetExecutableDir() == exe_path.parent_path());
    CHECK(anet::GetExecutableConfigDir() == anet::GetExecutableRootDir() / "config");
}
