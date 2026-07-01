#include "catch.hpp"

#include "anet/test_util.hpp"

#ifdef _WIN32
#include <windows.h>
#endif
#include <clocale>

namespace {

void SetupUtf8Console()
{
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::setlocale(LC_CTYPE, ".UTF-8");
}

} // namespace

int main(int argc, char* argv[])
{
    SetupUtf8Console();
    anet::test::StderrLogGuard log_target_guard;

    Catch::Session session;
    session.configData().showDurations = Catch::ShowDurations::Always;
    return session.run(argc, argv);
}
