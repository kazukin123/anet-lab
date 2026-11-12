#include "anet/catch_test.hpp"

#include "anet/test_util.hpp"

#ifdef _WIN32
#include <windows.h>
#endif
#include <clocale>
#include <exception>

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

    anet::test::PreparedTestArgs test_args;
    try {
        test_args = anet::test::PrepareTestArgs(argc, argv);
    } catch (const std::exception& e) {
        return anet::test::ReportTestArgsError(e);
    }
    anet::test::SetupTestFailureDialog(test_args.failure_dialog_enabled);

    anet::test::StderrLogGuard log_target_guard;

    // どのテストより先に、プロセス全体の MKLDNN を 1 回だけ切る。
    // Debug 版 libtorch の oneDNN は CPU の Conv で thread 数を確かめる assert（nthr_ == nthr）を起こし、
    // 実行体ごと止まる（PyTorch issue #104421）。Catch2 の既定の実行順はランダムなので、
    // テストの中で切ると、そのテストが先に回ったかで止まるかが変わる（PRD124）。
    // ビルド構成を問わず切る。Release 版 libtorch と GPU の本番では起きないため、本番の初期化には入れない（PRD028、PRD124 D4）。
    at::globalContext().setUserEnabledMkldnn(false);

    Catch::Session session;
    session.configData().showDurations = Catch::ShowDurations::Always;
    return session.run(test_args.Argc(), test_args.Argv());
}
