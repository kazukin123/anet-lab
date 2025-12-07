// app.cpp
#include "app.hpp"
#include <filesystem>
#include <wx/stdpaths.h>
#include <wx/cmdline.h>
#include <wx/filename.h>
#include "anet/metrics_logger.hpp"
#include "CartPoleFrame.hpp"

wxString GetExeDir() {
    wxStandardPaths& sp = wxStandardPaths::Get();
    wxString exe_path = sp.GetExecutablePath();      // フルパス (C:\proj\bin\myapp.exe 等)
    wxFileName fn(exe_path);
    return fn.GetPath();                            // ディレクトリ部分を返す
}

std::filesystem::path GetProjectRootDir()
{
    std::filesystem::path exePath = GetExeDir().ToStdString();  // 既存の GetExeDir を利用
    return exePath.parent_path().parent_path();    // exe の親ディレクトリを返す
}

std::string GetConfigFilePath() {
    return (GetProjectRootDir() / "config" / "CartPoleRLGUI.txt").string();  // パスを結合
}

std::string GetLogsPath() {
    return (GetProjectRootDir() / "runs").string();
}

static wxCmdLineEntryDesc desc[] = {
    // kind,              short-name, long-name, usage,      type,                  flags
    //{ wxCMD_LINE_SWITCH, "v",         "verbose", "エラー表示を饒舌に" }, // wxCMD_LINE_SWITCH:A boolean argument of the program;    e.g. -v to enable verbose mode.
    //{ wxCMD_LINE_OPTION, "f",         "file",    "設定ファイルのパス" }, // wxCMD_LINE_OPTION:An argument with an associated value; e.g. -o filename

    {
        wxCMD_LINE_PARAM,              // 種別：位置パラメータ
        nullptr,                       // 短いオプション名なし
        nullptr,                       // 長いオプション名なし
        "key=value pairs",             // 説明文
        wxCMD_LINE_VAL_STRING,         // 文字列として受け取る
        wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE      // 複数 OK
    },
    //{ wxCMD_LINE_PARAM,  NULL,        NULL,  "引数",     wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },  // A parameter: a required program argument.
    { wxCMD_LINE_USAGE_TEXT, NULL,    NULL,    "CartPoleRLGUI.exe key1=value1 key2=value2" },     //  Additional usage text.
    { wxCMD_LINE_NONE } // 終了マーク
};

bool MyApp::OnInit() {
    wxInitAllImageHandlers();

    wxCmdLineParser cmdline_(desc, argc, (wchar_t**)argv);
    if (cmdline_.Parse(true))
        return false;

    config_mgr_ = std::make_unique<anet::ConfigManager>(GetConfigFilePath(), &cmdline_);
    anet::MetricsLogger::Init(std::make_unique<anet::JsonlBackend>(), GetLogsPath());
    auto* frame = new CartPoleFrame("CartPole RL");
    frame->Show(true);
    return true;
}

int MyApp::OnExit()
{
    anet::MetricsLogger::Reset();
    return 0;
}

wxIMPLEMENT_APP(MyApp);

