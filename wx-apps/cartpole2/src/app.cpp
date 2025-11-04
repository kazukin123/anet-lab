
#include "app.hpp"
#include "CartPoleFrame.hpp"
#include <wx/cmdline.h>
#include <wx/stdpaths.h>
#include <wx/filename.h>
#include <filesystem>

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
    return (GetProjectRootDir() / "logs").string();
}

static wxCmdLineEntryDesc desc[] = {
    // kind,              short-name, long-name, usage,      type,                  flags
    //{ wxCMD_LINE_SWITCH, "v",         "verbose", "エラー表示を饒舌に" }, // wxCMD_LINE_SWITCH:A boolean argument of the program;    e.g. -v to enable verbose mode.
    //{ wxCMD_LINE_OPTION, "f",         "file",    "設定ファイルのパス" }, // wxCMD_LINE_OPTION:An argument with an associated value; e.g. -o filename

    { wxCMD_LINE_OPTION, "a",         "agent",   "agent.presetの上書き値", wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL }, // wxCMD_LINE_OPTION:An argument with an associated value; e.g. -o filename
    { wxCMD_LINE_OPTION, "t",         "train",   "train.presetの上書き値", wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL }, // wxCMD_LINE_OPTION:An argument with an associated value; e.g. -o filename

    //{ wxCMD_LINE_PARAM,  NULL,        NULL,      "引数",     wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },  // A parameter: a required program argument.
    { wxCMD_LINE_USAGE_TEXT, NULL,    NULL,      "CartPoleRLGUI.exe --agent=agent_rb" },     //  Additional usage text.
    { wxCMD_LINE_NONE } // 終了マーク
};

bool MyApp::OnInit() {
    wxInitAllImageHandlers();

    cmdline_ = std::make_unique<wxCmdLineParser>(desc, argc, (wchar_t**)argv);
    if (cmdline_->Parse(true)) {
        return false;
    }
    properties_ = std::make_unique<anet::Properties>(GetConfigFilePath());
    anet::MetricsLogger::Init(std::make_unique<anet::JsonlBackend>(), GetLogsPath());

    bool enable_image_log;
    properties_->Read("log.enable_image_log", enable_image_log, true);
    anet::MetricsLogger::Instance()->SetEnableImageLog(enable_image_log);

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

