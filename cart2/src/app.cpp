
#include "app.hpp"
#include "CartPoleFrame.hpp"
#include <wx/fileconf.h>
#include <wx/cmdline.h>

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

    properties_ = std::make_unique<Properties>("CartPoleRLGUI.txt");

    auto backend = std::make_unique<JsonlBackend>();
	mt_logger = std::make_unique<MetricsLogger>(std::move(backend), "logs");

    bool enable_image_log;
    properties_->Read("log.enable_image_log", enable_image_log, true);
    mt_logger->SetEnableImageLog(enable_image_log);

    auto* frame = new CartPoleFrame("CartPole RL");
    frame->Show(true);
    return true;
}

wxIMPLEMENT_APP(MyApp);

