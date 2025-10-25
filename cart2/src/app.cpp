
#include "app.hpp"
#include "CartPoleFrame.hpp"
#include <wx/fileconf.h>
#include <wx/cmdline.h>


bool MyApp::OnInit() {

    //wxCmdLineParser parser(argc, (wchar_t**)argv);
    ////parser.Parse(false);
    //int size = parser.GetParamCount();
    //for (int i = 0; i < parser.GetParamCount(); i++) {
    //    wxString param = parser.GetParam(i);
    //}

    properties_ = std::make_unique<Properties>("CartPoleRLGUI.txt");

    auto backend = std::make_unique<JsonlBackend>();
	mt_logger = std::make_unique<MetricsLogger>(std::move(backend), "logs");

    auto* frame = new CartPoleFrame("CartPole RL");
    frame->Show(true);
    return true;
}

wxIMPLEMENT_APP(MyApp);

