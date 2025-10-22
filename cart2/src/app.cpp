
#include "app.hpp"
#include "CartPoleFrame.hpp"

bool MyApp::OnInit() {
    auto backend = std::make_unique<JsonlBackend>();
	mt_logger = std::make_unique<MetricsLogger>(std::move(backend), "logs");

    auto* frame = new CartPoleFrame("CartPole RL");
    frame->Show(true);
    return true;
}

wxIMPLEMENT_APP(MyApp);
