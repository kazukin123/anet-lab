
#include "app.hpp"

bool MyApp::OnInit() {
    tb_logger = std::make_unique<simplelog::JsonlLogger>("train.jsonl");
    auto* frame = new CartPoleFrame("CartPole RL");
    frame->Show(true);
    return true;
}

wxIMPLEMENT_APP(MyApp);
