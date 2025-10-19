#include "CartPoleFrame.hpp"
#include "CartPoleCanvas.hpp"
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <torch/torch.h>
#include <iomanip>
#include <sstream>

wxBEGIN_EVENT_TABLE(CartPoleFrame, wxFrame)
EVT_TIMER(wxID_ANY, CartPoleFrame::OnTimer)
wxEND_EVENT_TABLE()

CartPoleFrame::CartPoleFrame(const wxString& title)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(800, 800)),
    device(torch::kCPU),
    timer(this, wxID_ANY)
{
    // --- GUIレイアウト ---
    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    canvas = new CartPoleCanvas(this);
    plotPanel = new PlotPanel(this);
    logBox = new wxTextCtrl(this, wxID_ANY, wxEmptyString,
        wxDefaultPosition, wxSize(800, 150),
        wxTE_MULTILINE | wxTE_READONLY);

    canvas->SetMinSize(wxSize(-1, 280));  // ← 上部の描画エリア固定高さ
    canvas->SetMaxSize(wxSize(-1, 280));  // （上下方向のリサイズ禁止）

    logBox->SetMinSize(wxSize(-1, 150));  // ← 下部ログ固定高さ
    logBox->SetMaxSize(wxSize(-1, 150));

    vbox->Add(canvas, 0, wxEXPAND | wxALL, 5);
    vbox->Add(plotPanel, 1, wxEXPAND | wxALL, 5);
    vbox->Add(logBox, 0, wxEXPAND | wxALL, 5);
    SetSizer(vbox);
    Layout();

    // --- RL初期化 ---
    env = std::make_unique<CartPoleEnv>();
    agent = std::make_unique<RLAgent>(4, 2, device);
    state = env->reset();  // ← reset() は at::Tensor を返す

    // --- タイマー開始 ---
    Bind(wxEVT_TIMER, &CartPoleFrame::OnTimer, this);
    timer.Start(20);  // 20msごとに描画更新

    logBox->AppendText("CartPoleRLGUI started.\n");
}

void CartPoleFrame::OnTimer(wxTimerEvent& event) {
    // --- 学習ステップを高速で複数回回す ---
    for (int i = 0; i < 10; ++i) {  // 10 step per draw frame
        auto action = agent->select_action(state);
        auto [next_state, reward, done] = env->step(action.item<int>());
        agent->update(state, next_state, reward, done);
        state = next_state.clone();

        step_count++;
//        plotPanel->AddReward(reward);
        if (done) {
            episode_count++;
            float total_reward = env->get_total_reward();

            std::ostringstream ss;
            ss << "Episode " << episode_count
                << " finished, total_reward = "
                << std::fixed << std::setprecision(3)
                << total_reward << "\n";
            logBox->AppendText(ss.str());

            plotPanel->AddReward(total_reward);
            state = env->reset();
            break;
        }
    }

    // --- カート位置・角度の描画更新 ---
    canvas->SetState(env->get_x(), env->get_theta());
    canvas->Refresh();

    // --- 100 step ごとにログを出力 ---
    if (step_count % 100 == 0) {
        std::ostringstream ss;
        ss << "Step " << step_count
            << ", eps=" << std::fixed << std::setprecision(2) << agent->epsilon
            << ", latest_loss=" << agent->latest_loss
            << "\n";
        logBox->AppendText(ss.str());
    }
}
