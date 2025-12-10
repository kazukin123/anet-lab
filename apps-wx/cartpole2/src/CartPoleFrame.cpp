#include "CartPoleFrame.hpp"
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <torch/torch.h>
#include <wx/log.h>
#include <wx/sizer.h>
#include "anet/tensor_utils.hpp"
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/observers.hpp"
#include "anet/replay_buffer.hpp"
#include "CartPoleCanvas.hpp"
#include "UISnapshot.hpp"
#include "app.hpp"

wxBEGIN_EVENT_TABLE(CartPoleFrame, wxFrame)
EVT_TIMER(wxID_ANY, CartPoleFrame::OnTimer)
EVT_LEFT_DOWN(CartPoleFrame::OnMouseClick)
wxEND_EVENT_TABLE()

namespace LOG = anet::log;


CartPoleFrame::CartPoleFrame(const wxString& title, int timer_ms)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(800, 800))
    , timer_ms_(timer_ms), timer(this, wxID_ANY)
{
    // GUIレイアウト ---
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

    // ログレベル
#if ANET_ENABLE_DEBUGINFO
    wxLog::SetLogLevel(wxLOG_Debug);
#endif

    // ログ出力先をこのクラスに設定
    wxLog::SetActiveTarget(this);
    LOG::info() << "CartPoleRLGUI started.";

    //// Train開始！
    //wxGetApp().StartTraining();

    // Train終了イベント
    Bind(wxEVT_TRAINER_EXIT, [this](wxCommandEvent&) {
        LOG::info() << "Stop training requested. Exiting.";
        //wxGetApp().StopTraining();
        //wxGetApp().Exit();                // UI スレッドで呼ぶ
        Close(true);    // Frameを閉じる
        });
    //Bind(wxEVT_CLOSE_WINDOW, &CartPoleFrame::OnClose, this);

    // タイマー開始
    Bind(wxEVT_TIMER, &CartPoleFrame::OnTimer, this);
    timer.Start(timer_ms_);  // 学習＆描画更新
}

CartPoleFrame::~CartPoleFrame()
{
    wxLog::SetActiveTarget(NULL);
}

void CartPoleFrame::OnTimer(wxTimerEvent& event)
{
    anet::ProfileRange r("CartPoleFrame::OnTimer");

    // 再入防止
    timer.Stop();

    // スナップショットから画面描画
    auto snapshot = wxGetApp().GetUISnapshot();
    if (snapshot.has_value()) {
        this->canvas->SetUIData(*snapshot);
    }

    // 画面表示更新
    Refresh();

    // タイマー再開
    this->timer.Start();
}

void CartPoleFrame::DoLogText(const wxString& msg)
{
    this->logBox->AppendText(msg);
    this->logBox->AppendText("\n");
}

void CartPoleFrame::AddPlotData(float value)
{
    plotPanel->AddData(value);
}

void CartPoleFrame::OnMouseClick(wxMouseEvent& event)
{
    wxGetApp().ToggleTraining();
}
