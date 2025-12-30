// EvalPanel.cpp

#include "EvalPanel.hpp"
#include "RunnerApp.hpp"
#include "anet/log.hpp"
#include "anet/config.hpp"
#include "anet/profile.hpp"
#include "anet/rl.hpp"
#include "anet/observers.hpp"

namespace LOG = anet::log;


EvalPanel::EvalPanel(wxWindow* parent, const EvalPanelConfig& config)
	: wxPanel(parent), config_(config), update_timer_(this, wxID_ANY)
{
}

void EvalPanel::Initialize(std::shared_ptr<anet::rl::DefaultTrainer> trainer)
{
	// View生成
	view_ = wxGetApp().CreateExperinceView(this);
	view_window_ = view_->AsWindow();

	// レイアウト
	wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
	sizer->Add(view_window_, 1, wxEXPAND | wxALL, 5);
	this->SetSizer(sizer);
	this->Layout();

	// EvalRunner生成(これで独自の評価エピソードを回す。ENVはTrainと同等の別インスタンス、AGENTはTrainと共用)
	eval_runner_ = trainer->CreateEvalRunner();

	// Observer生成
	auto notifier = eval_runner_->GetNotifier();
	this->observer_ = notifier->Attach<anet::rl::FunctionTrainObserver>(
		[this](const anet::rl::TrainEvent& event)
		{
			view_->UpdateViewData(event, true);		// Step毎に画面反映したいかもなのでforce=true
			//view_->CaptureViewData();
		},
		"EvalPanel");

	// Timer開始
	Bind(wxEVT_TIMER, &EvalPanel::OnTimer, this, update_timer_.GetId());
	int interval = 1000 / config_.fps;
	ANET_LOG_DEBUG("interval=" << interval);
	update_timer_.Start(interval); 
}

void EvalPanel::DoStep(int64_t action)
{
	eval_runner_->DoStep(action);
	view_->CaptureViewData();
	Refresh();
}

void EvalPanel::TogglePause()
{
	is_pause_ = !is_pause_;
	LOG::info() << "Eval " << (is_pause_ ? "paused." : " resumed.");
}

void EvalPanel::OnTimer(wxTimerEvent& event)
{
	anet::ProfileRange r("EvalPanel::OnTimer");

	if (is_pause_) return;

	// 評価エピソードを回す（Observer経由でeval_canvas更新)
	if (config_.step_per_frame > 0) {
		anet::ProfileRange r("LunarLanderFrame::OnEvalTimer.DoUpdateFrame");
		eval_runner_->DoUpdateFrame(config_.step_per_frame);
	}

	// データ断面をキャプチャ
	view_->CaptureViewData();

	// 表示反映（リクエスト）
	Refresh();
}

void EvalPanel::OnClose(wxCloseEvent& event)
{
	auto notifier = wxGetApp().GetTrainer()->GetNotifier();
	notifier->Detach(this->observer_);
}
