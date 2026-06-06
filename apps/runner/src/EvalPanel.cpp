// EvalPanel.cpp

#include "EvalPanel.hpp"
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/config.hpp"
#include "anet/rl.hpp"
#include "anet/observers.hpp"
#include "RunnerApp.hpp"


namespace LOG = anet::log;


EvalPanel::EvalPanel(wxWindow* parent, const EvalPanelConfig& config)
	: wxPanel(parent), config_(config), update_timer_(this, wxID_ANY)
{
	is_pause_ = config_.auto_start ? false : true;
}

void EvalPanel::DoClose()
{
	if (update_timer_.IsRunning()) {
		update_timer_.Stop();
	}
	if (observer_) {
		auto notifier = wxGetApp().GetRunManager().GetNotifier();
		if (notifier) {
			notifier->Detach(this->observer_);
		}
		observer_ = nullptr;
	}
}

EvalPanel::~EvalPanel()
{
	DoClose();
}

void EvalPanel::Initialize(std::shared_ptr<anet::rl::RunManager> run_manager, std::shared_ptr<anet::rl::EvalRunner> runner)
{
	// View生成
	view_ = wxGetApp().CreateExperinceView(this);
	view_window_ = view_->AsWindow();

	// レイアウト
	wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
	sizer->Add(view_window_, 1, wxEXPAND | wxALL, 5);
	this->SetSizer(sizer);
	this->Layout();

	// EvalRunnerを保存(これで独自の評価エピソードを回す。ENVはTrainと同等の別インスタンス、AGENTはTrainと共用)
	runner_ = runner;

	// Observer生成
	auto notifier = run_manager->GetNotifier();
	this->observer_ = notifier->AttachScoped<anet::rl::FunctionTrainObserver>(
		runner,
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

void EvalPanel::DoStep()
{
	runner_->DoStep();
	view_->CaptureViewData();
	Refresh();
}

void EvalPanel::DoStep(int64_t action)
{
	runner_->DoStep(action);
	view_->CaptureViewData();
	Refresh();
}

void EvalPanel::TogglePause()
{
	is_pause_ = !is_pause_;
	auto log_str = std::string("Eval ") + (is_pause_ ? "paused." : "resumed.");
	LOG::info() << log_str;
	wxGetApp().GetMainFrame()->SetStatusText(log_str);
	wxGetApp().FlushRunOutputs();
}

void EvalPanel::OnTimer(wxTimerEvent& event)
{
	ANET_PROFILE_FUNC();

	if (is_pause_) return;

	// 評価エピソードを回す（Observer経由でeval_canvas更新)
	if (config_.step_per_frame > 0) {
		ANET_PROFILE_SCOPE_FULL(eval_update_frame, "LunarLanderFrame::OnEvalTimer.DoUpdateFrame");
		runner_->DoUpdateFrame(config_.step_per_frame);
	}

	// データ断面をキャプチャ
	view_->CaptureViewData();

	// 表示反映（リクエスト）
	Refresh();
}

void EvalPanel::OnClose(wxCloseEvent& event)
{
	update_timer_.Stop();
	auto notifier = wxGetApp().GetRunManager().GetNotifier();
	notifier->Detach(this->observer_);
}
