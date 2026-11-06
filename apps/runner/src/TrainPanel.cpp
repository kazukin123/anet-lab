// TrainPanel.cpp

#include "TrainPanel.hpp"
#include <algorithm>
#include <cmath>
#include "RunnerApp.hpp"
#include "anet/config.hpp"
#include "anet/rl.hpp"
#include "anet/observers.hpp"

namespace train_panel_detail {

void ValidateTrainFps(float fps, const char* key)
{
	if (!std::isfinite(fps) || fps < 0.0f || fps > 1000.0f) {
		ANET_SYSTEM_ERROR("Invalid " << key << ": value=" << fps
			<< " (expected: finite number in [0, 1000])");
	}
}

}  // namespace train_panel_detail

using namespace train_panel_detail;


void TrainPanelConfig::Validate() const
{
	ValidateTrainFps(fps, "app.train_panel.fps");
}


TrainPanel::TrainPanel(wxWindow* parent, const TrainPanelConfig& config)
	: anet::rl::gui::Panel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize,
		wxTAB_TRAVERSAL | wxFULL_REPAINT_ON_RESIZE),
	fps_(config.fps), update_timer_(this, wxID_ANY)
{
	config.Validate();
	SetBackgroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_BACKGROUND));
	Bind(wxEVT_SIZE, &TrainPanel::OnSize, this);
}

void TrainPanel::Initialize(std::shared_ptr<anet::rl::RunManager> run_manager)
{
	// View生成
	view_ = wxGetApp().CreateExperinceView(this);
	view_window_ = view_->AsWindow();

	// レイアウト
	wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
	sizer->Add(view_window_, 1, wxEXPAND | wxALL, 5);
	this->SetSizer(sizer);
	this->Layout();

	// Observer生成
	auto train_runner = run_manager->GetTrainRunner();
	auto notifier = run_manager->GetNotifier();
	this->observer_ = notifier->AttachScoped<anet::rl::FunctionTrainObserver>(
		train_runner,
		[this](const anet::rl::TrainEvent& event)
		{
			view_->UpdateViewData(event);
		},
		"TrainPanel");

	// 初期化前に受け取った表示状態と選択FPSを、そのまま初回の判定に使う。
	Bind(wxEVT_TIMER, &TrainPanel::OnTimer, this, update_timer_.GetId());
	initialized_ = true;
	UpdateTimerState();
}

void TrainPanel::SetFps(float fps)
{
	ValidateTrainFps(fps, "TrainPanel::SetFps fps");

	// 非表示中と初期化前も、次に表示するときのFPSを保持する。
	fps_ = fps;
	UpdateTimerState();
}

void TrainPanel::SetViewVisible(bool visible)
{
	view_visible_ = visible;
	UpdateTimerState();
}

void TrainPanel::UpdateTimerState()
{
	// Initialize が timer の購読を用意するまでは、timer 自体には触らない。
	if (!initialized_) return;
	if (!view_visible_ || fps_ == 0.0f) {
		if (update_timer_.IsRunning()) update_timer_.Stop();
		return;
	}

	// 同じ表示状態の再適用で、稼働中timerの周期をやり直さない。
	const int interval = std::max(1, static_cast<int>(std::lround(1000.0 / fps_)));
	if (!update_timer_.IsRunning() || update_timer_.GetInterval() != interval) {
		update_timer_.Start(interval);
	}
}

void TrainPanel::OnTimer(wxTimerEvent& event)
{
	//ANET_LOG_DEBUG("TrainPanel::OnTimer size=(" << this->GetSize().x << ", " << this->GetSize().y << ")");
	wxWindow* parent = GetParent();
	wxSize parent_size = parent ? parent->GetClientSize() : wxSize(-1, -1);
	//ANET_LOG_DEBUG("TrainPanel::OnTimer parent_size=(" << parent_size.x << ", " << parent_size.y << ")");

	// データ断面をキャプチャ
	view_->CaptureViewData();

	// 表示反映（リクエスト）
	RefreshViewSurface();
}

void TrainPanel::OnClose(wxCloseEvent& event)
{
	update_timer_.Stop();
	auto notifier = wxGetApp().GetRunManager().GetNotifier();
	notifier->Detach(this->observer_);
}

void TrainPanel::OnSize(wxSizeEvent& event)
{
	RefreshViewSurface();
	event.Skip();
}

void TrainPanel::RefreshViewSurface()
{
	Refresh(true);
	if (view_window_) {
		view_window_->Refresh(false);
	}
}
