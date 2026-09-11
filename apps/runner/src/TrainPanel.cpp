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
	config_(config), update_timer_(this, wxID_ANY)
{
	config_.Validate();
	SetBackgroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_BACKGROUND));
	SetDoubleBuffered(true);
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

	// Timer開始。0 は表示更新を意図的に停止する。
	Bind(wxEVT_TIMER, &TrainPanel::OnTimer, this, update_timer_.GetId());
	SetFps(config_.fps);
}

void TrainPanel::SetFps(float fps)
{
	ValidateTrainFps(fps, "TrainPanel::SetFps fps");

	// 実行時変更では既存 timer を止めてから、新しい周期を確定する。
	update_timer_.Stop();
	if (fps == 0.0f) {
		ANET_LOG_DEBUG("TrainPanel view updates disabled");
		return;
	}

	const int interval = std::max(1, static_cast<int>(std::lround(1000.0 / fps)));
	ANET_LOG_DEBUG("interval=" << interval);
	update_timer_.Start(interval);
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
