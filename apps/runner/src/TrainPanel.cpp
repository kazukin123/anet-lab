// TrainPanel.cpp

#include "TrainPanel.hpp"
#include "RunnerApp.hpp"
#include "anet/config.hpp"
#include "anet/rl.hpp"
#include "anet/observers.hpp"


TrainPanel::TrainPanel(wxWindow* parent, const TrainPanelConfig& config)
	: anet::rl::gui::Panel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize,
		wxTAB_TRAVERSAL | wxFULL_REPAINT_ON_RESIZE),
	config_(config), update_timer_(this, wxID_ANY)
{
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

	// Timer開始
	Bind(wxEVT_TIMER, &TrainPanel::OnTimer, this, update_timer_.GetId());
	int interval = 1000 / config_.fps;
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
