// EvalPanel.cpp

#include "EvalPanel.hpp"
#include <algorithm>
#include <cmath>
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/config.hpp"
#include "anet/rl.hpp"
#include "anet/observers.hpp"
#include "RunnerApp.hpp"


namespace LOG = anet::log;

namespace eval_panel_detail {

void ValidateEvalFps(float fps, const char* key)
{
	if (!std::isfinite(fps) || fps < 0.0f || fps > 1000.0f) {
		ANET_SYSTEM_ERROR("Invalid " << key << ": value=" << fps
			<< " (expected: finite number in [0, 1000])");
	}
}

}  // namespace eval_panel_detail

using namespace eval_panel_detail;


EvalPanelModelSyncMode EvalPanelModelSyncConfig::GetMode() const
{
	const auto normalized = anet::ToLower(anet::TrimCopy(mode));
	if (normalized == "shared") return EvalPanelModelSyncMode::Shared;
	if (normalized == "frame") return EvalPanelModelSyncMode::Frame;
	if (normalized == "time") return EvalPanelModelSyncMode::Time;
	if (normalized == "episode") return EvalPanelModelSyncMode::Episode;
	ANET_SYSTEM_ERROR("Invalid app.eval_panel.model_sync.mode: " << mode
		<< " (expected: shared, frame, time, episode)");
	return EvalPanelModelSyncMode::Shared;
}

bool EvalPanelModelSyncConfig::UsesClonedModel() const
{
	return GetMode() != EvalPanelModelSyncMode::Shared;
}

void EvalPanelModelSyncConfig::Validate() const
{
	const auto sync_mode = GetMode();
	if (sync_mode == EvalPanelModelSyncMode::Frame && frame_interval <= 0) {
		ANET_SYSTEM_ERROR("Invalid app.eval_panel.model_sync.frame_interval: " << frame_interval
			<< " (expected: positive integer)");
	}
	if (sync_mode == EvalPanelModelSyncMode::Time && time_interval_ms <= 0) {
		ANET_SYSTEM_ERROR("Invalid app.eval_panel.model_sync.time_interval_ms: " << time_interval_ms
			<< " (expected: positive integer)");
	}
	if (sync_mode == EvalPanelModelSyncMode::Episode && episode_interval <= 0) {
		ANET_SYSTEM_ERROR("Invalid app.eval_panel.model_sync.episode_interval: " << episode_interval
			<< " (expected: positive integer)");
	}
}

void EvalPanelConfig::Validate() const
{
	ValidateEvalFps(fps, "app.eval_panel.fps");
	model_sync.Validate();
}

EvalPanel::EvalPanel(wxWindow* parent, const EvalPanelConfig& config)
	: wxPanel(parent), config_(config), update_timer_(this, wxID_ANY)
{
	config_.Validate();
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

	// Timer開始。Eval FPS は表示だけでなく Eval step の実行周期を表す。
	Bind(wxEVT_TIMER, &EvalPanel::OnTimer, this, update_timer_.GetId());
	SetFps(config_.fps);
}

void EvalPanel::SetFps(float fps)
{
	ValidateEvalFps(fps, "EvalPanel::SetFps fps");

	// 実行時変更では既存 timer を止めてから、新しい周期を確定する。
	update_timer_.Stop();
	const int interval = std::max(1, static_cast<int>(std::lround(1000.0 / fps)));
	ANET_LOG_DEBUG("interval=" << interval);
	if (interval <= 0) {
		update_timer_.Stop();
	} else {
		update_timer_.Start(interval);
	}
}

void EvalPanel::DoStep()
{
	SyncBeforeManualStep();
	runner_->DoStep();
	SyncAfterStep(runner_->GetCounts());
	view_->CaptureViewData();
	Refresh();
}

void EvalPanel::DoStep(int64_t action)
{
	SyncBeforeManualStep();
	runner_->DoStep(action);
	SyncAfterStep(runner_->GetCounts());
	view_->CaptureViewData();
	Refresh();
}

void EvalPanel::TogglePause()
{
	is_pause_ = !is_pause_;
	if (!is_pause_ && UsesClonedModel()) {
		SyncModel();
	}
	auto log_str = std::string("Evaluation ") + (is_pause_ ? "paused" : "resumed");
	LOG::info() << log_str;
	wxGetApp().GetMainFrame()->SetStatusText(log_str);
	wxGetApp().FlushRunOutputs();
}

bool EvalPanel::UsesClonedModel() const
{
	return config_.model_sync.UsesClonedModel();
}

void EvalPanel::SyncModel()
{
	ANET_PROFILE_FUNC();

	if (!UsesClonedModel() || runner_ == nullptr) return;

	runner_->Sync();
	frames_since_model_sync_ = 0;
	episodes_since_model_sync_ = 0;
	last_model_sync_time_ = std::chrono::steady_clock::now();
}

void EvalPanel::SyncBeforeFrame()
{
	if (!UsesClonedModel()) return;

	const auto sync_mode = config_.model_sync.GetMode();
	if (sync_mode == EvalPanelModelSyncMode::Frame) {
		++frames_since_model_sync_;
		if (frames_since_model_sync_ >= config_.model_sync.frame_interval) {
			SyncModel();
		}
	} else if (sync_mode == EvalPanelModelSyncMode::Time) {
		const auto now = std::chrono::steady_clock::now();
		const auto interval = std::chrono::milliseconds(config_.model_sync.time_interval_ms);
		if (last_model_sync_time_ == std::chrono::steady_clock::time_point{}
			|| now - last_model_sync_time_ >= interval) {
			SyncModel();
		}
	}
}

void EvalPanel::SyncBeforeManualStep()
{
	if (!UsesClonedModel()) return;

	const auto sync_mode = config_.model_sync.GetMode();
	if (sync_mode == EvalPanelModelSyncMode::Frame || sync_mode == EvalPanelModelSyncMode::Time) {
		SyncBeforeFrame();
	}
}

anet::rl::ControlSignal EvalPanel::SyncAfterStep(const anet::rl::StepCounts&)
{
	ANET_PROFILE_FUNC();

	if (UsesClonedModel()
		&& config_.model_sync.GetMode() == EvalPanelModelSyncMode::Episode
		&& runner_ != nullptr
		&& runner_->LastStepHadEpisodeEnd()) {
		++episodes_since_model_sync_;
		if (episodes_since_model_sync_ >= config_.model_sync.episode_interval) {
			SyncModel();
		}
	}
	return anet::rl::ControlSignal::CONTINUE;
}

void EvalPanel::OnTimer(wxTimerEvent& event)
{
	ANET_PROFILE_FUNC();

	if (is_pause_) return;

	// 評価エピソードを回す（Observer経由でeval_canvas更新)
	if (config_.step_per_frame > 0) {
		ANET_PROFILE_SCOPE_FULL(eval_update_frame, "EvalPanel::OnEvalTimer.DoUpdateFrame");
		SyncBeforeFrame();
		runner_->DoUpdateFrame(
			config_.step_per_frame,
			nullptr,
			[this](const anet::rl::StepCounts& counts)
			{
				return SyncAfterStep(counts);
			});
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
