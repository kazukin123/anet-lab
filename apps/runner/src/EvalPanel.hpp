// EvalPanel.hpp

#pragma once

#include <chrono>
#include <memory>
#include <wx/wx.h>
#include "anet/config.hpp"
#include "anet/rl.hpp"
#include "anet/gui.hpp"
#include "anet/trainer.hpp"

enum class EvalPanelModelSyncMode {
	Shared,
	Frame,
	Time,
	Episode,
};

struct EvalPanelModelSyncConfig {
	std::string mode = "frame";
	int frame_interval = 1;
	int time_interval_ms = 1000;
	int episode_interval = 1;

	EvalPanelModelSyncMode GetMode() const;
	bool UsesClonedModel() const;
	void Validate() const;
};

struct EvalPanelConfig {
	float fps = 30.0f;
	int step_per_frame = 3;
	bool auto_start = true;
	std::string eval_config_tag;
	EvalPanelModelSyncConfig model_sync;

	void Validate() const;
};

class EvalPanel final : public wxPanel {
public:
	EvalPanel(wxWindow* parent, const EvalPanelConfig& config);
	~EvalPanel();

	void Initialize(std::shared_ptr<anet::rl::RunManager> run_manager, std::shared_ptr<anet::rl::EvalRunner> runner);
	void DoClose();

	const EvalPanelConfig& GetConfig() const { return config_; }
	bool IsPaused() const { return is_pause_; }
	void SetFps(float fps);
	void TogglePause();
	void DoStep();
	void DoStep(int64_t action);
protected:
	void OnTimer(wxTimerEvent& event);
	void OnClose(wxCloseEvent& event);
private:
	bool UsesClonedModel() const;
	void SyncModel();
	void SyncBeforeFrame();
	void SyncBeforeManualStep();
	anet::rl::ControlSignal SyncAfterStep(const anet::rl::StepCounts& counts);
private:
	const EvalPanelConfig config_;
	std::shared_ptr<anet::rl::EvalRunner> runner_ = nullptr;
	std::shared_ptr<anet::rl::gui::View> view_ = nullptr;
	std::shared_ptr<anet::rl::TrainObserver> observer_ = nullptr;
	wxWindow* view_window_ = nullptr;
	wxTimer update_timer_;
	int frames_since_model_sync_ = 0;
	int episodes_since_model_sync_ = 0;
	std::chrono::steady_clock::time_point last_model_sync_time_;
	bool is_pause_ = false;
};
