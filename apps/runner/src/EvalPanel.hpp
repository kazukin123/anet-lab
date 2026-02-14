// EvalPanel.hpp

#pragma once

#include <memory>
#include <wx/wx.h>
#include "anet/config.hpp"
#include "anet/rl.hpp"
#include "anet/gui.hpp"
#include "anet/trainer.hpp"


struct EvalPanelConfig {
	float fps = 30.0f;
	int step_per_frame = 3;
	bool auto_start = true;
};

class EvalPanel final : public wxPanel {
public:
	EvalPanel(wxWindow* parent, const EvalPanelConfig& config);
	~EvalPanel();

	void Initialize(std::shared_ptr<anet::rl::RunManager> run_manager, std::shared_ptr<anet::rl::EvalRunner> runner);
	void DoClose();

	void TogglePause();
	void DoStep();
	void DoStep(int64_t action);
protected:
	void OnTimer(wxTimerEvent& event);
	void OnClose(wxCloseEvent& event);
private:
	const EvalPanelConfig config_;
	std::shared_ptr<anet::rl::EvalRunner> runner_ = nullptr;
	std::shared_ptr<anet::rl::gui::View> view_ = nullptr;
	std::shared_ptr<anet::rl::TrainObserver> observer_ = nullptr;
	wxWindow* view_window_ = nullptr;
	wxTimer update_timer_;
	bool is_pause_ = false;
};
