// EvalPanel.hpp

#pragma once

#include <wx/wx.h>
#include "anet/config.hpp"
#include "anet/rl.hpp"
#include "anet/gui.hpp"
#include "anet/trainer.hpp"

#include "anet/trainer.hpp"


struct EvalPanelConfig {
	float fps = 30.0f;
	int step_per_frame = 3;
};

class EvalPanel : public wxPanel {
public:
	EvalPanel(wxWindow* parent, const EvalPanelConfig& config);

	void Initialize(std::shared_ptr<anet::rl::DefaultTrainer> trainer);

	void DoStep(int64_t action);
	void TogglePause();
protected:
	void OnTimer(wxTimerEvent& event);
	void OnClose(wxCloseEvent& event);
private:
	const EvalPanelConfig config_;
	std::shared_ptr<anet::rl::EvalRunner> eval_runner_;
	std::shared_ptr<anet::rl::gui::View> view_;
	std::shared_ptr<anet::rl::TrainObserver> observer_;
	wxWindow* view_window_;
	wxTimer update_timer_;
	bool is_pause_ = false;
};
