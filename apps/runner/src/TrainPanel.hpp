// TrainPanel.hpp

#pragma once

#include <wx/wx.h>
#include "anet/config.hpp"
#include "anet/rl.hpp"
#include "anet/trainer.hpp"
#include "anet/gui.hpp"

struct TrainPanelConfig {
	float fps = 10.0f;
};

class TrainPanel : public anet::rl::gui::Panel {
public:
	TrainPanel(wxWindow* parent, const TrainPanelConfig& config);

	void Initialize(std::shared_ptr<anet::rl::RunManager> run_manager);
protected:
	void OnTimer(wxTimerEvent& event);
	void OnClose(wxCloseEvent& event);
	void OnSize(wxSizeEvent& event);
	//void OnMouseLeftClick(wxMouseEvent& event);
	//void OnMouseRightClick(wxMouseEvent& event);
private:
	void RefreshViewSurface();

	const TrainPanelConfig config_;
	std::shared_ptr<anet::rl::gui::View> view_;
	std::shared_ptr<anet::rl::TrainObserver> observer_;
	wxWindow* view_window_ = nullptr;
	wxTimer update_timer_;
};
