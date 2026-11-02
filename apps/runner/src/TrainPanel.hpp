// TrainPanel.hpp

#pragma once

#include <wx/wx.h>
#include "anet/config.hpp"
#include "anet/rl.hpp"
#include "anet/trainer.hpp"
#include "anet/gui.hpp"

struct TrainPanelConfig {
	float fps = 10.0f;

	void Validate() const;
};

class TrainPanel : public anet::rl::gui::Panel {
public:
	TrainPanel(wxWindow* parent, const TrainPanelConfig& config);

	void Initialize(std::shared_ptr<anet::rl::RunManager> run_manager);
	void SetFps(float fps);
	void SetViewVisible(bool visible);
protected:
	void OnTimer(wxTimerEvent& event);
	void OnClose(wxCloseEvent& event);
	void OnSize(wxSizeEvent& event);
	//void OnMouseLeftClick(wxMouseEvent& event);
	//void OnMouseRightClick(wxMouseEvent& event);
private:
	void RefreshViewSurface();
	void UpdateTimerState();

	float fps_;
	bool view_visible_ = true;
	bool initialized_ = false;
	std::shared_ptr<anet::rl::gui::View> view_;
	std::shared_ptr<anet::rl::TrainObserver> observer_;
	wxWindow* view_window_ = nullptr;
protected:
	wxTimer update_timer_;
};
