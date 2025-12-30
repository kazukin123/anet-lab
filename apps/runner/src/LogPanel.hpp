// LogPanel.hpp

#pragma once

#include <wx/wx.h>
#include "anet/gui.hpp"


class LogPanel : public anet::rl::gui::Panel {
public:
    explicit LogPanel(wxWindow* parent);
    virtual ~LogPanel();

private:
    /**
        * @brief 内部コンポーネントの作成と配置
        */
    void SetupControls();

    /**
        * @brief wxLog のターゲットをこのビューに設定する
        */
    void SetupLogTarget();

    /**
        * @brief wxLog のターゲットを元に戻す
        */
    void RestoreLogTarget();

private:
    wxTextCtrl* text_ctrl_ = nullptr;
    wxLog* old_log_target_ = nullptr; // 以前のログターゲット（復元用）
};

