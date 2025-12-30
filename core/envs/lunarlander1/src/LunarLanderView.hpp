// LunarLanderView.hpp

#pragma once

#include <optional>
#include <vector>
#include <memory>
#include <wx/wx.h>
#include <torch/torch.h>
#include "anet/rl.hpp"
#include "anet/gui.hpp"



// =============================================================
// LunarLanderData
// =============================================================

struct TerrainPoint {
    float x;
    float y;
};

struct UISegment {
    float x0, y0;
    float x1, y1;
};

struct Legs {
    UISegment left_leg;       // 左脚の座標情報
    UISegment right_leg;      // 右脚の座標情報
};

struct TerrainPolyline {
    std::vector<TerrainPoint> points;
};

struct LunarLanderData {
    // Step
    anet::rl::step_t step;

    // RL由来情報
    anet::rl::SingleState state;
    std::int64_t action;
    float reward;

    // ENV由来情報
    std::unordered_map<std::string, torch::Tensor> aux;

    // 表示用情報
    std::optional<float> total_reward;

    /// @todo 毎回取る情報と取らない情報を分離整理

    static LunarLanderData Create(anet::rl::TrainEvent event);
};


// =============================================================
// LunarLanderCanvas
// =============================================================

class LunarLanderView;

class LunarLanderPanel : public anet::rl::gui::Panel {
public:
    //explicit LunarLanderPanel(wxWindow* parent, std::shared_ptr<LunarLanderView> view, int update_timer_ms);
    explicit LunarLanderPanel(wxWindow* parent);

    void ApplyData(LunarLanderView& view);
    //void SetData(const LunarLanderData& data);
protected:
    void OnPaint(wxPaintEvent& event);
    //void OnMouseLeftClick(wxMouseEvent& event);
    //void OnMouseRightClick(wxMouseEvent& event);
    void OnClose(wxCloseEvent& event);
    //void OnTimer(wxTimerEvent& event);
private:

    // world座標→画面座標変換
    wxPoint WorldToScreen(float wx, float wy, int width, int height) const;
    int WorldToScreen(float size, int width, int height) const;

    // 描画ヘルパ
    void DrawRL(wxDC& dc);
    void DrawWorldLine(wxDC& dc, int width, int height);
    void DrawTerrain(wxDC& dc, int width, int height);
    void DrawPad(wxDC& dc, int width, int height);
    void DrawLander(wxDC& dc, int width, int height);
    void DrawLegs(wxDC& dc, int width, int height);
    void DrawThrust(wxDC& dc, int width, int height);
    void DrawWind(wxDC& dc, int width, int height);
private:
    // 最新のUIデータ（スナップショット）
    bool has_snapshot_ = false;
    LunarLanderData snapshot_;

    // world座標系の範囲
    float world_min_x_ = -10.0f;
    float world_max_x_ = 10.0f;
    float world_min_y_ = 0.0f;
    float world_max_y_ = 15.0f;

    wxDECLARE_EVENT_TABLE();
};


// =============================================================
// LunarLanderView
// =============================================================

class LunarLanderCanvas;

class LunarLanderView : public anet::rl::gui::ViewBase, public std::enable_shared_from_this<LunarLanderView> {
public:
    //explicit LunarLanderView(
    //    const LunarLanderViewConfig& config, wxWindow* parent, std::shared_ptr<anet::rl::Notifier> notifier);
    explicit LunarLanderView(wxWindow* parent);
    void Initialize();

    void CaptureViewData() override;
    void UpdateViewData(const anet::rl::TrainEvent& event, bool force) override;
    //void OnTrain(const anet::rl::TrainEvent& event) override;
    //void OnLearn(const anet::rl::LearnEvent& event) override;
    //std::string ToString() const override;
    wxWindow* AsWindow() override { return panel_; }

    std::optional<LunarLanderData> PopData() { return data_store_.Get(); }
    //void DetachFromNotifier();

    ~LunarLanderView() = default;
private:
    //LunarLanderViewConfig config_;
    wxWindow* parent_;
    LunarLanderPanel* panel_;
    anet::rl::gui::UIDataStore<LunarLanderData> data_store_;
};


// =============================================================
// LunarLanderViewCreator
// =============================================================

class LunarLanderViewCreator : public anet::rl::gui::ViewCreator {
public:
    LunarLanderViewCreator() = default;

    std::shared_ptr<anet::rl::gui::View> CreateView(
        wxWindow* parent, const anet::ConfigData config_data, std::shared_ptr<anet::rl::Notifier> notifier) const
    {
        //LunarLanderViewConfig config(config_data);
        //auto ret = std::make_shared<LunarLanderView>(config, parent, notifier);
        auto ret = std::make_shared<LunarLanderView>(parent);
        ret->Initialize();
        return ret;
    }

    std::string GetTargetClassId() const override
    {
        return "LunarLanderEnv";
    }
};
