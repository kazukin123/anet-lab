#include "AtariView.hpp"

#include <algorithm>
#include <cmath>

#include <wx/dcclient.h>

#include "anet/profile.hpp"

using namespace anet::rl::env;

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

static float GetAuxFloat(const anet::rl::AuxData& aux, const std::string& key, float fallback)
{
    const auto it = aux.find(key);
    return it == aux.end() ? fallback : it->second.item<float>();
}

static int64_t GetAuxInt64(const anet::rl::AuxData& aux, const std::string& key, int64_t fallback)
{
    const auto it = aux.find(key);
    return it == aux.end() ? fallback : it->second.item<int64_t>();
}

AtariPanel::AtariPanel(wxWindow* parent)
    : Panel(parent, wxID_ANY, wxDefaultPosition, wxSize(760, 520))
{
    // GL描画面の前面へwxのテキスト行を重ねる。
    auto* main_sizer = new wxBoxSizer(wxVERTICAL);
    wxGLAttributes attributes;
    attributes.PlatformDefaults().DoubleBuffer().EndList();
    canvas_ = new wxGLCanvas(
        this, attributes, wxID_ANY, wxDefaultPosition, wxDefaultSize,
        wxFULL_REPAINT_ON_RESIZE);
    context_ = new wxGLContext(canvas_);
    main_sizer->Add(canvas_, 1, wxEXPAND);
    SetSizer(main_sizer);

    overlay_ = new wxStaticText(
        canvas_, wxID_ANY, "Atari: waiting for data", wxPoint(8, 8),
        wxDefaultSize, wxBORDER_SIMPLE | wxST_NO_AUTORESIZE);
    overlay_->SetForegroundColour(*wxWHITE);
    overlay_->SetBackgroundColour(*wxBLACK);
    overlay_->SetSize(overlay_->GetBestSize());

    canvas_->Bind(wxEVT_PAINT, &AtariPanel::OnPaint, this);
    canvas_->Bind(wxEVT_SIZE, &AtariPanel::OnSize, this);

    // 全面を覆う子windowから、Panel共通のRunner操作イベントへmouse入力を戻す。
    canvas_->Bind(wxEVT_LEFT_DOWN, &AtariPanel::OnChildMouse, this);
    canvas_->Bind(wxEVT_RIGHT_DOWN, &AtariPanel::OnChildMouse, this);
    canvas_->Bind(wxEVT_LEFT_DCLICK, &AtariPanel::OnChildMouse, this);
    overlay_->Bind(wxEVT_LEFT_DOWN, &AtariPanel::OnChildMouse, this);
    overlay_->Bind(wxEVT_RIGHT_DOWN, &AtariPanel::OnChildMouse, this);
    overlay_->Bind(wxEVT_LEFT_DCLICK, &AtariPanel::OnChildMouse, this);
}

AtariPanel::~AtariPanel()
{
    delete context_;
}

void AtariPanel::ApplyData(const AtariViewData& data)
{
    data_ = data;
    overlay_->SetLabel(wxString::Format(
        "Score(raw): %.0f   Lives: %lld   Game step: %lld   Frame: %lld   Action: %s   Reward: %.1f",
        data_.game_score,
        static_cast<long long>(data_.lives),
        static_cast<long long>(data_.game_len),
        static_cast<long long>(data_.game_frames),
        data_.action_name.c_str(),
        data_.reward));
    overlay_->SetSize(overlay_->GetBestSize());
    overlay_->Raise();
    canvas_->Refresh(false);
}

void AtariPanel::OnPaint(wxPaintEvent& event)
{
    wxPaintDC dc(canvas_);
    Render();
}

void AtariPanel::OnSize(wxSizeEvent& event)
{
    canvas_->Refresh(false);
    event.Skip();
}

void AtariPanel::OnChildMouse(wxMouseEvent& event)
{
    OnMouse(event);
}

void AtariPanel::DrawTensor(
    const torch::Tensor& image, int x, int y, int width, int height)
{
    if (!image.defined() || image.dim() != 3 || image.size(2) != 3) return;

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB,
        static_cast<GLsizei>(image.size(1)), static_cast<GLsizei>(image.size(0)),
        0, GL_RGB, GL_UNSIGNED_BYTE, image.data_ptr<uint8_t>());

    const wxSize canvas_size = canvas_->GetSize() * canvas_->GetContentScaleFactor();
    const float left = 2.0f * static_cast<float>(x) / canvas_size.x - 1.0f;
    const float right = 2.0f * static_cast<float>(x + width) / canvas_size.x - 1.0f;
    const float top = 1.0f - 2.0f * static_cast<float>(y) / canvas_size.y;
    const float bottom = 1.0f - 2.0f * static_cast<float>(y + height) / canvas_size.y;

    glEnable(GL_TEXTURE_2D);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(left, top);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(right, top);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(right, bottom);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(left, bottom);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &texture);
}

void AtariPanel::Render()
{
    if (context_ == nullptr || canvas_ == nullptr) return;
    canvas_->SetCurrent(*context_);

    const wxSize size = canvas_->GetSize() * canvas_->GetContentScaleFactor();
    if (size.x <= 0 || size.y <= 0) return;
    glViewport(0, 0, size.x, size.y);
    glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 各領域に収まる最大整数倍率で、RGBを主、前処理gridを副として並べる。
    const int margin = 10;
    const int main_area_width = std::max(1, size.x * 2 / 3 - margin * 2);
    const int side_area_width = std::max(1, size.x - size.x * 2 / 3 - margin * 2);
    if (data_.rgb_hwc.defined()) {
        const int scale = std::max(1, std::min(
            main_area_width / static_cast<int>(data_.rgb_hwc.size(1)),
            (size.y - margin * 2) / static_cast<int>(data_.rgb_hwc.size(0))));
        const int width = static_cast<int>(data_.rgb_hwc.size(1)) * scale;
        const int height = static_cast<int>(data_.rgb_hwc.size(0)) * scale;
        DrawTensor(data_.rgb_hwc, margin + (main_area_width - width) / 2,
            (size.y - height) / 2, width, height);
    }
    if (data_.grid_hwc.defined()) {
        const int scale = std::max(1, std::min(
            side_area_width / static_cast<int>(data_.grid_hwc.size(1)),
            (size.y - margin * 2) / static_cast<int>(data_.grid_hwc.size(0))));
        const int width = static_cast<int>(data_.grid_hwc.size(1)) * scale;
        const int height = static_cast<int>(data_.grid_hwc.size(0)) * scale;
        const int area_x = size.x * 2 / 3 + margin;
        DrawTensor(data_.grid_hwc, area_x + (side_area_width - width) / 2,
            (size.y - height) / 2, width, height);
    }

    glFlush();
    canvas_->SwapBuffers();
}

AtariView::AtariView(wxWindow* parent)
    : ViewBaseType(parent)
{
    window_ = new AtariPanel(parent_);
}

AtariViewData AtariView::CreateViewData(const anet::rl::TrainEvent& event) const
{
    ANET_PROFILE_FUNC();
    AtariViewData data;
    constexpr int64_t lane = 0;

    if (const auto rgb = event.env->GetTensor("rgb_frame", lane); rgb.has_value()) {
        data.rgb_hwc = rgb->detach().to(torch::kCPU).permute({ 1, 2, 0 }).contiguous();
    }
    if (event.step_result && event.step_result->next_state.obs.Contains(anet::rl::ObsKeys::kGrid)) {
        auto grid = event.step_result->next_state.obs.At(anet::rl::ObsKeys::kGrid)[lane]
            .detach().to(torch::kCPU);
        data.grid_hwc = grid.squeeze(0).unsqueeze(2).repeat({ 1, 1, 3 }).contiguous();

        const auto aux_list = event.step_result->GetAuxDataList(static_cast<int>(lane));
        if (!aux_list.empty()) {
            data.game_score = GetAuxFloat(aux_list[0], "game_score", 0.0f);
            data.lives = GetAuxInt64(aux_list[0], "lives", 0);
            data.game_len = GetAuxInt64(aux_list[0], "game_len", 0);
            data.game_frames = GetAuxInt64(aux_list[0], "game_frames", 0);
        }
    }

    if (event.experience.reward.defined() && event.experience.reward.numel() > lane) {
        data.reward = event.experience.reward[lane].item<float>();
    }
    if (event.action_info) {
        const auto actions = event.action_info->GetAction(torch::kCPU);
        if (actions.defined() && actions.numel() > lane) {
            const auto action = actions[lane].item<int64_t>();
            const auto labels = event.env->GetSpec().action_spec.value_labels;
            if (0 <= action && action < static_cast<int64_t>(labels.size())) {
                data.action_name = labels[static_cast<size_t>(action)];
            }
        }
    }
    return data;
}

std::shared_ptr<anet::rl::gui::View> AtariViewCreator::CreateView(
    wxWindow* parent,
    const anet::ConfigData& config_data,
    std::shared_ptr<anet::rl::Notifier> notifier) const
{
    return std::make_shared<AtariView>(parent);
}
