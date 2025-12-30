#include "anet/gui.hpp"

namespace anet::rl::gui {
    // イベントIDの実体化
    wxDEFINE_EVENT(EVT_FORWARDED_MOUSE, anet::rl::gui::ForwardedMouseEvent);
    wxDEFINE_EVENT(EVT_FORWARDED_KEY, anet::rl::gui::ForwardedKeyEvent);
}

using namespace anet::rl::gui;
namespace LOG = anet::log;


// =============================================================
// ForwardedMouseEvent
// =============================================================

ForwardedMouseEvent::ForwardedMouseEvent(wxEventType commandType, int id)
    : wxCommandEvent(commandType, id), source_window_(nullptr)
{
    ;
}

ForwardedMouseEvent::ForwardedMouseEvent(const ForwardedMouseEvent& event)
    : wxCommandEvent(event), raw_event_(event.raw_event_), source_window_(event.source_window_)
{
    ;
}

wxEvent* ForwardedMouseEvent::Clone() const
{
    return new ForwardedMouseEvent(*this);
}


// =============================================================
// ForwardedKeyEvent
// =============================================================

ForwardedKeyEvent::ForwardedKeyEvent(wxEventType commandType, int id)
    : wxCommandEvent(commandType, id), source_window_(nullptr)
{
    ;
}

ForwardedKeyEvent::ForwardedKeyEvent(const ForwardedKeyEvent& event)
    : wxCommandEvent(event), raw_event_(event.raw_event_), source_window_(event.source_window_)
{
    ;
}

wxEvent* ForwardedKeyEvent::Clone() const
{
    return new ForwardedKeyEvent(*this);
}


// =============================================================
// Panel
// =============================================================

Panel::Panel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
    : wxPanel(parent, id, pos, size, style, name)
{
    Bind(wxEVT_LEFT_DOWN, &Panel::OnMouse, this);
    Bind(wxEVT_RIGHT_DOWN, &Panel::OnMouse, this);
    Bind(wxEVT_LEFT_DCLICK, &Panel::OnMouse, this);
    Bind(wxEVT_CHAR_HOOK, &Panel::OnKey, this);
}

void Panel::OnMouse(wxMouseEvent& event)
{
    SetFocus();
    ForwardedMouseEvent new_event(EVT_FORWARDED_MOUSE, GetId());
    new_event.SetEventObject(this);
    new_event.SetWindow(this);
    new_event.SetMouseEvent(event);
    ProcessEvent(new_event);
    event.Skip();
}

void Panel::OnKey(wxKeyEvent& event)
{
    LOG::info() << "Panel::OnKey() type=" << event.GetEventType();

    // 多重発火防止： 自分が「最下層のPanel」なのか、それとも「すでに処理済みのイベントを受け取った親Panel」なのかを判定
    wxWindow* target = dynamic_cast<wxWindow*>(event.GetEventObject());
    bool childPanelExists = false;

    if (target && target != this) {
        // 発生元(target)から、自分(this)に辿り着くまでの親を順にチェック
        wxWindow* p = target->GetParent();
        while (p && p != this) {
            // もし途中にPanelクラスがあれば、その下位パネルがすでにイベントを発行しているはずなのでスルー
            if (dynamic_cast<Panel*>(p)) {
                childPanelExists = true;
                break;
            }
            p = p->GetParent();
        }
    }

    // すでに下位のPanelが存在するなら、自分は何もせず標準動作(Skip)のみ行う
    if (childPanelExists) {
        event.Skip();
        return;
    }

    ForwardedKeyEvent new_event(EVT_FORWARDED_KEY, GetId());
    new_event.SetEventObject(this);
    new_event.SetWindow(this);
    new_event.SetKeyEvent(event);
    ProcessEvent(new_event);
    event.Skip();
}


// =============================================================
// ViewRepository
// =============================================================

void ViewRepository::Register(std::shared_ptr<ViewCreator> creator)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto class_id = creator->GetTargetClassId();
    creators_[class_id] = creator;
}

std::shared_ptr<ViewCreator> ViewRepository::GetViewCreator(const std::string& class_id) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = creators_.find(class_id);
    if (it == creators_.end()) return nullptr;
    return it->second;
}


// =============================================================
// DefaultViewFactory
// =============================================================

DefaultViewFactory::DefaultViewFactory(const anet::ConfigData& config_data)
    : config_data_(config_data)
{
    ;
}

std::shared_ptr<View> DefaultViewFactory::CreateView(
    wxWindow* parent, const std::string& class_id, std::shared_ptr<Notifier> notifier) const
{
    auto creator = ViewRepository::Instance().GetViewCreator(class_id);
    if (creator == nullptr)
        return nullptr;

    auto view  = creator->CreateView(parent, config_data_, notifier);

    return view;
}
