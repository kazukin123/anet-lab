#include "anet/catch_test.hpp"

#include <array>
#include <memory>
#include <wx/frame.h>
#include "anet/gui.hpp"

TEST_CASE("Panel forwards each char hook once from the innermost panel", "[prd122][panel_keys]")
{
    using namespace anet::rl::gui;
    // 実際の発生元へCHAR_HOOKを渡し、親へ伝わる経路を通して回数と発行元を数える。
    const int scenario = GENERATE(0, 1, 2, 3);
    auto frame = std::make_unique<wxFrame>(nullptr, wxID_ANY, "PRD122 key characterization");
    auto* outer = new Panel(frame.get());
    auto* inner = new Panel(outer);
    auto* nested = new wxWindow(inner, wxID_ANY);
    auto* direct = new wxWindow(outer, wxID_ANY);
    auto* outside = new wxWindow(frame.get(), wxID_ANY);
    const std::array<wxWindow*, 4> sources{{nested, inner, direct, outside}};
    const std::array<Panel*, 4> expected_panels{{inner, inner, outer, nullptr}};
    wxWindow* source = sources[scenario];
    Panel* expected = expected_panels[scenario];
    int forwarded_count = 0;
    frame->Bind(EVT_FORWARDED_KEY, [&](ForwardedKeyEvent& event) {
        ++forwarded_count;
        CHECK(event.GetEventObject() == expected);
        CHECK(event.GetWindow() == expected);
        CHECK(event.GetKeyEvent().GetEventObject() == source);
        CHECK(event.GetKeyEvent().GetKeyCode() == WXK_LEFT);
        if (expected) CHECK(event.GetId() == expected->GetId());
    });
    wxKeyEvent key(wxEVT_CHAR_HOOK);
    key.SetEventObject(source);
    key.m_keyCode = WXK_LEFT;
    const bool handled = source->GetEventHandler()->ProcessEvent(key);
    CHECK_FALSE(handled);
    CHECK(forwarded_count == (expected ? 1 : 0));
}
