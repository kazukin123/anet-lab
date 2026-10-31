#include "anet/catch_test.hpp"
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include <wx/aui/auibar.h>
#include <wx/clipbrd.h>
#include <wx/evtloop.h>
#include <wx/settings.h>
#include <wx/srchctrl.h>
#include <wx/timer.h>
#include <wx/utils.h>
#include "LogPanel.hpp"
#include "anet/config.hpp"

namespace {
    // 実際の wx の子ウィンドウを、利用者と同じツール操作で動かす。
    struct LogUiFixture {
        // テストの process は main loop を回さないため、Yield が窓のメッセージとタイマーを配れるよう、ループを有効にしておく。
        wxEventLoop event_loop;
        wxEventLoopActivator event_loop_activator{&event_loop};
        // LogPanel は wxLog 全体の時刻書式を変えるため、同じ process の後続テストへ残さないよう戻す。
        const wxString previous_timestamp = wxLog::GetTimestamp();
        wxFrame frame{nullptr, wxID_ANY, "Logs acceptance", wxDefaultPosition, wxSize(900, 400)};
        LogPanel* panel = nullptr;
        wxTextCtrl* text = nullptr;
        wxSearchCtrl* search = nullptr;
        std::vector<wxAuiToolBar*> toolbars;

        ~LogUiFixture() { wxLog::SetTimestamp(previous_timestamp); }
        explicit LogUiFixture(const anet::ConfigData& config = anet::ConfigData())
        {
            panel = new LogPanel(&frame, config);
            for (auto* child : panel->GetChildren()) {
                if (auto* candidate = dynamic_cast<wxTextCtrl*>(child)) text = candidate;
                if (auto* candidate = dynamic_cast<wxSearchCtrl*>(child)) search = candidate;
                if (auto* candidate = dynamic_cast<wxAuiToolBar*>(child)) toolbars.push_back(candidate);
            }
            REQUIRE(text != nullptr);
            REQUIRE(search != nullptr);
            REQUIRE(toolbars.size() == 5);
            frame.Show();
            frame.SendSizeEvent();
            wxTheApp->Yield(true);
        }

        std::pair<wxAuiToolBar*, wxAuiToolBarItem*> Find(const wxString& help) const
        {
            for (auto* toolbar : toolbars) {
                for (size_t i = 0; i < toolbar->GetToolCount(); ++i) {
                    auto* tool = toolbar->FindToolByIndex(static_cast<int>(i));
                    if (tool->GetShortHelp() == help) return {toolbar, tool};
                }
            }
            FAIL("tool not found: " << help.ToStdString());
            return {};
        }

        // チェック型は、実際のクリックと同じく状態を切り替えてから通知する。
        void Press(wxAuiToolBar* toolbar, wxAuiToolBarItem* tool)
        {
            if (tool->GetKind() == wxITEM_CHECK) toolbar->ToggleTool(tool->GetId(), !toolbar->GetToolToggled(tool->GetId()));
            wxCommandEvent event(wxEVT_TOOL, tool->GetId());
            event.SetEventObject(toolbar);
            toolbar->GetEventHandler()->ProcessEvent(event);
        }
        void Press(const wxString& help)
        {
            const auto [toolbar, tool] = Find(help);
            Press(toolbar, tool);
        }
        // 未確認の警告は ▾ 付きのツールで見分ける。
        std::pair<wxAuiToolBar*, wxAuiToolBarItem*> FindAlert() const
        {
            for (auto* toolbar : toolbars) {
                for (size_t i = 0; i < toolbar->GetToolCount(); ++i) {
                    auto* tool = toolbar->FindToolByIndex(static_cast<int>(i));
                    if (tool->HasDropDown()) return {toolbar, tool};
                }
            }
            FAIL("alert tool not found");
            return {};
        }
        void PressAlert()
        {
            const auto [toolbar, tool] = FindAlert();
            Press(toolbar, tool);
        }
        bool AlertShown() const { return FindAlert().first->IsShown(); }
        wxString AlertLabel() const { return FindAlert().second->GetLabel(); }
        bool Toggled(const wxString& help) const
        {
            const auto [toolbar, tool] = Find(help);
            return toolbar->GetToolToggled(tool->GetId());
        }
        bool AutoScroll() const { return Toggled("Auto-scroll"); }
        int AutoScrollX() const { return Find("Auto-scroll").first->GetPosition().x; }
        // 未読の行数は、表示されているときだけ件数を返す。
        wxString Unread() const
        {
            const auto [toolbar, tool] = Find("Show the next unread lines");
            return toolbar->IsShown() ? tool->GetLabel() : wxString();
        }
        long UnreadCount() const
        {
            long count = 0;
            if (!Unread().empty()) Unread().Mid(1).ToLong(&count);
            return count;
        }
        bool SearchArrowsEnabled() const
        {
            const auto [older_toolbar, older] = Find("Older match");
            const auto [newer_toolbar, newer] = Find("Newer match");
            return older_toolbar->GetToolEnabled(older->GetId()) && newer_toolbar->GetToolEnabled(newer->GetId());
        }
        long VisibleAt(int y) const
        {
            long position = 0;
            text->HitTest(wxPoint(2, y), &position);
            return position;
        }
        long FirstVisible() const { return VisibleAt(0); }
        long LastVisible() const { return VisibleAt(text->GetClientSize().y - 4); }
        long LastLineStart() const { return static_cast<long>(text->GetValue().rfind('\n') + 1); }
        // 目印は行の途中（時刻の後ろ）にあるため、見えているかは行の番号で比べる。
        long LineOf(long position) const
        {
            long column = 0;
            long line = 0;
            text->PositionToXY(position, &column, &line);
            return line;
        }
        bool IsVisible(const wxString& marker) const
        {
            const long line = LineOf(StartOf(marker));
            return LineOf(FirstVisible()) <= line && line <= LineOf(LastVisible());
        }
        // 利用者のスクロールは止まるのを待ってから判定されるので、その判定まで進める。
        static void Settle()
        {
            for (int i = 0; i < 25; ++i) {
                wxTheApp->Yield(true);
                wxMilliSleep(20);
            }
        }
        long StartOf(const wxString& marker) const { return static_cast<long>(text->GetValue().find(marker)); }
        long LineStartOf(const wxString& marker) const
        {
            const auto newline = text->GetValue().rfind('\n', static_cast<size_t>(StartOf(marker)));
            return newline == wxString::npos ? 0 : static_cast<long>(newline + 1);
        }
        wxString Selected() const { return text->GetStringSelection(); }
    };
}

static void Log(wxLogLevel level, const wxString& message)
{
    wxLogGeneric(level, "%s", message);
}

TEST_CASE("Logs append records without a trailing empty line", "[log_panel]")
{
    LogUiFixture ui;
    Log(wxLOG_Message, "first");
    Log(wxLOG_Message, "second");
    Log(wxLOG_Message, "third");
    CHECK_FALSE(ui.text->GetValue().EndsWith("\n"));
    CHECK(ui.text->GetValue().Freq('\n') == 2);
    CHECK(ui.text->GetNumberOfLines() == 3);

    // Clear の後の 1 行目にも区切りの改行を付けない。
    ui.Press("Clear log");
    CHECK(ui.text->IsEmpty());
    Log(wxLOG_Message, "after clear");
    CHECK(ui.text->GetValue().Contains("after clear"));
    CHECK(ui.text->GetValue().Freq('\n') == 0);
}

TEST_CASE("Logs tint warning error and verbose lines for the theme", "[log_panel]")
{
    LogUiFixture ui;
    ui.panel->SetLogLevel("verbose");
    Log(wxLOG_Message, "plain");
    Log(wxLOG_Warning, "careful");
    Log(wxLOG_Error, "broken");
    Log(wxLOG_Info, "detail");
    // 起動時に選んだテーマ（PRD094_UI_DARK）の配色を確かめていることを先に確認する。
    const bool dark = wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOW).GetLuminance() < 0.5;
    CHECK(dark == (std::getenv("PRD094_UI_DARK") != nullptr));

    // 時刻の後ろに ": " を付けず、半角スペース 1 つで種類へ続ける。
    const wxString first_line = ui.text->GetValue().BeforeFirst('\n');
    CHECK(first_line.length() == wxString("12:34:56.789 [I] plain").length());
    CHECK(first_line.EndsWith(" [I] plain"));
    CHECK_FALSE(first_line.Contains(": [I]"));

    const auto colour_at = [&ui](long position) {
        wxTextAttr style;
        REQUIRE(ui.text->GetStyle(position, style));
        return style.GetTextColour();
    };
    const auto check_palette = [&] {
        // 本文は種類ごとの色、行の頭の時刻は種類によらず同じ薄い色。
        CHECK(colour_at(ui.StartOf("plain")) == wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT));
        CHECK(colour_at(ui.StartOf("careful")) == (dark ? wxColour(0xF0, 0xB0, 0x40) : wxColour(0xA6, 0x54, 0x00)));
        CHECK(colour_at(ui.StartOf("broken")) == (dark ? wxColour(0xFF, 0x6B, 0x6B) : wxColour(0xC6, 0x28, 0x28)));
        CHECK(colour_at(ui.StartOf("detail")) == (dark ? wxColour(0x9A, 0x9A, 0x9A) : wxColour(0x6E, 0x6E, 0x6E)));
        const auto timestamp = dark ? wxColour(0x7A, 0x8B, 0xA1) : wxColour(0x5F, 0x78, 0x94);
        for (const char* marker : {"plain", "careful", "broken", "detail"}) {
            CHECK(colour_at(ui.LineStartOf(marker)) == timestamp);
        }
    };
    check_palette();

    // テーマ変更の通知で塗り直しても、同じ色の割り当てを保つ。
    wxSysColourChangedEvent changed;
    ui.panel->GetEventHandler()->ProcessEvent(changed);
    check_palette();
}

TEST_CASE("Logs keep the selection and follow or hold the view", "[log_panel]")
{
    LogUiFixture ui;
    for (int i = 0; i < 200; ++i) Log(wxLOG_Message, wxString::Format("line %d", i));
    CHECK(ui.AutoScroll());
    CHECK(ui.LastVisible() >= ui.LastLineStart());

    // 自動スクロール中も、利用者の選択は追記で消えない（最下から離れないよう、見えている最終行の中を選ぶ）。
    const long selected_from = ui.LastLineStart();
    const long selected_to = selected_from + 5;
    ui.text->SetSelection(selected_from, selected_to);
    Log(wxLOG_Message, "kept selection");
    long from = 0;
    long to = 0;
    ui.text->GetSelection(&from, &to);
    CHECK(from == selected_from);
    CHECK(to == selected_to);
    CHECK(ui.AutoScroll());
    CHECK(ui.LastVisible() >= ui.LastLineStart());

    // 止めている間は、表示の先頭も選択も動かさず、まだ映っていない行数を示す。
    ui.Press("Auto-scroll");
    CHECK_FALSE(ui.AutoScroll());
    CHECK(ui.Unread().empty());
    ui.text->ShowPosition(ui.text->XYToPosition(0, 50));
    const long first_visible = ui.FirstVisible();
    for (int i = 0; i < 5; ++i) Log(wxLOG_Message, wxString::Format("held %d", i));
    CHECK(ui.FirstVisible() == first_visible);
    ui.text->GetSelection(&from, &to);
    CHECK(from == selected_from);
    CHECK(to == selected_to);
    CHECK(ui.Unread() == "+5");

    // 再開すると末尾へ飛び、件数を消す。
    ui.Press("Auto-scroll");
    CHECK(ui.AutoScroll());
    CHECK(ui.Unread().empty());
    CHECK(ui.LastVisible() >= ui.LastLineStart());
}

TEST_CASE("Logs count only lines not yet shown and page through them", "[log_panel]")
{
    LogUiFixture ui;
    for (int i = 0; i < 200; ++i) Log(wxLOG_Message, wxString::Format("line %d", i));

    // 数は、止めてから届いた行のうち、表示の下端より下にある行（表示の下端の行までは映ったとみなす）。
    const auto held = [](int i) { return wxString::Format("held %02d", i); };
    const auto count_below_view = [&] {
        long bottom = 0;
        ui.text->HitTest(wxPoint(0, ui.text->GetClientSize().y - 1), &bottom);
        long below = 0;
        for (int i = 0; i < 60; ++i) below += ui.LineOf(ui.StartOf(held(i))) > ui.LineOf(bottom) ? 1 : 0;
        return below;
    };
    const auto x_of_tool = [&ui](const wxString& help) { return ui.Find(help).first->GetPosition().x; };
    const int auto_scroll_x = x_of_tool("Auto-scroll");
    const int wrap_x = x_of_tool("Wrap long lines");
    ui.Press("Auto-scroll");
    REQUIRE_FALSE(ui.AutoScroll());
    for (int i = 0; i < 60; ++i) Log(wxLOG_Message, held(i));
    const long all_held = ui.UnreadCount();
    CHECK(all_held >= 59);
    CHECK(all_held == count_below_view());

    // 件数は Copy all の右（検索欄の左）に出て、Auto-scroll・Wrap の位置は動かさない。
    CHECK(x_of_tool("Show the next unread lines") > x_of_tool("Copy all"));
    CHECK(x_of_tool("Show the next unread lines") < ui.search->GetPosition().x);
    CHECK(x_of_tool("Auto-scroll") == auto_scroll_x);
    CHECK(x_of_tool("Wrap long lines") == wrap_x);

    // スクロールで読み進めた分だけ減り、上へ戻っても増えない。
    ui.text->ScrollLines(5);
    ui.Settle();
    const long after_scroll = ui.UnreadCount();
    CHECK(after_scroll < all_held);
    CHECK(after_scroll == count_below_view());
    ui.text->ScrollLines(-20);
    ui.Settle();
    CHECK(ui.UnreadCount() == after_scroll);
    CHECK_FALSE(ui.AutoScroll());

    // 押すと、次の未読の行を直前の行と一緒に出し、映った分だけ減る。Auto-scroll は止めたまま。
    const int first_unread = static_cast<int>(60 - after_scroll);
    ui.Press("Show the next unread lines");
    CHECK(ui.IsVisible(held(first_unread)));
    CHECK(ui.IsVisible(held(first_unread - 1)));
    const long after_press = ui.UnreadCount();
    CHECK(after_press < after_scroll);
    CHECK(after_press == count_below_view());
    CHECK_FALSE(ui.AutoScroll());

    // 押した後に届いた行も、未読に足される。
    for (int i = 0; i < 3; ++i) Log(wxLOG_Message, wxString::Format("late %d", i));
    CHECK(ui.UnreadCount() == after_press + 3);

    // 押すたびに続きへ進み、最下まで映ると件数が消える。最下でも Auto-scroll は止めたまま。
    long previous = ui.UnreadCount();
    for (int i = 0; i < 20 && previous > 0; ++i) {
        ui.Press("Show the next unread lines");
        CHECK(ui.UnreadCount() < previous);
        previous = ui.UnreadCount();
    }
    CHECK(ui.Unread().empty());
    CHECK(ui.IsVisible("late 2"));
    CHECK_FALSE(ui.AutoScroll());

    // Auto-scroll を ON にすると、未読は消える。
    ui.text->ShowPosition(0);
    for (int i = 0; i < 30; ++i) Log(wxLOG_Message, wxString::Format("more %d", i));
    CHECK_FALSE(ui.Unread().empty());
    ui.Press("Auto-scroll");
    CHECK(ui.Unread().empty());

    // Clear で消え、映る位置に届いた行は、止めていても数えない。
    ui.Press("Auto-scroll");
    ui.Press("Clear log");
    for (int i = 0; i < 3; ++i) Log(wxLOG_Message, wxString::Format("short %d", i));
    CHECK_FALSE(ui.AutoScroll());
    CHECK(ui.Unread().empty());
}

TEST_CASE("Logs wrap long lines on demand", "[log_panel]")
{
    // 既定は折り返しで、その間は横スクロールバーを出さない（その分だけ表示が広い）。
    LogUiFixture ui;
    Log(wxLOG_Message, wxString('x', 600));
    CHECK(ui.text->GetNumberOfLines() > 1);
    const int wrapped_height = ui.text->GetClientSize().y;
    ui.Press("Wrap long lines");
    CHECK(ui.text->GetNumberOfLines() == 1);
    CHECK(ui.text->GetClientSize().y < wrapped_height);
    ui.Press("Wrap long lines");
    CHECK(ui.text->GetNumberOfLines() > 1);
    CHECK(ui.text->GetClientSize().y == wrapped_height);
}

TEST_CASE("Logs indent wrapped lines to the message on demand", "[log_panel]")
{
    // 既定は字下げありで、折り返している間だけ押せる。
    LogUiFixture ui;
    const auto [toolbar, indent] = ui.Find("Indent wrapped lines");
    CHECK(toolbar->GetToolToggled(indent->GetId()));
    CHECK(toolbar->GetToolEnabled(indent->GetId()));

    // 折り返した 2 行目の頭の横位置を、本文の頭・行の頭と比べる。
    const auto x_of = [&ui](long position) { return ui.text->PositionToCoords(position).x; };
    const auto second_row_x = [&](const wxString& marker) {
        long column = 0;
        long line = 0;
        ui.text->PositionToXY(ui.StartOf(marker), &column, &line);
        return x_of(ui.text->XYToPosition(0, line + 1));
    };
    const auto aligned = [](int x, int expected) { return std::abs(x - expected) <= 2; };
    Log(wxLOG_Message, "first " + wxString('x', 600));
    CHECK(aligned(second_row_x("first"), x_of(ui.StartOf("first"))));

    // OFF にすると、既にある行も、後から足す行も、行の頭へ戻る。
    ui.Press("Indent wrapped lines");
    CHECK(aligned(second_row_x("first"), x_of(ui.LineStartOf("first"))));
    Log(wxLOG_Warning, "second " + wxString('y', 600));
    CHECK(aligned(second_row_x("second"), x_of(ui.LineStartOf("second"))));

    // ON に戻すと本文の頭にそろい、折り返さない間は押せなくなる。
    ui.Press("Indent wrapped lines");
    CHECK(aligned(second_row_x("first"), x_of(ui.StartOf("first"))));
    CHECK(aligned(second_row_x("second"), x_of(ui.StartOf("second"))));
    ui.Press("Wrap long lines");
    CHECK_FALSE(toolbar->GetToolEnabled(indent->GetId()));
    ui.Press("Wrap long lines");
    CHECK(toolbar->GetToolEnabled(indent->GetId()));
}

TEST_CASE("Logs read the level and toggle defaults from the config", "[log_panel]")
{
    // 設定が無ければ、3 つのトグルは ON、表示 level は info。
    {
        LogUiFixture ui;
        CHECK(ui.AutoScroll());
        CHECK(ui.Toggled("Wrap long lines"));
        CHECK(ui.Toggled("Indent wrapped lines"));
        Log(wxLOG_Info, "verbose hidden");
        CHECK_FALSE(ui.text->GetValue().Contains("verbose hidden"));
    }

    // app.log_panel.* で、起動時の状態を変えられる。
    anet::ConfigData config;
    config.Set("app.log_panel.level", "verbose");
    config.Set("app.log_panel.auto_scroll", false);
    config.Set("app.log_panel.wrap", false);
    config.Set("app.log_panel.indent", false);
    LogUiFixture ui(config);
    CHECK_FALSE(ui.AutoScroll());
    CHECK_FALSE(ui.Toggled("Wrap long lines"));
    CHECK_FALSE(ui.Toggled("Indent wrapped lines"));
    Log(wxLOG_Info, "verbose shown");
    CHECK(ui.text->GetValue().Contains("verbose shown"));
    Log(wxLOG_Message, wxString('x', 600));
    CHECK(ui.text->GetNumberOfLines() == 2);
    CHECK(ui.Unread().empty());
}

TEST_CASE("Logs stop auto-scroll when the end leaves the view and resume at the end", "[log_panel]")
{
    LogUiFixture ui;
    for (int i = 0; i < 200; ++i) Log(wxLOG_Message, wxString::Format("line %d", i));
    REQUIRE(ui.AutoScroll());

    // スクロールで最下が見えなくなると、次の行を待たずに止まり、表示を動かさずに新着を数える。
    ui.text->ScrollLines(-20);
    ui.Settle();
    CHECK_FALSE(ui.AutoScroll());
    const long first_visible = ui.FirstVisible();
    Log(wxLOG_Message, "while away");
    CHECK(ui.FirstVisible() == first_visible);
    CHECK(ui.Unread() == "+1");

    // スクロールで最下まで戻すと再開し、件数を消す。
    ui.text->ScrollPages(100);
    ui.Settle();
    CHECK(ui.AutoScroll());
    CHECK(ui.Unread().empty());

    // 通知の後もスクロールが続き、少し遅れて最下に着いた場合（ホイールのなめらかなスクロール）も再開する。
    ui.text->ScrollLines(-20);
    ui.Settle();
    REQUIRE_FALSE(ui.AutoScroll());
    wxMouseEvent wheel(wxEVT_MOUSEWHEEL);
    wheel.SetEventObject(ui.text);
    ui.text->GetEventHandler()->ProcessEvent(wheel);
    wxTheApp->Yield(true);
    ui.text->ShowPosition(ui.text->GetLastPosition());
    ui.Settle();
    CHECK(ui.AutoScroll());

    // 判定を待っている間に行が届いても、最下に着いていれば足す前に再開する。
    ui.text->ScrollLines(-20);
    ui.Settle();
    REQUIRE_FALSE(ui.AutoScroll());
    ui.text->ScrollPages(100);
    Log(wxLOG_Message, "arrived while settling");
    CHECK(ui.AutoScroll());
    CHECK(ui.LastVisible() >= ui.LastLineStart());
    ui.Settle();
    CHECK(ui.AutoScroll());

    // 通知の無いスクロールで離れていても、次の行を足す前に止めて、表示を動かさない。
    ui.text->ShowPosition(0);
    const long top = ui.FirstVisible();
    Log(wxLOG_Message, "silent scroll");
    CHECK_FALSE(ui.AutoScroll());
    CHECK(ui.FirstVisible() == top);

    // 最下にいるままの操作では、手で止めた状態を再開しない。
    ui.Press("Auto-scroll");
    CHECK(ui.AutoScroll());
    ui.Press("Auto-scroll");
    CHECK_FALSE(ui.AutoScroll());
    ui.text->ScrollLines(1);
    ui.Settle();
    CHECK_FALSE(ui.AutoScroll());

    // 窓を縮めても、自動スクロール中は最下を見せ続け、止めたと取り違えない。
    ui.Press("Auto-scroll");
    ui.frame.SetSize(ui.frame.GetSize() - wxSize(0, 150));
    ui.Settle();
    Log(wxLOG_Message, "after resize");
    CHECK(ui.AutoScroll());
    CHECK(ui.LastVisible() >= ui.LastLineStart());
}

TEST_CASE("Logs search older and newer matches without case and continue from the other end", "[log_panel]")
{
    LogUiFixture ui;
    Log(wxLOG_Message, "Alpha one");
    Log(wxLOG_Message, "beta");
    Log(wxLOG_Message, "ALPHA two");
    Log(wxLOG_Message, "gamma");
    Log(wxLOG_Message, "alpha three");
    CHECK_FALSE(ui.search->IsCancelButtonVisible());
    CHECK_FALSE(ui.SearchArrowsEnabled());
    ui.search->SetValue("alpha");
    CHECK(ui.search->IsCancelButtonVisible());
    CHECK(ui.SearchArrowsEnabled());
    const auto selected_line = [&ui] {
        long from = 0;
        long to = 0;
        ui.text->GetSelection(&from, &to);
        return ui.text->GetValue().Mid(static_cast<size_t>(from)).BeforeFirst('\n');
    };

    // 選択が無ければ末尾から古い方へ進み、端まで行ったら末尾から続ける。
    ui.Press("Older match");
    CHECK(ui.Selected().Lower() == "alpha");
    CHECK(selected_line() == "alpha three");
    CHECK_FALSE(ui.AutoScroll());
    ui.Press("Older match");
    CHECK(selected_line() == "ALPHA two");
    ui.Press("Older match");
    CHECK(selected_line() == "Alpha one");
    ui.Press("Older match");
    CHECK(selected_line() == "alpha three");

    // 新しい方へも端から続き、Enter（検索の通知）は古い方へ進む。
    ui.Press("Newer match");
    CHECK(selected_line() == "Alpha one");
    wxCommandEvent enter(wxEVT_SEARCH, ui.search->GetId());
    enter.SetEventObject(ui.search);
    enter.SetString("alpha");
    ui.search->GetEventHandler()->ProcessEvent(enter);
    CHECK(selected_line() == "alpha three");

    // 一致が無ければ検索欄の背景で示し、語を変えると戻す。
    const auto normal = wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOW);
    ui.search->SetValue("zzz");
    ui.Press("Older match");
    CHECK(ui.search->GetBackgroundColour() != normal);
    CHECK(selected_line() == "alpha three");
    ui.search->SetValue("zz");
    CHECK(ui.search->GetBackgroundColour() == normal);
    ui.search->SetValue("");
    CHECK_FALSE(ui.search->IsCancelButtonVisible());
    CHECK_FALSE(ui.SearchArrowsEnabled());
}

TEST_CASE("Logs count unviewed warnings and open the oldest first", "[log_panel]")
{
    LogUiFixture ui;
    // 0 件の間は欄ごと隠し、残りを左端へ詰める。
    CHECK_FALSE(ui.AlertShown());
    CHECK(ui.AutoScrollX() == 0);
    Log(wxLOG_Warning, "first warning");
    Log(wxLOG_Message, "plain");
    Log(wxLOG_Error, "an error");
    Log(wxLOG_Warning, "second warning");
    auto alerts = ui.panel->GetUnviewedAlerts();
    CHECK(alerts.warnings == 2);
    CHECK(alerts.errors == 1);
    CHECK(ui.AlertShown());
    CHECK(ui.AutoScrollX() > 0);
    CHECK(ui.AlertLabel() == "3");
    CHECK(ui.FindAlert().second->GetShortHelp() == "2 warnings, 1 error not viewed");

    // 古い順に開き、開いた行は見たことにする。
    ui.PressAlert();
    CHECK(ui.Selected().Contains("first warning"));
    CHECK_FALSE(ui.AutoScroll());
    alerts = ui.panel->GetUnviewedAlerts();
    CHECK(alerts.warnings == 1);
    CHECK(alerts.errors == 1);
    ui.PressAlert();
    CHECK(ui.Selected().Contains("an error"));
    alerts = ui.panel->GetUnviewedAlerts();
    CHECK(alerts.warnings == 1);
    CHECK(alerts.errors == 0);

    // 表示しないレベルの記録は数えない。Clear で 0 に戻る。
    ui.panel->SetLogLevel("error");
    Log(wxLOG_Warning, "hidden warning");
    CHECK_FALSE(ui.text->GetValue().Contains("hidden warning"));
    CHECK(ui.panel->GetUnviewedAlerts().Total() == 1);
    ui.Press("Clear log");
    CHECK(ui.panel->GetUnviewedAlerts().Total() == 0);
    CHECK_FALSE(ui.AlertShown());
    CHECK(ui.AutoScrollX() == 0);
}

TEST_CASE("Logs copy all displayed lines to the clipboard", "[.][log_panel_clipboard]")
{
    // 実行すると利用者のクリップボードを書き換えるため、明示したときだけ走らせる。
    LogUiFixture ui;
    Log(wxLOG_Message, "copy one");
    Log(wxLOG_Warning, "copy two");
    ui.Press("Copy all");
    wxClipboardLocker locker;
    REQUIRE(wxTheClipboard->IsOpened());
    wxTextDataObject data;
    REQUIRE(wxTheClipboard->GetData(data));
    wxString copied = data.GetText();
    copied.Replace("\r\n", "\n");
    CHECK(copied == ui.text->GetValue());
}

TEST_CASE("Logs demo window emits every level until closed", "[.][log_panel_demo]")
{
    // 手で操作を試すための窓。Run を起こさずに、各 level の行を 1 秒ごとに足し、窓を閉じるまで動かす。
    LogUiFixture ui;
    ui.frame.SetTitle("Logs demo (close this window to finish)");
    ui.frame.SetSize(ui.frame.FromDIP(wxSize(1000, 500)));
    ui.panel->SetLogLevel("verbose");
    bool closed = false;
    ui.frame.Bind(wxEVT_CLOSE_WINDOW, [&closed](wxCloseEvent&) { closed = true; });
    wxTimer timer;
    int tick = 0;
    timer.Bind(wxEVT_TIMER, [&tick](wxTimerEvent&) {
        // INFO を基本に、ときどき VERBOSE・WARN・ERROR と、折り返しを試す長い行を混ぜる。
        ++tick;
        Log(wxLOG_Message, wxString::Format("demo info tick=%d", tick));
        if (tick % 3 == 0) Log(wxLOG_Info, wxString::Format("demo verbose tick=%d", tick));
        if (tick % 5 == 0) Log(wxLOG_Warning, wxString::Format("demo warning tick=%d", tick));
        if (tick % 7 == 0) Log(wxLOG_Message, wxString::Format("demo long line tick=%d ", tick) + wxString('x', 400));
        if (tick % 11 == 0) Log(wxLOG_Error, wxString::Format("demo error tick=%d", tick));
    });
    timer.Start(1000);
    while (!closed) {
        wxTheApp->Yield(true);
        wxMilliSleep(10);
    }
    timer.Stop();
}
