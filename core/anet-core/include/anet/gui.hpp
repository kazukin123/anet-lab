// anet/gui.hpp

#pragma once

#include <chrono>
#include <deque>
#include <optional>
#include <string>
#include <functional>
#include <memory>
#include <vector>
#include <wx/wx.h>
#include <wx/aui/aui.h>
#include <wx/aui/serializer.h>
#include "anet/common.hpp"
#include "anet/rl.hpp"
#include "anet/profile.hpp"

namespace anet::rl { struct StepTraceColumnSpec; }

namespace anet::rl::gui {


    // ============================================================================
    // イベントタグ宣言
    // 　デフォルトのキーボード・マウス系イベントはバブリングされなくて不便なので
    // ============================================================================

    class ForwardedMouseEvent;
    class ForwardedKeyEvent;
    wxDECLARE_EVENT(EVT_FORWARDED_MOUSE, anet::rl::gui::ForwardedMouseEvent);
    wxDECLARE_EVENT(EVT_FORWARDED_KEY, anet::rl::gui::ForwardedKeyEvent);


    // ============================================================================
    // イベントクラス定義
    // ============================================================================

    class ForwardedMouseEvent : public wxCommandEvent {
    public:
        ForwardedMouseEvent(wxEventType commandType = EVT_FORWARDED_MOUSE, int id = 0);
        ForwardedMouseEvent(const ForwardedMouseEvent& event);
        virtual wxEvent* Clone() const override;

        void SetMouseEvent(const wxMouseEvent& evt) { raw_event_ = evt; }
        wxMouseEvent GetMouseEvent() const { return raw_event_; }

        void SetWindow(wxWindow* window) { source_window_ = window; }
        wxWindow* GetWindow() const { return source_window_; }
    private:
        wxMouseEvent raw_event_;
        wxWindow* source_window_;
    };

    class ForwardedKeyEvent : public wxCommandEvent {
    public:
        ForwardedKeyEvent(wxEventType commandType = EVT_FORWARDED_KEY, int id = 0);
        ForwardedKeyEvent(const ForwardedKeyEvent& event);
        virtual wxEvent* Clone() const override;

        void SetKeyEvent(const wxKeyEvent& evt) { raw_event_ = evt; }
        wxKeyEvent GetKeyEvent() const { return raw_event_; }

        void SetWindow(wxWindow* window) { source_window_ = window; }
        wxWindow* GetWindow() const { return source_window_; }
    private:
        wxKeyEvent raw_event_;
        wxWindow* source_window_;
    };


    // =============================================================
    // Panel
    // =============================================================

    class Panel : public wxPanel
    {
    public:
        Panel(wxWindow* parent, wxWindowID id = wxID_ANY,
            const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize,
            long style = wxTAB_TRAVERSAL, const wxString& name = wxPanelNameStr);
        virtual ~Panel() {}

    protected:
        virtual void OnMouse(wxMouseEvent& event);
        virtual void OnKey(wxKeyEvent& event);
    };


    // =============================================================
    // AuiLayoutFrame
    // =============================================================

    /// 最大化による一時非表示を含め、利用者が開いているpaneかを返す。
    bool IsPaneOpen(const wxAuiPaneInfo& pane, bool has_maximized_pane);

    // wxAUI レイアウト管理 Frame の共通基底クラス (制約吸収層。docs/adr/0016 参照)。
    //
    // wxAUI の dock サイズは内部 dock 構造体が持ち、pane.dock_size は dock が
    // (再)生成される瞬間しか読まれない。既存 dock のサイズを設定する公開 API は無く、
    // pane を隠して dock が空になると dock ごと削除されてサイズは失われ、
    // サッシ確定を通知するイベントも無い。本クラスはこれらの制約を
    //   - 構造化シリアライザ往復 (TakeLayoutSnapshot → 編集 → ApplyLayoutSnapshot)
    //   - 遷移時同期 (隠す直前に実 dock サイズを pane.dock_size へ書き戻す)
    // で吸収し、pane⇄メニュー連動とポリシー適用タイミングの制御だけを担う。
    // pane の中身とレイアウトポリシーは派生クラスが決める。
    class AuiLayoutFrame : public wxFrame {
    public:
        AuiLayoutFrame(const wxString& title, const wxSize& size);
        virtual ~AuiLayoutFrame();
    protected:
        // ---- 派生クラスが実装する層 ----

        /// レイアウトポリシー適用フック。表示遷移後・リサイズ後・restore 後に呼ばれる。
        virtual void OnApplyLayoutPolicy() {}
        
        /// pane 非表示化の直前 (メニュー OFF / ✕ボタン共通) の追加処理。
        virtual void OnPaneHiding(wxAuiPaneInfo& WXUNUSED(pane)) {}

        /// pane 表示の直前 (メニュー ON) の追加処理。
        virtual void OnPaneShowing(wxAuiPaneInfo& WXUNUSED(pane)) {}

    protected:
        // ---- 派生クラスが使う道具 ----

        // レイアウトポリシーを適用する (フックの呼び出し口。派生からの明示適用もこれを使う)
        void ApplyLayoutPolicy() { OnApplyLayoutPolicy(); }
 
        // pane とメニュー項目を対応付け、表示トグル・✕連動・チェック同期を有効にする
        void RegisterPaneMenu(int menu_id, wxWindow* window);
 
        // 登録済み pane の表示状態をメニューのチェックへ反映する
        void UpdatePaneMenuChecks();
 
        // ApplyLayoutPolicy の遅延実行 (多重呼び出しは 1 回に集約される)
        void SchedulePolicy();
 
        // 表示中 pane が属する dock の実サイズを pane.dock_size へ書き戻す (遷移時同期)
        void SyncDockSizesToPanes();
 
        bool HasMaximizedPane() const;
 
        // 現在レイアウトのスナップショット (非表示・浮動 pane の dock_size は記憶値で補完済み)
        std::vector<wxAuiPaneLayoutInfo> TakeLayoutSnapshot();
 
        // 編集済み snapshot の反映 (内部で Update が 1 回走る)。
        // 注意: LoadLayout は pane 配列を丸ごと差し替えるため、取得済みの wxAuiPaneInfo
        // 参照はこの後すべて無効になる。必要なら GetPane で再取得する。
        void ApplyLayoutSnapshot(std::vector<wxAuiPaneLayoutInfo> panes);
 
        static wxAuiPaneLayoutInfo* FindPaneLayout(std::vector<wxAuiPaneLayoutInfo>& panes, const wxString& name);
 
        // pane が指定 dock (方向+layer) にドッキング表示中か。rect 幾何に依存しない定位置判定
         static bool IsInHomeDock(const wxAuiPaneInfo& pane, int direction, int layer);
 
        // 生成時刻で一意化した名前を持つ pane 情報を作る (動的追加 pane 用)
        wxAuiPaneInfo MakeUniquePaneInfo(const wxString& name, const wxString& caption, const wxString& sub_caption = wxEmptyString);
    protected:
        wxAuiManager aui_mgr_;      ///< フラグ・DockSizeConstraint の設定は派生クラスの責務
    private:
        // pane とメニュー項目の対応
        struct PaneMenuBinding {
            int menu_id;
            wxWindow* window;
        };
    private:
        void OnFrameSize(wxSizeEvent& event);
        void OnTogglePaneMenu(wxCommandEvent& event);
        void OnAuiPaneClose(wxAuiManagerEvent& event);
        void OnAuiPaneMaximize(wxAuiManagerEvent& event);
        void OnAuiPaneRestore(wxAuiManagerEvent& event);
        const PaneMenuBinding* FindBindingByMenuId(int menu_id) const;
        const PaneMenuBinding* FindBindingByWindow(wxWindow* window) const;
        void CheckPaneMenuItem(int menu_id, bool checked);
    private:
        std::vector<PaneMenuBinding> pane_menu_bindings_;
        bool layout_policy_scheduled_ = false;
    };


    // =============================================================
    // UIDataStore
    // =============================================================

    template<typename UIDataType>
    class UIDataStore {
    private:
        using Clock = std::chrono::steady_clock;
        using TimePoint = Clock::time_point;
        using Duration = std::chrono::milliseconds;
    public:
        explicit UIDataStore(int force_update_interval_ms = 200)
            : force_update_interval_(std::chrono::milliseconds(force_update_interval_ms))
        {
            auto now_rep = Clock::now().time_since_epoch().count();
            last_update_time_rep_.store(now_rep);
        }

        // Updateすべきか判定
        bool ShouldUpdate()
        {
            // データが空ならtrue
            if (data_request_.load(std::memory_order_acquire))
                return true;

            // 強制更新時間間隔指定が無効なら終了
            if (force_update_interval_.count() <= 0)
                return false;

            // 経過時間算出
            auto now = Clock::now();
            auto now_rep = now.time_since_epoch().count();
            auto last_rep = last_update_time_rep_.load(std::memory_order_acquire);
            auto diff_rep = now_rep - last_rep;

            // 時間差分を duration 型に変換して比較
            if (Clock::duration(diff_rep) >= force_update_interval_) {
                // 未吸い出しのデータがあっても一定時間以上経過していたらUpdate必要と判定
                return true;
            }

            return false;
        }

        // Trainer側から更新（コピーを保持）
        void Update(const UIDataType& data)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            data_ = data;
            data_exists_.store(true);
            data_request_.store(false);
        }

        // UI側から取得（コピーを返す）
        std::optional<UIDataType> Get()
        {
            if (data_exists_.load() == false) return std::nullopt;
            std::lock_guard<std::mutex> lock(mutex_);
            data_request_.store(true);
            return data_;
        }

    private:
        const std::chrono::milliseconds force_update_interval_; ///< 強制更新間隔（ミリ秒単位）の指定値
        std::mutex mutex_;
        UIDataType data_;
        std::atomic<bool> data_exists_{ false };          ///< データ保持有無（一度でもUpdateしたらtrueになる）
        std::atomic<bool> data_request_{ true };          ///< データ更新要否（Getで吸い出されたらtrue、Updateでfalseになる）
        std::atomic<Clock::rep> last_update_time_rep_;    ///< 前回のUpdateからの経過時間カウント値(Atomic)
    };


    // =============================================================
    // View
    // =============================================================

    struct TraceSnapshot {
        std::function<void()> show;
    };

    class TraceFollower {
    public:
        virtual std::shared_ptr<const TraceSnapshot> CreateTraceSnapshot(const TrainEvent& event) = 0;
        virtual void ShowTraceSnapshot(std::shared_ptr<const TraceSnapshot> snapshot) = 0;
        virtual void ReturnToLive() = 0;
        virtual ~TraceFollower() = default;
    };

    class View : public TraceFollower, public std::enable_shared_from_this<View> {
    public:
        virtual void UpdateViewData(const anet::rl::TrainEvent& event, bool force = false) = 0;     ///< Trainerスレッド/Mainスレッド
        virtual void UpdateViewData(const anet::rl::LearnEvent& event, bool force = false) = 0;     ///< Trainerスレッド/Mainスレッド
        virtual void CaptureViewData() = 0;                                     ///< Mainスレッド
        virtual wxWindow* AsWindow() = 0;
		virtual ~View() = default;
	};

    template<typename ViewDataType, typename ViewWindowType>
    class ViewBase : public View {
    protected:
        using ViewBaseType = ViewBase<ViewDataType, ViewWindowType>;
    public:
        explicit ViewBase(wxWindow* parent) : parent_(parent) {}

        virtual ViewDataType CreateViewData(const anet::rl::TrainEvent& event) const = 0;

        void CaptureViewData() override
        {
            // Trace中はlive storeを表示へ流さず、選択したsnapshotを固定する。
            if (trace_mode_) {
                if (trace_data_) window_->ApplyData(*trace_data_);
                return;
            }
            const auto frame = data_store_.Get();
            if (frame && frame->data) window_->ApplyData(*frame->data);
        }
        void UpdateViewData(const anet::rl::LearnEvent& event, bool force) override {}
        void UpdateViewData(const anet::rl::TrainEvent& event, bool force = false) override
        {
            // 既存の生成頻度を維持し、immutable dataをLiveと履歴で共有する。
            if (!force && !data_store_.ShouldUpdate()) return;
            auto data = std::make_shared<const ViewDataType>(CreateViewData(event));
            data_store_.Update(Frame{ .step = event.counts.exp_step, .data = std::move(data) });
        }
        std::shared_ptr<const TraceSnapshot> CreateTraceSnapshot(const TrainEvent& event) override
        {
            ANET_PROFILE_FUNC();
            // 同じ通知で既に作ったViewDataは再計算しない。
            auto frame = data_store_.Get();
            if (!frame || frame->step != event.counts.exp_step) {
                auto data = std::make_shared<const ViewDataType>(CreateViewData(event));
                frame = Frame{ .step = event.counts.exp_step, .data = std::move(data) };
                data_store_.Update(*frame);
            }
            auto weak = std::weak_ptr<ViewBaseType>(
                std::static_pointer_cast<ViewBaseType>(this->shared_from_this()));
            return std::make_shared<const TraceSnapshot>(TraceSnapshot{
                .show = [weak, data = frame->data] {
                    if (auto view = weak.lock()) {
                        view->trace_data_ = data;
                        view->window_->Show();
                        view->window_->ApplyData(*data);
                    }
                } });
        }
        void ShowTraceSnapshot(std::shared_ptr<const TraceSnapshot> snapshot) override
        {
            trace_mode_ = true;
            trace_data_.reset();
            if (snapshot && snapshot->show) snapshot->show();
            else window_->Hide();
        }
        void ReturnToLive() override
        {
            trace_mode_ = false;
            trace_data_.reset();
            window_->Show();
            CaptureViewData();
        }
        wxWindow* AsWindow() override { return window_; }
        virtual ~ViewBase() = default;
    protected:
        struct Frame {
            anet::rl::step_t step = 0;
            std::shared_ptr<const ViewDataType> data;
        };
        wxWindow* parent_;
        anet::rl::gui::UIDataStore<Frame> data_store_;
        ViewWindowType* window_ = nullptr;
        std::shared_ptr<const ViewDataType> trace_data_;
        bool trace_mode_ = false;
    };


    // =============================================================
    // ViewCreator
    // =============================================================

	class ViewCreator {
	public:
		virtual std::shared_ptr<View> CreateView(
            wxWindow* parent, const anet::ConfigData& config_data, std::shared_ptr<Notifier> notifier) const = 0;
        virtual std::string GetTargetClassId() const = 0;
        virtual  ~ViewCreator() = default;
	};


    // =============================================================
    // ViewRepository
    // =============================================================

    class ViewRepository {
    public:
        static ViewRepository& Instance()
        {
            static ViewRepository inst;
            return inst;
        }

        void Register(std::shared_ptr<ViewCreator> factory);
        std::shared_ptr<ViewCreator> GetViewCreator(const std::string& class_id) const;
    private:
        ViewRepository() = default;
    private:
        mutable std::mutex mtx_;
        std::unordered_map<std::string, std::shared_ptr<ViewCreator>> creators_;
    };

    template<typename T, class... Args>
    inline void RegisterViewCreator(Args&&... args)
    {
        auto factory = std::make_shared<T>(std::forward<Args>(args)...);
        ViewRepository::Instance().Register(factory);
    }


    // =============================================================
    // DefaultViewFactory
    // =============================================================

    class DefaultViewFactory {
    public:
        DefaultViewFactory(const anet::ConfigData& config_data);

        std::shared_ptr<View> CreateView(wxWindow* parent, const std::string& class_id, std::shared_ptr<Notifier> notifier) const;
    private:
        anet::ConfigData config_data_;
    };

    // Evaluation Traceの表示と再生を、生成側から独立して扱う。
    float AppendTracePercentile(std::deque<float>& values, float value, size_t window_size);

    // 同じ16 DIPの記号をメインと履歴の操作バーで共有する。
    enum class PlaybackIcon { Play, Pause, Stop, StepBack, StepForward, Slower, Faster };
    wxBitmapBundle MakePlaybackBitmap(PlaybackIcon icon, const wxColour& colour);

    class TracePlayer {
    public:
        void SetRows(std::vector<int64_t> oldest_to_newest);
        void Select(int64_t id, bool keep_playing = false);
        bool CanPlay(double base_steps_per_second) const;
        std::string Play();
        std::string Pause();
        std::string Stop();
        std::string StepForward();
        std::string StepBack();
        std::string Faster();
        std::string Slower();
        std::string Tick(double seconds, double base_steps_per_second);
        std::optional<int64_t> GetSelected() const { return selected_; }
        bool IsPlaying() const { return playing_; }
        double GetSpeed() const;
        std::string GetSpeedLabel() const;
    private:
        std::vector<int64_t> rows_;
        std::optional<int64_t> selected_;
        bool playing_ = false;
        int speed_index_ = 3;
        double progress_ = 0;
    };

    std::string FormatTraceValue(const StepTraceColumnSpec& column, float value, const ActionSpec& actions);


}   // namespace anet::rl::gui


#define ANET_REGISTER_VIEW_CREATOR(CreatorType) \
    namespace { \
        struct CreatorType##AutoRegister { \
            CreatorType##AutoRegister() { \
                anet::rl::RegisterViewCreator(std::make_shared<CreatorType>()); \
            } \
        }; \
        static CreatorType##AutoRegister global_##CreatorType##_auto_register; \
    }

