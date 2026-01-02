// anet/gui.hpp

#pragma once

#include <chrono>
#include <wx/wx.h>
#include "anet/common.hpp"
#include "anet/rl.hpp"

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

            // 強制更新設定が無効なら終了
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

    class View {
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
            auto data = data_store_.Get();
            if (!data.has_value()) return;
            window_->ApplyData(*data);
        }
        void UpdateViewData(const anet::rl::LearnEvent& event, bool force) override {}
        void UpdateViewData(const anet::rl::TrainEvent& event, bool force = false)
        {
            // Update不要ならスキップ
            if (!force && !data_store_.ShouldUpdate())
                return;

            // UIデータを生成
            ViewDataType ui_data = CreateViewData(event);

            // UIデータを保存
            data_store_.Update(ui_data);
        }
        wxWindow* AsWindow() override { return window_; }
        
        virtual ~ViewBase() = default;
    protected:
        wxWindow* parent_;
        anet::rl::gui::UIDataStore<ViewDataType> data_store_;
        ViewWindowType* window_ = nullptr;
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

