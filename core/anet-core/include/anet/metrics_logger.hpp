#pragma once

#include <fstream>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <wx/image.h>
#include <wx/process.h>
#include <wx/app.h>
#include <nlohmann/json.hpp>
#include "anet/heat_map.hpp"
#include "anet/config.hpp"

wxDECLARE_EVENT(wxEVT_APP_EXECUTE_START, wxThreadEvent);

namespace anet {
    using json = nlohmann::json;

    //----------------------------------------------
    // Backendインターフェース
    //----------------------------------------------
    class IBackend {
    public:
        virtual ~IBackend() = default;
        virtual void Open(const std::string& root_dir, const std::string& run_name) = 0;
        virtual void WriteJsonl(const json& obj) = 0;
        virtual void Flush() = 0;
    };

    //----------------------------------------------
    // JSONLバックエンド
    //----------------------------------------------
    class JsonlBackend : public IBackend {
    private:
        std::ofstream ofs;
    public:
        void Open(const std::string& root_dir, const std::string& run_name) override;
        void WriteJsonl(const json& obj) override;
        void Flush() override;
    };

    //----------------------------------------------
    // ExecuteStarter メインスレッドで(ffmpeg起動するための仕掛け）
    //----------------------------------------------
    class ExecuteStarter {
    public:
        ExecuteStarter(const wxString& command, wxProcess* process)
            : command_(command), process_(process)
        {
            ;
        }

        void Execute()
        {
            std::unique_lock lock(mutex_);
            wxThreadEvent* ev = new wxThreadEvent(wxEVT_APP_EXECUTE_START);
            ev->SetPayload<ExecuteStarter*>(this);
            wxQueueEvent(wxTheApp, ev);
            cv_.wait(lock, [&] { return started_; });
        }

        void OnMainStart()
        {
            std::unique_lock lock(mutex_);
            wxExecute(command_, wxEXEC_ASYNC, process_);
            started_ = true;
            cv_.notify_one();
        }

        wxString GetCommand() { return command_; }
    private:
        wxProcess* process_;
        std::mutex mutex_;
        std::condition_variable cv_;
        wxString command_;
        bool started_ = false;
    };

    //----------------------------------------------
    // VideoLogger (ffmpegパイプで動画出力)
    //----------------------------------------------
    class VideoLogger {
    private:
        wxProcess* process_ = nullptr;
        wxOutputStream* stream_ = nullptr;
        int width_ = 0, height_ = 0;
        std::string path_;
        int fps_;
        int in_rate_;
        std::string codec_;

        void Close();
    public:
        VideoLogger(const std::string& path, int width, int height, int fps = 30, const std::string& codec = "mjpeg");
        ~VideoLogger() { Close(); }

        void WriteFrame(const wxImage& img);
    };

    //----------------------------------------------
    // MetricsLogger 本体
    //----------------------------------------------
    class MetricsLogger {
    public:
        explicit MetricsLogger(
            std::unique_ptr<IBackend> backend, const std::string& root = "runs", const std::string& run_name_tmpl = "");

        MetricsLogger(const MetricsLogger&) = delete;
        MetricsLogger& operator=(const MetricsLogger&) = delete;

        // --- Singleton API ---
        static std::shared_ptr<MetricsLogger> Instance();
        static void Init(std::unique_ptr<IBackend> backend, const std::string& root = "runs", const std::string& run_name_tmpl = "run_{t}");
        static void Reset();

        static std::string GetRunName(const std::string& run_name_tmpl);

        inline void LogScalar(const std::string& tag, int64_t step, double value) {
            json obj = {
                {"type", "scalar"},
                {"tag", tag},
                {"step", step},
                {"value", value}
            };
            backend_->WriteJsonl(obj);
        }

        void Log(const anet::Config& config);
        void LogJson(const std::string& tag, const json& data);

        void LogImage(const std::string& tag, int step, const wxImage& image);
        void LogImage(const std::string& tag, int step, const anet::ImageSource& src, int width = -1, int height = -1);

        inline std::string GetRunName() const { return run_name_; }
        inline std::string GetOutDir() const { return std::filesystem::relative(root_dir_ + "/" + run_name_).string(); }
        inline void Flush() { backend_->Flush(); }
    private:
        std::string CreateTimeStampStr() const;
        std::string CreateRunName(const std::string& run_name_tmpl) const;
    private:
        std::unique_ptr<IBackend> backend_;
        std::string root_dir_;
        std::string run_name_;
        std::mutex config_mutex_;
        
        // 画像・動画用連番管理
        std::unordered_map<std::string, uint64_t> image_seq_;
        std::unordered_map<std::string, std::unique_ptr<VideoLogger>> video_loggers_;

        static json round_numbers(const json& j, int precision = 6);
        static std::string current_time_str();
        static std::string sanitize_filename(const std::string& s);

        // 内部実装
        void LogImage_subtyped(const std::string& tag,
            int step,
            const wxImage& image,
            const std::string& subtype_or_empty);

        // --- Singleton管理 ---
        static std::shared_ptr<MetricsLogger> instance_;
        static std::mutex instance_mutex_;
    };
}   // namespace anet
