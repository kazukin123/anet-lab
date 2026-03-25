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
#include "anet/graphviz.hpp"


wxDECLARE_EVENT(wxEVT_APP_EXECUTE_START, wxThreadEvent);

namespace anet {

    //----------------------------------------------
    // Backendインターフェース
    //----------------------------------------------
    class IBackend {
    public:
        virtual ~IBackend() = default;
        virtual void Open(const std::filesystem::path& runs_dir, const std::string& run_name) = 0;
        virtual void WriteJsonl(const json& obj) = 0;
        virtual void Flush() = 0;
    };

    //----------------------------------------------
    // JSONLバックエンド
    //----------------------------------------------
    class JsonlBackend final : public IBackend {
    public:
        void Open(const std::filesystem::path& runs_dir, const std::string& run_name) override;
        void WriteJsonl(const json& obj) override;
        void Flush() override;
    private:
        std::ofstream ofs;
        std::mutex mtx_;
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
    public:
        VideoLogger(const std::string& path, int width, int height, const std::string& codec = "libx264", int fps = 30);
        ~VideoLogger() { Close(); }

        void WriteFrame(const wxImage& img);
    private:
        void Close();
    private:
        wxProcess* process_ = nullptr;
        wxOutputStream* stream_ = nullptr;
        std::mutex write_mutex_;
        int width_ = 0;
        int height_ = 0;
        std::string path_;
        std::string codec_;
        int fps_;
        int in_rate_;
    };

    //----------------------------------------------
    // MetricsLogger
    //----------------------------------------------

    struct MetricsLoggerConfig {
        std::string runs_dir = "runs";
        std::string run_name_tmpl = "run_{t}";
        std::string video_codec = "libx264";
        int video_fps = 30;
        bool use_png_dump = false;
    };

    class MetricsLogger {
    public:
        explicit MetricsLogger(
        	std::unique_ptr<IBackend> backend, const MetricsLoggerConfig& config, const std::filesystem::path& root_dir);

        MetricsLogger(const MetricsLogger&) = delete;
        MetricsLogger& operator=(const MetricsLogger&) = delete;

        inline void LogScalar(const std::string& tag, int64_t step, double value) {
            json obj = {
                {"type", "scalar"},
                {"tag", tag},
                {"step", step},
                {"value", value}
            };
            backend_->WriteJsonl(obj);
        }

        void Log(const std::string& tag, const anet::Config& config);
        void Log(const anet::Config& config);

        void Log(const std::string& tag, const json& data);
        void Log(const std::string& tag, anet::rl::step_t step, const json& data);

        void Log(const std::string& tag, anet::rl::step_t step, const wxImage& image);
        void Log(const std::string& tag, anet::rl::step_t step, const anet::ImageSource& src, int width = -1, int height = -1);

        void Log(const std::string& tag, anet::rl::step_t step, const anet::graphviz::GraphViz& viz);

        inline std::string GetRunName() const { return run_name_; }
        inline std::filesystem::path GetRunDir() const { return run_dir_; }
        inline void Flush() { backend_->Flush(); }
    public:
        // --- Singleton API ---
        static std::shared_ptr<MetricsLogger> Instance();
        static void Init(std::unique_ptr<IBackend> backend, const MetricsLoggerConfig& config, const std::filesystem::path& root_dir);
        static void Reset();
    private:
        std::string CreateTimeStampStr() const;
        std::string CreateRunName(const std::string& run_name_tmpl) const;
        void LogJsonInternal(const std::string& tag, const json& data);
        void LogImage_subtyped(const std::string& tag, anet::rl::step_t step, const wxImage& image, const std::string& subtype_or_empty);
    private:
        static std::string GetCurrentTimeStr();
        static std::string SanitizeFilename(const std::string& s);
    private:
        std::unique_ptr<IBackend> backend_;
        MetricsLoggerConfig config_;
        std::string run_name_;
        std::filesystem::path run_dir_;
        
        std::mutex log_mutex_;
        std::mutex video_mutex_;

        // 画像・動画用連番管理
        std::unordered_map<std::string, uint64_t> image_seq_;
        std::unordered_map<std::string, std::unique_ptr<VideoLogger>> video_loggers_;
    private:
        // --- Singleton管理 ---
        static std::shared_ptr<MetricsLogger> instance_;
        static std::mutex instance_mutex_;
    };

}   // namespace anet
