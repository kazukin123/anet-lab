#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <wx/image.h>
#include <wx/process.h>
#include "anet/heat_map.hpp"

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
        VideoLogger(const std::string& path, int width, int height, int fps = 10, int in_rate = 30, const std::string& codec = "mjpeg");
        ~VideoLogger() { Close(); }

        void WriteFrame(const wxImage& img);
    };

    //----------------------------------------------
    // MetricsLogger 本体
    //----------------------------------------------
    class MetricsLogger {
    public:
        explicit MetricsLogger(std::unique_ptr<IBackend> b,
            const std::string& root = "logs",
            const std::string& run = "");

        MetricsLogger(const MetricsLogger&) = delete;
        MetricsLogger& operator=(const MetricsLogger&) = delete;

        // --- Singleton API ---
        static std::shared_ptr<MetricsLogger> Instance();
        static void Init(std::unique_ptr<IBackend> backend, const std::string& root = "logs", const std::string& run = "");
        static void Reset();

        void SetEnableImageLog(bool enable_image_log) { enable_image_log_ = enable_image_log; }

        inline void LogScalar(const std::string& tag, int step, double value) {
            json obj = {
                {"type", "scalar"},
                {"tag", tag},
                {"step", step},
                {"value", value}
            };
            backend_->WriteJsonl(obj);
        }

        inline void LogJson(const std::string& tag, const json& data) {
            json rounded = round_numbers(data);
            json obj = {
                {"type", "json"},
                {"tag", tag},
                {"timestamp", current_time_str()},
                {"data", rounded}
            };
            backend_->WriteJsonl(obj);
        }

        inline void LogImage(const std::string& tag, int step, const wxImage& image) {
            if (!enable_image_log_) return;
            LogImage_subtyped(tag, step, image, "");
        }

        inline void LogImage(const std::string& tag, int step, const anet::ImageSource& src, int width = -1, int height = -1) {
            if (!enable_image_log_) return;
            auto img = src.Render(width, height);
            auto subtype = src.GetImageSubType();
            LogImage_subtyped(tag, step, img, subtype);
        }

        inline std::string GetRunName() const { return run_name_; }
        inline std::string GetOutDir() const { return std::filesystem::relative(root_dir_ + "/" + run_name_).string(); }
        inline void Flush() { backend_->Flush(); }
    private:
        std::unique_ptr<IBackend> backend_;
        std::string root_dir_;
        std::string run_name_;
        bool enable_image_log_ = true;

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
