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
        virtual void open(const std::string& root_dir, const std::string& run_name) = 0;
        virtual void write_jsonl(const json& obj) = 0;
        virtual void flush() = 0;
    };

    //----------------------------------------------
    // JSONLバックエンド
    //----------------------------------------------
    class JsonlBackend : public IBackend {
    private:
        std::ofstream ofs;
    public:
        void open(const std::string& root_dir, const std::string& run_name) override;
        void write_jsonl(const json& obj) override;
        void flush() override;
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

    public:
        VideoLogger(const std::string& path, int width, int height, int fps = 30, int in_rate = 120, const std::string& codec = "mjpeg");
        ~VideoLogger() { Close(); }

        void WriteFrame(const wxImage& img);
        void Close();
    };

    //----------------------------------------------
    // MetricsLogger 本体
    //----------------------------------------------
    class MetricsLogger {
    private:
        std::unique_ptr<IBackend> backend;
        std::string root_dir;
        std::string run_name;
        bool enable_image_log_ = true;

        // 画像・動画用連番管理
        std::unordered_map<std::string, uint64_t> image_seq_;
        std::unordered_map<std::string, std::unique_ptr<VideoLogger>> video_loggers_;

        static json round_numbers(const json& j, int precision = 6);
        static std::string current_time_str();
        static std::string sanitize_filename(const std::string& s);

        // 内部実装
        void log_image_subtyped(const std::string& tag,
            int step,
            const wxImage& image,
            const std::string& subtype_or_empty);

        // --- Singleton管理 ---
        static std::shared_ptr<MetricsLogger> instance_;
        static std::mutex instance_mutex_;
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

        inline void log_scalar(const std::string& tag, int step, double value) {
            json obj = {
                {"type", "scalar"},
                {"tag", tag},
                {"step", step},
                {"value", value}
            };
            backend->write_jsonl(obj);
        }

        inline void log_json(const std::string& tag, const json& data) {
            json rounded = round_numbers(data);
            json obj = {
                {"type", "json"},
                {"tag", tag},
                {"timestamp", current_time_str()},
                {"data", rounded}
            };
            backend->write_jsonl(obj);
        }

        inline void log_image(const std::string& tag, int step, const wxImage& image) {
            if (!enable_image_log_) return;
            log_image_subtyped(tag, step, image, "");
        }

        inline void log_image(const std::string& tag, int step, const anet::ImageSource& src, int width = -1, int height = -1) {
            if (!enable_image_log_) return;
            auto img = src.Render(width, height);
            auto subtype = src.GetImageSubType();
            log_image_subtyped(tag, step, img, subtype);
        }

        inline std::string get_run_name() const { return run_name; }
        inline std::string get_out_dir() const { return std::filesystem::relative(root_dir + "/" + run_name).string(); }
        inline void flush() { backend->flush(); }
    };
}   // namespace anet
