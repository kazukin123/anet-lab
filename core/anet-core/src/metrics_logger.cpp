#include "anet/metrics_logger.hpp"
#include <stdexcept>
#include <algorithm>
#include <wx/process.h>
#include <wx/image.h>
#include <wx/filename.h>
#include "anet/log.hpp"
#include "anet/str_util.hpp"
#include "anet/profile.hpp"

namespace LOG = anet::log;

wxDEFINE_EVENT(wxEVT_APP_EXECUTE_START, wxThreadEvent);

namespace anet {

    //----------------------------------------------
    // JsonlBackend 実装
    //----------------------------------------------
    void JsonlBackend::Open(const std::string& root_dir, const std::string& run_name) {
        std::filesystem::create_directories(root_dir + "/" + run_name);
        auto path = root_dir + "/" + run_name + "/metrics.jsonl";
        ofs.open(path, std::ios::app);
        if (!ofs) throw std::runtime_error("Failed to open: " + path);
    }

    void JsonlBackend::WriteJsonl(const json& obj) {
        ofs << obj.dump() << "\n";
        //ofs.flush();
    }

    void JsonlBackend::Flush() { ofs.flush(); }

    //----------------------------------------------
    // VideoLogger 実装
    //----------------------------------------------
    VideoLogger::VideoLogger(const std::string& path, int width, int height, int fps, const std::string& codec)
        : width_(width), height_(height), path_(path), fps_(fps), codec_(codec)
    {
        wxFileName fn(wxString::FromUTF8(path_));
        wxFileName::Mkdir(fn.GetPath(), wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);

        wxString cmd = wxString::Format(
            "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %dx%d -framerate %d "
            "-i - -thread_queue_size 512 -f matroska -c:v %s -q:v 2 \"%s\"",
            width_, height_, fps_, wxString::FromUTF8(codec_), wxString::FromUTF8(path_)
        );

        process_ = new wxProcess();
        process_->Redirect();  // 標準入出力をリダイレクト

        if (!wxThread::IsMain()) {
            ANET_LOG_DEBUG("Sending ffmpeg execue request into main thread. command=" << cmd);
            ExecuteStarter executer(cmd, process_);
            executer.Execute();
            ANET_LOG_DEBUG("ffmpeg execute done. pid=" << process_->GetPid());
        } else {
            long pid = wxExecute(cmd, wxEXEC_ASYNC | wxEXEC_HIDE_CONSOLE, process_);
            if (pid == 0)
                throw std::runtime_error("Failed to launch ffmpeg process");
        }

        // 書き込みストリーム取得
        stream_ = process_->GetOutputStream();
        if (!stream_)
            throw std::runtime_error("Failed to get ffmpeg stdin stream");
    }

    void VideoLogger::WriteFrame(const wxImage& img) {
        if (!stream_ || !stream_->IsOk()) return;
        const unsigned char* data = img.GetData();
        size_t nbytes = width_ * height_ * 3;

        size_t written = 0;
        while (written < nbytes) {
            stream_->Write(data + written, nbytes - written);
            if (!stream_->IsOk()) {
                LOG::error() << "ffmpeg pipe write failed";
                return;
            }
            written += stream_->LastWrite();
        }
    }

    void VideoLogger::Close() {
        if (stream_) {
            stream_ = nullptr;
        }
        if (process_) {
            process_->SetNextHandler(nullptr);
            process_->CloseOutput();
            delete process_;
            process_ = nullptr;
        }
    }

    //----------------------------------------------
    // MetricsLogger 内部ヘルパ
    //----------------------------------------------
    json MetricsLogger::round_numbers(const json& j, int precision) {
        if (j.is_number_float()) {
            double val = j.get<double>();
            double scale = std::pow(10.0, precision);
            return std::round(val * scale) / scale;
        }
        else if (j.is_object()) {
            json res;
            for (auto& [k, v] : j.items()) res[k] = round_numbers(v, precision);
            return res;
        }
        else if (j.is_array()) {
            json arr = json::array();
            for (auto& v : j) arr.push_back(round_numbers(v, precision));
            return arr;
        }
        return j;
    }

    std::string MetricsLogger::current_time_str() {
        auto t = std::chrono::system_clock::now();
        std::time_t tt = std::chrono::system_clock::to_time_t(t);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &tt);
#else
        localtime_r(&tt, &tm);
#endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
        return buf;
    }

    std::string MetricsLogger::sanitize_filename(const std::string& s) {
        std::string r = s;
        for (char& c : r) {
            switch (c) {
            case '/': case '\\': case ':': case '*': case '?':
            case '"': case '<': case '>': case '|':
                c = '-';
                break;
            default:
                break;
            }
        }
        return r;
    }

    //----------------------------------------------
    // MetricsLogger コンストラクタ
    //----------------------------------------------
    MetricsLogger::MetricsLogger(std::unique_ptr<IBackend> backend, const std::string& root, const std::string& run_name_tmpl)
        : backend_(std::move(backend)), root_dir_(root)
    {
        if (run_name_tmpl.empty()) {
            run_name_ = "run_" + CreateTimeStampStr();
        } else {
            run_name_ = CreateRunName(run_name_tmpl);
        }

        backend_->Open(root_dir_, run_name_);
        json meta = { {"type","meta"}, {"event","start"}, {"timestamp", current_time_str()} };
        backend_->WriteJsonl(meta);
    }

    std::string MetricsLogger::CreateTimeStampStr() const
    {
        auto t = std::chrono::system_clock::now();
        std::time_t tt = std::chrono::system_clock::to_time_t(t);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &tt);
#else
        localtime_r(&tt, &tm);
#endif
        char buf_ts[64];
        std::strftime(buf_ts, sizeof(buf_ts), "%Y%m%d-%H%M%S", &tm);

        return buf_ts;
    }

    std::string MetricsLogger::CreateRunName(const std::string& run_name_tmpl) const
    {
        auto t = CreateTimeStampStr();
        auto run_name = anet::ReplaceAll(run_name_tmpl, "{t}", t);
        return run_name;
    }

    void MetricsLogger::LogJson(const std::string& tag, const json& data)
    {
        json rounded = round_numbers(data);
        json obj = {
            {"type", "json"},
            {"tag", tag},
            {"timestamp", current_time_str()},
            {"data", rounded}
        };
        backend_->WriteJsonl(obj);

        std::string safe_tag = sanitize_filename(tag);
        std::string full_dir = root_dir_ + "/" + run_name_ + "/json";
        std::string full_path = root_dir_ + "/" + run_name_ + "/json/" + safe_tag + ".json";
        std::filesystem::create_directories(full_dir);
        std::ofstream ofs(full_path);  // ファイルを開く
        ofs << obj.dump(4) << std::endl;     // インデント幅 4 で書き出
    }

    void MetricsLogger::LogImage(const std::string& tag, int step, const wxImage& image)
    {
        ProfileRange r("MetricsLogger::LogImage1");

        LogImage_subtyped(tag, step, image, "");
    }

    void MetricsLogger::LogImage(const std::string& tag, int step, const anet::ImageSource& src, int width, int height)
    {
        ProfileRange r("MetricsLogger::LogImage2");

        auto image = src.Render(width, height);
        auto subtype = src.GetImageSubType();
        LogImage_subtyped(tag, step, image, subtype);
    }

    //----------------------------------------------
    // 画像・動画出力
    //----------------------------------------------
    void MetricsLogger::LogImage_subtyped(const std::string& tag,
        int step,
        const wxImage& image,
        const std::string& subtype_or_empty)
    {
        ProfileRange r("MetricsLogger::LogImage_subtyped");

        ProfileRange r1("MetricsLogger::::LogImage_subtyped.prepare");

        // タグを安全なファイル名に変換
        std::string safe_tag = sanitize_filename(tag);

        // ---- 画像書き込み (個別PNG保存は無効化) ----
        if (false) {
            uint64_t seq = image_seq_[tag]++;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%06llu", static_cast<unsigned long long>(seq));

            std::string rel_dir = "images/" + safe_tag;
            std::string rel_path = rel_dir + "/" + safe_tag + "_" + buf + ".png";
            std::string full_dir = root_dir_ + "/" + run_name_ + "/" + rel_dir;
            std::string full_path = root_dir_ + "/" + run_name_ + "/" + rel_path;

            std::filesystem::create_directories(full_dir);
            image.SaveFile(full_path, wxBITMAP_TYPE_PNG);
        }

        // ---- 動画書き込み ----
        auto vid_path = root_dir_ + "/" + run_name_ + "/videos/" + safe_tag + ".mkv";
        auto it = video_loggers_.find(tag);
        if (it == video_loggers_.end()) {
            ProfileRange r2("MetricsLogger::::LogImage_subtyped.make_VideoLogger");

            auto vlog = std::make_unique<VideoLogger>(
                vid_path, image.GetWidth(), image.GetHeight(), 30);
            json vmeta = {
                {"type", "video"},
                {"tag", tag},
                {"path", "videos/" + safe_tag + ".mkv"},
                {"timestamp", current_time_str()}
            };
            backend_->WriteJsonl(vmeta);
            it = video_loggers_.emplace(tag, std::move(vlog)).first;
        }

        ProfileRange r3("MetricsLogger::::LogImage_subtyped.writeFrame", r1);
        it->second->WriteFrame(image);

        /// @todo 動画フレーム情報Metrics出力

        // ---- JSONL (画像単体情報) ----
        //json obj = {
        //    {"run", run_name},
        //    {"type", "image"},
        //    {"tag", tag},
        //    {"step", step},
        //    {"path", rel_path},
        //    {"timestamp", current_time_str()}
        //};
        //if (!subtype_or_empty.empty()) obj["subtype"] = subtype_or_empty;
        //backend->write_jsonl(obj);
    }


    // ---- シングルトン管理 ----

    std::shared_ptr<MetricsLogger> MetricsLogger::instance_ = nullptr;
    std::mutex MetricsLogger::instance_mutex_;

    std::shared_ptr<MetricsLogger> MetricsLogger::Instance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        return instance_;
    }

    void MetricsLogger::Init(std::unique_ptr<IBackend> backend, const std::string& root, const std::string& run)
    {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = std::make_shared<MetricsLogger>(std::move(backend), root, run);
        }
    }

    void MetricsLogger::Reset() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        instance_.reset();
    }
} // namespace anet
