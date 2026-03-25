#include "anet/metrics_logger.hpp"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <format>
#include <wx/process.h>
#include <wx/image.h>
#include <wx/filename.h>
#include "anet/log.hpp"
#include "anet/str_util.hpp"
#include "anet/profile.hpp"

using namespace anet;
namespace LOG = anet::log;

wxDEFINE_EVENT(wxEVT_APP_EXECUTE_START, wxThreadEvent);


//----------------------------------------------
// JsonlBackend
//----------------------------------------------

void JsonlBackend::Open(const std::filesystem::path& runs_dir, const std::string& run_name)
{
    auto run_dir = runs_dir / run_name;
    std::filesystem::create_directories(run_dir);
    auto jsonl_path = run_dir / "metrics.jsonl";
    ofs.open(jsonl_path, std::ios::app);
    ANET_CHECK_MSG(ofs, "Failed to open: " << jsonl_path);
}

void JsonlBackend::WriteJsonl(const json& obj)
{
    std::string line = obj.dump() + "\n";
    std::lock_guard<std::mutex> lock(mtx_);
    ofs << line;
}

void JsonlBackend::Flush()
{
    std::lock_guard<std::mutex> lock(mtx_);
    ofs.flush();
}


//----------------------------------------------
// VideoLogger 実装
//----------------------------------------------

VideoLogger::VideoLogger(const std::string& path, int width, int height, const std::string& codec, int fps)
    : width_(width), height_(height), path_(path), codec_(codec), fps_(fps)
{
    ANET_CHECK_MSG(width <= 8192, "invalid Image size.");
    ANET_CHECK_MSG(height <= 4320, "invalid Image size.");

    wxFileName fn(wxString::FromUTF8(path_));
    wxFileName::Mkdir(fn.GetPath(), wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);

    // エンコード用の出力オプションをコーデックによって切り替え
    wxString output_options;
    if (codec_ == "libx264") {
        // H.264用: 超高速(CPU最小)、色差間引きなし
        output_options = wxString::Format(
            "-c:v libx264 -preset ultrafast -crf 15 -pix_fmt yuv444p -g %d -keyint_min 30 -sc_threshold 0 -tune fastdecode ",
            fps);
    } else if (codec_ == "h264_nvenc") {
        output_options = wxString::Format(
            "-c:v h264_nvenc -preset p1 -rc vbr -cq 15 -pix_fmt yuv444p -g %d  -forced-idr 1",
            fps);
    } else {
        // MJPEG等: 従来の可変ビットレート品質指定 (2は最高品質クラス)
        output_options = wxString::Format("-c:v %s -q:v 2", wxString::FromUTF8(codec_));
    }

    // コマンドライン
    wxString cmd = wxString::Format(
        "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %dx%d -framerate %d -threads 2 "
        //"-report "
        "-thread_queue_size 512 -i - -f matroska %s \"%s\"",
        width_, height_, fps_, output_options, wxString::FromUTF8(path_)
    );
    //ANET_LOG_DEBUG("cmd=" << cmd.c_str());
    LOG::info() << "VideoLogger: cmd=" << cmd.c_str();

    process_ = new wxProcess();
    process_->Redirect();  // 標準入出力をリダイレクト

    if (!wxThread::IsMain()) {
        ANET_LOG_DEBUG("Sending ffmpeg execue request into main thread. command=" << cmd);
        ExecuteStarter executer(cmd, process_);
        executer.Execute();
        ANET_LOG_DEBUG("ffmpeg execute done. pid=" << process_->GetPid());
        LOG::info() << "ffmpeg started from thread. pid=" << process_->GetPid();
    } else {
        long pid = wxExecute(cmd, wxEXEC_ASYNC | wxEXEC_HIDE_CONSOLE, process_);
        if (pid == 0)
            throw std::runtime_error("Failed to launch ffmpeg process");
        ANET_LOG_DEBUG("ffmpeg started from main thread. pid=" << pid);
        LOG::info() << "ffmpeg started from main thread. pid=" << pid;
    }

    // 書き込みストリーム取得
    stream_ = process_->GetOutputStream();
    if (!stream_)
        throw std::runtime_error("Failed to get ffmpeg stdin stream");
}

void VideoLogger::WriteFrame(const wxImage& img)
{
    std::lock_guard<std::mutex> lock(write_mutex_);
    ANET_CHECK(stream_ != nullptr);
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

void VideoLogger::Close()
{
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
// MetricsLogger
//----------------------------------------------

std::string MetricsLogger::GetCurrentTimeStr()
{
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

std::string MetricsLogger::SanitizeFilename(const std::string& s)
{
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

MetricsLogger::MetricsLogger(std::unique_ptr<IBackend> backend, const MetricsLoggerConfig& config, const std::filesystem::path& root_dir)
    : backend_(std::move(backend)), config_(config)
{
    if (config_.run_name_tmpl.empty()) {
        run_name_ = "run_" + CreateTimeStampStr();
    } else {
        run_name_ = CreateRunName(config_.run_name_tmpl);
    }

    auto runs_dir = root_dir / config_.runs_dir;
    backend_->Open(runs_dir, run_name_);
    json meta = { {"type","meta"}, {"event","start"}, {"timestamp", GetCurrentTimeStr()} };
    backend_->WriteJsonl(meta);

    run_dir_ = root_dir / config_.runs_dir / run_name_;
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

void MetricsLogger::Log(const anet::Config& config)
{
    auto tag = config.GetConfigPrefix();
    Log(tag, config);
}

void MetricsLogger::LogJsonInternal(const std::string& tag, const json& data)
{
    json rounded = round_numbers(data);
    json obj = {
        {"type", "json"},
        {"tag", tag},
        {"data", rounded}
    };

    std::string safe_tag = SanitizeFilename(tag);
    auto json_dir = run_dir_ / "json";
    std::filesystem::create_directories(json_dir);
    auto json_path = json_dir / (safe_tag + ".json");
    std::ofstream ofs(json_path);
    ofs << obj.dump(4) << std::endl;

    obj["timestamp"] = GetCurrentTimeStr();
    backend_->WriteJsonl(obj);
}

void MetricsLogger::Log(const std::string& tag, const anet::Config& config)
{
    std::lock_guard<std::mutex> lock(log_mutex_);

    // JSONとして書き込み
    auto json_data = config.ToJson();
    LogJsonInternal(tag, json_data);

    auto config_prefix = config.GetConfigPrefix();
    auto config_str = config.ToConfigString();

    // config.txtに追記
    auto common_txt_path = this->run_dir_ / "config.txt";
    {
        std::ofstream ofs(common_txt_path, std::ios_base::app);  // 追記モードでファイルを開く
        ofs << config_str;
    }

    // バラのファイルにダンプ
    std::string safe_tag = SanitizeFilename(tag);
    auto config_dir = this->run_dir_ / "config";
    std::filesystem::create_directories(config_dir);
    auto config_txt_path = config_dir / (safe_tag + ".txt");
    {
        std::ofstream ofs(config_txt_path, std::ios_base::out);
        ofs << config_str;
    }
}

void MetricsLogger::Log(const std::string& tag, const json& data)
{
    std::lock_guard<std::mutex> lock(log_mutex_);
    LogJsonInternal(tag, data);
}

void MetricsLogger::Log(const std::string& tag, anet::rl::step_t step, const json& data)
{
    std::lock_guard<std::mutex> lock(log_mutex_);

    json rounded = round_numbers(data);
    json obj = {
        {"type", "json"},
        {"tag", tag},
        {"data", rounded}
    };

    std::string safe_tag = SanitizeFilename(tag);
    auto json_dir = run_dir_ / "json";
    std::filesystem::create_directories(json_dir);
    auto full_path = json_dir / (safe_tag + std::format("_{}.json", step));

    std::ofstream ofs(full_path);  // ファイルを開く
    ofs << obj.dump(4) << std::endl;     // インデント幅 4 で書出

    obj["timestamp"] = GetCurrentTimeStr();
    backend_->WriteJsonl(obj);
}

void MetricsLogger::Log(const std::string& tag, anet::rl::step_t step, const wxImage& image)
{
    ProfileRange r("MetricsLogger::LogImage1");

    LogImage_subtyped(tag, step, image, "");
}

void MetricsLogger::Log(const std::string& tag, anet::rl::step_t step, const anet::ImageSource& src, int width, int height)
{
    ProfileRange r("MetricsLogger::LogImage2");

    auto image = src.Render(width, height);
    auto subtype = src.GetImageSubType();
    LogImage_subtyped(tag, step, image, subtype);
}

void MetricsLogger::LogImage_subtyped(const std::string& tag, anet::rl::step_t step, const wxImage& image, const std::string& subtype_or_empty)
{
    ProfileRange r("MetricsLogger::LogImage_subtyped");

    ProfileRange r1("MetricsLogger::::LogImage_subtyped.prepare");

    // タグを安全なファイル名に変換
    std::string safe_tag = SanitizeFilename(tag);

    // ---- 画像書き込み (個別PNG保存はデバッグ用) ----
    if (config_.use_png_dump) {
        uint64_t seq = image_seq_[tag]++;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%06llu", static_cast<unsigned long long>(seq));

        auto images_dir = run_dir_ / "images" / safe_tag;
        std::filesystem::create_directories(images_dir);
        auto image_path = images_dir / (safe_tag + "_" + buf + ".png");
        image.SaveFile(image_path.string(), wxBITMAP_TYPE_PNG);
    }

    VideoLogger* target_logger = nullptr;
    {
        // ロックを取って、既にロガーが存在するか確認
        std::lock_guard<std::mutex> lock(video_mutex_);
        auto it = video_loggers_.find(tag);
        if (it != video_loggers_.end()) {
            target_logger = it->second.get();
        }
    }

    // 一時退避用
    std::unique_ptr<VideoLogger> temp_vlog;

    // VideoLoggerが存在しない場合は作る
    if (!target_logger) {
        // VideoLogger を作る
        auto vid_path = run_dir_ / "videos" / (safe_tag + ".mkv");
        temp_vlog = std::make_unique<VideoLogger>(vid_path.string(), image.GetWidth(), image.GetHeight(), config_.video_codec, config_.video_fps);

        // マップ登録用にロックを取る
        std::lock_guard<std::mutex> lock(video_mutex_);

        // 作っている間に、別のスレッドが同じタグのロガーを作ってしまったか再確認（Double-Checked Locking）
        auto it = video_loggers_.find(tag);
        if (it == video_loggers_.end()) {
            // Mapに登録
            ProfileRange r2("MetricsLogger::::LogImage_subtyped.make_VideoLogger");
            target_logger = temp_vlog.get();
            video_loggers_[tag] = std::move(temp_vlog);

            // メタデータを一回だけ書き込み
            json vmeta = {
                {"type", "video"},
                {"tag", tag},
                {"path", "videos/" + safe_tag + ".mkv"},
                {"timestamp", GetCurrentTimeStr()}
            };
            backend_->WriteJsonl(vmeta);
        } else {
            // もし別スレッドが先に作っていたら、そっちを使う（temp_vlog は自動で破棄される）
            target_logger = it->second.get();
        }
    }

    ProfileRange r3("MetricsLogger::::LogImage_subtyped.writeFrame", r1);
    target_logger->WriteFrame(image);

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

void MetricsLogger::Log(const std::string& tag, anet::rl::step_t step, const anet::graphviz::GraphViz& viz)
{
    std::lock_guard<std::mutex> lock(log_mutex_);

    std::string safe_tag = SanitizeFilename(tag);
    auto dot_dir = run_dir_ / "dot";
    std::filesystem::create_directories(dot_dir);
    auto tag_dir = dot_dir / safe_tag;
    std::filesystem::create_directories(tag_dir);
    auto full_path = tag_dir / (safe_tag + std::format("_{:010}.dot", step));

    std::ofstream ofs(full_path);
    ofs << viz.ToDotString() << std::endl;
}


// ---- シングルトン管理 ----

std::shared_ptr<MetricsLogger> MetricsLogger::instance_ = nullptr;
std::mutex MetricsLogger::instance_mutex_;

std::shared_ptr<MetricsLogger> MetricsLogger::Instance()
{
    std::lock_guard<std::mutex> lock(instance_mutex_);
    return instance_;
}

void MetricsLogger::Init(std::unique_ptr<IBackend> backend, const MetricsLoggerConfig& config, const std::filesystem::path& root_dir)
{
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = std::make_shared<MetricsLogger>(std::move(backend), config, root_dir);
    }
}

void MetricsLogger::Reset() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    instance_.reset();
}

