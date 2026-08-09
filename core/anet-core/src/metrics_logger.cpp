#include "anet/metrics_logger.hpp"
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <format>
#include <sstream>
#include <thread>
#include <wx/process.h>
#include <wx/image.h>
#include <wx/filename.h>
#include "anet/json_util.hpp"
#include "anet/log.hpp"
#include "anet/str_util.hpp"
#include "anet/profile.hpp"
#include "metrics_logger_impl.hpp"

using namespace anet;
namespace LOG = anet::log;

wxDEFINE_EVENT(wxEVT_APP_EXECUTE_START, wxThreadEvent);

namespace {

std::string ConfigDataToConfigString(const ConfigData& config_data)
{
    std::ostringstream oss;
    for (const auto& kv : config_data.Map()) {
        oss << kv.first << " = " << kv.second << std::endl;
    }
    return oss.str();
}

constexpr size_t kMaxCapturedStderr = 4096;
constexpr auto kFfmpegStartupCheckDelay = std::chrono::milliseconds(250);
// wx の Windows pipe は非ブロッキングなので、大きい frame は小分けに書き込む。
constexpr size_t kFfmpegWriteChunkBytes = 4 * 1024;
constexpr int kFfmpegZeroWriteRetryLimit = 1000;
constexpr auto kFfmpegZeroWriteRetryDelay = std::chrono::milliseconds(1);

std::string ToUtf8String(const wxString& value)
{
    const wxCharBuffer buffer = value.utf8_str();
    return buffer.data() ? std::string(buffer.data()) : std::string();
}

wxString BuildVideoOutputOptions(const std::string& codec, int fps)
{
    if (codec == "libx264") {
        // H.264用: 超高速(CPU最小)、色差間引きなし
        return wxString::Format(
            "-c:v libx264 -preset ultrafast -crf 15 -pix_fmt yuv444p -g %d -keyint_min 30 -sc_threshold 0 -tune fastdecode ",
            fps);
    }
    if (codec == "h264_nvenc") {
        return wxString::Format(
            "-c:v h264_nvenc -preset p1 -rc vbr -cq 15 -pix_fmt yuv444p -g %d  -forced-idr 1",
            fps);
    }

    // MJPEG等: 従来の可変ビットレート品質指定 (2は最高品質クラス)
    return wxString::Format("-c:v %s -q:v 2", wxString::FromUTF8(codec));
}

wxString BuildFfmpegCommand(const std::string& path, int width, int height, const std::string& codec, int fps)
{
    const wxString output_options = BuildVideoOutputOptions(codec, fps);
    return wxString::Format(
        "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %dx%d -framerate %d -threads 2"
        //"-report "
        " -hide_banner -loglevel error -nostats"
        " -thread_queue_size 512 -i - -f matroska %s \"%s\"",
        width, height, fps, output_options, wxString::FromUTF8(path)
    );
}

} // namespace

namespace anet::detail {

bool IsNvencEligibleVideoSize(int width, int height)
{
    return (width % 2 == 0)
        && (height % 2 == 0)
        && width >= kNvencMinWidth
        && height >= kNvencMinHeight;
}

VideoCodecDecision ResolveVideoCodec(const std::string& requested_codec, int width, int height, const std::string& path)
{
    const bool nvenc_eligible = IsNvencEligibleVideoSize(width, height);

    if (requested_codec == "auto") {
        return VideoCodecDecision{
            .codec = nvenc_eligible ? "h264_nvenc" : "libx264",
            .requested_auto = true,
            .nvenc_eligible = nvenc_eligible,
        };
    }

    if (requested_codec == "h264_nvenc" && !nvenc_eligible) {
        ANET_SYSTEM_ERROR(
            "metrics_logger.video_codec=h264_nvenc is not supported for video size "
            << width << "x" << height
            << ". Expected even width/height and at least "
            << kNvencMinWidth << "x" << kNvencMinHeight
            << ". Use metrics_logger.video_codec=auto or libx264. path=" << path);
    }

    return VideoCodecDecision{
        .codec = requested_codec,
        .requested_auto = false,
        .nvenc_eligible = nvenc_eligible,
    };
}

} // namespace anet::detail


namespace anet::detail {

namespace {

std::optional<JsonlIoFailure> TakeJsonlIoFailureOnce(
    const std::ios& stream,
    const std::filesystem::path& path,
    JsonlIoOperation operation,
    bool& error_reported)
{
    if (!stream.fail() || error_reported) {
        return std::nullopt;
    }

    error_reported = true;
    return JsonlIoFailure{
        .path = path,
        .operation = operation,
        .fail = stream.fail(),
        .bad = stream.bad(),
    };
}

const char* JsonlIoOperationName(JsonlIoOperation operation)
{
    switch (operation) {
    case JsonlIoOperation::kWrite:
        return "write";
    case JsonlIoOperation::kFlush:
        return "flush";
    }

    return "unknown";
}

} // namespace

std::optional<JsonlIoFailure> WriteJsonlLine(
    std::ostream& stream,
    std::string_view line,
    const std::filesystem::path& path,
    bool& error_reported)
{
    stream.write(line.data(), static_cast<std::streamsize>(line.size()));
    return TakeJsonlIoFailureOnce(stream, path, JsonlIoOperation::kWrite, error_reported);
}

std::optional<JsonlIoFailure> FlushJsonl(
    std::ostream& stream,
    const std::filesystem::path& path,
    bool& error_reported)
{
    stream.flush();
    return TakeJsonlIoFailureOnce(stream, path, JsonlIoOperation::kFlush, error_reported);
}

void LogJsonlIoFailure(const JsonlIoFailure& failure)
{
    LOG::error()
        << "Metrics JSONL I/O failed: operation=" << JsonlIoOperationName(failure.operation)
        << " path=" << failure.path
        << " fail=" << (failure.fail ? "true" : "false")
        << " bad=" << (failure.bad ? "true" : "false")
        << ". The run is not stopped, but metrics may no longer be recorded."
        << " Check free disk space and filesystem health."
        << " Further errors for this file are suppressed.";
}

} // namespace anet::detail


//----------------------------------------------
// JsonlBackend
//----------------------------------------------

void JsonlBackend::Open(const std::filesystem::path& runs_dir, const std::string& run_name)
{
    auto run_dir = runs_dir / run_name;
    std::filesystem::create_directories(run_dir);
    jsonl_path_ = run_dir / "metrics.jsonl";
    io_error_reported_ = false;
    ofs.open(jsonl_path_, std::ios::app);
    ANET_CHECK_MSG(ofs, "Failed to open: " << jsonl_path_);
}

void JsonlBackend::WriteJsonl(const json& obj)
{
    std::string line = obj.dump() + "\n";
    std::optional<detail::JsonlIoFailure> failure;
    {
        // JSONLの書き込みとone-shot失敗claimを同じ排他区間で確定する。
        std::lock_guard<std::mutex> lock(mtx_);
        failure = detail::WriteJsonlLine(ofs, line, jsonl_path_, io_error_reported_);
    }

    // 外部logger callbackによる再入やlock順序の逆転を避けるため、排他解除後に通知する。
    if (failure) {
        detail::LogJsonlIoFailure(*failure);
    }
}

void JsonlBackend::Flush()
{
    std::optional<detail::JsonlIoFailure> failure;
    {
        // バッファ排出で初めて顕在化するI/O失敗もwriteと共通のlatchで捕捉する。
        std::lock_guard<std::mutex> lock(mtx_);
        failure = detail::FlushJsonl(ofs, jsonl_path_, io_error_reported_);
    }

    // LogPanel/FileLoggerへの配送はmetrics I/Oの排他区間外で行う。
    if (failure) {
        detail::LogJsonlIoFailure(*failure);
    }
}


//----------------------------------------------
// VideoLogger 実装
//----------------------------------------------

class anet::VideoLogger::Process final : public wxProcess {
public:
    void DisableTerminationCapture()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        capture_termination_ = false;
    }

    void OnTerminate(int pid, int status) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (capture_termination_) {
            terminated_ = true;
            pid_ = pid;
            exit_code_ = status;
            has_exit_code_ = true;
            DrainStderrLocked();
        }
        wxProcess::OnTerminate(pid, status);
    }

    bool CopyTerminationState(long expected_pid, int* exit_code, bool* has_exit_code, std::string* captured_stderr) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!terminated_) {
            return false;
        }
        if (expected_pid != 0 && pid_ != 0 && pid_ != expected_pid) {
            return false;
        }
        *exit_code = exit_code_;
        *has_exit_code = has_exit_code_;
        captured_stderr->append(captured_stderr_);
        return true;
    }

private:
    void DrainStderrLocked()
    {
        wxInputStream* error_stream = GetErrorStream();
        if (!error_stream) {
            return;
        }

        char buffer[1024];
        while (IsErrorAvailable()) {
            error_stream->Read(buffer, sizeof(buffer));
            const size_t bytes_read = error_stream->LastRead();
            if (bytes_read == 0) {
                break;
            }
            captured_stderr_.append(buffer, bytes_read);
            if (captured_stderr_.size() > kMaxCapturedStderr) {
                captured_stderr_.erase(0, captured_stderr_.size() - kMaxCapturedStderr);
            }
        }
    }

    mutable std::mutex mutex_;
    bool capture_termination_ = true;
    bool terminated_ = false;
    long pid_ = 0;
    int exit_code_ = 0;
    bool has_exit_code_ = false;
    std::string captured_stderr_;
};

VideoLogger::VideoLogger(const std::string& path, int width, int height, const std::string& codec, int fps)
    : width_(width), height_(height), path_(path), codec_(codec), fps_(fps)
{
    ANET_CHECK_MSG(width <= 8192, "invalid Image size. width=" << width);
    ANET_CHECK_MSG(height <= 4320, "invalid Image size  height=" << height);

    wxFileName fn(wxString::FromUTF8(path_));
    wxFileName::Mkdir(fn.GetPath(), wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);

    const auto decision = detail::ResolveVideoCodec(codec, width_, height_, path_);
    if (decision.requested_auto) {
        LOG::info() << "VideoLogger(auto): " << width_ << "x" << height_ << " -> " << decision.codec;
    }

    LaunchFfmpeg(decision.codec);
    if (!DiedAtStartup()) {
        return;
    }

    if (decision.requested_auto && decision.codec == "h264_nvenc") {
        std::string failure_message;
        {
            std::lock_guard<std::mutex> lock(write_mutex_);
            failure_message = BuildFfmpegFailureMessageLocked("ffmpeg startup failed for auto-selected h264_nvenc");
        }
        LOG::warn() << failure_message << " Retrying with libx264. path=" << path_;

        LaunchFfmpeg("libx264");
        if (!DiedAtStartup()) {
            return;
        }
    }

    std::lock_guard<std::mutex> lock(write_mutex_);
    const std::string failure_message = BuildFfmpegFailureMessageLocked("ffmpeg startup failed");
    CloseProcessLocked();
    ANET_SYSTEM_ERROR(failure_message);
}

void VideoLogger::LaunchFfmpeg(const std::string& codec)
{
    std::lock_guard<std::mutex> lock(write_mutex_);
    CloseProcessLocked();

    codec_ = codec;
    ffmpeg_dead_.store(false);
    exit_code_ = 0;
    has_exit_code_ = false;
    pid_ = 0;
    captured_stderr_.clear();
    launch_cmd_.clear();
    stream_ = nullptr;

    const wxString cmd = BuildFfmpegCommand(path_, width_, height_, codec_, fps_);
    launch_cmd_ = ToUtf8String(cmd);
    LOG::info() << "VideoLogger: cmd=" << launch_cmd_;

    process_ = new Process();
    process_->Redirect();  // 標準入出力をリダイレクト

    if (!wxThread::IsMain()) {
        ANET_LOG_DEBUG("Sending ffmpeg execute request into main thread. command=" << launch_cmd_);
        ExecuteStarter executer(cmd, process_);
        executer.Execute();
        pid_ = process_->GetPid();
        ANET_LOG_DEBUG("ffmpeg execute done. pid=" << pid_);
        LOG::info() << "ffmpeg started from thread. pid=" << pid_;
    } else {
        pid_ = wxExecute(cmd, wxEXEC_ASYNC | wxEXEC_HIDE_CONSOLE, process_);
        ANET_LOG_DEBUG("ffmpeg started from main thread. pid=" << pid_);
        LOG::info() << "ffmpeg started from main thread. pid=" << pid_;
    }

    if (pid_ == 0) {
        ffmpeg_dead_.store(true);
        const std::string failure_message = BuildFfmpegFailureMessageLocked("Failed to launch ffmpeg process");
        CloseProcessLocked();
        ANET_SYSTEM_ERROR(failure_message);
    }

    // 書き込みストリーム取得
    stream_ = process_->GetOutputStream();
    if (!stream_) {
        const std::string failure_message = BuildFfmpegFailureMessageLocked("Failed to get ffmpeg stdin stream. nullptr");
        CloseProcessLocked();
        ANET_SYSTEM_ERROR(failure_message);
    }
    if (!stream_->IsOk()) {
        const std::string failure_message = BuildFfmpegFailureMessageLocked("Failed to get ffmpeg stdin stream. Is not OK");
        CloseProcessLocked();
        ANET_SYSTEM_ERROR(failure_message);
    }
}

bool VideoLogger::DiedAtStartup()
{
    std::this_thread::sleep_for(kFfmpegStartupCheckDelay);

    std::lock_guard<std::mutex> lock(write_mutex_);
    UpdateFfmpegDeathFromPidLocked();
    return ffmpeg_dead_.load();
}

void VideoLogger::UpdateFfmpegDeathFromPidLocked()
{
    if (ffmpeg_dead_.load() || pid_ == 0) {
        return;
    }
    if (process_ && process_->CopyTerminationState(pid_, &exit_code_, &has_exit_code_, &captured_stderr_)) {
        if (captured_stderr_.size() > kMaxCapturedStderr) {
            captured_stderr_.erase(0, captured_stderr_.size() - kMaxCapturedStderr);
        }
        ffmpeg_dead_.store(true);
        stream_ = nullptr;
        return;
    }
    if (!wxProcess::Exists(static_cast<int>(pid_))) {
        DrainStderrLocked();
        ffmpeg_dead_.store(true);
        stream_ = nullptr;
    }
}

void VideoLogger::DrainStderrLocked()
{
    if (!process_) {
        return;
    }
    wxInputStream* error_stream = process_->GetErrorStream();
    if (!error_stream) {
        return;
    }

    char buffer[1024];
    while (process_->IsErrorAvailable()) {
        error_stream->Read(buffer, sizeof(buffer));
        const size_t bytes_read = error_stream->LastRead();
        if (bytes_read == 0) {
            break;
        }
        AppendCapturedStderrLocked(buffer, bytes_read);
    }
}

void VideoLogger::AppendCapturedStderrLocked(const char* data, size_t size)
{
    if (size == 0) {
        return;
    }
    captured_stderr_.append(data, size);
    if (captured_stderr_.size() > kMaxCapturedStderr) {
        captured_stderr_.erase(0, captured_stderr_.size() - kMaxCapturedStderr);
    }
}

std::string VideoLogger::BuildFfmpegFailureMessageLocked(const std::string& context) const
{
    std::ostringstream oss;
    oss << context
        << ". path=" << path_
        << " codec=" << codec_
        << " exit_code=";
    if (has_exit_code_) {
        oss << exit_code_;
    } else {
        oss << "unknown";
    }
    oss << " cmd=\"" << launch_cmd_ << "\""
        << " stderr=\"" << StderrExcerptLocked() << "\"";
    return oss.str();
}

std::string VideoLogger::StderrExcerptLocked() const
{
    if (captured_stderr_.empty()) {
        return "(empty)";
    }
    return captured_stderr_;
}

void VideoLogger::WriteFrame(const wxImage& img)
{
    std::lock_guard<std::mutex> lock(write_mutex_);
    UpdateFfmpegDeathFromPidLocked();
    if (ffmpeg_dead_.load() || stream_ == nullptr) {
        ANET_SYSTEM_ERROR(BuildFfmpegFailureMessageLocked("ffmpeg process is not available before frame write"));
    }
    if (!stream_->IsOk()) {
        DrainStderrLocked();
        ANET_SYSTEM_ERROR(BuildFfmpegFailureMessageLocked("ffmpeg stdin stream is not OK before frame write"));
    }

    const unsigned char* data = img.GetData();
    size_t nbytes = width_ * height_ * 3;

    size_t written = 0;
    int zero_write_retries = 0;
    while (written < nbytes) {
        const size_t write_size = std::min(kFfmpegWriteChunkBytes, nbytes - written);
        stream_->Write(data + written, write_size);
        if (!stream_->IsOk()) {
            UpdateFfmpegDeathFromPidLocked();
            DrainStderrLocked();
            ANET_SYSTEM_ERROR(BuildFfmpegFailureMessageLocked("ffmpeg pipe write failed"));
        }
        const size_t last_write = stream_->LastWrite();
        if (last_write == 0) {
            UpdateFfmpegDeathFromPidLocked();
            DrainStderrLocked();
            if (ffmpeg_dead_.load() || stream_ == nullptr || !stream_->IsOk()) {
                ANET_SYSTEM_ERROR(BuildFfmpegFailureMessageLocked("ffmpeg pipe wrote zero bytes after ffmpeg stopped"));
            }
            ++zero_write_retries;
            if (zero_write_retries > kFfmpegZeroWriteRetryLimit) {
                ANET_SYSTEM_ERROR(BuildFfmpegFailureMessageLocked("ffmpeg pipe wrote zero bytes repeatedly"));
            }
            std::this_thread::sleep_for(kFfmpegZeroWriteRetryDelay);
            continue;
        }
        zero_write_retries = 0;
        written += last_write;
    }
}

void VideoLogger::CloseProcessLocked()
{
    if (stream_) {
        stream_ = nullptr;
    }
    if (process_) {
        process_->DisableTerminationCapture();
        process_->SetNextHandler(nullptr);
        process_->CloseOutput();
        delete process_;
        process_ = nullptr;
    }
    pid_ = 0;
}

void VideoLogger::Close()
{
    std::lock_guard<std::mutex> lock(write_mutex_);
    CloseProcessLocked();
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

void MetricsLogger::Log(const std::string& tag, const ConfigData& config_data)
{
    std::lock_guard<std::mutex> lock(log_mutex_);

    auto config_str = ConfigDataToConfigString(config_data);

    // Configと同じディレクトリにバラのファイルとしてダンプ
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
    ANET_PROFILE_FUNC();

    LogImage_subtyped(tag, step, image, "");
}

void MetricsLogger::Log(const std::string& tag, anet::rl::step_t step, const anet::ImageSource& src, int width, int height)
{
    ANET_PROFILE_FUNC();

    auto image = src.Render(width, height);
    auto subtype = src.GetImageSubType();
    LogImage_subtyped(tag, step, image, subtype);
}

void MetricsLogger::LogImage_subtyped(const std::string& tag, anet::rl::step_t step, const wxImage& image, const std::string& subtype_or_empty)
{
    ANET_PROFILE_FUNC();

    ANET_PROFILE_SCOPE(prepare);

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
            ANET_PROFILE_SCOPE(make_video_logger);
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

    ANET_PROFILE_SCOPE_NEXT(write_frame);
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

void MetricsLogger::Log(const std::string& tag, const anet::graphviz::GraphViz& viz)
{
    std::lock_guard<std::mutex> lock(log_mutex_);

    std::string safe_tag = SanitizeFilename(tag);
    auto dot_dir = run_dir_ / "dot";
    std::filesystem::create_directories(dot_dir);
    auto full_path = dot_dir / (safe_tag + ".dot");

    std::ofstream ofs(full_path);
    ofs << viz.ToDotString() << std::endl;
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
//    if (instance_) {
//        instance_->Flush();
//	}
    instance_.reset();
}
