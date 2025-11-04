#include "anet/MetricsLogger.hpp"
#include <stdexcept>
#include <wx/process.h>
#include <wx/wfstream.h>
#include <wx/image.h>
#include <wx/filename.h>

//----------------------------------------------
// JsonlBackend 実装
//----------------------------------------------
void JsonlBackend::open(const std::string& root_dir, const std::string& run_name) {
    std::filesystem::create_directories(root_dir + "/" + run_name);
    auto path = root_dir + "/" + run_name + "/metrics.jsonl";
    ofs.open(path, std::ios::app);
    if (!ofs) throw std::runtime_error("Failed to open: " + path);
}

void JsonlBackend::write_jsonl(const json& obj) {
    ofs << obj.dump() << "\n";
    ofs.flush();
}

void JsonlBackend::flush() { ofs.flush(); }

//----------------------------------------------
// VideoLogger 実装
//----------------------------------------------
VideoLogger::VideoLogger(const std::string& path, int width, int height, int fps,int in_rate, const std::string& codec)
        : width_(width), height_(height), path_(path), fps_(fps), in_rate_(in_rate), codec_(codec)
{
    wxFileName fn(wxString::FromUTF8(path_));
    wxFileName::Mkdir(fn.GetPath(), wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);

    wxString cmd = wxString::Format(
        "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %dx%d -r %d -framerate %d "
        "-i - -f matroska -c:v %s -q:v 2 \"%s\"",
        width_, height_, in_rate, fps_, wxString::FromUTF8(codec_), wxString::FromUTF8(path_)
    );

    process_ = new wxProcess();
    process_->Redirect();  // 標準入出力をリダイレクト
    long pid = wxExecute(cmd, wxEXEC_ASYNC | wxEXEC_HIDE_CONSOLE, process_);
    if (pid == 0)
        throw std::runtime_error("Failed to launch ffmpeg process");

    // 書き込みストリーム取得
    stream_ = process_->GetOutputStream();
    if (!stream_)
        throw std::runtime_error("Failed to get ffmpeg stdin stream");
}

void VideoLogger::WriteFrame(const wxImage& img) {
    if (!stream_ || !stream_->IsOk()) return;
    const unsigned char* data = img.GetData();
    size_t nbytes = width_ * height_ * 3;
    stream_->Write(data, nbytes);
}

void VideoLogger::Close() {
    if (stream_) {
        stream_->Close();
        stream_ = nullptr;
    }
    if (process_) {
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
MetricsLogger::MetricsLogger(std::unique_ptr<IBackend> b,
    const std::string& root,
    const std::string& run)
    : backend(std::move(b)), root_dir(root)
{
    if (run.empty()) {
        auto t = std::chrono::system_clock::now();
        std::time_t tt = std::chrono::system_clock::to_time_t(t);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &tt);
#else
        localtime_r(&tt, &tm);
#endif
        char buf_ts[64];
        std::strftime(buf_ts, sizeof(buf_ts), "run_%Y%m%d-%H%M%S", &tm);
        run_name = buf_ts;
    }
    else {
        run_name = run;
    }

    backend->open(root_dir, run_name);
    json meta = { {"type","meta"}, {"event","start"}, {"timestamp", current_time_str()} };
    backend->write_jsonl(meta);
}

//----------------------------------------------
// 画像・動画出力
//----------------------------------------------
void MetricsLogger::log_image_subtyped(const std::string& tag,
    int step,
    const wxImage& image,
    const std::string& subtype_or_empty)
{
	// タグを安全なファイル名に変換
    std::string safe_tag = sanitize_filename(tag);

	// ---- 画像書き込み (個別PNG保存は無効化) ----
    if (false) {
        uint64_t seq = image_seq_[tag]++;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%06llu", static_cast<unsigned long long>(seq));

        std::string rel_dir = "images/" + safe_tag;
        std::string rel_path = rel_dir + "/" + safe_tag + "_" + buf + ".png";
        std::string full_dir = root_dir + "/" + run_name + "/" + rel_dir;
        std::string full_path = root_dir + "/" + run_name + "/" + rel_path;

        std::filesystem::create_directories(full_dir);
        image.SaveFile(full_path, wxBITMAP_TYPE_PNG);
    }

    // ---- 動画書き込み ----
    auto vid_path = root_dir + "/" + run_name + "/videos/" + safe_tag + ".mkv";
    auto it = video_loggers_.find(tag);
    if (it == video_loggers_.end()) {
        auto vlog = std::make_unique<VideoLogger>(vid_path, image.GetWidth(), image.GetHeight());
        json vmeta = {
            //{"run", run_name},
            {"type", "video"},
            {"tag", tag},
            {"path", "videos/" + safe_tag + ".mkv"},
            {"fps", 30},
            {"timestamp", current_time_str()}
        };
        backend->write_jsonl(vmeta);
        it = video_loggers_.emplace(tag, std::move(vlog)).first;
    }
    it->second->WriteFrame(image);

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
