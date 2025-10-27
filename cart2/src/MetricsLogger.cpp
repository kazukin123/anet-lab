#include "anet/MetricsLogger.hpp"

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

void JsonlBackend::flush() {
    ofs.flush();
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

//----------------------------------------------
// MetricsLogger 本体
//----------------------------------------------
MetricsLogger::MetricsLogger(std::unique_ptr<IBackend> b,
    const std::string& root,
    const std::string& run)
    : backend(std::move(b)), root_dir(root)
{
    // 自動 run_name 付与
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

    json meta = {
        {"type", "meta"},
        {"event", "start"},
        {"timestamp", current_time_str()}
    };
    backend->write_jsonl(meta);
}

static std::string sanitize_filename(const std::string& s) {
    std::string r = s;
    for (char& c : r) {
        switch (c) {
        case '/': case '\\': case ':':
        case '*': case '?': case '"':
        case '<': case '>': case '|':
            c = '-';
            break;
        default:
            break;
        }
    }
    return r;
}

void MetricsLogger::log_image_subtyped(const std::string& tag,
    int step,
    const wxImage& image,
    const std::string& subtype_or_empty)
{
    // 連番
    uint64_t seq = image_seq_[tag]++;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%06llu", static_cast<unsigned long long>(seq));

    // **タグ名のファイル用安全化**
    std::string safe_tag = sanitize_filename(tag);

    // 保存パス（ファイル名のみ safe_tag に置換）
    std::string rel_path = "images/" + safe_tag + "/" + safe_tag + "_" + buf + ".png";
    std::string full_dir = root_dir + "/" + run_name + "/images/" + safe_tag;
    std::string full_path = root_dir + "/" + run_name + "/" + rel_path;

    std::filesystem::create_directories(full_dir);
    image.SaveFile(full_path, wxBITMAP_TYPE_PNG);

    // JSONL には元の tag をそのまま残す（ログ解析との整合性保持）
    json obj = {
        {"run", run_name},
        {"type", "image"},
        {"tag", tag},                // ← ここは元の tag
        {"step", step},
        {"path", rel_path},          // ← ここは safe_tag を含むパス
        {"timestamp", current_time_str()}
    };
    if (!subtype_or_empty.empty()) obj["subtype"] = subtype_or_empty;

    backend->write_jsonl(obj);
}

