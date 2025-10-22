#pragma once
#include <fstream>
#include <mutex>
#include <string>
#include <memory>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

namespace fs = std::filesystem;

//--------------------------------------------
// Backend Interface
//--------------------------------------------
class LogBackend {
public:
    virtual void open(const std::string& root_dir, const std::string& run_name) = 0;
    virtual void write(const nlohmann::json& j) = 0;
    virtual void flush() {}
    virtual ~LogBackend() = default;
};

//--------------------------------------------
// JSON Lines Backend
//--------------------------------------------
class JsonlBackend : public LogBackend {
    std::ofstream file;
    std::mutex mtx;
    std::string current_path;
public:
    void open(const std::string& root_dir, const std::string& run_name) override {
        fs::path run_dir = fs::path(root_dir) / run_name;
        if (!fs::exists(run_dir))
            fs::create_directories(run_dir);
        current_path = (run_dir / "metrics.jsonl").string();
        file.open(current_path, std::ios::out | std::ios::app);
    }

    void write(const nlohmann::json& j) override {
        std::lock_guard<std::mutex> lock(mtx);
        file << j.dump() << '\n';
    }

    void flush() override {
        std::lock_guard<std::mutex> lock(mtx);
        file.flush();
    }

    std::string get_current_path() const { return current_path; }
};

//--------------------------------------------
// Utility: timestamp & run name
//--------------------------------------------
inline std::string current_timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

inline std::string make_run_name() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << "run_" << std::put_time(&tm, "%Y%m%d-%H%M%S");
    return oss.str();
}

//--------------------------------------------
// Metrics Logger（run名生成→backend.open）
//--------------------------------------------
class MetricsLogger {
    std::unique_ptr<LogBackend> backend;
    std::string root_dir;
    std::string run_name;

public:
    MetricsLogger(std::unique_ptr<LogBackend> backend,
        const std::string& root_dir,
        const std::string& run = "")
        : backend(std::move(backend)), root_dir(root_dir)
    {
        run_name = run.empty() ? make_run_name() : run;
        this->backend->open(root_dir, run_name);  // ✅ ← ここでバックエンド初期化
        log_meta_start();
    }

    // メタ情報出力
    void log_meta_start() {
        std::string device_str = "CPU";
        //if (torch::cuda::is_available()) {
        //    int dev_idx = 0;
        //    auto prop = torch::cuda::getDeviceProperties(dev_idx);
        //    std::ostringstream devinfo;
        //    devinfo << "CUDA: GPU" << dev_idx << ": " << prop.name;
        //    device_str = devinfo.str();
        //}

        nlohmann::json meta = {
            {"type", "meta"},
            {"event", "start"},
            {"timestamp", current_timestamp()},
            {"torch_version", TORCH_VERSION},
            {"device", device_str}
        };
        backend->write(meta);
        backend->flush();
    }

    void log_scalar(const std::string& tag, int step, double value) {
        nlohmann::json j = {
            {"run", run_name},
            {"type", "scalar"},
            {"tag", tag},
            {"step", step},
            {"value", value}
        };
        backend->write(j);
    }

    void log_vector(const std::string& tag, int step, const std::vector<float>& vec) {
        nlohmann::json j = {
            {"run", run_name},
            {"type", "vector"},
            {"tag", tag},
            {"step", step},
            {"values", vec}
        };
        backend->write(j);
    }

    void log_tensor_stats(const std::string& tag, int step, const torch::Tensor& t) {
        nlohmann::json j = {
            {"run", run_name},
            {"type", "tensor"},
            {"tag", tag},
            {"step", step},
            {"shape", t.sizes()},
            {"mean", t.mean().item<double>()},
            {"std",  t.std().item<double>()}
        };
        backend->write(j);
    }

    void log_json(const nlohmann::json& custom) {
        nlohmann::json j = custom;
        j["run"] = run_name;
        backend->write(j);
    }

    void flush() { backend->flush(); }

    std::string get_run_name() const { return run_name; }
};
