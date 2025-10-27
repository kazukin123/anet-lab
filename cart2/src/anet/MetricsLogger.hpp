#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <iostream>
#include <memory>
#include <cmath>

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
    void open(const std::string& root_dir, const std::string& run_name) override {
        std::filesystem::create_directories(root_dir + "/" + run_name);
        auto path = root_dir + "/" + run_name + "/metrics.jsonl";
        ofs.open(path, std::ios::app);
        if (!ofs) throw std::runtime_error("Failed to open: " + path);
    }

    void write_jsonl(const json& obj) override {
        ofs << obj.dump() << "\n";
        ofs.flush();
    }

    void flush() override {
        ofs.flush();
	}
};

//----------------------------------------------
// MetricsLogger本体
//----------------------------------------------
class MetricsLogger {
private:
    std::unique_ptr<IBackend> backend;
    std::string root_dir;
    std::string run_name;
    std::string device_name;
    std::string torch_version;

    // float/doubleを丸める関数
    static json round_numbers(const json& j, int precision = 6) {
        if (j.is_number_float()) {
            double val = j.get<double>();
            double scale = std::pow(10.0, precision);
            return std::round(val * scale) / scale;
        }
        else if (j.is_object()) {
            json res;
            for (auto& [k, v] : j.items()) {
                res[k] = round_numbers(v, precision);
            }
            return res;
        }
        else if (j.is_array()) {
            json arr = json::array();
            for (auto& v : j) arr.push_back(round_numbers(v, precision));
            return arr;
        }
        return j;
    }

public:
    explicit MetricsLogger(std::unique_ptr<IBackend> b,
        const std::string& root = "logs",
        const std::string& run = "")
        : backend(std::move(b)), root_dir(root)
    {
        // 自動run名（タイムスタンプ付与）
        if (run.empty()) {
            auto t = std::chrono::system_clock::now();
            std::time_t tt = std::chrono::system_clock::to_time_t(t);
            std::tm tm{};
#ifdef _WIN32
            localtime_s(&tm, &tt);
#else
            localtime_r(&tt, &tm);
#endif
            char buf[64];
            std::strftime(buf, sizeof(buf), "run_%Y%m%d-%H%M%S", &tm);
            run_name = buf;
        }
        else {
            run_name = run;
        }
        backend->open(root_dir, run_name);

        // 起動メタ
        json meta = {
            {"type", "meta"},
            {"event", "start"},
            {"timestamp", current_time_str()},
            {"torch_version", "2.9.0"},
            {"device", "CPU"}
        };
        backend->write_jsonl(meta);
    }

    static std::string current_time_str() {
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

    void log_scalar(const std::string& tag, int step, double value) {
        json obj = {
            {"run", run_name},
            {"tag", tag},
            {"step", step},
            {"value", value},
            {"type", "scalar"}
        };
        backend->write_jsonl(obj);
    }

    void log_json(const std::string& tag, const json& data) {
        // 数値を丸めたJSONを出力
        json rounded = round_numbers(data);
        json obj = {
            {"run", run_name},
            {"tag", tag},
            {"timestamp", current_time_str()},
            {"type", "json"},
            {"data", rounded}
        };
        backend->write_jsonl(obj);
    }

    std::string get_run_name() const { return run_name; }

    std::string get_out_dir() const {
		return std::filesystem::relative(root_dir + "/" + run_name).string();
    }

    void flush() {
        backend->flush();
	}
};
