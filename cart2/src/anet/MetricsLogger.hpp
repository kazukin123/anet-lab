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
#include <unordered_map>
#include <cstdio>
#include <wx/image.h>

#include "anet/HeatMap.hpp"  // anet::ImageSource を含む

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
// MetricsLogger 本体
//----------------------------------------------
class MetricsLogger {
private:
    std::unique_ptr<IBackend> backend;
    std::string root_dir;
    std::string run_name;

    // 画像連番（tag ごと）
    std::unordered_map<std::string, uint64_t> image_seq_;

    static json round_numbers(const json& j, int precision = 6);
    static std::string current_time_str();

    // subtype 付き内部実装（重い処理は cpp 側）
    void log_image_subtyped(const std::string& tag,
        int step,
        const wxImage& image,
        const std::string& subtype_or_empty);

public:
    explicit MetricsLogger(std::unique_ptr<IBackend> b,
        const std::string& root = "logs",
        const std::string& run = "");

    // 軽量メソッドはヘッダ内に実装
    inline void log_scalar(const std::string& tag, int step, double value) {
        json obj = {
            {"run", run_name},
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
            {"run", run_name},
            {"type", "json"},
            {"tag", tag},
            {"timestamp", current_time_str()},
            {"data", rounded}
        };
        backend->write_jsonl(obj);
    }

    // 汎用画像（subtype なし）
    inline void log_image(const std::string& tag, int step, const wxImage& image) {
        log_image_subtyped(tag, step, image, "");
    }

    // 可視化オブジェクト（subtype は anet::ImageSource 側が返す）
    inline void log_image(const std::string& tag, int step, const anet::ImageSource& src) {
        auto img = src.Render();
        auto subtype = src.GetImageSubType();
        log_image_subtyped(tag, step, img, subtype);
    }

    inline std::string get_run_name() const { return run_name; }

    inline std::string get_out_dir() const {
        return std::filesystem::relative(root_dir + "/" + run_name).string();
    }

    inline void flush() { backend->flush(); }
};
