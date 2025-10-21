#pragma once
#include <fstream>
#include <string>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <torch/torch.h>

#ifdef _WIN32
#include <windows.h>
#endif

// CUDAが使える場合のみGPU情報取得APIを有効化
#if defined(__CUDACC__) || defined(_MSC_VER)
#include <cuda_runtime.h>
#define USE_CUDA_API
#endif

// ======================================================
// 拡張版 tb_logger.hpp
// - 外部依存なし（nlohmann/json不要）
// - メタ情報出力（Torch版数・GPU名・学習時間）
// - CPU/非CUDA環境でも安全
// ======================================================
namespace simplelog {

    // ---------- ISO8601時刻 ----------
    inline std::string iso_timestamp() {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto t = system_clock::to_time_t(now);
        std::tm tm;
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream ss;
        ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
        return ss.str();
    }

    // ---------- GPU名を安全に取得 ----------
    inline std::string detect_device_name() {
        if (torch::cuda::is_available()) {
#ifdef USE_CUDA_API
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
                return "CUDA (device not found)";
            std::ostringstream oss;
            for (int i = 0; i < count; ++i) {
                cudaDeviceProp prop;
                if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                    if (i > 0) oss << ", ";
                    oss << "GPU" << i << ": " << prop.name;
                }
            }
            return "CUDA: " + oss.str();
#else
            return "CUDA (runtime headers unavailable)";
#endif
        }
        else {
            return "CPU";
        }
    }

    // ---------- JSON文字列エスケープ ----------
    inline std::string escape_json(const std::string& s) {
        std::ostringstream o;
        for (auto c : s) {
            switch (c) {
            case '\"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default: o << c; break;
            }
        }
        return o.str();
    }

    // ======================================================
    // クラス本体
    // ======================================================
    class JsonlLogger {
    public:
        explicit JsonlLogger(const std::string& path)
            : ofs_(path) {
            start_time_ = std::chrono::steady_clock::now();
            write_meta_start();
        }

        ~JsonlLogger() {
            write_meta_end();
            ofs_.flush();
        }

        // --- スカラー値（TensorBoardでグラフ化） ---
        void scalar(const std::string& tag, double value, int64_t step) {
            std::lock_guard<std::mutex> lk(mu_);
            ofs_ << "{\"type\":\"scalar\",\"tag\":\"" << tag
                << "\",\"step\":" << step
                << ",\"value\":" << value << "}\n";
        }

        // --- テキスト出力（任意情報） ---
        void text(const std::string& tag, const std::string& value, int64_t step) {
            std::lock_guard<std::mutex> lk(mu_);
            ofs_ << "{\"type\":\"text\",\"tag\":\"" << tag
                << "\",\"step\":" << step
                << ",\"value\":\"" << escape_json(value) << "\"}\n";
        }

        // --- 設定出力（バッチサイズ・エポックなど） ---
        void config(const std::string& key, const std::string& value) {
            std::lock_guard<std::mutex> lk(mu_);
            ofs_ << "{\"type\":\"config\",\"key\":\"" << key
                << "\",\"value\":\"" << escape_json(value) << "\"}\n";
        }

        // --- メタ情報: 開始時 ---
        void write_meta_start() {
            std::lock_guard<std::mutex> lk(mu_);
            std::string device_name = detect_device_name();
            ofs_ << "{\"type\":\"meta\",\"event\":\"start\""
                << ",\"timestamp\":\"" << iso_timestamp() << "\""
                << ",\"torch_version\":\""
                << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << "\""
                << ",\"device\":\"" << escape_json(device_name) << "\""
                << "}\n";
        }

        // --- メタ情報: 終了時 ---
        void write_meta_end() {
            std::lock_guard<std::mutex> lk(mu_);
            auto end_time = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(end_time - start_time_).count();
            ofs_ << "{\"type\":\"meta\",\"event\":\"end\""
                << ",\"timestamp\":\"" << iso_timestamp() << "\""
                << ",\"train_time_sec\":" << elapsed_sec
                << "}\n";
        }

        void flush() {
            ofs_.flush();
        }
    private:
        std::ofstream ofs_;
        std::mutex mu_;
        std::chrono::steady_clock::time_point start_time_;
    };



} // namespace simplelog
