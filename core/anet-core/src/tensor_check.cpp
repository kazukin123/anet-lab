#include "anet/tensor_check.hpp"

//----------------------------------------------
// Device チェック実体
//----------------------------------------------
void _anet_check_device_impl(
    const torch::Tensor& t, const torch::Device& expect,
    const char* msg, const char* file, int line)
{
    const bool expect_cpu = expect.is_cpu();
    const bool actual_cpu = t.device().is_cpu();
    if (expect_cpu != actual_cpu) {
        std::stringstream ss;
        ss << "Device mismatch: " << msg
            << " | tensor=" << t.device()
            << " | expected=" << expect
            << " | File: " << file << ":" << line;
        throw std::runtime_error(ss.str());
    }
}

//----------------------------------------------
// Shape（単一）チェック実体
//----------------------------------------------
void _anet_check_shape_impl(const torch::Tensor& t,
    const std::vector<int64_t>& expect,
    const char* msg,
    const char* file,
    int line)
{
    auto actual = t.sizes().vec();
    if (actual != expect) {
        std::stringstream ss;
        ss << "Shape mismatch: " << msg
            << " | tensor=" << t.sizes()
            << " | expected=[";
        for (auto v : expect) ss << v << " ";
        ss << "]"
            << " | File: " << file << ":" << line;
        throw std::runtime_error(ss.str());
    }
}

//----------------------------------------------
// Shape-OR（複数 shape 許可）チェック実体
//----------------------------------------------
void _anet_check_shape_or_impl(const torch::Tensor& t,
    const std::vector<std::vector<int64_t>>& expects,
    const char* msg,
    const char* file,
    int line)
{
    auto actual = t.sizes().vec();

    for (const auto& e : expects) {
        if (actual.size() != e.size()) continue;

        bool ok = true;

        for (size_t i = 0; i < e.size(); ++i) {
            if (e[i] == ANET_SHAPE_ENDANY) break;  // ここから先は全て許容
            if (e[i] == ANET_SHAPE_ANY) continue;  // 任意次元
            if (actual[i] != e[i]) {
                ok = false;
                break;
            }
        }

        if (ok) return;  // いずれかにマッチ
    }

    std::stringstream ss;
    ss << "Shape mismatch: " << msg
        << " | actual=" << t.sizes()
        << " | allowed=[";
    for (const auto& e : expects) {
        ss << "[";
        for (auto v : e) ss << v << " ";
        ss << "] ";
    }
    ss << "]"
        << " | File: " << file << ":" << line;

    throw std::runtime_error(ss.str());
}
