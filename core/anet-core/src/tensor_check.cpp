#include "anet/tensor_check.hpp"

//----------------------------------------------
// Device チェック実体
//----------------------------------------------
void _anet_check_device_impl(const torch::Tensor& t, const torch::Device& expect, const char* msg, const char* file, int line)
{
    // タイプ（CPUかCUDAか）のチェック
    if (expect.type() != t.device().type()) {
        std::stringstream ss;
        ss << "Device type mismatch: " << msg
            << " | tensor=" << t.device()
            << " | expected=" << expect
            << " | File: " << file << ":" << line;
        throw std::runtime_error(ss.str());
    }

    // CUDAの場合、要求側に明示的なIndexが指定されていればIndexもチェック
    if (expect.has_index() && expect.index() >= 0 && t.device().index() != expect.index()) {
        std::stringstream ss;
        ss << "Device index mismatch: " << msg
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

    if (expects.size() == 0) {
        if (t.sizes().size() == 0) return;
    } else {
        for (const auto& e : expects) {
            bool ok = true;
            bool has_endany = false;

            for (size_t i = 0; i < e.size(); ++i) {
                if (e[i] == ANET_SHAPE_ENDANY) {
                    has_endany = true;
                    break; // ENDANY以降はチェック不要
                }
                if (i >= actual.size()) {
                    ok = false; // actualの方が短くてENDANYも無い
                    break;
                }
                if (e[i] == ANET_SHAPE_ANY) continue;
                if (actual[i] != e[i]) {
                    ok = false; // 次元サイズ不一致
                    break;
                }
            }

            // すべての次元が一致し、かつ「余分な未チェック次元」が存在しないか確認
            if (ok && (has_endany || actual.size() == e.size())) {
                return; // 完全一致！
            }
        }
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

void _anet_check_nan_impl(const torch::Tensor& t, const char* msg, const char* file, int line)
{
    if (torch::isnan(t).any().item<bool>() || torch::isinf(t).any().item<bool>()) {
        std::stringstream ss;
        ss << "NaN detected: " <<  msg
            << " | tensor=" << t
            << " | File: " << file << ":" << line;

        throw std::runtime_error(ss.str());
    }
}
