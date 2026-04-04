

#include "anet/tensor_util.hpp"
#include <functional>
#include <iostream>
#include "anet/common.hpp"

namespace anet {

    torch::Device MakeDevice(int type, int index)
    {
        if (type == 0) return torch::Device(torch::kCPU);
        if (type == 1) return torch::Device(torch::kCUDA, index);
        ANET_ASSERT_MSG(false, "Invalid device type");
        return torch::Device(torch::kCPU);
    }

    std::string ToDefString(const torch::Tensor& t)
    {
        std::ostringstream oss;
        oss << "(" << t.device() << " " << t.dtype() << ") " << t.sizes();
        return oss.str();
    }

    std::string ToString(const torch::Tensor& t, int precision)
    {
        std::ostringstream oss;
        if (t.defined()) {
            oss.precision(precision);
            oss << "(" << t.device() << " " << t.dtype() << ") " << t.sizes() << " -> ";
            printTensorAsNestedBrackets(t, oss); //oss << "\n";
            //printTensorAsRows(t, oss);
        } else {
            oss << "undefined";
        }
        return oss.str();
    }

    void printTensorAsRows(const torch::Tensor& t, std::ostream& os)
    {
        auto sizes = t.sizes();
        int dim = sizes.size();

        std::vector<int64_t> idx(dim, 0);

        std::function<void(int)> rec = [&](int d) {
            if (d == dim) {
                // TensorIndex に変換
                std::vector<at::indexing::TensorIndex> tidx;
                tidx.reserve(dim);
                for (auto v : idx) tidx.push_back(v);

                float v = t.index(tidx).item<float>();

                os << "\n[";
                for (int i = 0; i < dim; i++) {
                    os << idx[i];
                    if (i + 1 < dim) os << ",";
                }
                os << "] = " << v;
                return;
            }

            for (int64_t i = 0; i < sizes[d]; i++) {
                idx[d] = i;
                rec(d + 1);
            }
            };

        rec(0);
    }

    float tensorValueAt(const torch::Tensor& t, const std::vector<int64_t>& idx)
    {
        std::vector<at::indexing::TensorIndex> tidx;
        tidx.reserve(idx.size());
        for (auto v : idx) {
            tidx.push_back(v);  // int64_t → TensorIndex に変換
        }
        return t.index(tidx).item<float>();
    }

    void printTensorAsNestedBrackets(const torch::Tensor& t, std::ostream& os)
    {
        auto sizes = t.sizes();
        int dim = sizes.size();

        if (dim == 0) {
            // スカラー
            os << "[] = " << t.item<float>();
            return;
        }

        std::vector<int64_t> idx(dim, 0);

        std::function<void(int)> rec = [&](int d) {
            // 最後の次元（行として展開する次元）
            if (d == dim - 1) {
                // 行ラベル
                os << "[";
                if (dim == 1) {
                    os << ":";   // 1D のときは [:] みたいにしておく
                }
                else {
                    for (int i = 0; i < dim - 1; i++) {
                        os << idx[i];
                        if (i + 1 < dim - 1) os << ",";
                    }
                }
                os << "] ";

                // 最後の次元を横に並べる
                for (int64_t j = 0; j < sizes[dim - 1]; j++) {
                    idx[dim - 1] = j;
                    float v = tensorValueAt(t, idx);
                    os << v;
                    if (j + 1 < sizes[dim - 1]) os << " ";
                }
                os;
                return;
            }

            // 再帰的に前の次元を回す
            for (int64_t i = 0; i < sizes[d]; i++) {
                idx[d] = i;
                rec(d + 1);
                // ブロックごとに空行を入れたいならここで入れる
                // if (dim >= 3 && d == dim - 2 && i + 1 < sizes[d]) os << "\n";
            }
            };

        rec(0);
    }

}

torch::Tensor anet::GetOrFail(const anet::TensorDict& dict, const std::string& key, const std::string& msg)
{
    auto opt = dict.Get(key);
    if (!opt.has_value()) {
        ANET_SYSTEM_ERROR("expected key '" << key << "' in TensorDict, but it was not found. " << msg);
        return torch::Tensor(); // 到達不可
    }
    return *opt;
}
