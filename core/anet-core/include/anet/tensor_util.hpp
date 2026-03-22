#pragma once
#include <string>
#include <sstream>
#include <torch/torch.h>

namespace anet {

    torch::Device MakeDevice(int type, int index);

    // Device と dtype を一元管理する軽量コンテキスト
    struct TensorContext {
        torch::Device device;
        torch::Dtype float_dtype = torch::kFloat;
        torch::Dtype long_dtype = torch::kLong;
        torch::Dtype bool_dtype = torch::kBool;

        explicit TensorContext(torch::Device dev = torch::kCPU) : device(dev) {}

        inline torch::TensorOptions FloatOpt() const {
            return torch::TensorOptions().dtype(float_dtype).device(device);
        }
        inline torch::TensorOptions LongOpt() const {
            return torch::TensorOptions().dtype(long_dtype).device(device);
        }
        inline torch::TensorOptions BoolOpt() const {
            return torch::TensorOptions().dtype(bool_dtype).device(device);
        }
    };

    // =========================
    // 転送／基本ユーティリティ
    // =========================

    // 必要なときだけ .to(device) する安全版
    //inline torch::Tensor ToDevice(const torch::Tensor& t, const torch::Device& dev) {
    //    return (t.device() == dev) ? t : t.to(dev);
    //}

    // detach + to(device)
    inline torch::Tensor ToDeviceDetached(const torch::Tensor& t, const torch::Device& dev) {
        return t.detach().to(dev);
    }

    // スカラ取得（GPU→CPU同期を含む）
    template<typename T = float>
    inline T ToScalar(const torch::Tensor& t) {
        return t.cpu().item<T>();
    }

    inline float ToFloat(const torch::Tensor& t) {
        return t.cpu().item<float>();
    }

    inline bool ToBool(const torch::Tensor& t) {
        return t.cpu().item<bool>();
    }

    // =========================
    // テンソル生成ヘルパ
    // =========================

    inline torch::Tensor FullLike(const torch::Tensor& ref, float val, const TensorContext& ctx) {
        return torch::full(ref.sizes(), val, ctx.FloatOpt());
    }
    inline torch::Tensor OnesLike(const torch::Tensor& ref, const TensorContext& ctx) {
        return torch::ones(ref.sizes(), ctx.FloatOpt());
    }
    inline torch::Tensor ZerosLike(const torch::Tensor& ref, const TensorContext& ctx) {
        return torch::zeros(ref.sizes(), ctx.FloatOpt());
    }
    inline torch::Tensor BoolFullLike(const torch::Tensor& ref, bool val, const TensorContext& ctx) {
        return torch::full(ref.sizes(), val, ctx.BoolOpt());
    }

    // デバッグ用：テンソルを文字列化
    void printTensorAsNestedBrackets(const torch::Tensor& t, std::ostream& os);
    void printTensorAsRows(const torch::Tensor& t, std::ostream& os);
    std::string ToDefString(const torch::Tensor& t);
    std::string ToString(const torch::Tensor& t, int precision = 4);

    inline float itemf(const at::Tensor& t) {
        auto s = t.detach().to(torch::kCPU);
        TORCH_CHECK(s.numel() == 1, "itemf expects scalar, got numel=", s.numel(), " shape=", s.sizes());
        return s.item<float>();
    }

    //inline std::string ToString(const anet::rl::AuxData& aux_data)
    //{
    //    std::ostringstream oss;
    //    for (auto kv : aux_data) {
    //        oss << kv.first << " : " << anet::ToString(kv.second) << "\n";
    //    }
    //    return oss.str();
    //}


    // =========================
    // TensorDict
    // =========================

    class TensorDict {
    public:

        // ---------------------------------------------------------
        // コンストラクタ
        // ---------------------------------------------------------

        TensorDict() = default;


        // ---------------------------------------------------------
        // 基本機能
        // ---------------------------------------------------------

        // テンソルの追加・上書き
        void Set(const std::string& key, const torch::Tensor& tensor)
        {
            dict_[key] = tensor;
        }

        // テンソルの取得（存在しない場合は nullopt）
        std::optional<torch::Tensor> Get(const std::string& key) const
        {
            auto it = dict_.find(key);
            if (it != dict_.end()) {
                return it->second;
            }
            return std::nullopt;
        }

        // テンソルの取得（存在しない場合は例外スロー。確実に存在する場合用）
        torch::Tensor At(const std::string& key) const
        {
            return dict_.at(key); // std::out_of_range を投げる
        }

        // 存在チェック
        bool Contains(const std::string& key) const
        {
            return dict_.find(key) != dict_.end();
        }

        bool Empty() const { return dict_.empty(); }
        size_t Size() const { return dict_.size(); }

        std::string ToString() const
        {
            std::stringstream ss;
            ss << "------------\n";
            for (const auto& kv : dict_) {
                ss << kv.first << ": ";
                ss << anet::ToString(kv.second) << "\n";
            }
            ss << "------------\n";
            return ss.str();
        }

        std::string ToDefString() const
        {
            std::stringstream ss;
            ss << "------------\n";
            for (const auto& kv : dict_) {
                ss << kv.first << ": ";
                ss << anet::ToDefString(kv.second) << "\n";
            }
            ss << "------------\n";
            return ss.str();
        }

        // ---------------------------------------------------------
        // イテレータ (Range-based for用)
        // ---------------------------------------------------------

        auto begin() { return dict_.begin(); }
        auto end() { return dict_.end(); }
        auto begin() const { return dict_.begin(); }
        auto end() const { return dict_.end(); }


        // ---------------------------------------------------------
        // TensorDictならではの便利機能
        // ---------------------------------------------------------

        // 全テンソルを一括で指定デバイスに転送して新しいTensorDictを返す
        TensorDict To(torch::Device device, bool non_blocking = false) const
        {
            TensorDict res;
            for (const auto& kv : dict_) {
                res.Set(kv.first, kv.second.to(device, non_blocking));
            }
            return res;
        }

        // 全テンソルを一括でDetachして新しいTensorDictを返す
        TensorDict Detach() const
        {
            TensorDict res;
            for (const auto& kv : dict_) {
                res.Set(kv.first, kv.second.detach());
            }
            return res;
        }

        // ---------------------------------------------------------
        // Operator
        // ---------------------------------------------------------

        torch::Tensor& operator[](const std::string& key)
        {
            return dict_[key];
        }

    private:
        // ソート順を保証して描画順序を安定させるため std::map を採用
        std::map<std::string, torch::Tensor> dict_;
    };

} // namespace anet::util
