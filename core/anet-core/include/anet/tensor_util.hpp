#pragma once
#include <string>
#include <sstream>
#include <limits>
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


    // =========================
    // デバッグ用：テンソルを文字列化
    // =========================

    void printTensorAsNestedBrackets(const torch::Tensor& t, std::ostream& os);
    void printTensorAsRows(const torch::Tensor& t, std::ostream& os);
    std::string ToDefString(const torch::Tensor& t);
    std::string ToString(const torch::Tensor& t, int precision = 4);


    // =========================
    // ツール小物
    // =========================

    inline float ItemF(const at::Tensor& t)
    {
        //auto s = t.detach().to(torch::kCPU);
        //ANET_ASSERT_MSG(t.numel() == 1, "itemf expects scalar, got numel=" <<  t.numel() <<  " sizes=" << t.sizes());
        return t.item<float>();
    }

    template<typename T>
    T ItemOrNaN(const torch::Tensor& t)
    {
        return t.defined() ? t.item<T>() : std::numeric_limits<T>::quiet_NaN();
    };

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

        /// デフォルトコンストラクタ
        TensorDict() = default;


        /// 単一 Keyで初期化するコンストラクタ.
        /// 使い方： { anet::rl::ObsKeys::kVector, torch::tensor(...) }
        TensorDict(const std::string& key, const torch::Tensor& tensor)
        {
            Set(key, tensor);
        }

        /// 複数のKey-Valueをリストで一括初期化するコンストラクタ.
        /// 使い方： { 
        ///    { anet::rl::ObsKeys::kGrid, board_tensor },
        ///    { anet::rl::ObsKeys::kVector, dropper_tensor }
        /// }
        TensorDict(std::initializer_list<std::pair<std::string, torch::Tensor>> init)
        {
            for (const auto& kv : init) {
                Set(kv.first, kv.second);
            }
        }

        // ---------------------------------------------------------
        // C++標準コンテナメソッド
        // ---------------------------------------------------------

        auto begin() { return dict_.begin(); }
        auto end() { return dict_.end(); }
        auto begin() const { return dict_.begin(); }
        auto end() const { return dict_.end(); }
        bool empty() const { return dict_.empty(); }
        size_t size() const { return dict_.size(); }


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
            return dict_.at(key); // Keyが無い場合はstd::out_of_range が投げられる
        }

        // 存在チェック
        bool Contains(const std::string& key) const
        {
            return dict_.find(key) != dict_.end();
        }

        /// 空ではない、かつテンソルディクトが有効な状態（定義済み）かを判定
        bool Defined() const
        {
            // 要素が空の場合は、未初期化（undefined）とみなす
            if (dict_.empty()) return false;

            // 全ての要素が defined() かどうかチェック
            for (const auto& kv : dict_) {
                if (!kv.second.defined()) return false;
            }
            return true;
        }

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
        // TensorDictならではの便利機能
        // ---------------------------------------------------------

        /// 全テンソルを一括で指定デバイスに転送して新しいTensorDictを返す
        TensorDict To(torch::Device device, bool non_blocking = false) const
        {
            TensorDict result;
            for (const auto& kv : dict_) {
                result.Set(kv.first, kv.second.to(device, non_blocking));
            }
            return result;
        }

        /// 全テンソルを一括で型変換して新しいTensorDictを返す
        TensorDict To(torch::ScalarType dtype) const
        {
            TensorDict result;
            for (const auto& kv : dict_) {
                result.Set(kv.first, kv.second.to(dtype));
            }
            return result;
        }

        /// 内部テンソルの代表デバイスを取得する
        torch::Device device() const
        {
            if (dict_.empty()) return torch::Device(torch::kCPU); // 空なら安全側に倒してCPU
            return dict_.begin()->second.device();
        }

        /// 全てのテンソルをCPUに転送した新しいTensorDictを返す
        TensorDict Cpu() const
        {
            TensorDict result;
            for (const auto& kv : dict_) {
                result.Set(kv.first, kv.second.cpu());
            }
            return result;
        }

        /// 全てのテンソルをCUDA（GPU）に転送した新しいTensorDictを返す
        TensorDict Cuda() const
        {
            TensorDict result;
            for (const auto& kv : dict_) {
                result.Set(kv.first, kv.second.cuda());
            }
            return result;
        }

        /// 全テンソルを一括でDetachして新しいTensorDictを返す
        TensorDict Detach() const
        {
            TensorDict result;
            for (const auto& kv : dict_) {
                result.Set(kv.first, kv.second.detach());
            }
            return result;
        }

        /// 全テンソルを一括でCloneして新しいTensorDictを返す
        TensorDict Clone() const
        {
            TensorDict result;
            for (const auto& kv : dict_) {
                result.Set(kv.first, kv.second.clone());
            }
            return result;
        }

         /// 指定した次元のサイズを返す
        int64_t Size(int64_t dim) const
        {
            // 空だったら0
            if (dict_.empty()) return 0;

            // 最初の要素のサイズを代表として返す
            return dict_.begin()->second.size(dim);
        }

        /// 指定した次元（通常はバッチ次元）で分割し、単一要素のTensorDictのリストを返す
        std::vector<TensorDict> Unbind(int64_t dim = 0) const
        {
            if (dict_.empty()) return {};

            int64_t batch_size = Size(dim);
            std::vector<TensorDict> result(batch_size);

            for (const auto& kv : dict_) {
                const auto& key = kv.first;
                // 各テンソルを一括でunbind
                std::vector<torch::Tensor> tensors = kv.second.unbind(dim);

                // 結果の配列に分配
                for (int64_t i = 0; i < batch_size; ++i) {
                    result[i].Set(key, tensors[i]);
                }
            }
            return result;
        }

        /// 内部テンソル群の整合性チェック。不整合が無ければtrue。
        bool IsValid() const
        {
            if (dict_.empty()) return true;

            auto it = dict_.begin();
            const int64_t expected_batch_size = it->second.size(0);
            const torch::Device expected_device = it->second.device();

            for (; it != dict_.end(); ++it) {
                const auto& t = it->second;

                // バッチサイズ（dim 0）の一致確認
                if (t.size(0) != expected_batch_size) {
                    return false;
                }

                // デバイスの一致確認
                if (t.device() != expected_device) {
                    return false;
                }
            }
            return true;
        }

        /// バッチ要素のインプレース更新 (代入)
        void CopyBatchItem(int64_t batch_index, const TensorDict& single_dict)
        {
            for (auto& kv : dict_) {
                const auto& key = kv.first;
                // コピー元に同じキーが存在する場合のみコピーを実行
                if (single_dict.Contains(key)) {
                    // LibTorchでは tensor[index] が i番目のスライス（ビュー）を返すため、
                    // そのまま copy_ を呼ぶことで元のテンソルのメモリが直接書き換わる
                    kv.second[batch_index].copy_(single_dict.At(key));
                }
            }
        }


        // ---------------------------------------------------------
        // Operators
        // ---------------------------------------------------------

        // Keyに該当するTensorを返す。無かったら作る。
        torch::Tensor& operator[](const std::string& key)
        {
            return dict_[key];
        }

        // バッチの指定インデックスの要素を抽出し、バッチ次元を持たない新しいTensorDictを返す。
        TensorDict operator[](int64_t index) const
        {
            TensorDict res;
            for (const auto& kv : dict_) {
                res.Set(kv.first, kv.second[index]);
            }
            return res;
        }


        // ---------------------------------------------------------
        // utility
        // ---------------------------------------------------------

        /// 複数のTensorDictをKeyごとにスタックし、バッチ次元(デフォルトdim=0)を作る
        static TensorDict Stack(const std::vector<TensorDict>& dicts, int dim = 0)
        {
            if (dicts.empty()) return TensorDict();

            TensorDict result;
            // 最初のDictのKeyを基準に回す（すべてのDictが同じKeyを持つ前提）
            for (const auto& kv : dicts[0]) {
                const std::string& key = kv.first;

                std::vector<torch::Tensor> tensors;
                tensors.reserve(dicts.size());

                // 全Dictから同じKeyのテンソルをかき集める
                for (const auto& d : dicts) {
                    tensors.push_back(d.At(key));
                }

                // かき集めたテンソルをスタックして新しいDictにセット
                result.Set(key, torch::stack(tensors, dim));
            }
            return result;
        }

    private:
        // ソート順を保証して描画順序を安定させるため std::map を採用
        std::map<std::string, torch::Tensor> dict_;
    };

} // namespace anet::util
