// anet/common.hpp

#pragma once

#include <memory>
#include <optional>
#include <type_traits>
#include <functional>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <torch/torch.h>
#include "anet/config.hpp"
#include "anet/diag.hpp"
#include "anet/json_util.hpp"
#include "anet/tensor_util.hpp"

namespace anet {

    // ===========================================================================
	// Tensor Function Providers
    // ===========================================================================

    using TensorDictFunction = std::function<anet::TensorDict(const anet::TensorDict&)>;
    using TraceCallback = std::function<void(std::string_view, const torch::Tensor&)>;

    class TensorDictFunctionProvider {
    public:
        virtual std::optional<TensorDictFunction> GetTensorDictFunction(const std::string& key) { return std::nullopt; }
        virtual ~TensorDictFunctionProvider() = default;
    };


    // ===========================================================================
    // Module
    // ===========================================================================

    //class Module : public std::enable_shared_from_this<Module> {
    //public:
    //    virtual std::string GetClassName() = 0;
    //    virtual std::string GetInstanceName() = 0;
    //    //virtual OrderedMap<std::string, Module>  GetChildlen() = 0;
    //public:
    //    virtual anet::OrderedMap<std::string, std::string> GetParameters() = 0;
    //    virtual anet::OrderedMap<std::string, std::vector<torch::Tensor>> GetTensorVector() = 0;

    //    virtual std::optional<std::string> GetParam(const std::string& key) = 0;
    //    virtual std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string key) = 0;
    //public:
    //    virtual void Load(std::istream& stream) const = 0;
    //    virtual void Save(std::ostream& stream) const = 0;
    //    virtual void Print(std::ostream& stream) const = 0;
    //    virtual anet::json ToJson(bool recursive = true) = 0;
    //public:
    //    virtual ~Module() = default;
    //};

    class Module {
    public:
        virtual std::optional<ConfigData> GetConfigData() const { return std::nullopt; }
        /// 指定 key が未知の場合だけ std::nullopt を返す。
        /// key が既知だが現在値を出力できない場合は quiet_NaN() を返す。
        virtual std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const = 0;
        virtual std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const = 0;
        virtual std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const = 0;
        virtual ~Module() = default;
    };

    class ModuleBase : virtual public Module {
    public:
        virtual std::optional<float> GetScalar(const std::string& key, int64_t = -1) const override
            { return std::nullopt; }
        virtual std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t = -1) const override
            { return std::nullopt; }
        virtual std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t = -1) const override
            { return std::nullopt; }
        virtual ~ModuleBase() = default;
    };


    // ===========================================================================
    // TensorSpec
    // ===========================================================================

    // 空間トポロジー
    enum class SpaceType {
        Vector,     // 順序や空間関係を持たない1D配列 (通常はLinearで受ける)
        Grid,       // 空間的な隣接関係を持つ配列 (通常はConvで受ける)
        Sequence    // 時間的な順序を持つ系列 (通常はTransformer/RNNで受ける)
    };

    /// Tensorの仕様情報
    struct TensorSpec {
        /// 空間トポロジー
        SpaceType type = SpaceType::Vector;

		/// Tensor形状(バッチ次元を除く). 基本、Vectorなら[dim]、Gridなら[channel, height, width]、Sequenceなら[channel, length]
        std::vector<std::int64_t> shape;

        /// データ・タイプ
        torch::Dtype dtype = torch::kFloat32;

        /// 離散値向けのクラス数。連続値(Continuous)なら0、離散値(Discrete)なら1以上。
        int64_t num_classes = 0;

        // Viewer表示用ラベル（Flatten要素数と同サイズ、または空）
        std::vector<std::string> labels;

        /// 最小値 (サイズ1なら全体適用(Broadcast)、要素数と同じなら要素別)
        std::vector<double> min_values;

        /// 最大値 (サイズ1なら全体適用(Broadcast)、要素数と同じなら要素別)
        std::vector<double> max_values;

        // --- ユーティリティメソッド ---

        bool IsDiscrete() const { return num_classes > 0; }

        bool HasValidLabels() const
        {
            if (labels.empty()) return true; // 省略(空)は常にOKとする

            if (type == SpaceType::Vector) {
                // Vectorなら、フラット化した全体の要素数と一致しているか
                return labels.size() == CalcFlattenDim();
            } else if (type == SpaceType::Grid || type == SpaceType::Sequence) {
                // Grid/Sequenceなら、チャネル次元(shape[0])の数と一致しているか
                return !shape.empty() && labels.size() == shape[0];
            }
            return false;
        }

        std::int64_t CalcFlattenDim() const
        {
            std::int64_t dim = 1;
            for (auto s : shape) dim *= s;
            return dim;
        }

        std::optional<double> GetMin(size_t index = 0) const
        {
            if (min_values.empty()) return std::nullopt;
            if (min_values.size() == 1) return min_values[0];
            if (index < min_values.size()) return min_values[index];
            return std::nullopt;
        }

        std::optional<double> GetMax(size_t index = 0) const {
            if (max_values.empty()) return std::nullopt;
            if (max_values.size() == 1) return max_values[0];
            if (index < max_values.size()) return max_values[index];
            return std::nullopt;
        }

        anet::json ToJson() const;
        std::string ToString() const;
    };

    using TensorSpecMap = std::unordered_map<std::string, anet::TensorSpec>;


    // ToString() を持つかどうか判定するメタ関数
    //template<typename T>
    //using has_ToString = decltype(std::declval<const T&>().ToString());

    // ToString()があれば何でもOKな operator<<
    //template<typename T,typename = has_ToString<T>>
    //std::ostream& operator<<(std::ostream& os, const T& v)
    //{
    //    return (os << v.ToString());
    //}

}   // namespace anet
