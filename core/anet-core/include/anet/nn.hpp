// anet/nn.hpp

#pragma once

#include <memory>
#include <string>
#include <optional>
#include <vector>
#include <functional>
#include <map>
#include <utility>
#include <torch/torch.h>
#include "anet/config.hpp"
#include "anet/common.hpp"
#include "anet/graphviz.hpp"


namespace anet::nn {


    // ===========================================================================
    // Config Structures
    // ===========================================================================

    struct NetworkBlockConfig {
        std::string type;             // "Conv2d", "Linear", "ResBlock" etc.
        anet::ConfigData config_data; // Specific config data for the block. "kernel_size"=3 etc.
    };

    struct NetworkBranchConfig {
        std::string name;                      ///< ブランチ名（＝出力キー）
        std::vector<std::string> bind_keys;    ///< 入力として要求するキーのリスト
        std::vector<std::string> raw_keys;     ///< bind_keys のうち、"(raw)" が指定されたキーのリスト
        bool auto_format = true;               ///< ブランチ全体での自動フォーマット有効/無効
        std::string structure_str;             ///< ブランチ内部の直列パイプライン構造
    };

    struct ConfigProfileConfig {
        std::string type = "linear";   ///< 補間ルール（初期実装は linear のみ）
        double start = 0.0;            ///< i=0 の値
        double end = 0.0;              ///< i=N-1 の値
        bool has_end = false;          ///< end が明示指定されたか
    };

    struct NetworkConfig {
        std::map<std::string, NetworkBlockConfig> block_configs;
        std::map<std::string, NetworkBranchConfig> branches;
        std::map<std::string, std::string> output_keys;
        std::map<std::string, ConfigProfileConfig> config_profiles;

        NetworkConfig() = default;
        NetworkConfig(const anet::ConfigData& config_data, const std::string& config_prefix = "net");
		anet::json ToJson() const;
    };

    struct WeightInitConfig {
        std::string mode = "xavier";        ///< "default", "xavier", "he", "orthogonal", "constant"
        double manual_gain = 0.0f;          ///< Orthogonal時の手動ゲイン (0.0なら自動計算)
        std::string nonlinearity = "relu";  ///< "relu", "linear", "tanh" etc.
        double constant_val = 0.0;          ///< for Constant
    };

    struct NetworkGraphVizConfig {
        bool show_param_shapes = false;
        bool show_param_count = false;
        bool show_tensor_specs = false;
        bool show_branch_config = false;
        bool show_head_info = false;
        std::string layout = "LR";
        bool cluster_branches = true;
        int float_precision = 3;
    };

    void ValidateNetworkGraphVizConfig(const NetworkGraphVizConfig& config, const std::string& owner);


    // ===========================================================================
    // NetworkHead
    // ===========================================================================

    struct HeadGraphVizInfo {
        struct OutputInfo {
            std::string name;
            std::vector<int64_t> shape;
        };

        std::string type;
        std::vector<OutputInfo> outputs;
        std::vector<std::pair<std::string, std::string>> details;
    };

    class NetworkHead : public torch::nn::Module, public anet::TensorDictFunctionProvider {
    public:
        virtual anet::TensorDict Forward(const anet::TensorDict& feature_dict) = 0;
        virtual HeadGraphVizInfo GetGraphVizInfo() const { return {}; }
        virtual ~NetworkHead() = default;
    };

    class NetworkHeadFactory {
    public:
        virtual std::shared_ptr<NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const = 0;
        virtual ~NetworkHeadFactory() = default;
    };


    // ===========================================================================
	// Network (Body + Head)
    // ===========================================================================

    class NetworkBody;

    class Network : public torch::nn::Module, public anet::TensorDictFunctionProvider {
    public:
        Network(
            const NetworkConfig& config,
            const anet::TensorSpecMap& input_specs,
            std::shared_ptr<NetworkHeadFactory> head_factory,
            std::shared_ptr<NetworkBody> body,
            std::shared_ptr<NetworkHead> head);

        anet::TensorDict Forward(const anet::TensorDict& input, const anet::TraceSink& sink = {});

        std::shared_ptr<Network> Clone(std::optional<torch::Device> device = std::nullopt) const;                 /// 自身の完全な複製(別インスタンス)を生成
        void CopyTo(Network& target) const;                     /// ターゲットへ重みを完全上書き (Hard Update)
        void SoftCopyTo(Network& target, double tau) const;     /// ターゲットへ重みをブレンド (Soft Update)

    public: //可視化関連
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override;
        std::unique_ptr<anet::graphviz::GraphViz> MakeGraphViz(const NetworkGraphVizConfig& config) const;
    private:
        // 実行用の実体
        std::shared_ptr<NetworkBody> body_;
        std::shared_ptr<NetworkHead> head_;

        // 構築情報(Clone用)
        NetworkConfig config_;
        anet::TensorSpecMap input_specs_;
        std::shared_ptr<NetworkHeadFactory> head_factory_;
    };


    // ===========================================================================
    // NetworkBuilder
    // ===========================================================================

    class NetworkBuilder {
    public:
        static std::shared_ptr<Network> BuildNetwork(
            const NetworkConfig& network_config,
            const anet::TensorSpecMap& input_specs,
            std::shared_ptr<NetworkHeadFactory> head_factory,
            std::optional<torch::Device> device = std::nullopt);
    };

} // namespace anet::nn
