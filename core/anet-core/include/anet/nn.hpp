// anet/nn.hpp

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <torch/torch.h>
#include "anet/config.hpp"


namespace anet::nn {
	

    /// 予約タグ: Graphへの入力テンソルを指す
    static constexpr const char* kReservedTagInput = "input";

    /// 予約タグ: 直前ブロックの出力テンソルを指す
    static constexpr const char* kReservedTagPrev = "prev";

    /// モジュール生成時のコンテキスト情報（構造情報）
    struct ModuleContext {
        std::vector<std::string> input_tags;  ///< このモジュールへの入力として指定されたタグ一覧
        std::vector<int64_t> input_shape;     ///< 入力テンソルの形状
    };

    // ===========================================================================
    // Config Structures
    // ===========================================================================

    struct NetworkBlockConfig {
        std::string type;             // "Conv2d", "Linear", "ResBlock" etc.
        anet::ConfigData config_data; // Specific config data for the block. "kernel_size"=3 etc.
    };

    struct NetworkConfig {
        std::map<std::string, NetworkBlockConfig> block_configs;
        std::string structure_str;      // Flatten > Linear_128 > ReLU > Linear_256 > ReLU

        NetworkConfig() = default;
        NetworkConfig(const anet::ConfigData& config_data);
		anet::json ToJson() const;
    };

    struct WeightInitConfig {
        int mode = 1;             // 0:Default, 1:Xavier, 2:He, 3:Orthogonal
        float manual_gain = 0.0f; // Orthogonal時の手動ゲイン (0.0なら自動計算)
        std::string nonlinearity = "relu"; // "relu", "linear", "tanh" etc.
    };


    // ===========================================================================
    // Network Module Interface (Base Marker)
    // ===========================================================================

    /// @brief anet管理下のニューラルネットモジュール基底クラス
    class NetworkModule : public torch::nn::Module {
    public:
        /// @brief Forward実行
        ///        torch::nn::Module::forwardはテンプレートメソッドのため、
        ///        多態性を持たせるために純粋仮想関数として再定義する。
        virtual torch::Tensor Forward(torch::Tensor input) = 0;

        virtual ~NetworkModule() = default;
    };


    // ===========================================================================
    // NetworkHead
    // ===========================================================================

    class NetworkHead : public NetworkModule, public anet::TensorFunctionProvider {
    public:
        virtual ~NetworkHead() = default;
    };

    class NetworkHeadFactory {
    public:
		virtual std::shared_ptr<NetworkHead> CreateHead(int64_t feature_dim) const = 0;
        virtual ~NetworkHeadFactory() = default;
    };


    // ===========================================================================
	// Network (Body + Head)
    // ===========================================================================

    class NetworkBody;

    class Network : public torch::nn::Module, public anet::TensorFunctionProvider {
    public:
        Network(std::shared_ptr<NetworkBody> body, std::shared_ptr<NetworkHead> head);

        std::optional<TensorFunction> GetTensorFunction(const std::string& key) override;
        torch::Tensor Forward(torch::Tensor input); ///<  Input -> Body -> Feature -> Head -> Output
    private:
        std::shared_ptr<NetworkBody> body_;
        std::shared_ptr<NetworkHead> head_;
    };

    class NetworkBuilder {
    public:
        static std::shared_ptr<Network> BuildNetwork(
            const NetworkConfig& network_config,
            const std::vector<int64_t>& input_shape,   ///< InputのTensor形状 (例: {Channels, Height, Width} or {Channels, Length})
            std::shared_ptr<NetworkHeadFactory> head_factory
        );
    };

}
