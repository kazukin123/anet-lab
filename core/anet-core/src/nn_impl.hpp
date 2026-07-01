// nn_impl.hpp

#pragma once

#include "anet/nn.hpp" 

namespace anet::nn {


    // ===========================================================================
    //  WeightInitializer
    // ===========================================================================

    torch::nn::init::NonlinearityType GetNonlinearityType(const std::string& name);
    torch::Tensor SdpaSelfAttention(const torch::nn::MultiheadAttention& mha, const torch::Tensor& x);
    torch::Tensor DropPath(const torch::Tensor& x, double drop_prob, bool training);

    class WeightInitializer {
    public:
        template <typename T>
        static void Initialize(T& layer, const WeightInitConfig& config)
        {
            if (config.mode == "default") return;

            auto& weight = layer->weight;

            // 初期化中は勾配計算不要
            torch::NoGradGuard no_grad;

			// mode別に初期化実行
            if (config.mode == "xavier") {
                // Xavierは通常 gain=1.0 前提だが、calculate_gainを使う手もある
                torch::nn::init::xavier_uniform_(weight);
            } else if (config.mode == "he") {
                // 文字列からLibtorchの定数へ変換
                auto nonlinearity_mode = GetNonlinearityType(config.nonlinearity);

                // 重み初期化(kaiming_normal_ は内部で nonlinearity に応じた gain を計算してくれる)
                torch::nn::init::kaiming_normal_(weight, 0.0, torch::kFanOut, nonlinearity_mode);
            } else if (config.mode == "orthogonal") {
                double gain = 1.0;

                if (config.manual_gain > 0.0f) {
                    gain = config.manual_gain;
                } else {
                    // "relu" -> sqrt(2), "tanh" -> 5/3, "linear" -> 1.0 など
                    try {
                        auto nonlinearity_mode = GetNonlinearityType(config.nonlinearity);
                        gain = torch::nn::init::calculate_gain(nonlinearity_mode);
                    } catch (...) {
                        ANET_SYSTEM_ERROR("Unknown nonlinearity: " << config.nonlinearity);
                    }
                }

				// Orthogonal初期化
                torch::nn::init::orthogonal_(weight, gain);
            } else if (config.mode == "constant") {
                torch::nn::init::constant_(weight, config.constant_val);
            } else {
                ANET_SYSTEM_ERROR("Unknown WeightInitConfig.mode: \"" << config.mode
                    << "\" expected one of: default, xavier, he, orthogonal, constant");
            }

            // バイアスはゼロ初期化
            auto& bias = layer->bias;
            if (bias.defined() && config.mode != "default") {
                if (config.mode == "constant") {
                    // Constantモードならバイアスも同じ値で埋める (ZeroInit用など)
                    torch::nn::init::constant_(bias, config.constant_val);
                } else {
                    // 通常はゼロ初期化
                    torch::nn::init::constant_(bias, 0.0);
                }
            }
        }
    };


    // ===========================================================================
    // Network Internal Components
    // ===========================================================================

    /// anet管理下のニューラルネットモジュール基底クラス
    class NetworkModule : public torch::nn::Module {
    public:
        /// Forward実行.
        /// torch::nn::Module::forwardはテンプレートメソッドのため、多態性を持たせるために純粋仮想関数として再定義
        virtual torch::Tensor Forward(torch::Tensor input) = 0;
        virtual bool IsConv2dVisualizable() const { return false; }
        virtual anet::ConfigData GetCurrentConfigData() const { return {}; }
        virtual ~NetworkModule() = default;
    };

    /// 単一のモジュールをラップし、入出力タグ情報を保持するブロック
    class NetworkBlock : public torch::nn::Module {
    public:
        NetworkBlock(std::string name, std::shared_ptr<NetworkModule> module);

        torch::Tensor Forward(torch::Tensor input);

        const std::string& GetName() const { return name_; }
        std::shared_ptr<NetworkModule> GetModule() const { return module_; }
        bool IsConv2dVisualizable() const { return module_ ? module_->IsConv2dVisualizable() : false; }
    private:
        std::string name_;
        std::shared_ptr<NetworkModule> module_;
    };

    /// 複数のBlockを順次実行し、タグによるデータフローを解決する実行エンジン
    class NetworkStruct : public torch::nn::Module {
    public:
        explicit NetworkStruct(std::vector<std::shared_ptr<NetworkBlock>> blocks = {});

        torch::Tensor Forward(torch::Tensor input, const anet::TraceSink& sink = {});
        const std::vector<std::shared_ptr<NetworkBlock>>& GetBlocks() const { return blocks_; }
    private:
        std::vector<std::shared_ptr<NetworkBlock>> blocks_;
    };

    /// DAGを構成する一本のノード.
    /// 入力を結合し、内部の直列パイプラインに流す。
    class NetworkBranch : public torch::nn::Module {
    public:
        NetworkBranch(
            std::string name, std::vector<std::string> bind_keys, std::shared_ptr<NetworkStruct> network_struct);

        /// 現在のTensorDictから必要な入力(bind)を拾って結合し、処理を実行して結果をDictに書き戻す
        void Execute(anet::TensorDict& current_state, const anet::TraceSink& sink = {});

        const std::string& GetName() const { return name_; }
        const std::vector<std::string>& GetBindKeys() const { return bind_keys_; }
        std::shared_ptr<NetworkStruct> GetNetworkStruct() const { return network_struct_; }

    private:
        std::string name_;
        std::vector<std::string> bind_keys_;
        std::shared_ptr<NetworkStruct> network_struct_;
    };

    class NetworkStructBuilder {
    public:
        /// @brief Config全体と、パース対象の構造文字列を受け取り、NetworkStructを構築する
        static std::shared_ptr<NetworkStruct> Build(
            const NetworkConfig& root_config, const std::string& structure_str);
    };


    // ===========================================================================
	// NetworkBody related classes
    // ===========================================================================

    /// 環境からの生TensorDictを、NN入力用に最初に受け取ってチェック・変換する関所（NetworkBody内部クラス）
    class NetworkBoundaryPreprocessor {
    public:
        NetworkBoundaryPreprocessor(
            const anet::TensorSpecMap& specs, const std::vector<std::string>& raw_keys);

        /// フォーマット済みのTensorDictを生成して返す
        anet::TensorDict Format(const anet::TensorDict& raw_input) const;
    private:
        anet::TensorSpecMap specs_;
        std::set<std::string> raw_keys_; // 検索速度のため set に保持
    };

    class NetworkBody : public torch::nn::Module {
    public:
        NetworkBody(
            std::vector<std::shared_ptr<NetworkBranch>> branches,
            const anet::TensorSpecMap& specs,
            const std::vector<std::string>& raw_keys,
            std::map<std::string, std::string> output_keys);

        // DAG全体のForward実行
        anet::TensorDict Forward(const anet::TensorDict& input, const anet::TraceSink& sink = {});
        const std::vector<std::shared_ptr<NetworkBranch>>& GetBranches() const { return branches_; }
        const std::map<std::string, std::string>& GetOutputKeys() const { return output_keys_; }

    private:
        std::vector<std::shared_ptr<NetworkBranch>> branches_;
        NetworkBoundaryPreprocessor preprocessor_;
        std::map<std::string, std::string> output_keys_;
    };

    class NetworkBodyBuilder {
    public:
        /// 設定とSpecからDAG全体を解析・トポロジカルソートし、NetworkBodyを構築する
        static std::shared_ptr<NetworkBody> Build(
            const NetworkConfig& config, const anet::TensorSpecMap& input_specs);
    };


    // ===========================================================================
    // NetworkModule Factories & Repository
    // ===========================================================================

    /// モジュール生成時のコンテキスト情報（構造情報）
    struct ModuleContext {
    	// メンバ無し（今後の拡張用）
    };

    class NetworkModuleFactory {
    public:
        /// @brief Context (タグ情報等) を受け取り、適切なモジュールを生成する
        virtual std::shared_ptr<NetworkModule> CreateModule(
            const ConfigData& config_data, const ModuleContext& context) const = 0;
        virtual ~NetworkModuleFactory() = default;
    };

    class NetworkModuleRepository {
    public:
        static NetworkModuleRepository& Instance();
    public:
        void Register(const std::string& type_name, std::shared_ptr<NetworkModuleFactory> factory);
        std::shared_ptr<NetworkModuleFactory> GetFactory(const std::string& type_name) const;
    private:
        NetworkModuleRepository() = default;
    private:
        mutable std::mutex mtx_;
        std::map<std::string, std::shared_ptr<NetworkModuleFactory>> registry_;
    };

    template<typename T, class... Args>
    inline void RegisterNetworkModuleFactory(const std::string& type_name, Args&&... args)
    {
        auto factory = std::make_shared<T>(std::forward<Args>(args)...);
        NetworkModuleRepository::Instance().Register(type_name, factory);
    }


    // ===========================================================================
	// Global Initialization
    // ===========================================================================

    void InitNN();


} // namespace anet:nn
