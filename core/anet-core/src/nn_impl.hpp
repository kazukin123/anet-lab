// nn_impl.hpp

#pragma once

#include "anet/nn.hpp" 

namespace anet::nn {


    // ===========================================================================
    // Helper Functions (Internal)
    // ===========================================================================

    int ParseInt(const std::string& s, const std::string& param_name);


    // ===========================================================================
    //  WeightInitializer
    // ===========================================================================

    torch::nn::init::NonlinearityType GetNonlinearityType(const std::string& name);

    class WeightInitializer {
    public:
        template <typename T>
        static void Initialize(T& layer, const WeightInitConfig& config)
        {
            if (config.mode == 0) return;

            auto& weight = layer->weight;

            // 初期化中は勾配計算不要
            torch::NoGradGuard no_grad;

			// mode別に初期化実行
            if (config.mode == 1) { // Xavier Uniform
                // Xavierは通常 gain=1.0 前提だが、calculate_gainを使う手もある
                torch::nn::init::xavier_uniform_(weight);
            } else if (config.mode == 2) { // He Normal (Kaiming)
                // 文字列からLibtorchの定数へ変換
                auto nonlinearity_mode = GetNonlinearityType(config.nonlinearity);

                // 重み初期化(kaiming_normal_ は内部で nonlinearity に応じた gain を計算してくれる)
                torch::nn::init::kaiming_normal_(weight, 0.0, torch::kFanOut, nonlinearity_mode);
            } else if (config.mode == 3) { // Orthogonal
                double gain = 1.0;

                if (config.manual_gain > 0.0f) {
                    gain = config.manual_gain;
                } else {
                    // "relu" -> sqrt(2), "tanh" -> 5/3, "linear" -> 1.0 など
                    try {
                        auto nonlinearity_mode = GetNonlinearityType(config.nonlinearity);
                        gain = torch::nn::init::calculate_gain(nonlinearity_mode);
                    } catch (...) {
                        ANET_SYSTEM_ERROR("Unknown nonlinearity: " << config.nonlinearity << ". Using gain=1.0");
                    }
                }

				// Orthogonal初期化
                torch::nn::init::orthogonal_(weight, gain);
            } else if (config.mode == 4) { // Constant
                torch::nn::init::constant_(weight, config.constant_val);
            }

            // バイアスはゼロ初期化
            auto& bias = layer->bias;
            if (bias.defined() && config.mode != 0) {
                if (config.mode == 4) {
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
    // Internal Components (Block & Body)
    // ===========================================================================

    /// @brief 単一のモジュールをラップし、入出力タグ情報を保持するブロック
    class NetworkBlock : public torch::nn::Module {
    public:
        NetworkBlock(
            std::string name,
            std::shared_ptr<NetworkModule> module,
            std::vector<std::string> input_tagss,
            std::string output_tag
        );

        torch::Tensor Forward(torch::Tensor input);

        const std::string& GetName() const { return name_; }
        const std::vector<std::string>& GetInputTags() const { return input_tags_; }
        const std::string& GetOutputTag() const { return output_tag_; }
        std::shared_ptr<NetworkModule> GetModule() const { return module_; }
    private:
        std::string name_;
        std::shared_ptr<NetworkModule> module_;
        std::vector<std::string> input_tags_;
        std::string output_tag_;
    };

    /// @brief 複数のBlockを順次実行し、タグによるデータフローを解決する実行エンジン
    class NetworkStruct : public torch::nn::Module {
    public:
        explicit NetworkStruct(std::vector<std::shared_ptr<NetworkBlock>> blocks);

        /// @brief Graph全体への入力inputを受け取り、フローを実行して最終出力を返す
        torch::Tensor Forward(torch::Tensor input);

        /// @brief ダミー入力による出力次元計測
        int64_t InferFeatureDim(const std::vector<int64_t>& input_shape);
    private:
        std::vector<std::shared_ptr<NetworkBlock>> blocks_;
    };

    /// @brief NetworkStructを持つサブブロック用モジュール (CompositeModule)
    class CompositeModule : public NetworkModule {
    public:
        explicit CompositeModule(std::shared_ptr<NetworkStruct> graph);

        torch::Tensor Forward(torch::Tensor input) override;

        /// @todo nn_modules.cppに移す？
    private:
        std::shared_ptr<NetworkStruct> graph_;
    };

    /// @brief NetworkStructを持つルートオブジェクト (NetworkBody)
    class NetworkBody : public NetworkModule {
    public:
        explicit NetworkBody(std::shared_ptr<NetworkStruct> graph);

        torch::Tensor Forward(torch::Tensor input) override;
        int64_t InferFeatureDim(const std::vector<int64_t>& input_shape);  ///< ダミー入力による出力次元計測
    private:
        std::shared_ptr<NetworkStruct> graph_;
    };


    // ===========================================================================
    // Internal Factories & Repository
    // ===========================================================================

    class NetworkStructBuilder {
    public:
        /// @brief Config全体と、パース対象の構造文字列を受け取り、NetworkStructを構築する
        static std::shared_ptr<NetworkStruct> Build(
            const NetworkConfig& root_config,
            const std::string& structure_str
        );
    };

    class NetworkModuleFactory {
    public:
        /// @brief Context (タグ情報等) を受け取り、適切なモジュールを生成する
        virtual std::shared_ptr<NetworkModule> CreateModule(
            const ConfigData& config_data,const ModuleContext& context) const = 0;
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

	void InitNN();


} // namespace anet:nn
