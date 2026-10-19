// nn_impl.hpp

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <typeindex>

#include "anet/nn.hpp"

namespace anet::nn {


    // ===========================================================================
    //  WeightInitializer
    // ===========================================================================

    torch::nn::init::NonlinearityType GetNonlinearityType(const std::string& name);

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
            } else if (config.mode == "trunc_normal") {
                TruncNormal_(weight, config);
            } else {
                ANET_SYSTEM_ERROR("Unknown WeightInitConfig.mode: \"" << config.mode
                    << "\" expected one of: default, xavier, he, orthogonal, constant, trunc_normal");
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

    private:
        static double NormalCdf(double x)
        {
            return (1.0 + std::erf(x / std::sqrt(2.0))) / 2.0;
        }

        static void ValidateTruncNormalConfig(const WeightInitConfig& config)
        {
            if (!std::isfinite(config.trunc_std) || config.trunc_std <= 0.0) {
                ANET_SYSTEM_ERROR("Invalid WeightInitConfig.trunc_std: " << config.trunc_std
                    << " expected > 0 for mode=trunc_normal");
            }
            if (!std::isfinite(config.trunc_a) || !std::isfinite(config.trunc_b) || config.trunc_a >= config.trunc_b) {
                ANET_SYSTEM_ERROR("Invalid WeightInitConfig truncation range: trunc_a=" << config.trunc_a
                    << " trunc_b=" << config.trunc_b << " expected trunc_a < trunc_b for mode=trunc_normal");
            }
        }

        static void TruncNormal_(torch::Tensor& tensor, const WeightInitConfig& config)
        {
            ValidateTruncNormalConfig(config);

            const double low = NormalCdf(config.trunc_a / config.trunc_std);
            const double high = NormalCdf(config.trunc_b / config.trunc_std);
            tensor.uniform_(2.0 * low - 1.0, 2.0 * high - 1.0);
            tensor.erfinv_();
            tensor.mul_(config.trunc_std * std::sqrt(2.0));
            tensor.clamp_(config.trunc_a, config.trunc_b);
        }
    };


    // ===========================================================================
    // Network Internal Utils
    // ===========================================================================

    /// 実行オブジェクトが内部に持つ箱。計算部品だけが保持値と専用乱数を操作する。
    class NetworkForwardState {
    public:
        explicit NetworkForwardState(anet::seed_t seed);
        ~NetworkForwardState();
        NetworkForwardState(const NetworkForwardState&) = delete;
        NetworkForwardState& operator=(const NetworkForwardState&) = delete;

        template<typename T>
        T& Get(const void* owner)
        {
            auto& slot = GetSlot(owner, std::type_index(typeid(T)));
            if (!slot) slot = std::make_shared<T>();
            return *std::static_pointer_cast<T>(slot);
        }

        anet::RandomGenerator& GetRandom();
    private:
        std::shared_ptr<void>& GetSlot(const void* owner, std::type_index type);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// 同期Forwardの期間だけ、箱と不変の呼び出し情報を参照する。
    /// 名前付きContextには一時オブジェクトのCallInfoを渡さない。
    class NetworkForwardContext {
    public:
        static NetworkForwardContext GetDefault();
    public:
        NetworkForwardContext(NetworkForwardState* state, const NetworkForwardCallInfo& info,
            NetworkForwardCapture* capture = nullptr)
            : state_(state), info_(info), capture_(capture) {}

        bool HasCapture() const { return capture_ != nullptr; }
        NetworkForwardContext WithCapture(NetworkForwardCapture* capture) const;
        NetworkForwardContext WithCaptureBranch(std::string_view branch) const;
        NetworkForwardContext WithCaptureBlock(std::string_view block) const;
        bool WantsCapture(std::string_view key) const;
        void AddCapture(std::string_view key, torch::Tensor value) const;

        NetworkForwardState* GetState() const { return state_; }
        uint64_t GetClock() const { return info_.GetClock(); }
        const torch::Tensor& GetEpisodeStart() const { return info_.GetEpisodeStart(); }
        const std::optional<NetworkForwardRowMapping>& GetRowMapping() const { return info_.GetRowMapping(); }
    private:
        NetworkForwardState* const state_;
        const NetworkForwardCallInfo& info_;
        NetworkForwardCapture* capture_ = nullptr; ///< 呼び出し側所有。同期Forward中だけ借用する。
        std::string_view capture_branch_;
        std::string_view capture_block_;
        std::string MakeCaptureKey(std::string_view key) const;
    };

    // 射影の共用は数値演算だけとし、設定の解釈や保持境界は呼び出す具象クラスが所有する。
    std::pair<torch::Tensor, torch::Tensor> InitializeLinearSigma(torch::nn::Linear& layer, double sigma0);
    torch::Tensor AddFactorizedLinearNoise(
        const torch::Tensor& input, const torch::Tensor& mean,
        const torch::Tensor& sigma_weight, const torch::Tensor& sigma_bias,
        const torch::Tensor& input_factor, const torch::Tensor& output_factor);

    inline constexpr int kSpectralNormWarmStartIters = 15;

    struct SpectralNormState {
        torch::Tensor u;
        torch::Tensor v;
    };

    WeightNormMode ParseWeightNormMode(const std::string& mode);

    void PowerIterationStep(const torch::Tensor& weight_mat, SpectralNormState& state);
    SpectralNormState MakeSpectralNormState(
        const torch::Tensor& weight,
        WeightNormMode mode,
        const std::string& name,
        anet::RandomGenerator& rnd);

    torch::Tensor MakeSpectralNormalizedWeight(
        const torch::Tensor& weight,
        WeightNormMode mode,
        SpectralNormState& state,
        bool update_state);

    torch::Tensor SdpaSelfAttention(const torch::nn::MultiheadAttention& mha, const torch::Tensor& x);

    torch::Tensor SdpaSelfAttention(
        const torch::nn::MultiheadAttention& mha,
        const torch::Tensor& x,
        const torch::Tensor& in_proj_weight,
        const torch::Tensor& out_proj_weight);

    torch::Tensor DropPath(const torch::Tensor& x, double drop_prob, bool training);


    // ===========================================================================
    // Network Internal Components
    // ===========================================================================

    class ModuleRandomSource {
    public:
        explicit ModuleRandomSource(anet::seed_t base_seed) : seed_maker_(base_seed) { }
        std::shared_ptr<anet::RandomGenerator> Get(const std::string& purpose);
    private:
        anet::SeedMaker seed_maker_;
        std::mutex mutex_;
        std::map<std::string, std::shared_ptr<anet::RandomGenerator>> generators_;
    };

    // ModuleとHeadを作る側の契約。利用側は公開ヘッダの前方宣言だけを使う。
    struct WeightNormConfig {
        std::string mode = "none";      ///< "none", "spectral", "spectral_cap"
    };

    struct HeadGraphVizInfo {
        struct OutputInfo {
            std::string name;
            std::vector<int64_t> shape;
        };

        std::string type;
        std::vector<OutputInfo> outputs;
        std::vector<std::pair<std::string, std::string>> details;
    };

    class NetworkHeadExecutor {
    public:
        virtual anet::TensorDict Forward(const anet::TensorDict& features, const NetworkForwardContext& context) = 0;
        virtual ~NetworkHeadExecutor() = default;
    };

    class NetworkHead : public torch::nn::Module, public anet::TensorDictFunctionProvider, public NetworkHeadExecutor {
    public:
        virtual anet::TensorDict Forward(const anet::TensorDict& feature_dict, const NetworkForwardContext& context) = 0;
        /// 専用の実行器が不要ならnullptrを返し、通常の本体を直接呼ぶ。
        virtual std::unique_ptr<NetworkHeadExecutor> CreateExecutor(const NetworkForwardConfig& config, NetworkForwardClock clock_kind)
        {
            return nullptr;
        }
        virtual HeadGraphVizInfo GetGraphVizInfo() const { return {}; }
        virtual std::vector<NetworkNoisyParameterEntry> GetNoisyParameterEntries() const { return {}; }
        virtual void ValidateForwardReuse(const NetworkForwardConfig& source, const NetworkForwardConfig& requested) const {}
        virtual ~NetworkHead() = default;
    };

    class NetworkHeadFactory {
    public:
        virtual std::shared_ptr<NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const = 0;
        virtual ~NetworkHeadFactory() = default;
    };

    class NetworkModuleExecutor {
    public:
        virtual torch::Tensor Forward(torch::Tensor input, const NetworkForwardContext& context) = 0;
        virtual ~NetworkModuleExecutor() = default;
    };

    /// anet管理下のニューラルネットモジュール基底クラス
    class NetworkModule : public torch::nn::Module, public NetworkModuleExecutor {
    public:
        /// Forward実行.
        /// torch::nn::Module::forwardはテンプレートメソッドのため、多態性を持たせるために純粋仮想関数として再定義
        virtual torch::Tensor Forward(torch::Tensor input, const NetworkForwardContext& context) = 0;
        /// 専用の実行器が不要ならnullptrを返し、通常の本体を直接呼ぶ。
        virtual std::unique_ptr<NetworkModuleExecutor> CreateExecutor(
            const NetworkForwardConfig& config, NetworkForwardClock clock_kind)
        {
            return nullptr;
        }
        virtual bool IsConv2dVisualizable() const { return false; }
        virtual anet::ConfigData GetCurrentConfigData() const { return {}; }
        virtual void ValidateForwardReuse(const NetworkForwardConfig& source, const NetworkForwardConfig& requested) const {}
        virtual std::vector<SpectralNormEntry> GetSpectralNormEntries() const { return {}; }
        virtual std::vector<NetworkNoisyParameterEntry> GetNoisyParameterEntries() const { return {}; }
        virtual ~NetworkModule() = default;
    };

    // 子 Module の準備結果と固定した呼び先だけを持ち、通常の子は本体を直接呼ぶ。
    template <size_t N>
    class PreparedModuleChildren {
    public:
        PreparedModuleChildren(const std::array<NetworkModule*, N>& modules,
            const NetworkForwardConfig& config, NetworkForwardClock clock)
        {
            for (size_t i = 0; i < N; ++i) {
                if (!modules[i]) continue;
                owned_[i] = modules[i]->CreateExecutor(config, clock);
                calls_[i] = owned_[i] ? owned_[i].get() : modules[i];
            }
        }
        bool HasPrepared() const
        { return std::ranges::any_of(owned_, [](const auto& value) { return bool(value); }); }
        NetworkModuleExecutor* Get(size_t index) const { return calls_[index]; }
    private:
        std::array<std::unique_ptr<NetworkModuleExecutor>, N> owned_;
        std::array<NetworkModuleExecutor*, N> calls_{};
    };

    /// Body と Head が合成して使う具象の射影。設定型・状態型は Impl の内部に閉じる。
    class LinearProjectionModule final : public NetworkModule {
    public:
        explicit LinearProjectionModule(const anet::ConfigData& data);
        ~LinearProjectionModule() override;
        void Bind(torch::nn::Linear layer);
        class Executor : public NetworkModuleExecutor {
        public:
            virtual torch::Tensor ForwardWithMean(
                const torch::Tensor& input, const torch::Tensor& mean, const NetworkForwardContext& context) = 0;
        };
        std::unique_ptr<Executor> CreateProjectionExecutor(const NetworkForwardConfig& config, NetworkForwardClock clock);
        std::unique_ptr<NetworkModuleExecutor> CreateExecutor(
            const NetworkForwardConfig& config, NetworkForwardClock clock) override;
        torch::Tensor Forward(torch::Tensor input, const NetworkForwardContext& context) override;
        torch::Tensor ForwardWithMean(const torch::Tensor& input, const torch::Tensor& mean,
            const NetworkForwardContext& context);
        void ValidateForwardReuse(const NetworkForwardConfig& source, const NetworkForwardConfig& requested) const override;
        std::vector<NetworkNoisyParameterEntry> GetNoisyParameterEntries() const override;
        anet::ConfigData GetCurrentConfigData() const override;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };


    /// 単一のモジュールをラップし、入出力タグ情報を保持するブロック
    class NetworkBlock : public torch::nn::Module {
    public:
        NetworkBlock(std::string name, std::shared_ptr<NetworkModule> module);

        torch::Tensor Forward(torch::Tensor input, const NetworkForwardContext& context);

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

        torch::Tensor Forward(torch::Tensor input, const NetworkForwardContext& context, const anet::TraceCallback& callback = {},
            const std::vector<NetworkModuleExecutor*>* calls = nullptr);
        const std::vector<std::shared_ptr<NetworkBlock>>& GetBlocks() const { return blocks_; }
    private:
        std::vector<std::shared_ptr<NetworkBlock>> blocks_;
        std::vector<NetworkModuleExecutor*> default_calls_;
    };

    /// DAGを構成する一本のノード.
    /// 入力を結合し、内部の直列パイプラインに流す。
    class NetworkBranch : public torch::nn::Module {
    public:
        NetworkBranch(
            std::string name,
            std::vector<std::vector<std::string>> bind_terms,
            int64_t bind_concat_dim,
            std::shared_ptr<NetworkStruct> network_struct);

        /// 現在のTensorDictから必要な入力(bind)を拾って結合し、処理を実行して結果をDictに書き戻す
        void Execute(anet::TensorDict& current_state, const NetworkForwardContext& context, const anet::TraceCallback& callback = {},
            const std::vector<NetworkModuleExecutor*>* calls = nullptr);

        const std::string& GetName() const { return name_; }
        const std::vector<std::vector<std::string>>& GetBindTerms() const { return bind_terms_; }
        std::shared_ptr<NetworkStruct> GetNetworkStruct() const { return network_struct_; }

    private:
        std::string name_;
        std::vector<std::vector<std::string>> bind_terms_;
        int64_t bind_concat_dim_ = 1;
        std::shared_ptr<NetworkStruct> network_struct_;
    };

    class NetworkStructBuilder {
    public:
        /// Config全体と、パース対象の構造文字列を受け取り、NetworkStructを構築する
        static std::shared_ptr<NetworkStruct> Build(
            const NetworkConfig& root_config,
            const std::string& structure_str,
            std::shared_ptr<ModuleRandomSource> random_source);
        static std::shared_ptr<NetworkStruct> Build(
            const NetworkConfig& root_config,
            const std::string& structure_str,
            const std::map<std::string, ConfigProfileConfig>& config_profiles,
            std::shared_ptr<ModuleRandomSource> random_source);
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
        bool HasInputKey(const std::string& key) const { return specs_.contains(key); }
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
        anet::TensorDict Forward(
            const anet::TensorDict& input,
            const NetworkForwardContext& context,
            const anet::TraceCallback& callback = {},
            NetworkForwardCapture* capture = nullptr,
            const std::vector<std::vector<NetworkModuleExecutor*>>* calls = nullptr);
        anet::TensorDict ForwardUpTo(const anet::TensorDict& input, const NetworkForwardContext& context, const std::string& branch_key,
            NetworkForwardCapture* capture = nullptr);
        std::set<std::string> ComputeDependencyClosure(const std::string& branch_key) const;
        std::vector<std::string> GetBranchNames() const;
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
            const NetworkConfig& config,
            const anet::TensorSpecMap& input_specs,
            std::shared_ptr<ModuleRandomSource> random_source);
    };


    // ===========================================================================
    // NetworkModule Factories & Repository
    // ===========================================================================

    /// モジュール生成時のコンテキスト情報（構造情報）
    struct ModuleContext {
        std::shared_ptr<ModuleRandomSource> random_source;
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
