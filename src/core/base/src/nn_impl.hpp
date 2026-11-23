// nn_impl.hpp

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <mutex>
#include <string_view>
#include <typeindex>

#include "anet/nn_head.hpp"

namespace anet::nn {


    // ===========================================================================
    //  WeightInitializer
    // ===========================================================================

    /// 登録前の検査と重みへの適用で同じ半幅を求める。空の重みは半幅を持たない。
    std::optional<double> ComputeVarianceScalingBound(
        at::IntArrayRef weight_shape, at::ScalarType dtype, double variance_scale, std::string_view key);


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

    WeightNormMode ParseWeightNormMode(const std::string& mode, const std::string& key);

    void PowerIterationStep(const torch::Tensor& weight_mat, SpectralNormState& state);
    SpectralNormState MakeSpectralNormState(
        const torch::Tensor& weight,
        WeightNormMode mode,
        const std::string& name,
        anet::RandomGenerator& rnd,
        std::string_view mode_key,
        std::optional<std::string_view> init2_mode_key);

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
        ConfigFullKeyTable full_keys;
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

        torch::Tensor Forward(torch::Tensor input, const NetworkForwardContext& context,
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
        void Execute(anet::TensorDict& current_state, const NetworkForwardContext& context,
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
            std::shared_ptr<ModuleRandomSource> random_source,
            const std::string& branch_name);
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
            const std::vector<std::vector<NetworkModuleExecutor*>>* calls = nullptr);
        anet::TensorDict ForwardUpTo(const anet::TensorDict& input, const NetworkForwardContext& context, const std::string& branch_key);
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
