// anet/nn.hpp

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <optional>
#include <vector>
#include <functional>
#include <map>
#include <utility>
#include <torch/torch.h>
#include "anet/config.hpp"
#include "anet/common.hpp"
#include "anet/graphviz.hpp"
#include "anet/random.hpp"


namespace anet::nn {


    enum class PlasticityMetric {
        SRANK_DELTA_001,
        SRANK_DELTA_005,
        SRANK_DELTA_020,
        SRANK_RATIO_DELTA_001,
        SRANK_RATIO_DELTA_005,
        SRANK_RATIO_DELTA_020,
        DORMANT_RATIO,
        DEAD_RATIO,
        FEATURE_NORM,
        COUNT,
    };

    struct PlasticityMetricRequest {
        std::array<bool, static_cast<size_t>(PlasticityMetric::COUNT)> requested{};

        void Add(PlasticityMetric metric) {
            requested[static_cast<size_t>(metric)] = true;
        }

        bool Contains(PlasticityMetric metric) const {
            return requested[static_cast<size_t>(metric)];
        }

        bool Any() const {
            return std::ranges::any_of(requested, [](bool value) { return value; });
        }

        static PlasticityMetricRequest All() {
            PlasticityMetricRequest result;
            result.requested.fill(true);
            return result;
        }
    };

    struct PlasticityMetrics {
        std::array<std::optional<float>, 3> srank;
        std::array<std::optional<float>, 3> srank_ratio;
        std::optional<float> dormant_ratio;
        std::optional<float> dead_ratio;
        std::optional<float> feature_norm;

        std::optional<float> Get(PlasticityMetric metric) const;
    };

    // 無印の plasticity_srank / plasticity_srank_ratio は index 0、δ=0.01 を表す。
    inline constexpr std::array<float, 3> kPlasticitySrankDeltas{ 0.01f, 0.05f, 0.20f };
    inline constexpr float kPlasticityDormantTau = 0.025f;

    std::optional<PlasticityMetric> ParsePlasticityMetricSuffix(std::string_view suffix);
    PlasticityMetrics ComputePlasticityMetrics(
        const torch::Tensor& features,
        const PlasticityMetricRequest& request);


    // ===========================================================================
    // Config Structures
    // ===========================================================================

    /// 解決済みの設定データを不変に渡す。固有の型・解釈・検証は利用する具象クラスが所有する。
    class NetworkForwardConfig {
    public:
        NetworkForwardConfig(std::string key, const anet::ConfigData& data);
        static const NetworkForwardConfig& GetDefault();
        const anet::ConfigData& GetConfigData() const { return config_data_; }
        bool IsDefault() const { return is_default_; }
        const std::string key;
    private:
        NetworkForwardConfig();
        const anet::ConfigData config_data_;
        const bool is_default_;
    };

    /// 宣言の検証を登録済み Module factory へ委譲する。固有の項目は解釈しない。
    void ValidateNetworkForwardConfigDeclaration(const NetworkForwardConfig& config);

    /// 時計の種類。NN から見て、同じ行が次の呼び出しへ続くかどうかだけを表す。
    enum class NetworkForwardClock {
        kNone,            ///< 時計なし。各計算部品の既定動作だけ
        kTransientRows,   ///< 行は呼び出しごとに別物。保持は呼び出し単位だけ
        kContinuingRows,  ///< 同じ行が次の呼び出しへ続く。回数・episode での保持を使える
    };

    /// 呼び出し側が所有する箱。保持する値と乱数は NN の計算部品だけが操作する。
    class NetworkForwardState {
    public:
        ~NetworkForwardState();
        NetworkForwardState(const NetworkForwardState&) = delete;
        NetworkForwardState& operator=(const NetworkForwardState&) = delete;
    private:
        friend class Network;
        friend class NetworkForwardStateAccess;
        explicit NetworkForwardState(anet::seed_t seed);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// 元の行と結合入力の対応。一度確定した index は呼び出し中に変更しない。
    class NetworkForwardRowMapping {
    public:
        NetworkForwardRowMapping(const torch::Tensor& cpu_indices, int64_t source_rows, torch::Device device);
        int64_t GetSourceRows() const { return source_rows_; }
        int64_t GetRows() const { return indices_.size(0); }
        torch::Device GetDevice() const { return indices_.device(); }
        torch::Tensor GetIndices() const; ///< 検査用の独立した複製を返す。
        torch::Tensor SelectRows(const torch::Tensor& source) const;
    private:
        const int64_t source_rows_;
        const torch::Tensor indices_;
    };

    /// 呼び出し側が決める今回の情報。設定・箱・時計種別を含めない。
    class NetworkForwardCallInfo {
    public:
        struct Params {
            uint64_t clock = 0;
            torch::Tensor episode_start;
            std::optional<NetworkForwardRowMapping> row_mapping;
        };
        explicit NetworkForwardCallInfo(Params params) : params_(std::move(params)) {}
        uint64_t GetClock() const { return params_.clock; }
        const torch::Tensor& GetEpisodeStart() const { return params_.episode_start; }
        const std::optional<NetworkForwardRowMapping>& GetRowMapping() const { return params_.row_mapping; }
        NetworkForwardCallInfo WithRowMapping(std::optional<NetworkForwardRowMapping> row_mapping) const;
    private:
        const Params params_;
    };

    /// 呼び出しごとの不変な実行情報。設定と状態の所有権は呼び出し側にある。
    class NetworkForwardContext {
    public:
        /// 生成時の入力。生成後に変更しても Context のメンバは変更されない。
        struct Params {
            NetworkForwardState* state = nullptr;
            NetworkForwardClock clock_kind = NetworkForwardClock::kNone;
            uint64_t clock = 0;
            torch::Tensor episode_start;
            std::optional<NetworkForwardRowMapping> row_mapping;
        };

        explicit NetworkForwardContext(Params params);
        static NetworkForwardContext GetDefault();

        NetworkForwardState* GetState() const { return params_.state; }
        NetworkForwardClock GetClockKind() const { return params_.clock_kind; }
        uint64_t GetClock() const { return params_.clock; }
        const torch::Tensor& GetEpisodeStart() const { return params_.episode_start; }
        const std::optional<NetworkForwardRowMapping>& GetRowMapping() const { return params_.row_mapping; }

    private:
        const Params params_; ///< Tensor のストレージは共有する。呼び出し中の変更は禁止。
    };

    struct NetworkBlockConfig {
        std::string type;             ///< "Conv2d", "Linear", "ResBlock" etc.
        anet::ConfigData config_data; ///< Specific config data for the block. "kernel_size"=3 etc.
    };

    struct ConfigProfileConfig {
        std::string type = "linear";   ///< 補間ルール（初期実装は linear のみ）
        double start = 0.0;            ///< i=0 の値
        double end = 0.0;              ///< i=N-1 の値
        bool has_type = false;         ///< type が明示指定されたか
        bool has_start = false;        ///< start が明示指定されたか
        bool has_end = false;          ///< end が明示指定されたか
    };

    struct NetworkBranchConfig {
        std::string name;                      ///< ブランチ名（＝出力キー）
        std::vector<std::vector<std::string>> bind_terms; ///< 入力項のリスト（各項は積 factor のリスト）
        int64_t bind_concat_dim = 1;            ///< 複数項を結合する次元
        std::vector<std::string> raw_keys;      ///< bind factor のうち、"(raw)" が指定されたキーのリスト
        bool auto_format = true;               ///< ブランチ全体での自動フォーマット有効/無効
        std::string structure_str;             ///< ブランチ内部の直列パイプライン構造
        std::map<std::string, ConfigProfileConfig> config_profiles; ///< branch単位のconfig_profile上書き
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
        std::string mode = "xavier";        ///< "default", "xavier", "he", "orthogonal", "constant", "trunc_normal"
        double manual_gain = 0.0f;          ///< Orthogonal時の手動ゲイン (0.0なら自動計算)
        std::string nonlinearity = "relu";  ///< "relu", "linear", "tanh" etc.
        double constant_val = 0.0;          ///< for Constant
        double trunc_std = 0.02;            ///< for TruncNormal
        double trunc_a = -2.0;              ///< for TruncNormal
        double trunc_b = 2.0;               ///< for TruncNormal
    };

    struct WeightNormConfig {
        std::string mode = "none";      ///< "none", "spectral", "spectral_cap"
    };

    enum class WeightNormMode {
        kNone,
        kSpectral,
        kSpectralCap,
    };

    // Configのfloat値0.1fをdoubleへ拡幅した境界まで同値として受け入れる。
    inline constexpr double kSpectralNormMaxSoftUpdateTau = static_cast<double>(0.1f);

    struct SpectralNormEntry {
        std::string name;
        WeightNormMode mode = WeightNormMode::kNone;
        torch::Tensor weight;
        torch::Tensor u;
        torch::Tensor v;
    };

    torch::Tensor ComputeSpectralSigma(
        const torch::Tensor& weight_mat, const torch::Tensor& u, const torch::Tensor& v);

    /// optimizer 分類と診断だけに使う、登録済みの mean/sigma の対応。
    struct NetworkNoisyParameterEntry {
        torch::Tensor mean;
        torch::Tensor sigma;
        bool is_bias = false;
        std::string branch; ///< Head は空文字。Body の所属 branch は Network が付ける。
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

    class NetworkHeadExecution {
    public:
        virtual anet::TensorDict Forward(const anet::TensorDict& features, const NetworkForwardContext& context) = 0;
        virtual ~NetworkHeadExecution() = default;
    };

    class NetworkHead : public torch::nn::Module, public anet::TensorDictFunctionProvider, public NetworkHeadExecution {
    public:
        virtual anet::TensorDict Forward(const anet::TensorDict& feature_dict, const NetworkForwardContext& context) = 0;
        virtual std::unique_ptr<NetworkHeadExecution> PrepareForward(const NetworkForwardConfig& config, NetworkForwardClock clock_kind)
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
        virtual void ValidateForwardConfigDeclaration(const NetworkForwardConfig& config) const {}
        virtual ~NetworkHeadFactory() = default;
    };


    // ===========================================================================
	// Network (Body + Head)
    // ===========================================================================

    class NetworkBody;

    struct NetworkBranchCapture {
        std::string branch_key;
        torch::Tensor output;
    };

    struct NetworkParameterNormSplit {
        torch::Tensor feature;
        torch::Tensor readout;
        torch::Tensor feature_effective;
        torch::Tensor readout_effective;
        torch::Tensor sigma_feature_max;
        torch::Tensor sigma_readout_max;
        torch::Tensor invalid_count;
    };

    class NetworkForwardExecution;

    class Network : public torch::nn::Module, public anet::TensorDictFunctionProvider {
    public: // コンストラクタ
        Network(
            const NetworkConfig& config,
            const anet::TensorSpecMap& input_specs,
            std::shared_ptr<NetworkHeadFactory> head_factory,
            std::shared_ptr<NetworkBody> body,
            std::shared_ptr<NetworkHead> head,
            anet::seed_t construction_seed = 0);
    public: // Forward
        anet::TensorDict Forward(
            const anet::TensorDict& input,
            const anet::TraceCallback& callback = {},
            NetworkBranchCapture* capture = nullptr);
        anet::TensorDict ForwardUpTo(const anet::TensorDict& input, const std::string& branch_key);
        void ValidateForwardReuse(const NetworkForwardConfig& source, const NetworkForwardConfig& requested) const;
    public:
        /// shared_ptrで所有されたNetworkを準備し、実行オブジェクトにその寿命を保持させる。
        std::shared_ptr<NetworkForwardExecution> PrepareForward(
            const NetworkForwardConfig& config, NetworkForwardClock clock_kind, anet::seed_t seed);
	public: // 情報取得
        std::vector<std::string> GetBranchNames() const;
        std::shared_ptr<Network> Clone(std::optional<torch::Device> device = std::nullopt) const;                 /// 自身の完全な複製(別インスタンス)を生成
        void CopyTo(Network& target) const;                     /// ターゲットへ重みを完全上書き (Hard Update)
        void SoftCopyTo(Network& target, double tau) const;     /// ターゲットへ重みをブレンド (Soft Update)
	public: // パラメータ正規化関連
        std::vector<NetworkNoisyParameterEntry> GetNoisyParameterEntries() const;
        /// [feature weight,bias,readout weight,bias] の mean-abs、続いて同順の L2 ratio を FP32 [8] で返す。
        torch::Tensor ComputeNoisyParameterStatistics(const std::string& feature_key) const;
        NetworkParameterNormSplit ComputeParameterNormSplit(const std::string& feature_key) const;
        torch::Tensor ComputeSpectralNormValidity() const;
        std::vector<SpectralNormEntry> GetSpectralNormEntries() const;
    public: //可視化関連
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override;
        std::unique_ptr<anet::graphviz::GraphViz> MakeGraphViz(const NetworkGraphVizConfig& config) const;
    private:
        friend class NetworkForwardExecution;
        anet::TensorDict RunForward(
            const anet::TensorDict& input, const NetworkForwardContext& context,
            const anet::TraceCallback& callback, NetworkBranchCapture* capture,
            const std::vector<std::vector<class NetworkModuleExecution*>>* body_calls,
            NetworkHeadExecution* head_call);
        // 実行用の実体
        std::shared_ptr<NetworkBody> body_;
        std::shared_ptr<NetworkHead> head_;

        // 構築情報(Clone用)
        NetworkConfig config_;
        anet::TensorSpecMap input_specs_;
        std::shared_ptr<NetworkHeadFactory> head_factory_;
        anet::seed_t construction_seed_;
        bool has_spectral_norm_ = false;
    };


    /// 特定 Network と用途に結び付いた実行 Resource。同期・保存の対象にはしない。
    class NetworkForwardExecution {
    public:
        ~NetworkForwardExecution();
        NetworkForwardExecution(const NetworkForwardExecution&) = delete;
        NetworkForwardExecution& operator=(const NetworkForwardExecution&) = delete;
        anet::TensorDict Forward(
            const anet::TensorDict& input, const NetworkForwardCallInfo& info,
            const anet::TraceCallback& callback = {}, NetworkBranchCapture* capture = nullptr);
        anet::TensorDict ForwardUpTo(
            const anet::TensorDict& input, const NetworkForwardCallInfo& info, const std::string& branch_key);
        std::shared_ptr<NetworkForwardExecution> BindNetwork(
            std::shared_ptr<Network> network, const NetworkForwardConfig& config) const;
        const std::shared_ptr<Network>& GetNetwork() const;
    private:
        friend class Network;
        NetworkForwardExecution(std::shared_ptr<Network> network, const NetworkForwardConfig& config,
            NetworkForwardClock clock_kind, std::shared_ptr<NetworkForwardState> state);
        NetworkForwardContext MakeContext(const NetworkForwardCallInfo& info) const;
        struct Impl;
        const std::unique_ptr<Impl> impl_;
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
            anet::seed_t seed,
            std::optional<torch::Device> device = std::nullopt);
    };

} // namespace anet::nn
