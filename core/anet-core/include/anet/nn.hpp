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

    // ===========================================================================
    // 構造設定
    // ===========================================================================

    /// 再利用するblockの種別と設定。構築側が解決し、Module factoryが読む。
    struct NetworkBlockConfig {
        std::string type; ///< 登録済みModuleの種別名。
        anet::ConfigData config_data; ///< そのModuleへ渡す構造設定。
    };

    /// 同じ構造内の繰り返し位置に応じて設定値を補間する規則。
    struct ConfigProfileConfig {
        std::string type = "linear"; ///< 補間方式。現行はlinear。
        double start = 0.0; ///< 最初の位置の値。
        double end = 0.0; ///< 最後の位置の値。
        bool has_type = false; ///< 補間方式が明示されたか。
        bool has_start = false; ///< 開始値が明示されたか。
        bool has_end = false; ///< 終了値が明示されたか。
    };

    /// 入力の結合、構造、局所profileを表す1本のbranchの定義。
    struct NetworkBranchConfig {
        std::string name; ///< branch名。内部の出力キーにも使う。
        std::vector<std::vector<std::string>> bind_terms; ///< 結合する項の列。各項は要素積に使うfactor名の列。
        int64_t bind_concat_dim = 1; ///< 複数項を連結する次元。batch次元0は使わない。
        std::vector<std::string> raw_keys; ///< 自動整形を省略するraw入力キーの列。
        bool auto_format = true; ///< branchで入力の自動整形を使うか。
        std::string structure_str; ///< branch内の直列構造の記述。
        std::map<std::string, ConfigProfileConfig> config_profiles; ///< このbranchで上書きする補間profile。
    };

    /// block・branch・出力対応・profileを保持するNetworkの構築情報。
    struct NetworkConfig {
        std::map<std::string, NetworkBlockConfig> block_configs; ///< block名から種別・構造設定への対応。
        std::map<std::string, NetworkBranchConfig> branches; ///< branch名からbranch定義への対応。
        std::map<std::string, std::string> output_keys; ///< Headが受け取るキーからbranch名への対応。
        std::map<std::string, ConfigProfileConfig> config_profiles; ///< Network全体の補間profile。

        NetworkConfig() = default;
        NetworkConfig(const anet::ConfigData& config_data, const std::string& config_prefix = "net");
        anet::json ToJson() const;
    };

    /// 構築時にweightとbiasへ適用する初期化方式とその引数。
    struct WeightInitConfig {
        std::string mode = "xavier"; ///< default、xavier、he、orthogonal、constant、trunc_normal。
        double manual_gain = 0.0f; ///< orthogonalの手動gain。正値なら使用し、それ以外はnonlinearityから求める。
        std::string nonlinearity = "relu"; ///< heとorthogonalのgain計算に使う活性化名。
        double constant_val = 0.0; ///< constantでweightとbiasへ設定する値。
        double trunc_std = 0.02; ///< trunc_normalの標準偏差。
        double trunc_a = -2.0; ///< trunc_normalの切断区間の下限。
        double trunc_b = 2.0; ///< trunc_normalの切断区間の上限。
    };

    /// 初期化設定の共通条件を検証する。ownerは診断に用いる設定キーの接頭辞。
    void ValidateWeightInitConfig(const WeightInitConfig& config, const std::string& owner = "WeightInitConfig");

    /// 利用側が指定する構造図の表示項目とレイアウト。
    struct NetworkGraphVizConfig {
        bool show_param_shapes = false; ///< parameterのshapeを表示するか。
        bool show_param_count = false; ///< parameterの要素数を表示するか。
        bool show_tensor_specs = false; ///< Tensor仕様を表示するか。
        bool show_branch_config = false; ///< branchの設定を表示するか。
        bool show_head_info = false; ///< Headが提供する情報を表示するか。
        std::string layout = "LR"; ///< GraphVizのrankdir。既定はLR。
        bool cluster_branches = true; ///< branchごとに囲みを作るか。
        int float_precision = 3; ///< 数値表示の精度。
    };

    void ValidateNetworkGraphVizConfig(const NetworkGraphVizConfig& config, const std::string& owner);

    // ===========================================================================
    // 実行設定と呼び出し情報
    // ===========================================================================

    /// 選択済みの設定データを不変に渡す。固有の解釈と検証は利用する具象クラスが所有する。
    class NetworkForwardConfig {
    public:
        NetworkForwardConfig(std::string key, const anet::ConfigData& data);
        static const NetworkForwardConfig& GetDefault();
        const anet::ConfigData& GetConfigData() const { return config_data_; }
        bool IsDefault() const { return is_default_; }
        const std::string key; ///< 参照元を識別する設定キー。
    private:
        NetworkForwardConfig();
        const anet::ConfigData config_data_; ///< 選択済み設定の不変なコピー。
        const bool is_default_; ///< 具象設定を持たない既定要求か。
    };

    /// 同じ行が次の呼び出しへ続くかを表す時計種別。実行器の生成時に指定する。
    enum class NetworkForwardClock {
        kTransientRows,   ///< 行は呼び出しごとに別物。保持は呼び出し単位だけ
        kContinuingRows,  ///< 同じ行が次の呼び出しへ続く。回数・episode での保持を使える
    };

    /// 結合入力の各行を元の行へ対応させる不変の表。構築時に検証し、使用deviceへ一度転送する。
    class NetworkForwardRowMapping {
    public:
        /// CPU int64 [rows]のindexと元行数を検証し、指定deviceへ転送する。
        NetworkForwardRowMapping(const torch::Tensor& cpu_indices, int64_t source_rows, torch::Device device);
        int64_t GetSourceRows() const { return source_rows_; }
        int64_t GetRows() const { return indices_.size(0); }
        torch::Device GetDevice() const { return indices_.device(); }
        torch::Tensor GetIndices() const; ///< 検査用の独立した複製を返す。
        /// 元の行数を持つsourceから対応表の順に行を取り出す。
        torch::Tensor SelectRows(const torch::Tensor& source) const;
    private:
        const int64_t source_rows_; ///< 対応先となる元の行数。
        const torch::Tensor indices_; ///< 使用device上のint64 [rows]の対応表。
    };

    /// 呼び出し側が決める今回の時計値・境界・行対応。設定・箱・時計種別を含めない。
    class NetworkForwardCallInfo {
    public:
        /// 呼び出し情報の生成入力。メンバ値を取り込み、Tensorのストレージは共有する。
        struct Params {
            uint64_t clock = 0; ///< 用途内の離散的な時計番号。保持中は逆行させない。
            torch::Tensor episode_start; ///< bool [元行数B]の開始行マスク。利用しない場合は未定義でよい。
            std::optional<NetworkForwardRowMapping> row_mapping; ///< 結合入力の行対応。行の並びを変えない場合は省略する。
        };
        explicit NetworkForwardCallInfo(Params params) : params_(std::move(params)) {}
        uint64_t GetClock() const { return params_.clock; }
        const torch::Tensor& GetEpisodeStart() const { return params_.episode_start; }
        const std::optional<NetworkForwardRowMapping>& GetRowMapping() const { return params_.row_mapping; }
        /// 行対応だけを差し替えたコピーを返し、時計値と開始マスクを保つ。
        NetworkForwardCallInfo WithRowMapping(std::optional<NetworkForwardRowMapping> row_mapping) const;
    private:
        const Params params_; ///< 生成時に確定した呼び出し情報。Tensorのストレージは共有する。
    };

    /// 鍵で要求したbranch出力とmodule診断を、同期Forwardから持ち帰る器。
    class NetworkForwardCapture {
    public:
        using Entry = std::pair<std::string, torch::Tensor>;
        NetworkForwardCapture() = default;
        explicit NetworkForwardCapture(std::vector<std::string> requests);
        bool Wants(std::string_view key) const;
        void Add(std::string_view key, torch::Tensor value);
        torch::Tensor Get(std::string_view key) const;
        std::vector<Entry> GetAll(std::string_view prefix) const;
        std::vector<std::string> Unfulfilled() const;
        void Reset();
    private:
        std::vector<std::string> requests_;
        std::vector<Entry> entries_;
    };

    inline constexpr std::string_view kTransformerEncoderAttentionEntropyCaptureKey =
        "TransformerEncoder/attention_entropy";

    class NetworkBody;
    class NetworkHead;
    class NetworkHeadFactory;
    class NetworkForwardExecutor;
    struct NetworkNoisyParameterEntry;
    struct SpectralNormEntry;
    struct NetworkParameterNormSplit;
    struct NetworkParameterSplit;

    // ===========================================================================
    // Networkと用途別実行器
    // ===========================================================================

    /// BodyとHeadを所有し、Forward・重み同期・診断の入口となるNetwork。
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
        /// 入力を既定動作で計算する。capture指定時は要求されたbranch出力・module診断も取得する。
        anet::TensorDict Forward(
            const anet::TensorDict& input,
            const anet::TraceCallback& callback = {},
            NetworkForwardCapture* capture = nullptr);
        /// branch_keyまでを既定動作で計算し、Headを実行しない。
        anet::TensorDict ForwardUpTo(const anet::TensorDict& input, const std::string& branch_key,
            NetworkForwardCapture* capture = nullptr);
        /// sourceとrequestedの再利用条件を、それらを使う具象部品に検証させる。
        void ValidateForwardReuse(const NetworkForwardConfig& source, const NetworkForwardConfig& requested) const;
    public:
        /// shared_ptr所有のNetworkに対し、設定を準備する。clock_kindは保持制約、seedは用途専用の乱数に使う。
        /// 既定設定では部品を準備せずStateも作らないため、clock_kindとseedは使用しない。
        std::shared_ptr<NetworkForwardExecutor> CreateForwardExecutor(
            const NetworkForwardConfig& config, NetworkForwardClock clock_kind, anet::seed_t seed);
    public: // 情報取得
        std::vector<std::string> GetBranchNames() const;
        std::shared_ptr<Network> Clone(std::optional<torch::Device> device = std::nullopt) const;                 ///< 自身の完全な複製(別インスタンス)を生成
        void CopyTo(Network& target) const;                     ///< ターゲットへ重みを完全上書き (Hard Update)
        void SoftCopyTo(Network& target, double tau) const;     ///< ターゲットへ重みをブレンド (Soft Update)
    public: // パラメータ正規化関連
        /// 登録済みmean/sigmaの対を所属branchとともに列挙する。
        std::vector<NetworkNoisyParameterEntry> GetNoisyParameterEntries() const;
        /// feature_keyの閉包で分類し、FP32 [8]で返す。
        /// [feature weight, feature bias, readout weight, readout bias]のmean-abs、続いて同じ順のL2 ratio。
        torch::Tensor ComputeNoisyParameterStatistics(const std::string& feature_key) const;
        /// feature_keyの依存閉包でfeature/readoutを分けたnormとSN診断を返す。
        NetworkParameterNormSplit ComputeParameterNormSplit(const std::string& feature_key) const;
        /// feature依存閉包とreadoutに分類した全学習parameter。Noisy sigmaも含む。
        NetworkParameterSplit GetParameterSplit(const std::string& feature_key) const;
        /// 不正なSN spectral sigmaの数をFP32のdevice scalarで返す。
        torch::Tensor ComputeSpectralNormValidity() const;
        /// BodyのSN対象weightとbufferを診断用に列挙する。
        std::vector<SpectralNormEntry> GetSpectralNormEntries() const;
    public: //可視化関連
        /// Bodyを既定動作で通し、指定キーにHeadが提供する関数へつなぐ。
        /// Headがない場合はforwardキーでBodyのみの関数を返し、未対応キーはnulloptを返す。
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override;
        /// 表示設定に従って構造図を生成する。
        std::unique_ptr<anet::graphviz::GraphViz> MakeGraphViz(const NetworkGraphVizConfig& config) const;
    private:
        friend class NetworkForwardExecutor;
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

    /// 特定Networkと用途に結び付いた実行Resource。Networkの寿命を保持し、同期・保存には含めない。
    class NetworkForwardExecutor {
    public:
        ~NetworkForwardExecutor();
        NetworkForwardExecutor(const NetworkForwardExecutor&) = delete;
        NetworkForwardExecutor& operator=(const NetworkForwardExecutor&) = delete;

        /// 同期呼び出し中だけinfoを参照し、固定した処理を実行する。
        anet::TensorDict Forward(
            const anet::TensorDict& input, const NetworkForwardCallInfo& info,
            const anet::TraceCallback& callback = {}, NetworkForwardCapture* capture = nullptr);
    private:
        friend class Network;
        struct Impl;
        explicit NetworkForwardExecutor(std::unique_ptr<Impl> impl);
        const std::unique_ptr<Impl> impl_;
    };

    // ===========================================================================
    // 診断と可塑性
    // ===========================================================================

    /// weightのspectral sigmaに対する正規化方式。実効値の計算と診断が参照する。
    enum class WeightNormMode {
        kNone, ///< 正規化しない。
        kSpectral, ///< 常にspectral sigmaで割る。
        kSpectralCap, ///< spectral sigmaの絶対値が1を超える場合だけ割る。
    };

    /// SN対象weightとpower iterationのbufferを表す1件。Networkが診断用に列挙する。
    struct SpectralNormEntry {
        std::string name; ///< SN対象名。Networkからの列挙時はbranch.block名を含む。
        WeightNormMode mode = WeightNormMode::kNone; ///< 対象weightの正規化方式。
        torch::Tensor weight; ///< 正規化の対象weight [out, ...]。残る次元を平坦化してSNを計算する。
        torch::Tensor u; ///< power iterationの左特異ベクトル [out]。named buffer。
        torch::Tensor v; ///< power iterationの右特異ベクトル [平坦化した入力次元]。named buffer。
    };

    /// optimizer分類と診断だけに使う、登録済みmeanとsigmaのparameter対。
    struct NetworkNoisyParameterEntry {
        torch::Tensor mean; ///< sigmaに対応するweight [out, in]またはbias [out]のmean parameter。
        torch::Tensor sigma; ///< meanと同じshapeの、学習するノイズのスケールparameter。
        bool is_bias = false; ///< biasの対ならtrue、weightの対ならfalse。
        std::string branch; ///< Bodyの所属branch名。Headは空文字。
    };

    /// feature依存閉包の内外で分けた全学習parameter。Noisy sigmaも含む。
    struct NetworkParameterSplit {
        std::vector<torch::Tensor> feature;
        std::vector<torch::Tensor> readout;
    };

    /// 群別parameter normとSN診断。値はFP32のdevice scalar。
    struct NetworkParameterNormSplit {
        torch::Tensor feature; ///< feature依存閉包内の学習parameterのL2 norm。Noisy sigmaを除く。
        torch::Tensor readout; ///< 閉包外とHeadの同じ定義のL2 norm。
        torch::Tensor feature_effective; ///< feature群のSN対象weightを実効値に置換したL2 norm。
        torch::Tensor readout_effective; ///< readout群のSN対象weightを実効値に置換したL2 norm。
        torch::Tensor sigma_feature_max; ///< feature群のSN spectral sigmaの最大。capでは絶対値。対象なしはNaNのFP32 scalar。
        torch::Tensor sigma_readout_max; ///< readout群のSN spectral sigmaの最大。capでは絶対値。対象なしはNaNのFP32 scalar。
        torch::Tensor invalid_count; ///< 有限でないSN spectral sigmaの数。spectralでは0以下も数える。
    };

    /// 特徴量の可塑性を診断する指標と要求配列の添字。
    enum class PlasticityMetric {
        SRANK_DELTA_001, ///< delta=0.01のsrank。
        SRANK_DELTA_005, ///< delta=0.05のsrank。
        SRANK_DELTA_020, ///< delta=0.20のsrank。
        SRANK_RATIO_DELTA_001, ///< delta=0.01のsrank比率。
        SRANK_RATIO_DELTA_005, ///< delta=0.05のsrank比率。
        SRANK_RATIO_DELTA_020, ///< delta=0.20のsrank比率。
        DORMANT_RATIO, ///< tau=0.025の休眠比率。
        DEAD_RATIO, ///< 全サンプルで0となる特徴量の比率。
        FEATURE_NORM, ///< 特徴量ベクトルの平均L2 norm。
        COUNT, ///< 指標数。要求配列のサイズに使う。
    };

    /// 呼び出し側が測定を要求する可塑性指標の集合。
    struct PlasticityMetricRequest {
        std::array<bool, static_cast<size_t>(PlasticityMetric::COUNT)> requested{}; ///< 指標のenum添字ごとの測定要求。

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

    /// 要求した可塑性指標の測定値。未要求の値はnullopt。
    struct PlasticityMetrics {
        std::array<std::optional<float>, 3> srank; ///< delta=0.01/0.05/0.20のsrank（累積特異値が1-delta以上となる最小個数）。未要求はnullopt。
        std::array<std::optional<float>, 3> srank_ratio; ///< 同じdelta順のsrank/min(N, D)。未要求はnullopt。
        std::optional<float> dormant_ratio; ///< 平均絶対活性をunit間平均で正規化し、tau=0.025以下の特徴量比率。未要求はnullopt。
        std::optional<float> dead_ratio; ///< 全サンプルで0となる特徴量の比率。未要求はnullopt。
        std::optional<float> feature_norm; ///< 特徴量ベクトルの平均L2 norm。未要求はnullopt。

        std::optional<float> Get(PlasticityMetric metric) const;
    };

    // Configのfloat値0.1fをdoubleへ拡幅した境界まで同値として受け入れる。
    inline constexpr double kSpectralNormMaxSoftUpdateTau = static_cast<double>(0.1f);

    torch::Tensor ComputeSpectralSigma(
        const torch::Tensor& weight_mat, const torch::Tensor& u, const torch::Tensor& v);

    // 無印のsrankとsrank_ratioはindex 0のdelta=0.01を使う。
    inline constexpr std::array<float, 3> kPlasticitySrankDeltas{ 0.01f, 0.05f, 0.20f };
    inline constexpr float kPlasticityDormantTau = 0.025f;

    std::optional<PlasticityMetric> ParsePlasticityMetricSuffix(std::string_view suffix);
    /// FP32へ変換した特徴量 [N, D]から要求分だけ計算する。Nはサンプル数、Dは特徴量数。
    PlasticityMetrics ComputePlasticityMetrics(
        const torch::Tensor& features,
        const PlasticityMetricRequest& request);

    // ===========================================================================
    // Networkの構築
    // ===========================================================================

    /// 構造設定・入力仕様・Head factoryから完成したNetworkを構築する入口。
    class NetworkBuilder {
    public:
        /// 構造・入力仕様・Head factory・初期化seedから指定deviceのNetworkを作る。
        static std::shared_ptr<Network> BuildNetwork(
            const NetworkConfig& network_config,
            const anet::TensorSpecMap& input_specs,
            std::shared_ptr<NetworkHeadFactory> head_factory,
            anet::seed_t seed,
            std::optional<torch::Device> device = std::nullopt);
    };

} // namespace anet::nn
