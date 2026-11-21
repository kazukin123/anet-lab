#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "anet/nn.hpp"

namespace anet::nn {

    class NetworkForwardState;

    torch::nn::init::NonlinearityType GetNonlinearityType(const std::string& name);

    class WeightInitializer {
    public:
        template <typename T>
        static void Initialize(T& layer, const WeightInitConfig& config)
        {
            // 設定由来・直接指定のどちらも、重みを変更する前に共通条件を検証する。
            ValidateWeightInitConfig(config);
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
            } else if (config.mode == WeightInitConfig::kVarianceScalingMode) {
                InitializeVarianceScaling(weight, config.variance_scale);
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
        static void InitializeVarianceScaling(torch::Tensor& weight, double variance_scale);

        static double NormalCdf(double x)
        {
            return (1.0 + std::erf(x / std::sqrt(2.0))) / 2.0;
        }

        static void TruncNormal_(torch::Tensor& tensor, const WeightInitConfig& config)
        {
            const double low = NormalCdf(config.trunc_a / config.trunc_std);
            const double high = NormalCdf(config.trunc_b / config.trunc_std);
            tensor.uniform_(2.0 * low - 1.0, 2.0 * high - 1.0);
            tensor.erfinv_();
            tensor.mul_(config.trunc_std * std::sqrt(2.0));
            tensor.clamp_(config.trunc_a, config.trunc_b);
        }
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
        NetworkForwardCapture* GetCapture() const { return capture_; }
        std::string_view GetCaptureBranch() const { return capture_branch_; }
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

    /// Body と Head が合成して使う具象の射影。設定型・状態型は実装に閉じる。
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


    // 子の実行interfaceをHead固有の計算へ渡し、準備結果の寿命だけを共通管理する。
    template <typename Head, auto ForwardMethod, size_t N>
    class PreparedHeadExecutor final : public NetworkHeadExecutor {
    public:
        PreparedHeadExecutor(Head& owner, PreparedModuleChildren<N> children)
            : owner_(owner), children_(std::move(children)) {}
        anet::TensorDict Forward(const anet::TensorDict& input,
            const NetworkForwardContext& context) override
        {
            return ForwardImpl(input, context, std::make_index_sequence<N>{});
        }
    private:
        template <size_t... Indices>
        anet::TensorDict ForwardImpl(const anet::TensorDict& input,
            const NetworkForwardContext& context, std::index_sequence<Indices...>)
        {
            return (owner_.*ForwardMethod)(input, context, *children_.Get(Indices)...);
        }
        Head& owner_;
        const PreparedModuleChildren<N> children_;
    };

    template <typename Head, auto ForwardMethod, size_t N>
    std::unique_ptr<NetworkHeadExecutor> MakePreparedHeadExecutor(
        Head& owner, const std::array<NetworkModule*, N>& modules,
        const NetworkForwardConfig& config, NetworkForwardClock clock)
    {
        // 子を指定順に準備し、通常動作だけならHead本体への直接呼び出しを維持する。
        PreparedModuleChildren<N> children(modules, config, clock);
        if (!children.HasPrepared()) return nullptr;
        return std::make_unique<PreparedHeadExecutor<Head, ForwardMethod, N>>(
            owner, std::move(children));
    }

    static constexpr const char* kKey_DefaultOutput = "features";

    // ===========================================================================
    // PassThroughHead
    // ===========================================================================

    class PassThroughHead : public NetworkHead {
    public:
        explicit PassThroughHead(const std::string& output_key);
        anet::TensorDict Forward(const anet::TensorDict& feature_dict, const anet::nn::NetworkForwardContext& context) override;
        std::optional<TensorDictFunction> GetTensorDictFunction(const std::string& key) override;
        HeadGraphVizInfo GetGraphVizInfo() const override;
    private:
        std::string output_key_;
    };


    // ===========================================================================
    // LinearHead
    // ===========================================================================

    class LinearHead : public NetworkHead {
    public:
        LinearHead(
            int64_t in_features,
            int64_t out_features,
            std::string output_key,
            const WeightInitConfig& init_config);

        anet::TensorDict Forward(const anet::TensorDict& feature_dict, const anet::nn::NetworkForwardContext& context) override;
        std::optional<TensorDictFunction> GetTensorDictFunction(const std::string& key) override;
        HeadGraphVizInfo GetGraphVizInfo() const override;
    private:
        torch::nn::Linear linear_{ nullptr };
        int64_t out_features_;
        std::string output_key_;
    };

    class LinearHeadFactory : public NetworkHeadFactory {
    public:
        LinearHeadFactory(
            int64_t out_features,
            std::string output_key,
            const WeightInitConfig& init_config);

        std::shared_ptr<NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    private:
        int64_t out_features_;
        std::string output_key_;
        WeightInitConfig init_config_;
    };


} // namespace anet::nn
