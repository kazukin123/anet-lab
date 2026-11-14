// dqn_based_heads.hpp

#pragma once

#include <string>
#include "nn_impl.hpp"
#include "anet/agent.hpp"

namespace anet::rl::dqn {

    // ===========================================================================
    // Head Factories
    // ===========================================================================

    class HeadFactoryBase : public anet::nn::NetworkHeadFactory {
    public:
        static constexpr char kQOutputKey[] = "q";
        static constexpr char kQDistOutputKey[] = "q_dist";

        HeadFactoryBase(int64_t action_dim, const anet::nn::WeightInitConfig& init_config,
            const anet::ConfigData& projection_config = anet::EmptyConfigData);
        virtual ~HeadFactoryBase() = default;
    protected:
        int64_t action_dim_;
        anet::nn::WeightInitConfig init_config_;
        const anet::ConfigData projection_config_;

    };

    /// 既存の2つのHeadを同じNetworkの中で実行し、出力の意味を分ける。
    class DecomposedHeadFactory final : public anet::nn::NetworkHeadFactory {
    public:
        static constexpr char kExtrinsicHeadName[] = "extrinsic";
        static constexpr char kIntrinsicHeadName[] = "intrinsic";
        static constexpr char kHeadKeySeparator[] = "_";
        static constexpr char kHeadParameterPrefix[] = "head.";
        static constexpr char kModuleSeparator[] = ".";
        inline static const std::string kExtrinsicOutputPrefix = std::string(kExtrinsicHeadName) + kHeadKeySeparator;
        inline static const std::string kIntrinsicOutputPrefix = std::string(kIntrinsicHeadName) + kHeadKeySeparator;
        inline static const std::string kExtrinsicQOutputKey = kExtrinsicOutputPrefix + HeadFactoryBase::kQOutputKey;
        inline static const std::string kIntrinsicQOutputKey = kIntrinsicOutputPrefix + HeadFactoryBase::kQOutputKey;
        inline static const std::string kExtrinsicQDistOutputKey = kExtrinsicOutputPrefix + HeadFactoryBase::kQDistOutputKey;
        inline static const std::string kIntrinsicQDistOutputKey = kIntrinsicOutputPrefix + HeadFactoryBase::kQDistOutputKey;
        inline static const std::string kExtrinsicParameterPrefix = std::string(kHeadParameterPrefix) + kExtrinsicHeadName + kModuleSeparator;
        inline static const std::string kIntrinsicParameterPrefix = std::string(kHeadParameterPrefix) + kIntrinsicHeadName + kModuleSeparator;

        DecomposedHeadFactory(std::shared_ptr<anet::nn::NetworkHeadFactory> extrinsic,
            std::shared_ptr<anet::nn::NetworkHeadFactory> intrinsic, bool dedicated, bool intrinsic_dueling);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    private:
        std::shared_ptr<anet::nn::NetworkHeadFactory> extrinsic_, intrinsic_;
        bool dedicated_, intrinsic_dueling_;
    };

    class LinearHeadFactory final : public HeadFactoryBase {
    public:
        LinearHeadFactory(int64_t action_dim, const anet::nn::WeightInitConfig& init_config,
            const anet::ConfigData& projection_config = anet::EmptyConfigData);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    };

    /// skip の射影は primary と同じ分布表現を使い、行動・継続長の軸を公開する。
    class SkipHeadFactory final : public HeadFactoryBase {
    public:
        static constexpr const char* kKeyQ = "skip_q";
        static constexpr const char* kKeyQDist = "skip_q_dist";

        SkipHeadFactory(int64_t action_dim, int max_length, SkipNetworkMode network_mode,
            std::string quantile_mode, int num_quantiles, const anet::nn::WeightInitConfig& init_config,
            const anet::ConfigData& projection_config = anet::EmptyConfigData);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    private:
        int max_length_;
        SkipNetworkMode network_mode_;
        std::string quantile_mode_;
        int num_quantiles_;
    };

    class DuelingHeadFactory final : public HeadFactoryBase {
    public:
        DuelingHeadFactory(int64_t action_dim, const anet::nn::WeightInitConfig& init_config,
            const anet::ConfigData& projection_config = anet::EmptyConfigData);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    };

    class QuantileHeadFactory final : public HeadFactoryBase {
    public:
        QuantileHeadFactory(int64_t action_dim, int64_t num_quantiles, const anet::nn::WeightInitConfig& init_config,
            const anet::ConfigData& projection_config = anet::EmptyConfigData);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    private:
        int64_t num_quantiles_;
    };

    class QuantileDuelingHeadFactory final : public HeadFactoryBase {
    public:
        QuantileDuelingHeadFactory(int64_t action_dim, int64_t num_quantiles, const anet::nn::WeightInitConfig& init_config,
            const anet::ConfigData& projection_config = anet::EmptyConfigData);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    private:
        int64_t num_quantiles_;
    };

    class IQNHeadFactory final : public HeadFactoryBase {
    public:
        IQNHeadFactory(int64_t action_dim, const anet::nn::WeightInitConfig& init_config,
            const anet::ConfigData& projection_config = anet::EmptyConfigData);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    };

    class IQNDuelingHeadFactory final : public HeadFactoryBase {
    public:
        IQNDuelingHeadFactory(int64_t action_dim, const anet::nn::WeightInitConfig& init_config,
            const anet::ConfigData& projection_config = anet::EmptyConfigData);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    };

} // namespace anet::rl::dqn
