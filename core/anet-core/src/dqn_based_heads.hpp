// dqn_based_heads.hpp

#pragma once

#include "anet/nn.hpp"

namespace anet::rl::dqn {

    // ===========================================================================
    // Head Factories
    // ===========================================================================

    class HeadFactoryBase : public anet::nn::NetworkHeadFactory {
    public:
        HeadFactoryBase(int64_t action_dim, const anet::nn::WeightInitConfig& init_config);
        virtual ~HeadFactoryBase() = default;
    protected:
        int64_t action_dim_;
        anet::nn::WeightInitConfig init_config_;

    };

    class LinearHeadFactory final : public HeadFactoryBase {
    public:
        LinearHeadFactory(int64_t action_dim, const anet::nn::WeightInitConfig& init_config);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    };

    class DuelingHeadFactory final : public HeadFactoryBase {
    public:
        DuelingHeadFactory(int64_t action_dim, const anet::nn::WeightInitConfig& init_config);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    };

    class QuantileHeadFactory final : public HeadFactoryBase {
    public:
        QuantileHeadFactory(int64_t action_dim, int64_t num_quantiles, const anet::nn::WeightInitConfig& init_config);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    private:
        int64_t num_quantiles_;
    };

    class QuantileDuelingHeadFactory final : public HeadFactoryBase {
    public:
        QuantileDuelingHeadFactory(int64_t action_dim, int64_t num_quantiles, const anet::nn::WeightInitConfig& init_config);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    private:
        int64_t num_quantiles_;
    };

    class IQNHeadFactory final : public HeadFactoryBase {
    public:
        IQNHeadFactory(int64_t action_dim, const anet::nn::WeightInitConfig& init_config);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    };

    class IQNDuelingHeadFactory final : public HeadFactoryBase {
    public:
        IQNDuelingHeadFactory(int64_t action_dim, const anet::nn::WeightInitConfig& init_config);
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    };

} // namespace anet::rl::dqn
