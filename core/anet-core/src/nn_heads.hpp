// abet/heads.hpp

#pragma once

#include "nn_impl.hpp"

namespace anet::nn {


    /// @todo RL固有実装なのでanet::rlとかに移動?

    // ===========================================================================
    // Factories
    // ===========================================================================

    class LinearHeadFactory : public NetworkHeadFactory {
    public:
        LinearHeadFactory(int64_t action_dim, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(int64_t feature_dim) const override;
    private:
        int64_t action_dim_;
        WeightInitConfig init_config_;
    };

    class DuelingHeadFactory : public NetworkHeadFactory {
    public:
        DuelingHeadFactory(int64_t action_dim, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(int64_t feature_dim) const override;
    private:
        int64_t action_dim_;
        WeightInitConfig init_config_;
    };

    class QuantileHeadFactory : public NetworkHeadFactory {
    public:
        QuantileHeadFactory(int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(int64_t feature_dim) const override;
    private:
        int64_t action_dim_;
        int64_t num_quantiles_;
        WeightInitConfig init_config_;
    };

    class QuantileDuelingHeadFactory : public NetworkHeadFactory {
    public:
        QuantileDuelingHeadFactory(int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(int64_t feature_dim) const override;
    private:
        int64_t action_dim_;
        int64_t num_quantiles_;
        WeightInitConfig init_config_;
    };

} // namespace anet::nn
