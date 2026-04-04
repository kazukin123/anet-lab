// nn_heads.hpp

#pragma once

#include "nn_impl.hpp"

namespace anet::nn {


    /// @todo RL固有実装はAgent個別実装に移動

    static constexpr const char* kKey_DefaultOutput = "features";

    // ===========================================================================
    // PassThroughHead
    // ===========================================================================

    class PassThroughHead : public NetworkHead {
    public:
        explicit PassThroughHead(const std::string& output_key);
        anet::TensorDict Forward(const anet::TensorDict& feature_dict) override;
        std::optional<TensorDictFunction> GetTensorDictFunction(const std::string& key) override;
    private:
        std::string output_key_;
    };


    // ===========================================================================
    // Head Factories
    // ===========================================================================

    class HeadFactoryBase : public NetworkHeadFactory {
    public:
        HeadFactoryBase(int64_t action_dim, const WeightInitConfig& init_config);
        virtual ~HeadFactoryBase() = default;
    protected:
        int64_t action_dim_;
        WeightInitConfig init_config_;

    };

    class LinearHeadFactory final : public HeadFactoryBase {
    public:
        LinearHeadFactory(int64_t action_dim, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    };

    class DuelingHeadFactory final : public HeadFactoryBase {
    public:
        DuelingHeadFactory(int64_t action_dim, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    };

    class QuantileHeadFactory final : public HeadFactoryBase {
    public:
        QuantileHeadFactory(int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    private:
        int64_t num_quantiles_;
    };

    class QuantileDuelingHeadFactory final : public HeadFactoryBase {
    public:
        QuantileDuelingHeadFactory(int64_t action_dim, int64_t num_quantiles, const WeightInitConfig& init_config);
        std::shared_ptr<NetworkHead> CreateHead(const anet::TensorDict& dummy_features) const override;
    private:
        int64_t num_quantiles_;
    };

} // namespace anet::nn
