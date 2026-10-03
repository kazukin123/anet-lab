// nn_heads.hpp

#pragma once

#include "nn_impl.hpp"

namespace anet::nn {


    static constexpr const char* kKey_DefaultOutput = "features";
    static constexpr const char* kKey_Taus = "taus";

    // ===========================================================================
    // PassThroughHead
    // ===========================================================================

    class PassThroughHead : public NetworkHead {
    public:
        explicit PassThroughHead(const std::string& output_key);
        anet::TensorDict Forward(const anet::TensorDict& feature_dict) override;
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

        anet::TensorDict Forward(const anet::TensorDict& feature_dict) override;
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
