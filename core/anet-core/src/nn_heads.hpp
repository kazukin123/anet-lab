// nn_heads.hpp

#pragma once

#include "nn_impl.hpp"

namespace anet::nn {


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

} // namespace anet::nn
