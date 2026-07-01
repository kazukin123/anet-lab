// nn_heads.cpp

#include "nn_heads.hpp"
#include "anet/profile.hpp"


namespace anet::nn {

// ===========================================================================
// PassThroughHead
// ===========================================================================

PassThroughHead::PassThroughHead(const std::string& output_key)
    : output_key_(output_key)
{
    // 重みを持たないため register_module 等は不要
}

anet::TensorDict PassThroughHead::Forward(const anet::TensorDict& feature_dict)
{
    ANET_PROFILE_FUNC();

    torch::Tensor x = feature_dict.At(anet::nn::kKey_DefaultOutput);
    anet::TensorDict out;
    out.Set(output_key_, x);
    return out;
}

std::optional<anet::TensorDictFunction> PassThroughHead::GetTensorDictFunction(const std::string& key)
{
    if (key == "forward" || key == output_key_) {
        return [this](const anet::TensorDict& features) -> anet::TensorDict {
            torch::Tensor x = anet::GetOrFail(features, anet::nn::kKey_DefaultOutput);
            anet::TensorDict out;
            out.Set(output_key_, x);
            return out;
            };
    }
    return std::nullopt;
}

HeadGraphVizInfo PassThroughHead::GetGraphVizInfo() const
{
    HeadGraphVizInfo info;
    info.type = "PassThroughHead";
    info.outputs.push_back({ output_key_, {} });
    return info;
}

} // namespace anet::nn
