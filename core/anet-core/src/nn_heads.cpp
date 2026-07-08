// nn_heads.cpp

#include "nn_heads.hpp"
#include <utility>
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


// ===========================================================================
// LinearHead
// ===========================================================================

LinearHead::LinearHead(
    int64_t in_features,
    int64_t out_features,
    std::string output_key,
    const WeightInitConfig& init_config)
    : out_features_(out_features)
    , output_key_(std::move(output_key))
{
    // Body が作った特徴量を、タスク固有の出力キーへ射影する最終層を登録する。
    torch::nn::LinearOptions opts(in_features, out_features);
    opts.bias(true);
    linear_ = register_module("linear", torch::nn::Linear(opts));
    WeightInitializer::Initialize(linear_, init_config);
}

anet::TensorDict LinearHead::Forward(const anet::TensorDict& feature_dict)
{
    ANET_PROFILE_FUNC();

    // 共通 head 入力キーから特徴量を取り出し、呼び出し側が指定した出力名で返す。
    torch::Tensor x = feature_dict.At(anet::nn::kKey_DefaultOutput);
    anet::TensorDict out;
    out.Set(output_key_, linear_->forward(x));
    return out;
}

std::optional<anet::TensorDictFunction> LinearHead::GetTensorDictFunction(const std::string& key)
{
    if (key == "forward" || key == output_key_) {
        return [this](const anet::TensorDict& features) -> anet::TensorDict {
            torch::NoGradGuard no_grad;
            torch::Tensor x = anet::GetOrFail(
                features,
                anet::nn::kKey_DefaultOutput,
                "Please ensure 'net.body.output.[features]' is properly configured.");
            anet::TensorDict out;
            out.Set(output_key_, linear_->forward(x));
            return out;
            };
    }
    return std::nullopt;
}

HeadGraphVizInfo LinearHead::GetGraphVizInfo() const
{
    HeadGraphVizInfo info;
    info.type = "LinearHead";
    info.outputs.push_back({ output_key_, { out_features_ } });
    info.details.push_back({ "out_features", std::to_string(out_features_) });
    return info;
}

LinearHeadFactory::LinearHeadFactory(
    int64_t out_features,
    std::string output_key,
    const WeightInitConfig& init_config)
    : out_features_(out_features)
    , output_key_(std::move(output_key))
    , init_config_(init_config)
{
}

std::shared_ptr<NetworkHead> LinearHeadFactory::CreateHead(const anet::TensorDict& dummy_features) const
{
    // dummy forward の特徴量 shape から、最終次元を Linear 入力次元として確定する。
    torch::Tensor t = anet::GetOrFail(
        dummy_features,
        anet::nn::kKey_DefaultOutput,
        "Please ensure 'net.body.output.[features]' is properly configured.");
    const int64_t input_dim = t.size(-1);
    return std::make_shared<LinearHead>(input_dim, out_features_, output_key_, init_config_);
}

} // namespace anet::nn
