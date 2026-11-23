// nn_head.cpp

#include "anet/nn_head.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <ATen/Dispatch.h>
#include "anet/profile.hpp"
#include "nn_impl.hpp"


namespace anet::nn {

std::optional<double> ComputeVarianceScalingBound(
    at::IntArrayRef weight_shape, at::ScalarType dtype, double variance_scale, std::string_view key)
{
    // fan_in を定義でき、uniform_ が扱える重みであることを確かめる。
    if (weight_shape.size() < 2) {
        ANET_SYSTEM_ERROR("Invalid " << key << "=" << variance_scale
            << "; expected weight rank >= 2 (output and input axes), actual rank=" << weight_shape.size() << ".");
    }
    if (dtype != at::kHalf && dtype != at::kBFloat16 && dtype != at::kFloat && dtype != at::kDouble) {
        ANET_SYSTEM_ERROR("Invalid " << key << "=" << variance_scale
            << "; expected real floating-point weight dtype Half, BFloat16, Float or Double, actual dtype=" << dtype << ".");
    }
    if (std::find(weight_shape.begin(), weight_shape.end(), 0) != weight_shape.end()) return std::nullopt;

    // Convのweightの入力軸にはグループ当たりのチャネル数が入る。
    double fan_in = 1.0;
    for (const auto size : weight_shape.slice(1)) {
        fan_in *= static_cast<double>(size);
    }
    const double bound = std::sqrt(variance_scale) * std::sqrt(3.0 / fan_in);
    return AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, dtype, WeightInitConfig::kVarianceScalingMode, [&] {
            using Limits = std::numeric_limits<scalar_t>;
            const double min_bound = static_cast<double>(Limits::denorm_min());
            // uniform_は両端の差も要素型の最大値以下でなければならない。
            const double max_bound = static_cast<double>(Limits::max()) / 2.0;
            if (!(bound >= min_bound && bound <= max_bound)) {
                ANET_SYSTEM_ERROR("Invalid " << key << "=" << variance_scale
                    << "; expected uniform bound in [" << min_bound << ", " << max_bound
                    << "] for weight dtype=" << dtype << " and fan_in=" << fan_in << ", actual bound=" << bound << ".");
            }

            // uniform_ の端点が要素型への変換で外側へ丸められないようにする。
            // 非正規化数の区間では、最小の正規化数と同じ刻み幅を使う。
            const int exponent = std::max(std::ilogb(bound), Limits::min_exponent - 1);
            const double step = std::ldexp(1.0, exponent - (Limits::digits - 1));
            return std::floor(bound / step) * step;
        });
}

void WeightInitializer::InitializeVarianceScaling(torch::Tensor& weight, const WeightInitConfig& config)
{
    // 要素型で表せる内側の端点を確定してから重みを書き込む。
    if (const auto bound = ComputeVarianceScalingBound(
        weight.sizes(), weight.scalar_type(), config.variance_scale, config.full_keys.GetFullKey("variance_scale"))) {
        torch::nn::init::uniform_(weight, -*bound, *bound);
    }
}

// ===========================================================================
// PassThroughHead
// ===========================================================================

PassThroughHead::PassThroughHead(const std::string& output_key)
    : output_key_(output_key)
{
    // 重みを持たないため register_module 等は不要
}

anet::TensorDict PassThroughHead::Forward(const anet::TensorDict& feature_dict, const anet::nn::NetworkForwardContext& context)
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

anet::TensorDict LinearHead::Forward(const anet::TensorDict& feature_dict, const anet::nn::NetworkForwardContext& context)
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
