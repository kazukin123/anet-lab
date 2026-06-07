// dqn_based_heads.cpp

#include "dqn_based_heads.hpp"
#include "anet/profile.hpp"
#include "nn_heads.hpp"

#include <string>
#include <utility>


using namespace anet::rl::dqn;

static constexpr const char* kKey_ValueFeature = "value_feature";
static constexpr const char* kKey_AdvFeature = "adv_feature";

// Bodyから渡されたTensorDictから、指定キーの特徴量を抽出する安全なヘルパー
static torch::Tensor GetFeature(const anet::TensorDict& feature_dict, const std::string& key)
{
    auto opt = feature_dict.Get(key);
    if (!opt) {
        // 設定ファイルでの net.body.output.[feature]=XXX の指定漏れを親切に案内する
        ANET_SYSTEM_ERROR("NetworkHead expected key '" << key << "' in TensorDict, but it was not found. "
            "Please ensure 'net.body.output.[" << key << "]' is properly configured.");
        return torch::Tensor(); // 到達不可
    }
    return *opt;
}


// ===========================================================================
// LinearHead (Standard Q-Network Head)
// ===========================================================================

class LinearHead : public anet::nn::NetworkHead {
public:
    LinearHead(int64_t in_features, int64_t out_features, const anet::nn::WeightInitConfig& init_config)
        : out_features_(out_features)
    {
        torch::nn::LinearOptions opts(in_features, out_features);
        opts.bias(true);

        linear_ = register_module("linear", torch::nn::Linear(opts));
        anet::nn::WeightInitializer::Initialize(linear_, init_config);
    }

    anet::TensorDict Forward(const anet::TensorDict& feature_dict) override
    {
        ANET_PROFILE_FUNC();

        torch::Tensor x = feature_dict.At(anet::nn::kKey_DefaultOutput);
        anet::TensorDict out;
        out.Set("q", linear_->forward(x));
        return out;
    }

    std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override
    {
        if (key == "forward" || key == "forward.q" || key == "q_values") {
            return [this](const anet::TensorDict& features) -> anet::TensorDict {
                torch::NoGradGuard no_grad;
                torch::Tensor x = anet::GetOrFail(features, anet::nn::kKey_DefaultOutput);
                anet::TensorDict out;
                out.Set("q", linear_->forward(x));
                return out;
                };
        }
        return std::nullopt;
    }

    anet::nn::HeadGraphVizInfo GetGraphVizInfo() const override
    {
        anet::nn::HeadGraphVizInfo info;
        info.type = "LinearHead";
        info.outputs.push_back({ "q", { out_features_ } });
        info.details.push_back({ "action_dim", std::to_string(out_features_) });
        return info;
    }

private:
    torch::nn::Linear linear_{ nullptr };
    int64_t out_features_;
};


// ===========================================================================
// DuelingHead (V + A)
// ===========================================================================

class DuelingHead : public anet::nn::NetworkHead {
public:
    DuelingHead(
        int64_t value_in_features,
        int64_t adv_in_features,
        int64_t action_dim,
        std::string value_key,
        std::string adv_key,
        const anet::nn::WeightInitConfig& init_config)
        : action_dim_(action_dim)
        , value_key_(std::move(value_key))
        , adv_key_(std::move(adv_key))
    {
        torch::nn::LinearOptions v_opts(value_in_features, 1);
        v_opts.bias(true);
        value_ = register_module("value", torch::nn::Linear(v_opts));

        torch::nn::LinearOptions a_opts(adv_in_features, action_dim);
        a_opts.bias(true);
        adv_ = register_module("adv", torch::nn::Linear(a_opts));

        anet::nn::WeightInitializer::Initialize(value_, init_config);
        anet::nn::WeightInitializer::Initialize(adv_, init_config);
    }

    anet::TensorDict Forward(const anet::TensorDict& feature_dict) override
    {
        ANET_PROFILE_FUNC();

        torch::Tensor value_x = feature_dict.At(value_key_);
        torch::Tensor adv_x = feature_dict.At(adv_key_);

        auto v = value_->forward(value_x); // (B, 1)
        auto a = adv_->forward(adv_x);     // (B, A)
        auto a_mean = a.mean(/*dim=*/1, /*keepdim=*/true);
        auto q = v + (a - a_mean);

        anet::TensorDict out;
        out.Set("q", q);
        out.Set("v", v);
        out.Set("a", a);
        return out;
    }

    std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override
    {
        if (key == "forward" || key == "forward.q" || key == "q_values") {
            return [this](const anet::TensorDict& features) -> anet::TensorDict {
                torch::NoGradGuard no_grad;
                torch::Tensor value_x = GetFeature(features, value_key_);
                torch::Tensor adv_x = GetFeature(features, adv_key_);
                auto v = value_->forward(value_x);
                auto a = adv_->forward(adv_x);
                anet::TensorDict out;
                out.Set("q", v + (a - a.mean(1, true)));
                return out;
                };
        }
        if (key == "forward.v" || key == "v_values") {
            return [this](const anet::TensorDict& features) -> anet::TensorDict {
                torch::NoGradGuard no_grad;
                torch::Tensor x = GetFeature(features, value_key_);
                anet::TensorDict out;
                out.Set("v", value_->forward(x));
                return out;
                };
        }
        if (key == "forward.a" || key == "a_values") {
            return [this](const anet::TensorDict& features) -> anet::TensorDict {
                torch::NoGradGuard no_grad;
                torch::Tensor x = GetFeature(features, adv_key_);
                anet::TensorDict out;
                out.Set("a", adv_->forward(x));
                return out;
                };
        }
        return std::nullopt;
    }

    anet::nn::HeadGraphVizInfo GetGraphVizInfo() const override
    {
        anet::nn::HeadGraphVizInfo info;
        info.type = "DuelingHead";
        info.outputs.push_back({ "q", { action_dim_ } });
        info.outputs.push_back({ "v", { 1 } });
        info.outputs.push_back({ "a", { action_dim_ } });
        info.details.push_back({ "action_dim", std::to_string(action_dim_) });
        info.details.push_back({ "streams", "value, adv" });
        info.details.push_back({ "mode", value_key_ == adv_key_ ? "shared" : "branched" });
        info.details.push_back({ "value_input_key", value_key_ });
        info.details.push_back({ "adv_input_key", adv_key_ });
        return info;
    }

private:
    torch::nn::Linear value_{ nullptr };
    torch::nn::Linear adv_{ nullptr };
    int64_t action_dim_;
    std::string value_key_;
    std::string adv_key_;
};


// ===========================================================================
// QuantileHead (QR-DQN Plain)
// ===========================================================================

class QuantileHead : public anet::nn::NetworkHead {
public:
    QuantileHead(int64_t in_features, int64_t action_dim, int64_t num_quantiles, const anet::nn::WeightInitConfig& init_config)
        : action_dim_(action_dim), num_quantiles_(num_quantiles)
    {
        int64_t out_features = action_dim * num_quantiles;

        torch::nn::LinearOptions opts(in_features, out_features);
        opts.bias(true);
        linear_ = register_module("linear", torch::nn::Linear(opts));

        anet::nn::WeightInitializer::Initialize(linear_, init_config);
    }

    anet::TensorDict Forward(const anet::TensorDict& feature_dict) override
    {
        ANET_PROFILE_FUNC();
        torch::Tensor x = feature_dict.At(anet::nn::kKey_DefaultOutput);

        auto flat = linear_->forward(x); // (B, A*N)
        auto batch_size = flat.size(0);
        auto q_dist = flat.view({ batch_size, action_dim_, num_quantiles_ });
        auto q = q_dist.mean(2);

        anet::TensorDict out;
        out.Set("q_dist", q_dist);  // (B, A, N)
        out.Set("q", q);            // (B, A)
        return out;
    }

    std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override
    {
        if (key == "forward" || key == "forward.q" || key == "q_values") {
            return [this](const anet::TensorDict& features) -> anet::TensorDict {
                torch::NoGradGuard no_grad;
                torch::Tensor x = anet::GetOrFail(features, anet::nn::kKey_DefaultOutput);
                auto flat = linear_->forward(x);
                anet::TensorDict out;
                out.Set("q", flat.view({ flat.size(0), action_dim_, num_quantiles_ }).mean(2));
                return out;
                };
        }
        if (key == "forward.dist" || key == "distributions") {
            return [this](const anet::TensorDict& features) -> anet::TensorDict {
                torch::NoGradGuard no_grad;
                torch::Tensor x = anet::GetOrFail(features, anet::nn::kKey_DefaultOutput);
                auto flat = linear_->forward(x);
                anet::TensorDict out;
                out.Set("q_dist", flat.view({ flat.size(0), action_dim_, num_quantiles_ }));
                return out;
                };
        }
        return std::nullopt;
    }

    anet::nn::HeadGraphVizInfo GetGraphVizInfo() const override
    {
        anet::nn::HeadGraphVizInfo info;
        info.type = "QuantileHead";
        info.outputs.push_back({ "q", { action_dim_ } });
        info.outputs.push_back({ "q_dist", { action_dim_, num_quantiles_ } });
        info.details.push_back({ "action_dim", std::to_string(action_dim_) });
        info.details.push_back({ "num_quantiles", std::to_string(num_quantiles_) });
        return info;
    }

private:
    torch::nn::Linear linear_{ nullptr };
    int64_t action_dim_;
    int64_t num_quantiles_;
};


// ===========================================================================
// QuantileDuelingHead (QR-DQN + Dueling)
// ===========================================================================

class QuantileDuelingHead : public anet::nn::NetworkHead {
public:
    QuantileDuelingHead(
        int64_t value_in_features,
        int64_t adv_in_features,
        int64_t action_dim,
        int64_t num_quantiles,
        std::string value_key,
        std::string adv_key,
        const anet::nn::WeightInitConfig& init_config)
        : action_dim_(action_dim)
        , num_quantiles_(num_quantiles)
        , value_key_(std::move(value_key))
        , adv_key_(std::move(adv_key))
    {
        value_ = register_module("value", torch::nn::Linear(value_in_features, num_quantiles));
        adv_ = register_module("adv", torch::nn::Linear(adv_in_features, action_dim * num_quantiles));

        anet::nn::WeightInitializer::Initialize(value_, init_config);
        anet::nn::WeightInitializer::Initialize(adv_, init_config);
    }

    anet::TensorDict Forward(const anet::TensorDict& feature_dict) override
    {
        ANET_PROFILE_FUNC();
        torch::Tensor value_x = feature_dict.At(value_key_);
        torch::Tensor adv_x = feature_dict.At(adv_key_);

        auto batch_size = value_x.size(0);

        // V: (B, N) -> (B, 1, N)
        auto v = value_->forward(value_x).view({ batch_size, 1, num_quantiles_ });

        // A: (B, A*N) -> (B, A, N)
        auto a = adv_->forward(adv_x).view({ adv_x.size(0), action_dim_, num_quantiles_ });

        // Mean A over actions: (B, 1, N)
        auto a_mean = a.mean(1, true);

        // Q = V + (A - mean(A)) -> (B, A, N)
        auto q_dist = v + (a - a_mean);

        // Q分布の平均としてQ値を算出
        auto q = q_dist.mean(2); // (B, A)

        // 結果を返す
        anet::TensorDict out;
        out.Set("q_dist", q_dist);
        out.Set("q", q);
        out.Set("v_dist", v);
        out.Set("a_dist", a);
        return out;
    }

    std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override
    {
        if (key == "forward" || key == "forward.q" || key == "q_values") {
            return [this](const anet::TensorDict& features) -> anet::TensorDict {
                torch::NoGradGuard no_grad;
                torch::Tensor value_x = GetFeature(features, value_key_);
                torch::Tensor adv_x = GetFeature(features, adv_key_);
                auto v = value_->forward(value_x).view({ value_x.size(0), 1, num_quantiles_ });
                auto a = adv_->forward(adv_x).view({ adv_x.size(0), action_dim_, num_quantiles_ });
                auto q_dist = v + (a - a.mean(1, true));
                anet::TensorDict out;
                out.Set("q", q_dist.mean(2));
                return out;
                };
        }
        if (key == "forward.dist" || key == "distributions") {
            return [this](const anet::TensorDict& features) -> anet::TensorDict {
                torch::NoGradGuard no_grad;
                torch::Tensor value_x = GetFeature(features, value_key_);
                torch::Tensor adv_x = GetFeature(features, adv_key_);
                auto v = value_->forward(value_x).view({ value_x.size(0), 1, num_quantiles_ });
                auto a = adv_->forward(adv_x).view({ adv_x.size(0), action_dim_, num_quantiles_ });
                anet::TensorDict out;
                out.Set("q_dist", v + (a - a.mean(1, true)));
                return out;
                };
        }
        if (key == "forward.v" || key == "v_values") {
            return [this](const anet::TensorDict& features) -> anet::TensorDict {
                torch::NoGradGuard no_grad;
                torch::Tensor x = GetFeature(features, value_key_);
                auto v = value_->forward(x);
                anet::TensorDict out;
                out.Set("v_dist", v.view({ v.size(0), 1, num_quantiles_ }));
                return out;
                };
        }
        if (key == "forward.a" || key == "a_values") {
            return [this](const anet::TensorDict& features) -> anet::TensorDict {
                torch::NoGradGuard no_grad;
                torch::Tensor x = GetFeature(features, adv_key_);
                auto a = adv_->forward(x);
                anet::TensorDict out;
                out.Set("a_dist", a.view({ a.size(0), action_dim_, num_quantiles_ }));
                return out;
                };
        }
        return std::nullopt;
    }

    anet::nn::HeadGraphVizInfo GetGraphVizInfo() const override
    {
        anet::nn::HeadGraphVizInfo info;
        info.type = "QuantileDuelingHead";
        info.outputs.push_back({ "q", { action_dim_ } });
        info.outputs.push_back({ "q_dist", { action_dim_, num_quantiles_ } });
        info.outputs.push_back({ "v_dist", { 1, num_quantiles_ } });
        info.outputs.push_back({ "a_dist", { action_dim_, num_quantiles_ } });
        info.details.push_back({ "action_dim", std::to_string(action_dim_) });
        info.details.push_back({ "num_quantiles", std::to_string(num_quantiles_) });
        info.details.push_back({ "streams", "value, adv" });
        info.details.push_back({ "mode", value_key_ == adv_key_ ? "shared" : "branched" });
        info.details.push_back({ "value_input_key", value_key_ });
        info.details.push_back({ "adv_input_key", adv_key_ });
        return info;
    }

private:
    torch::nn::Linear value_{ nullptr };
    torch::nn::Linear adv_{ nullptr };
    int64_t action_dim_;
    int64_t num_quantiles_;
    std::string value_key_;
    std::string adv_key_;
};

// ===========================================================================
// Factories
// ===========================================================================

HeadFactoryBase::HeadFactoryBase(int64_t action_dim, const anet::nn::WeightInitConfig& init_config)
    : action_dim_(action_dim), init_config_(init_config)
{
}

LinearHeadFactory::LinearHeadFactory(int64_t action_dim, const anet::nn::WeightInitConfig& init_config)
    : HeadFactoryBase(action_dim, init_config)
{
}

std::shared_ptr<anet::nn::NetworkHead> LinearHeadFactory::CreateHead(const anet::TensorDict& dummy_features) const
{
    torch::Tensor t = GetFeature(dummy_features, anet::nn::kKey_DefaultOutput);
    int64_t input_dim = t.size(-1); // Flattenされている前提で最終次元を取得
    return std::make_shared<LinearHead>(input_dim, action_dim_, init_config_);
}

DuelingHeadFactory::DuelingHeadFactory(int64_t action_dim, const anet::nn::WeightInitConfig& init_config)
    : HeadFactoryBase(action_dim, init_config)
{
}

std::shared_ptr<anet::nn::NetworkHead> DuelingHeadFactory::CreateHead(const anet::TensorDict& dummy_features) const
{
    auto value_feature = dummy_features.Get(kKey_ValueFeature);
    auto adv_feature = dummy_features.Get(kKey_AdvFeature);
    if (value_feature && adv_feature) {
        return std::make_shared<DuelingHead>(
            value_feature->size(-1), adv_feature->size(-1), action_dim_, kKey_ValueFeature, kKey_AdvFeature, init_config_);
    }
    if (value_feature || adv_feature) {
        ANET_SYSTEM_ERROR("DuelingHead requires both 'value_feature' and 'adv_feature', or neither. "
            "Please configure both 'net.body.output.[value_feature]' and 'net.body.output.[adv_feature]'.");
    }

    torch::Tensor t = GetFeature(dummy_features, anet::nn::kKey_DefaultOutput);
    int64_t input_dim = t.size(-1);
    return std::make_shared<DuelingHead>(
        input_dim, input_dim, action_dim_, anet::nn::kKey_DefaultOutput, anet::nn::kKey_DefaultOutput, init_config_);
}

QuantileHeadFactory::QuantileHeadFactory(int64_t action_dim, int64_t num_quantiles, const anet::nn::WeightInitConfig& init_config)
    : HeadFactoryBase(action_dim, init_config)
    , num_quantiles_(num_quantiles)
{
}

std::shared_ptr<anet::nn::NetworkHead> QuantileHeadFactory::CreateHead(const anet::TensorDict& dummy_features) const
{
    torch::Tensor t = GetFeature(dummy_features, anet::nn::kKey_DefaultOutput);
    int64_t input_dim = t.size(-1);
    return std::make_shared<QuantileHead>(input_dim, action_dim_, num_quantiles_, init_config_);
}

QuantileDuelingHeadFactory::QuantileDuelingHeadFactory(int64_t action_dim, int64_t num_quantiles, const anet::nn::WeightInitConfig& init_config)
    : HeadFactoryBase(action_dim, init_config)
    , num_quantiles_(num_quantiles)
{
}

std::shared_ptr<anet::nn::NetworkHead> QuantileDuelingHeadFactory::CreateHead(const anet::TensorDict& dummy_features) const
{
    auto value_feature = dummy_features.Get(kKey_ValueFeature);
    auto adv_feature = dummy_features.Get(kKey_AdvFeature);
    if (value_feature && adv_feature) {
        return std::make_shared<QuantileDuelingHead>(
            value_feature->size(-1), adv_feature->size(-1),
            action_dim_, num_quantiles_, kKey_ValueFeature, kKey_AdvFeature, init_config_);
    }
    if (value_feature || adv_feature) {
        ANET_SYSTEM_ERROR("QuantileDuelingHead requires both 'value_feature' and 'adv_feature', or neither. "
            "Please configure both 'net.body.output.[value_feature]' and 'net.body.output.[adv_feature]'.");
    }

    torch::Tensor t = GetFeature(dummy_features, anet::nn::kKey_DefaultOutput);
    int64_t input_dim = t.size(-1);
    return std::make_shared<QuantileDuelingHead>(
        input_dim, input_dim, action_dim_, num_quantiles_,
        anet::nn::kKey_DefaultOutput, anet::nn::kKey_DefaultOutput, init_config_);
}
