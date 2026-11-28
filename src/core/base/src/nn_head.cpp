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


// ===========================================================================
// CompositeHead
// ===========================================================================

constexpr char kCompositeHeadType[] = "CompositeHead";
constexpr char kCompositeForwardFunction[] = "forward";

namespace {
    // 中の Head を名前つきで登録し、受け取るキーごとの値を渡して、出力を 1 つにまとめる。
    class CompositeHead final : public NetworkHead {
    public:
        struct Part {
            std::string name;
            std::string feature_key;
            std::shared_ptr<NetworkHead> head;
        };
    public:
        explicit CompositeHead(std::vector<Part> parts) : parts_(std::move(parts))
        {
            // 名前つきで登録するので、parameter の名前は head.<名前>.… になり、名前で組を分けられる。
            for (const auto& part : parts_) register_module(part.name, part.head);
        }

        anet::TensorDict Forward(const anet::TensorDict& features, const NetworkForwardContext& context) override
        {
            ANET_PROFILE_FUNC();
            anet::TensorDict result;
            for (const auto& part : parts_) {
                Merge(result, part.head->Forward(MakePartFeatures(features, part), context));
            }
            return result;
        }
        std::unique_ptr<NetworkHeadExecutor> CreateExecutor(
            const NetworkForwardConfig& config, NetworkForwardClock clock) override
        {
            // 中の Head のどれにも専用の実行器が要らなければ、本体を直接呼ぶ。
            std::vector<std::unique_ptr<NetworkHeadExecutor>> executors;
            bool has_prepared = false;
            for (const auto& part : parts_) {
                executors.push_back(part.head->CreateExecutor(config, clock));
                has_prepared = has_prepared || executors.back() != nullptr;
            }
            if (!has_prepared) return nullptr;
            return std::make_unique<Executor>(*this, std::move(executors));
        }
        void ValidateForwardReuse(const NetworkForwardConfig& source, const NetworkForwardConfig& requested) const override
        {
            for (const auto& part : parts_) part.head->ValidateForwardReuse(source, requested);
        }
        std::vector<NetworkNoisyParameterEntry> GetNoisyParameterEntries() const override
        {
            std::vector<NetworkNoisyParameterEntry> result;
            for (const auto& part : parts_) {
                auto entries = part.head->GetNoisyParameterEntries();
                result.insert(result.end(), entries.begin(), entries.end());
            }
            return result;
        }
        HeadGraphVizInfo GetGraphVizInfo() const override
        {
            HeadGraphVizInfo info;
            info.type = kCompositeHeadType;
            for (const auto& part : parts_) {
                auto part_info = part.head->GetGraphVizInfo();
                info.outputs.insert(info.outputs.end(), part_info.outputs.begin(), part_info.outputs.end());
                info.details.push_back({ part.name, part.feature_key + " -> " + part_info.type });
            }
            return info;
        }
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override
        {
            if (key != kCompositeForwardFunction) return std::nullopt;
            return [this](const anet::TensorDict& features) {
                torch::NoGradGuard no_grad;
                return Forward(features, NetworkForwardContext::GetDefault());
            };
        }
    private:
        class Executor final : public NetworkHeadExecutor {
        public:
            Executor(CompositeHead& owner, std::vector<std::unique_ptr<NetworkHeadExecutor>> executors)
                : owner_(owner), executors_(std::move(executors)) {}
            anet::TensorDict Forward(const anet::TensorDict& features, const NetworkForwardContext& context) override
            {
                ANET_PROFILE_FUNC();
                anet::TensorDict result;
                for (size_t i = 0; i < owner_.parts_.size(); ++i) {
                    const auto& part = owner_.parts_[i];
                    // 専用の実行器を持つ Head はそれを、持たない Head は本体を呼ぶ。
                    NetworkHeadExecutor& target = executors_[i] ? *executors_[i] : static_cast<NetworkHeadExecutor&>(*part.head);
                    Merge(result, target.Forward(MakePartFeatures(features, part), context));
                }
                return result;
            }
        private:
            CompositeHead& owner_; ///< 所有者の Network が、この実行器も所有する。
            std::vector<std::unique_ptr<NetworkHeadExecutor>> executors_;
        };
    private:
        static anet::TensorDict MakePartFeatures(const anet::TensorDict& features, const Part& part)
        {
            // 中の Head は、共通の入力名 features で 1 つの特徴量を受け取る。
            anet::TensorDict result;
            result.Set(kKey_DefaultOutput, anet::GetOrFail(features, part.feature_key,
                "CompositeHead expects this key in the Body outputs."));
            return result;
        }
        static void Merge(anet::TensorDict& result, const anet::TensorDict& outputs)
        {
            for (const auto& [key, value] : outputs) result.Set(key, value);
        }
    private:
        std::vector<Part> parts_;
    };
}

CompositeHeadFactory::CompositeHeadFactory(std::vector<CompositeHeadEntry> entries)
    : entries_(std::move(entries))
{
    // 組の一覧だけで決まる条件を、Network を組む前に確かめる。
    if (entries_.empty()) {
        ANET_SYSTEM_ERROR("Invalid CompositeHeadFactory entries; expected at least one entry.");
    }
    for (size_t i = 0; i < entries_.size(); ++i) {
        const auto& entry = entries_[i];
        // 名前は module の登録名になるので、区切りの「.」を含められない。
        if (entry.name.empty() || entry.name.find('.') != std::string::npos || entry.feature_key.empty()
            || !entry.factory) {
            ANET_SYSTEM_ERROR("Invalid CompositeHeadFactory entry index=" << i << " name='" << entry.name
                << "' feature_key='" << entry.feature_key
                << "'; expected a name without '.', a feature key and a Head factory.");
        }
        for (size_t j = 0; j < i; ++j) {
            if (entries_[j].name == entry.name) {
                ANET_SYSTEM_ERROR("Duplicate CompositeHeadFactory entry name='" << entry.name
                    << "'; expected unique names because they prefix the parameter names.");
            }
        }
    }
}

std::shared_ptr<NetworkHead> CompositeHeadFactory::CreateHead(const anet::TensorDict& dummy_features) const
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;
    std::vector<CompositeHead::Part> parts;
    std::vector<std::pair<std::string, std::string>> output_owners; // 出力のキーと、それを出す Head の名前。
    for (const auto& entry : entries_) {
        // 受け取るキーが Body の出力にあることを確かめ、その特徴量の形で中の Head を作る。
        const auto feature = dummy_features.Get(entry.feature_key);
        if (!feature) {
            std::string available;
            for (const auto& [key, value] : dummy_features) available += (available.empty() ? "" : ", ") + key;
            ANET_SYSTEM_ERROR("CompositeHead entry name='" << entry.name << "' feature_key='" << entry.feature_key
                << "' is missing in the Body outputs [" << available
                << "]; expected a Body output with this key.");
        }
        anet::TensorDict part_features;
        part_features.Set(kKey_DefaultOutput, *feature);
        auto head = entry.factory->CreateHead(part_features);
        if (!head) {
            ANET_SYSTEM_ERROR("CompositeHead entry name='" << entry.name << "' factory returned no Head.");
        }
        // 中の Head は CPU で作られる。仮の特徴量が CUDA にあっても仮の forward を通せるよう、特徴量の device へ移す
        // （Network の全体は、このあと NetworkBuilder が同じ device へ移す）。
        head->to(feature->device());
        // 出力のキーが重なると、あとの Head の値で黙って上書きされるので、仮の forward で確かめて止める。
        for (const auto& [key, value] : head->Forward(part_features, NetworkForwardContext::GetDefault())) {
            for (const auto& [owned_key, owner] : output_owners) {
                if (owned_key == key) {
                    ANET_SYSTEM_ERROR("Duplicate CompositeHead output key='" << key << "' from entries '" << owner
                        << "' and '" << entry.name << "'; expected distinct output keys.");
                }
            }
            output_owners.emplace_back(key, entry.name);
        }
        parts.push_back(CompositeHead::Part{ .name = entry.name, .feature_key = entry.feature_key, .head = std::move(head) });
    }
    return std::make_shared<CompositeHead>(std::move(parts));
}

} // namespace anet::nn
