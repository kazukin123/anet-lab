// util.cpp

#include "anet/util.hpp"

#include <string_view>

using namespace anet;


std::optional<ScalarAggregationKey> anet::ParseScalarAggregationKey(const std::string& key)
{
    static constexpr std::pair<std::string_view, ScalarAggregation> kPrefixes[] = {
        { "mean.", ScalarAggregation::MEAN },
        { "max.", ScalarAggregation::MAX },
        { "min.", ScalarAggregation::MIN },
        { "std.", ScalarAggregation::STD },
    };
    for (const auto& [prefix, aggregation] : kPrefixes) {
        if (key.starts_with(prefix) && key.size() > prefix.size()) {
            return ScalarAggregationKey{
                .aggregation = aggregation,
                .base_key = key.substr(prefix.size()),
            };
        }
    }
    return std::nullopt;
}

void ScalarSampleAccumulator::Reset()
{
    poisoned_ = false;
    count_ = 0;
    mean_ = 0.0;
    m2_ = 0.0;
    min_ = 0.0f;
    max_ = 0.0f;
}

void ScalarSampleAccumulator::Add(std::optional<float> sample)
{
    if (!sample.has_value()) {
        poisoned_ = true;
        return;
    }
    if (std::isnan(*sample)) return;

    // Welford 法で母分散を安定して更新する。
    count_++;
    const double delta = static_cast<double>(*sample) - mean_;
    mean_ += delta / static_cast<double>(count_);
    const double delta2 = static_cast<double>(*sample) - mean_;
    m2_ += delta * delta2;
    if (count_ == 1) {
        min_ = *sample;
        max_ = *sample;
    } else {
        min_ = std::min(min_, *sample);
        max_ = std::max(max_, *sample);
    }
}

std::optional<float> ScalarSampleAccumulator::Get(ScalarAggregation aggregation) const
{
    if (poisoned_) return std::nullopt;
    if (count_ == 0 || (aggregation == ScalarAggregation::STD && count_ < 2)) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    switch (aggregation) {
    case ScalarAggregation::MEAN:
        return static_cast<float>(mean_);
    case ScalarAggregation::MAX:
        return max_;
    case ScalarAggregation::MIN:
        return min_;
    case ScalarAggregation::STD:
        return static_cast<float>(std::sqrt(m2_ / static_cast<double>(count_)));
    default:
        ANET_SYSTEM_ERROR("Unknown ScalarAggregation=" << static_cast<int>(aggregation) << ".");
    }
    return std::nullopt;
}
