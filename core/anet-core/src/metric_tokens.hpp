// metric_tokens.hpp
#pragma once
#include "anet/rl.hpp"

namespace anet::rl::metric_tokens {
    enum class Kind { KEY, EVENT, STEP, TARGET, SCOPE, EMA, ATTRIBUTE };
    struct Token {
        Kind kind = Kind::KEY;
        std::string raw;
        std::string value;
        std::string attribute;
        std::optional<EventType> event;
        std::optional<StepAxis> step;
        std::optional<EventField> field;
    };
    std::vector<Token> ParseMetricTokens(const std::string& definition);
}
