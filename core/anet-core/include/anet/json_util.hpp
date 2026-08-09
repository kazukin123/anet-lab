// anet/json_util.hpp

#pragma once

#include <nlohmann/json.hpp>

namespace anet {

    using json = nlohmann::json;

    json round_numbers(const json& j, int precision = 6);

} // namespace anet
