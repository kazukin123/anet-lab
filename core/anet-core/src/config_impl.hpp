#pragma once

#include "anet/config.hpp"

namespace anet::detail {

    struct ConfigResolverResult {
        ConfigData::MapType effective_map;
        json resolution_json;
    };

    class ResolutionEngine;

    class ConfigResolver {
    public:
        static ConfigResolverResult Resolve(const ConfigData::MapType& source_map, const ConfigData::MapType& cli_overrides);
    };

} // namespace anet::detail
