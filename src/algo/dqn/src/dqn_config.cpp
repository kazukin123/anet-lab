#include "dqn_config.hpp"

std::string anet::rl::dqn::GetInitialProfiledValueKey(const anet::ProfiledValueConfig<float>& config)
{
    if (config.type == "phased") {
        const auto& phase = config.phase.Get(config.phases.front());
        return phase.full_keys.GetFullKey(phase.type == "constant" ? "value" : "start");
    }
    return config.full_keys.GetFullKey(config.type == "constant" ? "value" : "start");
}
