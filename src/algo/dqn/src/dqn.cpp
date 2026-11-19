#include "anet/algo/dqn.hpp"
#include "anet/agent.hpp"
#include "rainbow_agent.hpp"
#include "default_dqn_agent.hpp"

void anet::rl::algo::InitDqn()
{
    anet::rl::RegisterAgentFactory<anet::rl::dqn::RainbowAgentFactory>();
    anet::rl::RegisterAgentFactory<anet::rl::dqn::DefaultDQNAgentFactory>();
}
