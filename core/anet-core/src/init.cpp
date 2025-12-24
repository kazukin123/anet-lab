#include "anet/init.hpp"
#include "anet/dqn_agent.hpp"
#include "anet/rainbow_agent.hpp"
#include "anet/default_dqn_agent.hpp"

using namespace anet::rl;

void anet::rl::InitRL()
{
	RegisterAgentFactory<DQNAgentFactory>();
	RegisterAgentFactory<anet::rl::dqn::RainbowAgentFactory>();
	RegisterAgentFactory<anet::rl::dqn::DefaultDQNAgentFactory>();
}
