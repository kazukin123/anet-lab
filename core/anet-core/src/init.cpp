#include "anet/init.hpp"
#include "anet/dqn_agent.hpp"
#include "anet/rainbow_agent.hpp"

using namespace anet::rl;

void anet::rl::InitRL()
{
	RegisterAgentFactory<DQNAgentFactory>();
	RegisterAgentFactory<RainbowAgentFactory>();
}
