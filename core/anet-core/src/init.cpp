#include "anet/init.hpp"
#include "anet/dqn_agent.hpp"

using namespace anet::rl;

void anet::rl::InitRL()
{
	RegisterAgentFactory<DQNAgentFactory>();
}
