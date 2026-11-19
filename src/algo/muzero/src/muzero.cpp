#include "anet/algo/muzero.hpp"
#include "anet/agent.hpp"
#include "muzero_proto_agent.hpp"

void anet::rl::algo::InitMuZero()
{
    anet::rl::RegisterAgentFactory<anet::rl::muzero_proto::MuZeroAgentFactory>();
}
