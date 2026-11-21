#include "anet/algo/dqn.hpp"
#include "anet/init.hpp"
#include "anet/agent.hpp"
#include "rainbow_agent.hpp"
#include "default_dqn_agent.hpp"

void anet::rl::algo::InitDqn()
{
    static const auto failure = anet::rl::detail::CaptureRegistrationFailure([] {
        anet::rl::RegisterAgentFactory<anet::rl::dqn::RainbowAgentFactory>();
        anet::rl::RegisterAgentFactory<anet::rl::dqn::DefaultDQNAgentFactory>();
    });
    if (failure) std::rethrow_exception(failure);
}
