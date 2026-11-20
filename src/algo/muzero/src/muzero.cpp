#include "anet/algo/muzero.hpp"
#include "anet/init.hpp"
#include "anet/agent.hpp"
#include "muzero_proto_agent.hpp"

void anet::rl::algo::InitMuZero()
{
    static const auto failure = anet::rl::detail::CaptureRegistrationFailure([] {
        anet::rl::RegisterAgentFactory<anet::rl::muzero_proto::MuZeroAgentFactory>();
    });
    if (failure) std::rethrow_exception(failure);
}
