#include "anet/algo/ppo.hpp"
#include "anet/init.hpp"
#include "anet/agent.hpp"
#include "ppo_agent.hpp"

void anet::rl::algo::InitPpo()
{
    // 登録は 1 回だけ行う。初回の登録が失敗したら、以後の呼び出しも同じ失敗を返す。
    static const auto failure = anet::rl::detail::CaptureRegistrationFailure([] {
        anet::rl::RegisterAgentFactory<anet::rl::ppo::PPOAgentFactory>();
    });
    if (failure) std::rethrow_exception(failure);
}
