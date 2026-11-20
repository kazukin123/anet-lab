// LunarLander.cpp

#include "anet/env/DropMerge.hpp"
#include "anet/init.hpp"
#include "anet/env.hpp"
#include "anet/gui.hpp"
#include "DropMergeEnv.hpp"
#include "DropMergeView.hpp"

void anet::rl::env::InitDropMerge()
{
    static const auto failure = anet::rl::detail::CaptureRegistrationFailure([] {
        anet::rl::RegistEnvFactory(std::make_shared<anet::rl::env::drop_merge::DropMergeEnvFactory>());
        anet::rl::gui::RegisterViewCreator<DropMergeViewCreator>();
    });
    if (failure) std::rethrow_exception(failure);
}

