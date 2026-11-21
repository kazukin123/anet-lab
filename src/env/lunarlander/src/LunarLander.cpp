// LunarLander.cpp

#include "anet/env/LunarLander.hpp"
#include "anet/init.hpp"
#include "anet/env.hpp"
#include "anet/gui.hpp"
#include "LunarLanderEnv.hpp"
#include "LunarLanderView.hpp"

void anet::rl::env::InitLunarLander()
{
    static const auto failure = anet::rl::detail::CaptureRegistrationFailure([] {
        anet::rl::RegistEnvFactory(std::make_shared<LunarLanderEnvFactory>());
        anet::rl::gui::RegisterViewCreator<LunarLanderViewCreator>();
    });
    if (failure) std::rethrow_exception(failure);
}

