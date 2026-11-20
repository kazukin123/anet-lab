// CartPole.cpp

#include "anet/env/CartPole.hpp"
#include "anet/init.hpp"
#include "anet/env.hpp"
#include "anet/gui.hpp"
#include "CartPoleEnv.hpp"
#include "CartPoleView.hpp"

void anet::rl::env::InitCartPole()
{
    static const auto failure = anet::rl::detail::CaptureRegistrationFailure([] {
        anet::rl::RegistEnvFactory(std::make_shared<CartPoleEnvFactory>());
        anet::rl::gui::RegisterViewCreator<CartPoleViewCreator>();
    });
    if (failure) std::rethrow_exception(failure);
}

