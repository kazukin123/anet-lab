// CartPole.cpp

#include "anet/env/CartPole.hpp"
#include "anet/env.hpp"
#include "anet/gui.hpp"
#include "CartPoleEnv.hpp"
#include "CartPoleView.hpp"

void anet::rl::env::InitCartPole()
{
	anet::rl::RegistEnvFactory(std::make_shared<CartPoleEnvFactory>());
	anet::rl::gui::RegisterViewCreator<CartPoleViewCreator>();
}

