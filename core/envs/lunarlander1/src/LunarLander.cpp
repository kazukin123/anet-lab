
#include "anet/env/LunarLander.hpp"
#include "anet/env.hpp"
#include "anet/gui.hpp"
#include "LunarLanderEnv.hpp"
#include "LunarLanderView.hpp"

void anet::rl::env::InitLunarLander()
{
	anet::rl::RegistEnvFactory(std::make_shared<LunarLanderEnvFactory>());
	anet::rl::gui::RegisterViewCreator<LunarLanderViewCreator>();
}

