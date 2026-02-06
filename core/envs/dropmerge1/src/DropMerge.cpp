// LunarLander.cpp

#include "anet/env/DropMerge.hpp"
#include "anet/env.hpp"
#include "anet/gui.hpp"
#include "DropMergeEnv.hpp"
#include "DropMergeView.hpp"

void anet::rl::env::InitDropMerge()
{
	anet::rl::RegistEnvFactory(std::make_shared<DropMergeEnvFactory>());
	anet::rl::gui::RegisterViewCreator<DropMergeViewCreator>();
}

