#include "anet/env/Atari.hpp"

#include "AtariEnv.hpp"
#include "AtariView.hpp"
#include "anet/env.hpp"
#include "anet/gui.hpp"

using namespace anet::rl::env;

void anet::rl::env::InitAtari()
{
    RegistEnvFactory(std::make_shared<AtariEnvFactory>());
    anet::rl::gui::RegisterViewCreator<AtariViewCreator>();
}
