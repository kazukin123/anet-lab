#include "anet/env/Atari.hpp"
#include "anet/init.hpp"

#include "AtariEnv.hpp"
#include "AtariView.hpp"
#include "anet/env.hpp"
#include "anet/gui.hpp"

using namespace anet::rl::env;

void anet::rl::env::InitAtari()
{
    static const auto failure = anet::rl::detail::CaptureRegistrationFailure([] {
        RegistEnvFactory(std::make_shared<AtariEnvFactory>());
        anet::rl::gui::RegisterViewCreator<AtariViewCreator>();
    });
    if (failure) std::rethrow_exception(failure);
}
