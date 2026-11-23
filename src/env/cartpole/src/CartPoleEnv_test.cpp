#include "anet/test/catch.hpp"

#include "CartPoleEnv.hpp"

TEST_CASE("CartPole declares the vector observation unscaled", "[forwarder]")
{
    const CartPoleEnv env(CartPoleEnvConfig{}, torch::Device(torch::kCPU), "cartpole-scale", 1);
    const auto spec = env.GetSpec();
    CHECK(spec.state_spec.obs_spec.at(anet::rl::ObsKeys::kVector).scale == anet::ValueScale::Unscaled);
}
