#include "anet/catch_test.hpp"
#include "anet/nn_util.hpp"
#include <ATen/CPUGeneratorImpl.h>

TEST_CASE("Gradient noise uses pre-update Adam geometry without changing gradients or state", "[nn][gradnoise]")
{
    auto feature = torch::zeros({}, torch::requires_grad());
    auto readout = torch::zeros({}, torch::requires_grad());
    auto unused = torch::zeros({}, torch::requires_grad());
    torch::optim::AdamW optimizer(
        { feature, readout, unused }, torch::optim::AdamWOptions(0.01).betas(std::make_tuple(0.9, 0.5)).eps(0.25));
    for (const auto& parameter : { feature, readout }) {
        auto state = std::make_unique<torch::optim::AdamWParamState>();
        state->step(2);
        state->exp_avg(torch::zeros_like(parameter));
        state->exp_avg_sq(torch::full_like(parameter, parameter.is_same(feature) ? 0.046875 : 2.296875));
        optimizer.state()[parameter.unsafeGetTensorImpl()] = std::move(state);
        parameter.mutable_grad() = torch::full_like(parameter, 99);
    }
    unused.mutable_grad() = torch::full_like(unused, 999);
    const auto losses = feature * torch::tensor({ 1.0f, 3.0f, 5.0f, 7.0f }) + readout * torch::tensor({ 2.0f, 4.0f, 6.0f, 8.0f });
    const auto result = anet::MeasureGradientNoise(losses,
        { { .parameter = feature, .feature = true }, { .parameter = readout, .feature = false }, { .parameter = unused, .feature = true } },
        optimizer, 2, 0.25, { .raw = true, .adam = { true, true, true } });
    CHECK(result.raw.signal == Catch::Approx(33));
    CHECK(result.raw.variance == Catch::Approx(32));
    CHECK(result.adam[0].signal == Catch::Approx(53.25));
    CHECK(result.adam[0].variance == Catch::Approx(68));
    CHECK(result.adam[1].signal == Catch::Approx(48));
    CHECK(result.adam[2].signal == Catch::Approx(5.25));
    CHECK(result.adam[1].variance + result.adam[2].variance == Catch::Approx(result.adam[0].variance));
    CHECK(feature.grad().item<float>() == 99);
    CHECK(readout.grad().item<float>() == 99);
    CHECK(optimizer.state().size() == 2);
    CHECK(static_cast<const torch::optim::AdamWParamState&>(*optimizer.state().at(feature.unsafeGetTensorImpl())).step() == 2);
    CHECK(unused.grad().item<float>() == 999);
    for (const auto& parameter : { feature, readout }) {
        auto& state = static_cast<torch::optim::AdamWParamState&>(*optimizer.state().at(parameter.unsafeGetTensorImpl()));
        state.exp_avg_sq().fill_(0.421875);
    }
    const auto identity =
        anet::MeasureGradientNoise(losses, { { .parameter = feature, .feature = true }, { .parameter = readout, .feature = false } },
            optimizer, 2, 0.25, { .raw = true, .adam = { true, false, false } });
    CHECK(identity.adam[0].signal == Catch::Approx(identity.raw.signal));
    CHECK(identity.adam[0].variance == Catch::Approx(identity.raw.variance));
    losses.mean().backward();
    CHECK(feature.grad().item<float>() == Catch::Approx(103));
    CHECK(readout.grad().item<float>() == Catch::Approx(104));
}

TEST_CASE("Gradient noise keeps group-specific validity and skips unavailable EMA pairs", "[nn][gradnoise]")
{
    auto feature = torch::zeros({}, torch::requires_grad());
    auto readout = torch::zeros({}, torch::requires_grad());
    torch::optim::AdamW optimizer({ feature, readout }, torch::optim::AdamWOptions(0.01));
    auto state = std::make_unique<torch::optim::AdamWParamState>();
    state->step(1);
    state->exp_avg(torch::zeros_like(feature));
    state->exp_avg_sq(torch::ones_like(feature));
    optimizer.state()[feature.unsafeGetTensorImpl()] = std::move(state);
    const auto losses = feature * torch::tensor({ 1.0f, 3.0f, 5.0f, 7.0f }) + readout * torch::tensor({ 2.0f, 4.0f, 6.0f, 8.0f });
    const auto result =
        anet::MeasureGradientNoise(losses, { { .parameter = feature, .feature = true }, { .parameter = readout, .feature = false } },
            optimizer, 2, 1e-5, { .raw = true, .adam = { true, true, true } });
    CHECK(std::isfinite(result.raw.signal));
    CHECK(std::isfinite(result.adam[1].signal));
    CHECK(std::isnan(result.adam[0].signal));
    CHECK(std::isnan(result.adam[2].signal));
    const auto empty = anet::MeasureGradientNoise(feature * torch::ones({ 4 }), { { .parameter = feature, .feature = true } }, optimizer, 2,
        1e-5, { .raw = true, .adam = { true, true, true } });
    CHECK(std::isfinite(empty.adam[0].signal));
    CHECK(std::isnan(empty.adam[2].signal));
    CHECK(std::isnan(empty.adam[2].variance));
    anet::GradientNoiseEstimator ema(0.5);
    CHECK(ema.Update({ .signal = -2, .variance = 4 }).signal == Catch::Approx(-2));
    CHECK(std::isnan(ema.Update({}).signal));
    const auto resumed = ema.Update({ .signal = 2, .variance = 8 });
    CHECK(resumed.signal == Catch::Approx(2.0 / 3));
    CHECK(resumed.variance == Catch::Approx(20.0 / 3));
    auto subset = torch::cat({ feature.expand({ 2 }), readout.expand({ 2 }) });
    const auto partial = anet::MeasureGradientNoise(subset,
        { { .parameter = feature, .feature = true }, { .parameter = readout, .feature = false } }, optimizer, 2, 1e-5, { .raw = true });
    CHECK(partial.raw.signal == Catch::Approx(0));
    CHECK(partial.raw.variance == Catch::Approx(2));
}

TEST_CASE("Linear regression gradient noise converges to its analytic noise scale", "[nn][gradnoise]")
{
    auto parameter = torch::zeros({}, torch::requires_grad());
    torch::optim::AdamW optimizer({ parameter }, torch::optim::AdamWOptions(0.01));
    anet::GradientNoiseEstimator ema(0.05);
    const auto before = at::detail::getDefaultCPUGenerator().get_state();
    auto generator = at::make_generator<at::CPUGeneratorImpl>(920);
    anet::GradientNoiseEstimate smooth;
    for (int i = 0; i < 256; ++i) {
        const auto target = torch::randint(0, 2, { 64 }, generator, torch::TensorOptions().dtype(torch::kFloat32)) * 2 + 2;
        const auto losses = (parameter - target).square() * 0.5;
        smooth = ema.Update(anet::MeasureGradientNoise(losses, { { .parameter = parameter } }, optimizer, 8, 1e-5, { .raw = true }).raw);
    }
    CHECK(smooth.signal == Catch::Approx(9).epsilon(0.1));
    CHECK(smooth.variance == Catch::Approx(1).epsilon(0.25));
    CHECK(smooth.variance / smooth.signal == Catch::Approx(1.0 / 9).epsilon(0.25));
    CHECK(torch::equal(before, at::detail::getDefaultCPUGenerator().get_state()));
    CHECK_FALSE(parameter.grad().defined());
}
