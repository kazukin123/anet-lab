#include "anet/test/catch.hpp"
#include <array>
#include "anet/tensor_store.hpp"

TEST_CASE("Lane-time tensor store preserves stored observations by lane and time", "[rfd0191][tensor_store]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");
    constexpr char kInput[] = "input";
    constexpr char kLabel[] = "label";
    anet::LaneTimeTensorStore store(2);
    auto input = torch::arange(6, torch::kFloat32).reshape({2, 3}).to(device);
    const auto label = torch::tensor({1, 2}, torch::kInt64).to(device);
    store.Append({{kInput, input}, {kLabel, label}});
    input.add_(10);
    CHECK_THROWS(store.Append({{kInput, input.slice(1, 1)}, {kLabel, label}}));
    CHECK(store.GetTimeCount() == 1);
    store.Append({{kInput, input}, {kLabel, label + 2}});
    input.zero_();

    const std::array<anet::LaneTimeIndex, 2> indices{{{.lane = 0, .time = 1}, {.lane = 1, .time = 0}}};
    const auto selected = store.Gather(indices);
    CHECK(torch::equal(selected.At(kInput), torch::tensor({{10.f, 11.f, 12.f}, {3.f, 4.f, 5.f}}).to(device)));
    CHECK(torch::equal(selected.At(kLabel), torch::tensor({3, 2}, torch::kInt64).to(device)));
    CHECK_THROWS(store.Append({{kInput, input}, {kLabel, label}}));
    const std::array<anet::LaneTimeIndex, 1> invalid_lane{{{.lane = 2, .time = 0}}};
    CHECK_THROWS(store.Gather(invalid_lane));
    store.Clear();
    CHECK(store.GetTimeCount() == 0);
    CHECK_THROWS(store.Gather(indices));
    store.Append({{kInput, input}, {kLabel, label}});
    CHECK(store.GetTimeCount() == 1);
}
