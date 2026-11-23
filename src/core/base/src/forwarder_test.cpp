#include "anet/test/catch.hpp"

#include "anet/forwarder.hpp"
#include "anet/graphviz.hpp"
#include "anet/scaler.hpp"

constexpr char kStackBlock[] = "stack";
constexpr char kNormBlock[] = "norm";
constexpr char kVector[] = "vector";
constexpr char kAux[] = "aux";
constexpr char kFrameStackType[] = "FrameStack";
constexpr char kNormalizeType[] = "Normalize";
constexpr char kDropKeysType[] = "DropKeys";
constexpr char kListedKeys[] = "listed";
constexpr char kConstantMode[] = "constant";
constexpr char kNoneMode[] = "none";
constexpr char kRunningMode[] = "running";

TEST_CASE("Perception from a frame matches the stacked spec", "[perception]")
{
    anet::ConfigData config;
    int stack_count = 1;
    SECTION("Implicit stack") {}
    SECTION("Explicit stack with default count") {
        config.Set("perception.structure", kStackBlock);
        config.Set("perception.block.[stack].type", kFrameStackType);
    }
    SECTION("Four frames") {
        stack_count = 4;
        config.Set("perception.structure", kStackBlock);
        config.Set("perception.block.[stack].type", kFrameStackType);
        config.Set("perception.block.[stack].count", stack_count);
    }

    const anet::TensorSpec frame_spec{ .shape = { 3 } };
    const anet::TensorSpecMap frame_specs{ { kVector, frame_spec } };
    const anet::ForwarderChain chain(config, "perception", frame_specs, 42);
    CHECK(chain.GetStackKeys() == std::vector<std::string>{ kVector });
    const auto specs = chain.ForwardSpec(frame_specs);
    const auto& stacked_spec = specs.at(kVector);
    CHECK(stacked_spec.shape == std::vector<int64_t>{ stack_count, 3 });
    CHECK(stacked_spec.dtype == frame_spec.dtype);
    CHECK(stacked_spec.has_stack_axis);

    const auto vector = torch::arange(6, torch::TensorOptions().dtype(frame_spec.dtype)).reshape({ 2, 3 });
    const anet::TensorDict frame{ { kVector, vector } };
    const auto perception = chain.GetForwarder(anet::ForwardRequest::PerceptionFromFrame)->Forward(frame);
    const auto& stacked = perception.At(kVector);
    CHECK(stacked.sizes().slice(1).vec() == stacked_spec.shape);
    CHECK(stacked.scalar_type() == stacked_spec.dtype);
    CHECK(torch::equal(stacked, vector.unsqueeze(1).expand({ 2, stack_count, 3 })));
}

TEST_CASE("FrameStack aligns actor history and replay frames without changing stored input", "[perception]")
{
    constexpr char kUnstacked[] = "unstacked";
    const int count = GENERATE(1, 4);
    anet::ConfigData config;
    config.Set("perception.structure", kStackBlock);
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[stack].count", count);
    config.Set("perception.block.[stack].keys.mode", kListedKeys);
    config.Set("perception.block.[stack].keys.listed.names", kVector);
    const anet::TensorSpec spec{ .shape = { 1 } };
    const anet::TensorSpecMap frame_specs{ { kVector, spec }, { kUnstacked, spec } };
    const anet::ForwarderChain chain(config, "perception", frame_specs, 42);
    CHECK(chain.GetStackCount() == count);
    CHECK(chain.GetStackKeys() == std::vector<std::string>{ kVector });
    const auto output_specs = chain.ForwardSpec(frame_specs);
    CHECK(output_specs.at(kVector).has_stack_axis);
    CHECK_FALSE(output_specs.at(kUnstacked).has_stack_axis);

    const auto first = torch::tensor({ { 1.0f }, { 11.0f } });
    const auto second = torch::tensor({ { 2.0f }, { 12.0f } });
    const anet::TensorDict first_frame{ { kVector, first }, { kUnstacked, first } };
    const anet::TensorDict second_frame{ { kVector, second }, { kUnstacked, second } };
    auto actor = chain.CreateExecutor(2, torch::Device(torch::kCPU), 17);
    const auto initial = actor->Forward(first_frame, torch::tensor({ true, true }, torch::kBool));
    CHECK(torch::equal(initial.At(kVector), first.unsqueeze(1).expand({ 2, count, 1 })));
    const auto next = actor->Forward(second_frame, torch::tensor({ false, true }, torch::kBool));
    auto expected = second.unsqueeze(1).expand({ 2, count, 1 }).clone();
    expected.select(0, 0).narrow(0, 0, count - 1).fill_(1.0f);
    CHECK(torch::equal(next.At(kVector), expected));
    CHECK(torch::equal(next.At(kUnstacked), second));

    const anet::TensorDict replay{ { kVector, count == 1 ? second : expected }, { kUnstacked, second } };
    const auto sampled = chain.GetForwarder(anet::ForwardRequest::PerceptionFromStack)->Forward(replay);
    CHECK(torch::equal(sampled.At(kVector), expected));
    CHECK(sampled.At(kUnstacked).is_same(second));
    const auto stored = chain.GetForwarder(anet::ForwardRequest::FrameWithStatsUpdate)->Forward(second_frame);
    CHECK(stored.At(kVector).is_same(second));
    CHECK(stored.At(kUnstacked).is_same(second));
}

TEST_CASE("DropKeys hides keys from Perception and preserves them in the stored frame", "[perception]")
{
    const int stack_count = GENERATE(1, 4);
    anet::ConfigData config;
    config.Set("perception.structure", "stack > drop >");
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[stack].count", stack_count);
    config.Set("perception.block.[drop].type", kDropKeysType);
    config.Set("perception.block.[drop].keys", kAux);
    const anet::TensorSpec spec{ .shape = { 1 } };
    const anet::TensorSpecMap frame_specs{ { kVector, spec }, { kAux, spec } };
    const anet::ForwarderChain chain(config, "perception", frame_specs, 42);
    const auto output_specs = chain.ForwardSpec(frame_specs);
    CHECK(output_specs.at(kVector).has_stack_axis);
    CHECK_FALSE(output_specs.contains(kAux));

    const auto value = torch::ones({ 2, 1 });
    const anet::TensorDict frame{ { kVector, value }, { kAux, value } };
    auto executor = chain.CreateExecutor(2, torch::Device(torch::kCPU), 17);
    const auto actor = executor->Forward(frame, torch::ones({ 2 }, torch::kBool));
    const auto single = chain.GetForwarder(anet::ForwardRequest::PerceptionFromFrame)->Forward(frame);
    const auto stacked = value.unsqueeze(1).expand({2, stack_count, 1});
    const auto replay_input = stack_count == 1 ? frame : anet::TensorDict{{kVector, stacked}, {kAux, stacked}};
    const auto replay = chain.GetForwarder(anet::ForwardRequest::PerceptionFromStack)->Forward(replay_input);
    CHECK(single.At(kVector).sizes().slice(1).vec() == output_specs.at(kVector).shape);
    CHECK(single.At(kVector).scalar_type() == output_specs.at(kVector).dtype);
    for (const auto* perception : { &actor, &single, &replay }) {
        CHECK_FALSE(perception->Contains(kAux));
        CHECK(torch::equal(perception->At(kVector), stacked));
    }
    const auto stored = chain.GetForwarder(anet::ForwardRequest::FrameWithStatsUpdate)->Forward(frame);
    CHECK(stored.At(kAux).is_same(value));
}

TEST_CASE("Normalize applies one Frame's statistics across the stack", "[perception]")
{
    constexpr char kScaled[] = "scaled";
    const std::string mode = GENERATE(kNoneMode, kConstantMode, kRunningMode);
    const anet::TensorSpec unscaled{ .shape = { 8 }, .scale = anet::ValueScale::Unscaled };
    const anet::TensorSpec scaled{ .shape = { 8 } };
    const anet::TensorSpecMap specs{ { kVector, unscaled }, { kScaled, scaled } };
    anet::ConfigData config;
    config.Set("perception.structure", "stack > norm >");
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[stack].count", 4);
    config.Set("perception.block.[norm].type", kNormalizeType);
    config.Set("perception.block.[norm].mode", mode);
    config.Set("perception.block.[norm].constant.mean", "1 1 1 1 1 1 1 1");
    config.Set("perception.block.[norm].constant.std", "2 2 2 2 2 2 2 2");
    config.Set("perception.block.[norm].running.use_centering", true);
    const anet::ForwarderChain chain(config, "perception", specs, 42);
    const auto output_specs = chain.ForwardSpec(specs);
    CHECK(output_specs.at(kVector).scale == (mode == kNoneMode ? anet::ValueScale::Unscaled : anet::ValueScale::Scaled));
    CHECK(output_specs.at(kVector).has_stack_axis);

    const auto update = chain.GetForwarder(anet::ForwardRequest::FrameWithStatsUpdate);
    const auto sample = chain.GetForwarder(anet::ForwardRequest::PerceptionFromStack);
    const auto view = chain.GetForwarder(anet::ForwardRequest::PerceptionFromFrame);
    const anet::rl::ObservationNormalizerConfig reference_config{
        .pass_through = mode == kNoneMode,
        .constant_mean = std::vector<float>(8, 1.0f),
        .constant_std = std::vector<float>(8, 2.0f),
        .use_dynamic_scaling = mode == kRunningMode,
        .use_centering = mode == kRunningMode,
    };
    const anet::rl::StateSpec reference_spec{ .obs_spec = {{ kVector, unscaled }} };
    auto reference = anet::rl::ObservationNormalizerFactory(reference_config).CreateObservationNormalizer(reference_spec);
    std::vector<torch::Tensor> frames;
    for (int index = 0; index < 4; ++index) {
        auto value = (torch::arange(16, torch::kFloat32) + index).reshape({ 2, 8 });
        const anet::TensorDict frame{ { kVector, value }, { kScaled, value } };
        CHECK(update->Forward(frame).At(kVector).is_same(value));
        reference->UpdateStats(frame);
        frames.push_back(std::move(value));
    }
    const auto count_key = anet::rl::ObservationNormalizer::kKeyCount;
    REQUIRE(chain.GetScalar(count_key).has_value());
    CHECK(chain.GetScalar(count_key) == reference->GetScalar(count_key));

    const auto stacked = torch::stack(frames, 1);
    const anet::TensorDict stacked_input{ { kVector, stacked }, { kScaled, stacked } };
    std::vector<torch::Tensor> normalized;
    for (const auto& frame : frames) {
        normalized.push_back(reference->Normalize(anet::TensorDict{ { kVector, frame } }).At(kVector));
    }
    const auto sampled = sample->Forward(stacked_input);
    CHECK(torch::equal(sampled.At(kVector), torch::stack(normalized, 1)));
    CHECK(sampled.At(kScaled).is_same(stacked));

    const anet::TensorDict frame{ { kVector, frames.front() }, { kScaled, frames.front() } };
    const auto expected = normalized.front().unsqueeze(1).expand({ 2, 4, 8 });
    CHECK(torch::equal(view->Forward(frame).At(kVector), expected));
    auto actor = chain.CreateExecutor(2, torch::Device(torch::kCPU), 17);
    CHECK(torch::equal(actor->Forward(frame, torch::ones({ 2 }, torch::kBool)).At(kVector), expected));
    CHECK(chain.GetScalar(count_key) == reference->GetScalar(count_key));
}

TEST_CASE("Frame intake preserves the normalizer clip ratio without changing the frame", "[perception]")
{
    const std::string mode = GENERATE(kConstantMode, kRunningMode);
    constexpr float kClipRange = 0.5f;
    const anet::rl::StateSpec state_spec{
        .obs_spec = {{kVector, {.shape = {2}, .scale = anet::ValueScale::Unscaled}}},
    };
    anet::ConfigData config;
    config.Set("perception.structure", kNormBlock);
    config.Set("perception.block.[norm].type", kNormalizeType);
    config.Set("perception.block.[norm].mode", mode);
    config.Set("perception.block.[norm].constant.clip_range", kClipRange);
    config.Set("perception.block.[norm].running.clip_range", kClipRange);
    anet::ForwarderChain chain(config, "perception", state_spec.obs_spec, 161);
    const auto intake = chain.GetForwarder(anet::ForwardRequest::FrameWithStatsUpdate);
    const anet::rl::ObservationNormalizerConfig reference_config{
        .clip_range = kClipRange,
        .use_dynamic_scaling = mode == kRunningMode,
    };
    const auto reference = anet::rl::ObservationNormalizerFactory(reference_config).CreateObservationNormalizer(state_spec);

    const auto value = torch::full({2, 2}, 4.0f);
    const auto original = value.clone();
    const anet::TensorDict frame{{kVector, value}};
    const auto key = anet::rl::ObservationNormalizer::kKeyClipRatio;
    reference->NormalizeAndUpdateStats(frame);
    const auto expected = reference->GetScalar(key);
    REQUIRE(expected);
    REQUIRE(*expected > 0.0f);
    const auto stored = intake->Forward(frame);
    CHECK(stored.At(kVector).is_same(value));
    CHECK(torch::equal(stored.At(kVector), original));
    CHECK(chain.GetScalar(key) == expected);

    const anet::TensorDict zeros{{kVector, torch::zeros_like(value)}};
    auto actor = chain.CreateExecutor(2, torch::kCPU, 161);
    actor->Forward(zeros, torch::ones({2}, torch::kBool));
    chain.GetForwarder(anet::ForwardRequest::PerceptionFromFrame)->Forward(zeros);
    chain.GetForwarder(anet::ForwardRequest::PerceptionFromStack)->Forward(zeros);
    CHECK(chain.GetScalar(key) == expected);
}

TEST_CASE("Perception rejects invalid block declarations during construction", "[perception]")
{
    anet::ConfigData config;
    config.Set("perception.structure", "stack > norm");
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[norm].type", kNormalizeType);
    anet::TensorSpecMap specs{ { kVector, { .shape = { 8 }, .scale = anet::ValueScale::Unscaled } },
        { kAux, { .shape = { 2 } } } };
    std::string expected;
    SECTION("Unknown block") {
        config.Set("perception.structure", "stack > missing");
        expected = "perception.block.[missing].type";
    }
    SECTION("Unknown type") {
        config.Set("perception.block.[norm].type", "missing");
        expected = "perception.block.[norm].type";
    }
    SECTION("FrameStack after Normalize") {
        config.Set("perception.structure", "norm > stack");
        expected = "expected it first";
    }
    SECTION("Duplicate Normalize") {
        config.Set("perception.structure", "stack > norm > norm");
        expected = "at most one Normalize";
    }
    SECTION("Invalid stack count") {
        config.Set("perception.block.[stack].count", 0);
        expected = "perception.block.[stack].count";
    }
    SECTION("Unknown selected key") {
        config.Set("perception.block.[stack].keys.mode", kListedKeys);
        config.Set("perception.block.[stack].keys.listed.names", "missing");
        expected = "perception.block.[stack].keys.listed.names";
    }
    SECTION("Pre-stacked input") {
        specs.at(kVector).has_stack_axis = true;
        expected = "stack_axis=true";
    }
    SECTION("Empty input") {
        specs.clear();
        expected = "at least one input key";
    }
    SECTION("DropKeys leaves no output") {
        config.Set("perception.block.[norm].type", kDropKeysType);
        config.Set("perception.block.[norm].keys", "vector aux");
        expected = "at least one output key";
    }
    SECTION("Invalid inactive setting") {
        config.Set("perception.block.[norm].mode", kNoneMode);
        config.Set("perception.block.[norm].running.epsilon", 0.0f);
        expected = "perception.block.[norm].running.epsilon";
    }
    SECTION("Constant width does not match") {
        config.Set("perception.block.[norm].mode", kConstantMode);
        config.Set("perception.block.[norm].constant.mean", "1 2");
        expected = "perception.block.[norm].constant.mean";
    }
    SECTION("Unknown property in selected block") {
        config.Set("perception.block.[stack].unused", "wrong");
        expected = "perception.block.[stack].unused";
    }
    CHECK_THROWS_WITH(anet::ForwarderChain(config, "perception", specs, 42), Catch::Matchers::ContainsSubstring(expected));
}

TEST_CASE("Perception leaves former settings unread and ignores unused block definitions", "[perception]")
{
    constexpr char kFormerStackCount[] = "stucker.stack_count";
    anet::ConfigData config;
    config.Set(kFormerStackCount, 4);
    config.Set("perception.block.[unused].unexpected", "anything");
    const anet::TensorSpecMap specs{ { kVector, { .shape = { 8 } } } };
    const anet::ForwarderChain chain(config, "perception", specs, 42);
    CHECK(chain.GetStackCount() == 1);
    const auto unread = config.GetUnreadKeys();
    REQUIRE(unread.size() == 1);
    CHECK(unread.front().first == kFormerStackCount);
    CHECK_FALSE(chain.GetScalar(anet::rl::ObservationNormalizer::kKeyCount).has_value());

    config.Set("perception.structure", kNormBlock);
    config.Set("perception.block.[norm].type", kNormalizeType);
    config.Set("perception.block.[norm].mode", kConstantMode);
    config.Set("perception.block.[norm].constant.mean", "1 2");
    const anet::ForwarderChain scaled_only(config, "perception", specs, 42);
    CHECK_FALSE(scaled_only.GetScalar(anet::rl::ObservationNormalizer::kKeyCount).has_value());
    const auto tensor = torch::ones({ 2, 8 });
    const auto view = scaled_only.GetForwarder(anet::ForwardRequest::PerceptionFromFrame);
    CHECK(torch::equal(view->Forward(anet::TensorDict{ { kVector, tensor } }).At(kVector), tensor.unsqueeze(1)));
}

TEST_CASE("Perception diagrams describe ordered blocks, tensor shapes and named seeds", "[perception]")
{
    const anet::TensorSpecMap specs{ { kVector, { .shape = { 8 }, .scale = anet::ValueScale::Unscaled } },
        { kAux, { .shape = { 1 } } } };
    anet::ConfigData config;
    config.Set("perception.structure", "stack > norm > drop");
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[stack].count", 4);
    config.Set("perception.block.[norm].type", kNormalizeType);
    config.Set("perception.block.[norm].mode", kNoneMode);
    config.Set("perception.block.[drop].type", kDropKeysType);
    config.Set("perception.block.[drop].keys", kAux);
    const anet::ForwarderChain chain(config, "perception", specs, 42);
    const auto structure = chain.MakeGraphViz(false)->ToDotString();
    for (const auto* fragment : { kStackBlock, kFrameStackType, kNormBlock, kNormalizeType, kVector, kAux, "block_0\" -> \"block_1" }) {
        CHECK(structure.find(fragment) != std::string::npos);
    }
    const auto detail = chain.MakeGraphViz(true)->ToDotString();
    for (const auto* fragment : {"count", "keys.mode", "Unscaled", "Scaled", "stack_axis", "Float", "[4,8]", kNoneMode}) {
        CHECK(detail.find(fragment) != std::string::npos);
    }
    const auto seed = std::to_string(anet::SeedMaker(42).MakeNamedSeed(kNormBlock));
    CHECK(detail.find(seed) != std::string::npos);

    config.Set("perception.structure", "stack > drop > norm");
    const anet::ForwarderChain reordered(config, "perception", specs, 42);
    CHECK(reordered.MakeGraphViz(true)->ToDotString().find(seed) != std::string::npos);
}
