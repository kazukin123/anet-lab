#include "anet/test/catch.hpp"

#include "anet/forwarder.hpp"
#include "anet/graphviz.hpp"
#include "anet/scaler.hpp"

constexpr char kStackBlock[] = "stack";
constexpr char kNormBlock[] = "norm";
constexpr char kVector[] = "vector";
constexpr char kAux[] = "aux";
constexpr char kFrameStackType[] = "FrameStack";
constexpr char kSetSpecType[] = "SetSpec";
constexpr char kListedKeys[] = "listed";
constexpr char kAllKeys[] = "all";
constexpr char kKeepMode[] = "keep";
constexpr char kSetMode[] = "set";
constexpr char kScaledValue[] = "Scaled";
constexpr char kUnscaledValue[] = "Unscaled";
constexpr char kNormalizeType[] = "Normalize";
constexpr char kDropKeysType[] = "DropKeys";
constexpr char kConstantMode[] = "constant";
constexpr char kNoneMode[] = "none";
constexpr char kRunningMode[] = "running";
constexpr char kTestStartKey[] = "start";
constexpr char kIdentityType[] = "ForwarderTestIdentity";
constexpr char kIdentityKind[] = "ForwarderTest::Identity";

static anet::TensorSpecMap MakeSequenceSpecs(anet::TensorSpecMap specs)
{
    specs.emplace(kTestStartKey, anet::TensorSpec{ .dtype = torch::kBool });
    return specs;
}

static anet::TensorDict MakeSequenceInput(anet::TensorDict input, const torch::Tensor& starts)
{
    input.Set(kTestStartKey, starts);
    return input;
}

static anet::TensorSpecMap MakeSampleSpecs(anet::TensorSpecMap specs, int count, const std::vector<std::string>& keys)
{
    if (count == 1) return specs;
    for (const auto& key : keys) {
        auto& spec = specs.at(key);
        spec.shape.insert(spec.shape.begin(), count);
        spec.has_stack_axis = true;
    }
    return specs;
}

namespace {

    class IdentityModule final : public anet::ForwarderModule {
    public:
        std::shared_ptr<anet::SpecForwarder> CreateForwarder(const anet::ForwarderRequest&,
            const anet::TensorSpecMap& input_specs) const override
        {
            return anet::MakePassThroughForwarder(input_specs);
        }
    };

    class IdentityFactory final : public anet::ForwarderModuleFactory {
    public:
        std::optional<std::vector<anet::ForwardKind>> GetForwardKinds() const override
        {
            return std::vector<anet::ForwardKind>{ kIdentityKind };
        }
        void ReadConfigKeys(const anet::ConfigData& data) const override { (void)data.Get("version"); }
        std::shared_ptr<anet::ForwarderModule> CreateModule(const anet::ConfigData& data,
            const anet::ForwarderModuleContext&) const override
        {
            if (data.Get<int>("version", 1) != 1) {
                ANET_SYSTEM_ERROR("Invalid " << data.GetFullKey("version") << "; expected 1.");
            }
            return std::make_shared<IdentityModule>();
        }
    };

}

TEST_CASE("Builder executes a registered module for a test-defined forward kind", "[forwarder]")
{
    anet::RegisterForwarderModuleFactory(kIdentityType, std::make_shared<IdentityFactory>());
    anet::ConfigData config;
    config.Set("pipeline.structure", "identity > again");
    config.Set("pipeline.block.[identity].type", kIdentityType);
    config.Set("pipeline.block.[again].type", kIdentityType);
    const anet::ForwarderBuilder builder(config, "pipeline", 42);
    const anet::TensorSpecMap input_specs{ { kVector, { .shape = { 3 }, .dtype = torch::kFloat32 } } };

    const auto forwarder = builder.BuildForwarder(anet::ForwarderRequest(kIdentityKind), input_specs);
    REQUIRE(forwarder);
    REQUIRE(forwarder->GetInputSpecs().size() == 1);
    CHECK(forwarder->GetInputSpecs().at(kVector).shape == std::vector<int64_t>{ 3 });
    CHECK(forwarder->GetInputSpecs().at(kVector).dtype == torch::kFloat32);
    const auto values = torch::tensor({ { 1.0f, 2.0f, 3.0f } });
    const auto output = forwarder->Forward(anet::TensorDict{ { kVector, values } });
    CHECK(torch::equal(output.At(kVector), values));
    CHECK(forwarder->GetOutputSpecs().at(kVector).shape == output.At(kVector).sizes().slice(1).vec());
}

TEST_CASE("Builder rejects invalid registrations, configuration and requests", "[forwarder][perception]")
{
    constexpr char kRegisteredType[] = "ForwarderTestInvalidRequestIdentity";
    anet::RegisterForwarderModuleFactory(kRegisteredType, std::make_shared<IdentityFactory>());
    CHECK_THROWS(anet::RegisterForwarderModuleFactory(kRegisteredType, std::make_shared<IdentityFactory>()));

    anet::ConfigData config;
    config.Set("pipeline.structure", "identity");
    config.Set("pipeline.block.[identity].type", "UnregisteredForwarderTestType");
    CHECK_THROWS_WITH(anet::ForwarderBuilder(config, "pipeline", 42),
        Catch::Matchers::ContainsSubstring("pipeline.block.[identity].type"));

    config.Set("pipeline.block.[identity].type", kRegisteredType);
    const anet::ForwarderBuilder builder(config, "pipeline", 42);
    const anet::TensorSpecMap input_specs{ { kVector, { .shape = { 3 } } } };
    CHECK_THROWS(builder.BuildForwarder(anet::ForwarderRequest("UnsupportedKind"), input_specs));
    CHECK_THROWS(builder.BuildForwarder(anet::ForwarderRequest(kIdentityKind), {}));

    config.Set("pipeline.block.[identity].unexpected", "1");
    config.Set("pipeline.block.[identity].version", "not an integer");
    CHECK_THROWS_WITH(anet::ForwarderBuilder(config, "pipeline", 42),
        Catch::Matchers::ContainsSubstring("pipeline.block.[identity].unexpected"));
    config.Set("pipeline.structure", "another");
    config.Set("pipeline.block.[another].type", kRegisteredType);
    config.Set("pipeline.block.[another].version", "not an integer");
    CHECK_THROWS_WITH(anet::ForwarderBuilder(config, "pipeline", 42),
        Catch::Matchers::ContainsSubstring("pipeline.block.[another].version"));
}

TEST_CASE("Builder reports earlier configuration errors before later ones across blocks", "[forwarder][perception]")
{
    anet::ConfigData config;
    config.Set("perception.structure", "norm > set > drop");
    config.Set("perception.block.[norm].type", kNormalizeType);
    config.Set("perception.block.[set].type", kSetSpecType);
    config.Set("perception.block.[set].keys", kVector);
    config.Set("perception.block.[set].scale", kScaledValue);
    config.Set("perception.block.[set].forward_kinds.mode", kListedKeys);
    config.Set("perception.block.[set].forward_kinds.listed.names", kIdentityKind);
    config.Set("perception.block.[drop].type", kDropKeysType);

    SECTION("An unread key takes precedence over a value that cannot be parsed") {
        config.Set("perception.block.[norm].running.clip_range", "invalid");
        config.Set(GENERATE("perception.block.[norm].unexpected", "perception.block.[drop].unexpected"), "1");
        CHECK_THROWS_WITH(anet::ForwarderBuilder(config, "perception", 42),
            Catch::Matchers::ContainsSubstring("Unread config keys"));
    }
    SECTION("An invalid block value takes precedence over an unsupported kind in SetSpec") {
        CHECK_THROWS_WITH(anet::ForwarderBuilder(config, "perception", 42),
            Catch::Matchers::ContainsSubstring("perception.block.[drop].keys"));
    }
}

TEST_CASE("FrameStack and DropKeys declare actual output specs for each Forward kind", "[perception]")
{
    const int count = GENERATE(1, 4);
    anet::ConfigData config;
    config.Set("perception.structure", "stack > drop");
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[drop].type", kDropKeysType);
    config.Set("perception.block.[drop].keys", kAux);
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const anet::TensorSpecMap frame_specs{ { kVector, { .shape = { 3 } } }, { kAux, { .shape = { 1 } } } };
    const anet::TensorDict frame{ { kVector, torch::ones({ 2, 3 }) }, { kAux, torch::zeros({ 2, 1 }) } };
    const auto run = [&](const anet::ForwarderRequest& request, const anet::TensorSpecMap& specs,
        const anet::TensorDict& input, bool stacked) {
        const auto forwarder = builder.BuildForwarder(request, specs);
        const auto output = forwarder->Forward(input);
        const int expected_size = stacked ? 1 : 2;
        CHECK(output.size() == expected_size);
        CHECK(forwarder->GetOutputSpecs().size() == expected_size);
        CHECK(forwarder->GetOutputSpecs().at(kVector).has_stack_axis == stacked);
        if (stacked) CHECK(output.At(kVector).size(1) == count);
        for (const auto& [key, tensor] : output) {
            CAPTURE(key);
            const auto& spec = forwarder->GetOutputSpecs().at(key);
            CHECK(spec.shape == tensor.sizes().slice(1).vec());
            CHECK(spec.dtype == tensor.scalar_type());
        }
    };

    run(anet::rl::MakeFrameWithStatsUpdateRequest(), frame_specs, frame, false);
    run(anet::rl::MakePerceptionFromFrameRequest(count, { kVector }), frame_specs, frame, true);
    auto sequence_specs = frame_specs;
    auto sequence = frame;
    const std::string start_key = "example_start";
    sequence_specs.emplace(start_key, anet::TensorSpec{ .dtype = torch::kBool });
    sequence.Set(start_key, torch::ones({ 2 }, torch::kBool));
    run(anet::rl::MakePerceptionFromSequenceRequest(2, 42, start_key, count, { kVector }),
        sequence_specs, sequence, true);

    auto sample_specs = frame_specs;
    auto sample = frame;
    if (count > 1) {
        sample_specs.at(kVector).shape.insert(sample_specs.at(kVector).shape.begin(), count);
        sample_specs.at(kVector).has_stack_axis = true;
        sample.Set(kVector, torch::ones({ 2, count, 3 }));
    }
    run(anet::rl::MakePerceptionFromStackRequest(count, { kVector }), sample_specs, sample, true);
}

TEST_CASE("Running Normalize prepares its stats on intake and reuses them", "[perception]")
{
    anet::ConfigData config;
    config.Set("perception.structure", kNormBlock);
    config.Set("perception.block.[norm].type", kNormalizeType);
    config.Set("perception.block.[norm].mode", kRunningMode);
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const anet::TensorSpecMap specs{ { kVector, { .shape = { 2 }, .scale = anet::ValueScale::Unscaled } } };
    const auto view_request = anet::rl::MakePerceptionFromFrameRequest(1, { kVector });
    CHECK_THROWS_WITH(builder.BuildForwarder(view_request, specs), Catch::Matchers::ContainsSubstring(kVector));
    const auto intake_request = anet::rl::MakeFrameWithStatsUpdateRequest();
    const auto intake = builder.BuildForwarder(intake_request, specs);
    const auto values = torch::tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
    const anet::TensorDict obs{ { kVector, values } };
    CHECK(torch::equal(intake->Forward(obs).At(kVector), values));
    const auto key = anet::rl::ObservationNormalizer::kKeyCount;
    const auto count = builder.GetScalar(key);
    REQUIRE(count.has_value());
    CHECK(*count == 2);
    const auto view = builder.BuildForwarder(view_request, specs);
    CHECK(view->GetOutputSpecs().at(kVector).scale == anet::ValueScale::Scaled);
    view->Forward(obs);
    CHECK(builder.GetScalar(key) == count);
    builder.BuildForwarder(intake_request, specs);
    CHECK(builder.GetScalar(key) == count);

    auto different = specs;
    different.at(kVector).shape = { 3 };
    CHECK_THROWS(builder.BuildForwarder(intake_request, different));
    CHECK_THROWS(builder.BuildForwarder(view_request, different));
}

TEST_CASE("SetSpec changes only the requested declaration for the selected kind", "[perception]")
{
    anet::ConfigData config;
    config.Set("perception.structure", "set");
    config.Set("perception.block.[set].type", kSetSpecType);
    config.Set("perception.block.[set].keys", kVector);
    config.Set("perception.block.[set].scale", kUnscaledValue);
    config.Set("perception.block.[set].forward_kinds.mode", kListedKeys);
    config.Set("perception.block.[set].forward_kinds.listed.names", anet::rl::kForwardKindPerceptionFromFrame);
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const anet::TensorSpecMap specs{ { kVector, { .shape = { 2 } } } };
    const auto view = builder.BuildForwarder(anet::rl::MakePerceptionFromFrameRequest(1, { kVector }), specs);
    const auto intake = builder.BuildForwarder(anet::rl::MakeFrameWithStatsUpdateRequest(), specs);
    CHECK(view->GetOutputSpecs().at(kVector).scale == anet::ValueScale::Unscaled);
    CHECK(intake->GetOutputSpecs().at(kVector).scale == anet::ValueScale::Scaled);
    CHECK(view->GetOutputSpecs().at(kVector).labels.empty());
    CHECK(view->GetOutputSpecs().at(kVector).min_values.empty());
    CHECK(view->GetOutputSpecs().at(kVector).max_values.empty());
    const auto values = torch::ones({ 2, 2 });
    CHECK(view->Forward(anet::TensorDict{ { kVector, values } }).At(kVector).is_same(values));
}

TEST_CASE("SetSpec validates changed attributes against one frame", "[perception]")
{
    anet::ConfigData config;
    config.Set("perception.structure", "stack > set");
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[set].type", kSetSpecType);
    config.Set("perception.block.[set].keys", kVector);
    config.Set("perception.block.[set].labels.mode", kSetMode);
    config.Set("perception.block.[set].labels.set.names", "left right");
    config.Set("perception.block.[set].min_values.mode", kSetMode);
    config.Set("perception.block.[set].min_values.set.values", "-1.012345678901 -2");
    config.Set("perception.block.[set].max_values.mode", kSetMode);
    config.Set("perception.block.[set].max_values.set.values", "3");
    anet::TensorSpecMap specs{ { kVector, { .shape = { 2 } } } };
    const auto request = anet::rl::MakePerceptionFromFrameRequest(4, { kVector });
    SECTION("A stacked input uses one frame's attribute widths") {
        const anet::ForwarderBuilder builder(config, "perception", 42);
        const auto output = builder.BuildForwarder(request, specs);
        const auto& spec = output->GetOutputSpecs().at(kVector);
        CHECK(spec.shape == std::vector<int64_t>{ 4, 2 });
        CHECK(spec.labels == std::vector<std::string>{ "left", "right" });
        CHECK(spec.min_values == std::vector<double>{ -1.012345678901, -2.0 });
        CHECK(spec.max_values == std::vector<double>{ 3.0 });
    }
    SECTION("Rejects the wrong number of labels or range values") {
        const auto [field, value] = GENERATE(std::pair{ "labels.set.names", "left" },
            std::pair{ "min_values.set.values", "1 2 3" }, std::pair{ "max_values.set.values", "1 2 3" });
        const auto key = std::string("perception.block.[set].") + field;
        config.Set(key, value);
        const anet::ForwarderBuilder builder(config, "perception", 42);
        CHECK_THROWS_WITH(builder.BuildForwarder(request, specs), Catch::Matchers::ContainsSubstring(key));
    }
    SECTION("Rejects a key absent from the input specs") {
        config.Set("perception.block.[set].keys", "missing");
        const anet::ForwarderBuilder builder(config, "perception", 42);
        CHECK_THROWS_WITH(builder.BuildForwarder(request, specs),
            Catch::Matchers::ContainsSubstring("perception.block.[set].keys"));
    }
    SECTION("Rejects Unscaled for integral and discrete observations") {
        const bool discrete = GENERATE(false, true);
        config.Set("perception.block.[set].scale", kUnscaledValue);
        auto& spec = specs.at(kVector);
        spec.dtype = discrete ? torch::kBool : torch::kInt32;
        spec.num_classes = discrete ? 2 : 0;
        const anet::ForwarderBuilder builder(config, "perception", 42);
        CHECK_THROWS_WITH(builder.BuildForwarder(request, specs),
            Catch::Matchers::ContainsSubstring("perception.block.[set].scale"));
    }
    SECTION("Rejects an unsupported kind in a listed selection") {
        config.Set("perception.block.[set].forward_kinds.mode", kListedKeys);
        config.Set("perception.block.[set].forward_kinds.listed.names", "UnsupportedKind");
        CHECK_THROWS_WITH(anet::ForwarderBuilder(config, "perception", 42),
            Catch::Matchers::ContainsSubstring("perception.block.[set].forward_kinds.listed.names"));
    }
    SECTION("Reads a numeric list even in keep mode") {
        config.Set("perception.block.[set].min_values.mode", kKeepMode);
        config.Set("perception.block.[set].min_values.set.values", "1 wrong");
        CHECK_THROWS_WITH(anet::ForwarderBuilder(config, "perception", 42),
            Catch::Matchers::ContainsSubstring("perception.block.[set].min_values.set.values"));
    }
    SECTION("Rejects invalid or empty settings") {
        const auto [field, value] = GENERATE(
            std::pair{ "keys", "" }, std::pair{ "scale", "invalid" }, std::pair{ "labels.mode", "unset" },
            std::pair{ "min_values.mode", "unset" }, std::pair{ "max_values.mode", "unset" },
            std::pair{ "labels.set.names", "" }, std::pair{ "min_values.set.values", "" },
            std::pair{ "max_values.set.values", "" }, std::pair{ "forward_kinds.mode", "unset" });
        const auto key = std::string("perception.block.[set].") + field;
        config.Set(key, value);
        CHECK_THROWS_WITH(anet::ForwarderBuilder(config, "perception", 42), Catch::Matchers::ContainsSubstring(key));
    }
    SECTION("Rejects empty kind selections and declarations with no changes") {
        config.Set("perception.block.[set].forward_kinds.mode", kListedKeys);
        CHECK_THROWS_WITH(anet::ForwarderBuilder(config, "perception", 42),
            Catch::Matchers::ContainsSubstring("perception.block.[set].forward_kinds.listed.names"));
        config.Set("perception.block.[set].forward_kinds.mode", kAllKeys);
        config.Set("perception.block.[set].labels.mode", kKeepMode);
        config.Set("perception.block.[set].min_values.mode", kKeepMode);
        config.Set("perception.block.[set].max_values.mode", kKeepMode);
        CHECK_THROWS_WITH(anet::ForwarderBuilder(config, "perception", 42),
            Catch::Matchers::ContainsSubstring("at least one attribute to change"));
    }
}

TEST_CASE("Intake Normalize tracks a dropped key unless SetSpec marks it scaled", "[perception]")
{
    const bool set_scaled = GENERATE(false, true);
    anet::ConfigData config;
    config.Set("perception.structure", set_scaled ? "set > drop > norm" : "drop > norm");
    config.Set("perception.block.[drop].type", kDropKeysType);
    config.Set("perception.block.[drop].keys", kAux);
    config.Set("perception.block.[norm].type", kNormalizeType);
    if (set_scaled) {
        config.Set("perception.block.[set].type", kSetSpecType);
        config.Set("perception.block.[set].keys", kAux);
        config.Set("perception.block.[set].scale", kScaledValue);
    }
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const anet::TensorSpecMap specs{ { kVector, { .shape = { 1 } } },
        { kAux, { .shape = { 1 }, .scale = anet::ValueScale::Unscaled } } };
    const auto intake = builder.BuildForwarder(anet::rl::MakeFrameWithStatsUpdateRequest(), specs);
    const auto value = torch::ones({ 2, 1 });
    const anet::TensorDict input{ { kVector, value }, { kAux, value } };
    CHECK(intake->Forward(input).At(kAux).is_same(value));
    const auto count = builder.GetScalar(anet::rl::ObservationNormalizer::kKeyCount);
    if (set_scaled) {
        CHECK_FALSE(count.has_value());
    } else {
        REQUIRE(count.has_value());
        CHECK(*count == 2);
    }
    const auto view = builder.BuildForwarder(anet::rl::MakePerceptionFromFrameRequest(1, { kVector, kAux }), specs);
    CHECK_FALSE(view->Forward(input).Contains(kAux));
}

TEST_CASE("SetSpec can select a previously scaled float key for normalization", "[perception]")
{
    anet::ConfigData config;
    config.Set("perception.structure", "stack > set > norm");
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[set].type", kSetSpecType);
    config.Set("perception.block.[set].keys", kVector);
    config.Set("perception.block.[set].scale", kUnscaledValue);
    config.Set("perception.block.[norm].type", kNormalizeType);
    config.Set("perception.block.[norm].mode", kConstantMode);
    config.Set("perception.block.[norm].constant.mean", "2 2");
    config.Set("perception.block.[norm].constant.std", "2 2");
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const anet::TensorSpecMap specs{ { kVector, { .shape = { 2 } } } };
    const auto stacked = torch::full({ 2, 1, 2 }, 4.0f);
    const auto frame = stacked.squeeze(1);
    const auto starts = torch::ones({ 2 }, torch::kBool);
    const auto check = [&](const anet::ForwarderRequest& request, const anet::TensorSpecMap& input_specs,
        const anet::TensorDict& input) {
        const auto forwarder = builder.BuildForwarder(request, input_specs);
        const auto& spec = forwarder->GetOutputSpecs().at(kVector);
        CHECK(spec.has_stack_axis);
        CHECK(spec.scale == anet::ValueScale::Scaled);
        CHECK(torch::equal(forwarder->Forward(input).At(kVector), torch::ones_like(stacked)));
    };
    check(anet::rl::MakePerceptionFromFrameRequest(1, { kVector }), specs, anet::TensorDict{ { kVector, frame } });
    check(anet::rl::MakePerceptionFromStackRequest(1, { kVector }), specs, anet::TensorDict{ { kVector, frame } });
    check(anet::rl::MakePerceptionFromSequenceRequest(2, 42, kTestStartKey, 1, { kVector }),
        MakeSequenceSpecs(specs), MakeSequenceInput(anet::TensorDict{ { kVector, frame } }, starts));
}

TEST_CASE("Perception from a frame matches the stacked spec", "[perception]")
{
    const auto stack_count = GENERATE(1, 4);
    anet::ConfigData config;
    config.Set("perception.structure", kStackBlock);
    config.Set("perception.block.[stack].type", kFrameStackType);
    const anet::TensorSpec frame_spec{ .shape = { 3 } };
    const anet::TensorSpecMap frame_specs{ { kVector, frame_spec } };
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const auto forwarder = builder.BuildForwarder(anet::rl::MakePerceptionFromFrameRequest(stack_count, { kVector }), frame_specs);
    const auto& stacked_spec = forwarder->GetOutputSpecs().at(kVector);
    CHECK(stacked_spec.shape == std::vector<int64_t>{ stack_count, 3 });
    CHECK(stacked_spec.dtype == frame_spec.dtype);
    CHECK(stacked_spec.has_stack_axis);

    const auto vector = torch::arange(6, torch::TensorOptions().dtype(frame_spec.dtype)).reshape({ 2, 3 });
    const auto perception = forwarder->Forward(anet::TensorDict{ { kVector, vector } });
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
    const std::vector<std::string> stack_keys{ kVector };
    const anet::TensorSpec spec{ .shape = { 1 } };
    const anet::TensorSpecMap frame_specs{ { kVector, spec }, { kUnstacked, spec } };
    const anet::ForwarderBuilder builder(config, "perception", 42);
    auto actor = builder.BuildForwarder(anet::rl::MakePerceptionFromSequenceRequest(2, 17, kTestStartKey, count, stack_keys),
        MakeSequenceSpecs(frame_specs));
    const auto& output_specs = actor->GetOutputSpecs();
    CHECK(output_specs.at(kVector).has_stack_axis);
    CHECK_FALSE(output_specs.at(kUnstacked).has_stack_axis);

    const auto first = torch::tensor({ { 1.0f }, { 11.0f } });
    const auto second = torch::tensor({ { 2.0f }, { 12.0f } });
    const anet::TensorDict first_frame{ { kVector, first }, { kUnstacked, first } };
    const anet::TensorDict second_frame{ { kVector, second }, { kUnstacked, second } };
    const auto initial = actor->Forward(MakeSequenceInput(first_frame, torch::tensor({ true, true }, torch::kBool)));
    CHECK(torch::equal(initial.At(kVector), first.unsqueeze(1).expand({ 2, count, 1 })));
    const auto next = actor->Forward(MakeSequenceInput(second_frame, torch::tensor({ false, true }, torch::kBool)));
    auto expected = second.unsqueeze(1).expand({ 2, count, 1 }).clone();
    expected.select(0, 0).narrow(0, 0, count - 1).fill_(1.0f);
    CHECK(torch::equal(next.At(kVector), expected));
    CHECK(torch::equal(next.At(kUnstacked), second));

    const anet::TensorDict replay{ { kVector, count == 1 ? second : expected }, { kUnstacked, second } };
    const auto sample = builder.BuildForwarder(anet::rl::MakePerceptionFromStackRequest(count, stack_keys),
        MakeSampleSpecs(frame_specs, count, stack_keys));
    const auto sampled = sample->Forward(replay);
    CHECK(torch::equal(sampled.At(kVector), expected));
    CHECK(sampled.At(kUnstacked).is_same(second));
    const auto stored = builder.BuildForwarder(anet::rl::MakeFrameWithStatsUpdateRequest(), frame_specs)->Forward(second_frame);
    CHECK(stored.At(kVector).is_same(second));
    CHECK(stored.At(kUnstacked).is_same(second));
}

TEST_CASE("DropKeys hides keys from Perception and preserves them in the stored frame", "[perception]")
{
    const int stack_count = GENERATE(1, 4);
    const auto structure = GENERATE("stack > drop >", "drop > stack");
    const std::vector<std::string> stack_keys{ kVector, kAux };
    anet::ConfigData config;
    config.Set("perception.structure", structure);
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[drop].type", kDropKeysType);
    config.Set("perception.block.[drop].keys", kAux);
    const anet::TensorSpec spec{ .shape = { 1 } };
    const anet::TensorSpecMap frame_specs{ { kVector, spec }, { kAux, spec } };
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const auto view = builder.BuildForwarder(anet::rl::MakePerceptionFromFrameRequest(stack_count, stack_keys), frame_specs);
    const auto& output_specs = view->GetOutputSpecs();
    CHECK(output_specs.at(kVector).has_stack_axis);
    CHECK_FALSE(output_specs.contains(kAux));

    const auto value = torch::ones({ 2, 1 });
    const anet::TensorDict frame{ { kVector, value }, { kAux, value } };
    auto sequence = builder.BuildForwarder(anet::rl::MakePerceptionFromSequenceRequest(2, 17, kTestStartKey, stack_count, stack_keys),
        MakeSequenceSpecs(frame_specs));
    const auto actor = sequence->Forward(MakeSequenceInput(frame, torch::ones({ 2 }, torch::kBool)));
    const auto single = view->Forward(frame);
    const auto stacked = value.unsqueeze(1).expand({2, stack_count, 1});
    const auto replay_input = stack_count == 1 ? frame : anet::TensorDict{{kVector, stacked}, {kAux, stacked}};
    const auto sample = builder.BuildForwarder(anet::rl::MakePerceptionFromStackRequest(stack_count, stack_keys),
        MakeSampleSpecs(frame_specs, stack_count, stack_keys));
    const auto replay = sample->Forward(replay_input);
    CHECK(single.At(kVector).sizes().slice(1).vec() == output_specs.at(kVector).shape);
    CHECK(single.At(kVector).scalar_type() == output_specs.at(kVector).dtype);
    for (const auto* perception : { &actor, &single, &replay }) {
        CHECK_FALSE(perception->Contains(kAux));
        CHECK(torch::equal(perception->At(kVector), stacked));
    }
    const auto stored = builder.BuildForwarder(anet::rl::MakeFrameWithStatsUpdateRequest(), frame_specs)->Forward(frame);
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
    const auto structure = GENERATE("stack > norm", "norm > stack");
    const std::vector<std::string> stack_keys{ kVector, kScaled };
    config.Set("perception.structure", structure);
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[norm].type", kNormalizeType);
    config.Set("perception.block.[norm].mode", mode);
    config.Set("perception.block.[norm].constant.mean", "1 1 1 1 1 1 1 1");
    config.Set("perception.block.[norm].constant.std", "2 2 2 2 2 2 2 2");
    config.Set("perception.block.[norm].running.use_centering", true);
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const auto update = builder.BuildForwarder(anet::rl::MakeFrameWithStatsUpdateRequest(), specs);
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
    REQUIRE(builder.GetScalar(count_key).has_value());
    CHECK(builder.GetScalar(count_key) == reference->GetScalar(count_key));

    const auto sample = builder.BuildForwarder(anet::rl::MakePerceptionFromStackRequest(4, stack_keys),
        MakeSampleSpecs(specs, 4, stack_keys));
    const auto view = builder.BuildForwarder(anet::rl::MakePerceptionFromFrameRequest(4, stack_keys), specs);
    const auto& output_specs = view->GetOutputSpecs();
    CHECK(output_specs.at(kVector).scale == (mode == kNoneMode ? anet::ValueScale::Unscaled : anet::ValueScale::Scaled));
    CHECK(output_specs.at(kVector).has_stack_axis);

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
    auto actor = builder.BuildForwarder(anet::rl::MakePerceptionFromSequenceRequest(2, 17, kTestStartKey, 4, stack_keys),
        MakeSequenceSpecs(specs));
    anet::TensorDict actor_output;
    for (std::size_t index = 0; index < frames.size(); ++index) {
        const anet::TensorDict frame_at_step{ { kVector, frames[index] }, { kScaled, frames[index] } };
        actor_output = actor->Forward(MakeSequenceInput(frame_at_step, torch::full({ 2 }, index == 0, torch::kBool)));
        if (index == 0) CHECK(torch::equal(actor_output.At(kVector), expected));
    }
    CHECK(torch::equal(actor_output.At(kVector), sampled.At(kVector)));
    CHECK(builder.GetScalar(count_key) == reference->GetScalar(count_key));
}

TEST_CASE("FrameStack provides a one-frame axis to Normalize in each Perception kind", "[perception]")
{
    anet::ConfigData config;
    config.Set("perception.structure", "stack > norm");
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[norm].type", kNormalizeType);
    config.Set("perception.block.[norm].mode", kConstantMode);
    config.Set("perception.block.[norm].constant.mean", "1 2 3");
    config.Set("perception.block.[norm].constant.std", "2 2 2");
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const anet::TensorSpecMap specs{ { kVector, { .shape = { 3 }, .scale = anet::ValueScale::Unscaled } } };
    const anet::TensorDict frame{ { kVector, torch::tensor({ { 3.0f, 4.0f, 5.0f }, { 5.0f, 6.0f, 7.0f } }) } };
    const auto check = [&](const anet::ForwarderRequest& request, const anet::TensorSpecMap& input_specs,
        const anet::TensorDict& input) {
        const auto forwarder = builder.BuildForwarder(request, input_specs);
        const auto& output_spec = forwarder->GetOutputSpecs().at(kVector);
        CHECK(output_spec.has_stack_axis);
        CHECK(output_spec.shape == std::vector<int64_t>{ 1, 3 });
        CHECK(output_spec.scale == anet::ValueScale::Scaled);
        const auto values = forwarder->Forward(input).At(kVector);
        CHECK(torch::equal(values, torch::tensor({ { 1.0f, 1.0f, 1.0f }, { 2.0f, 2.0f, 2.0f } }).unsqueeze(1)));
    };
    check(anet::rl::MakePerceptionFromFrameRequest(1, { kVector }), specs, frame);
    check(anet::rl::MakePerceptionFromStackRequest(1, { kVector }), specs, frame);
    check(anet::rl::MakePerceptionFromSequenceRequest(2, 42, kTestStartKey, 1, { kVector }), MakeSequenceSpecs(specs),
        MakeSequenceInput(frame, torch::ones({ 2 }, torch::kBool)));
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
    const anet::ForwarderBuilder builder(config, "perception", 161);
    const auto intake = builder.BuildForwarder(anet::rl::MakeFrameWithStatsUpdateRequest(), state_spec.obs_spec);
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
    CHECK(builder.GetScalar(key) == expected);

    const anet::TensorDict zeros{{kVector, torch::zeros_like(value)}};
    auto actor = builder.BuildForwarder(anet::rl::MakePerceptionFromSequenceRequest(2, 161, kTestStartKey, 1, { kVector }),
        MakeSequenceSpecs(state_spec.obs_spec));
    actor->Forward(MakeSequenceInput(zeros, torch::ones({ 2 }, torch::kBool)));
    builder.BuildForwarder(anet::rl::MakePerceptionFromFrameRequest(1, { kVector }), state_spec.obs_spec)->Forward(zeros);
    builder.BuildForwarder(anet::rl::MakePerceptionFromStackRequest(1, { kVector }), state_spec.obs_spec)->Forward(zeros);
    CHECK(builder.GetScalar(key) == expected);
}

TEST_CASE("Perception rejects invalid block declarations during construction", "[perception]")
{
    anet::ConfigData config;
    config.Set("perception.structure", "stack > norm");
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[norm].type", kNormalizeType);
    anet::TensorSpecMap specs{ { kVector, { .shape = { 8 }, .scale = anet::ValueScale::Unscaled } },
        { kAux, { .shape = { 2 } } } };
    config.Set("perception.block.[norm].mode", kNoneMode);
    std::string expected;
    int count = 1;
    bool at_build = false;
    SECTION("Unknown block") {
        config.Set("perception.structure", "stack > missing");
        expected = "perception.block.[missing].type";
    }
    SECTION("Unknown type") {
        config.Set("perception.block.[norm].type", "missing");
        expected = "perception.block.[norm].type";
    }
    SECTION("Invalid stack count in the request") {
        count = 0;
        at_build = true;
        expected = "stack_count";
    }
    SECTION("Pre-stacked input") {
        specs.at(kVector).has_stack_axis = true;
        at_build = true;
        expected = "stack_axis=";
    }
    SECTION("Empty input") {
        at_build = true;
        specs.clear();
        expected = "at least one input key";
    }
    SECTION("DropKeys leaves no output") {
        config.Set("perception.structure", "stack > drop");
        config.Set("perception.block.[drop].type", kDropKeysType);
        config.Set("perception.block.[drop].keys", "vector aux");
        at_build = true;
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
        at_build = true;
        expected = "perception.block.[norm].constant.mean";
    }
    SECTION("Unknown property in selected block") {
        config.Set("perception.block.[stack].unused", "wrong");
        expected = "perception.block.[stack].unused";
    }
    if (at_build) {
        const anet::ForwarderBuilder builder(config, "perception", 42);
        CHECK_THROWS_WITH(builder.BuildForwarder(anet::rl::MakePerceptionFromFrameRequest(count, { kVector }), specs),
            Catch::Matchers::ContainsSubstring(expected));
    } else {
        CHECK_THROWS_WITH(anet::ForwarderBuilder(config, "perception", 42), Catch::Matchers::ContainsSubstring(expected));
    }
}

TEST_CASE("FrameStack rejects an invalid sequence start spec during construction", "[perception]")
{
    anet::ConfigData config;
    config.Set("perception.structure", kStackBlock);
    config.Set("perception.block.[stack].type", kFrameStackType);
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const auto request = anet::rl::MakePerceptionFromSequenceRequest(2, 42, kTestStartKey, 1, { kVector });
    auto specs = MakeSequenceSpecs({ { kVector, { .shape = { 3 } } } });

    SECTION("The start spec must declare bool") {
        specs.at(kTestStartKey).dtype = torch::kFloat32;
    }
    SECTION("One row of the start spec must be scalar") {
        specs.at(kTestStartKey).shape = { 1 };
    }
    CHECK_THROWS_WITH(builder.BuildForwarder(request, specs),
        Catch::Matchers::ContainsSubstring(kTestStartKey) && Catch::Matchers::ContainsSubstring("scalar bool"));
}

TEST_CASE("Perception leaves former settings unread and ignores unused block definitions", "[perception]")
{
    constexpr char kFormerStackCount[] = "stucker.stack_count";
    anet::ConfigData config;
    config.Set(kFormerStackCount, 4);
    config.Set("perception.block.[unused].unexpected", "anything");
    const anet::TensorSpecMap specs{ { kVector, { .shape = { 8 } } } };
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const auto view_request = anet::rl::MakePerceptionFromFrameRequest(1, { kVector });
    const auto bare = builder.BuildForwarder(view_request, specs);
    CHECK_FALSE(bare->GetOutputSpecs().at(kVector).has_stack_axis);
    const auto unread = config.GetUnreadKeys();
    REQUIRE(unread.size() == 1);
    CHECK(unread.front().first == kFormerStackCount);
    CHECK_FALSE(builder.GetScalar(anet::rl::ObservationNormalizer::kKeyCount).has_value());

    config.Set("perception.structure", kNormBlock);
    config.Set("perception.block.[norm].type", kNormalizeType);
    config.Set("perception.block.[norm].mode", kConstantMode);
    config.Set("perception.block.[norm].constant.mean", "1 2");
    const anet::ForwarderBuilder scaled_only(config, "perception", 42);
    CHECK_FALSE(scaled_only.GetScalar(anet::rl::ObservationNormalizer::kKeyCount).has_value());
    const auto tensor = torch::ones({ 2, 8 });
    const auto view = scaled_only.BuildForwarder(view_request, specs);
    CHECK(torch::equal(view->Forward(anet::TensorDict{ { kVector, tensor } }).At(kVector), tensor));
}

TEST_CASE("Perception diagrams describe ordered blocks, tensor shapes and named seeds", "[perception]")
{
    const anet::TensorSpecMap specs{ { kVector, { .shape = { 8 }, .scale = anet::ValueScale::Unscaled } },
        { kAux, { .shape = { 1 } } } };
    anet::ConfigData config;
    config.Set("perception.structure", "stack > norm");
    config.Set("perception.block.[stack].type", kFrameStackType);
    config.Set("perception.block.[norm].type", kNormalizeType);
    config.Set("perception.block.[norm].mode", kNoneMode);
    const auto request = anet::rl::MakePerceptionFromFrameRequest(4, { kVector, kAux });
    const anet::ForwarderBuilder builder(config, "perception", 42);
    const auto view = builder.BuildForwarder(request, specs);
    REQUIRE(view->MakeGraphViz(false));
    const auto structure = view->MakeGraphViz(false)->ToDotString();
    for (const auto* fragment : { anet::rl::kForwardKindPerceptionFromFrame, "input", "output", kStackBlock, kFrameStackType, kNormBlock, kNormalizeType, kVector, kAux, "block_0\" -> \"block_1" }) {
        CHECK(structure.find(fragment) != std::string::npos);
    }
    CHECK(structure.find("\"input\" -> \"block_0\"") != std::string::npos);
    CHECK(structure.find("\"block_1\" -> \"output\"") != std::string::npos);
    const auto detail = view->MakeGraphViz(true)->ToDotString();
    for (const auto* fragment : {kUnscaledValue, kScaledValue, "stack_axis", "Float", kNoneMode}) {
        CHECK(detail.find(fragment) != std::string::npos);
    }
    const auto node_attributes = [&](const std::string& node) {
        const auto start = detail.find("\"" + node + "\" [shape=");
        REQUIRE(start != std::string::npos);
        return detail.substr(start, detail.find('\n', start) - start);
    };
    CHECK(node_attributes("input").find("shape&quot;:[8]") != std::string::npos);
    const auto stack_node = node_attributes("block_0");
    CHECK(stack_node.find(">count</TD><TD ALIGN=\"RIGHT\">4</TD>") != std::string::npos);
    CHECK(stack_node.find("shape&quot;:[4,8]") != std::string::npos);
    CHECK(node_attributes("block_1").find("shape&quot;:[4,8]") != std::string::npos);
    CHECK(node_attributes("output").find("shape&quot;:[4,8]") != std::string::npos);
    const auto seed = std::to_string(anet::SeedMaker(42).MakeNamedSeed(kNormBlock));
    CHECK(detail.find(seed) != std::string::npos);

    config.Set("perception.structure", "stack > drop > norm");
    config.Set("perception.block.[drop].type", kDropKeysType);
    config.Set("perception.block.[drop].keys", kAux);
    const anet::ForwarderBuilder reordered(config, "perception", 42);
    const auto reordered_view = reordered.BuildForwarder(request, specs);
    CHECK(reordered_view->MakeGraphViz(true)->ToDotString().find(seed) != std::string::npos);
    CHECK_FALSE(anet::MakePassThroughForwarder(specs)->MakeGraphViz(false));
}
