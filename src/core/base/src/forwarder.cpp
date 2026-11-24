#include "anet/forwarder.hpp"

#include <algorithm>
#include <iterator>
#include <mutex>
#include "anet/init.hpp"
#include "anet/log.hpp"
#include "anet/profile.hpp"
#include "anet/str_util.hpp"
#include "anet/graphviz.hpp"
#include "anet/scaler.hpp"
#include "anet/stacker.hpp"

using namespace anet;

// 完全一致と末尾スラッシュの接頭辞は、同じ要求判定を共有する。
static bool MatchesCaptureRequest(std::string_view request, std::string_view key)
{
    return request.ends_with('/') ? key.starts_with(request) : key == request;
}

ForwardCapture::ForwardCapture(std::vector<std::string> requests)
    : requests_(std::move(requests)) {}

bool ForwardCapture::Wants(std::string_view key) const
{
    return std::ranges::any_of(requests_, [key](const auto& request) { return MatchesCaptureRequest(request, key); });
}

bool ForwardCapture::WantsAny(std::string_view prefix) const
{
    return std::any_of(requests_.begin(), requests_.end(), [prefix](const auto& request) {
        return std::string_view(request).starts_with(prefix)
            || (request.ends_with('/') && prefix.starts_with(request));
    });
}

void ForwardCapture::Add(std::string_view key, torch::Tensor value)
{
    // 要求外と二重書込みを発生点で検出し、上書きで不整合を隠さない。
    if (!Wants(key)) ANET_SYSTEM_ERROR("ForwardCapture: unrequested key='" << key << "'.");
    if (std::ranges::any_of(entries_, [key](const auto& entry) { return entry.first == key; })) {
        ANET_SYSTEM_ERROR("ForwardCapture: duplicate key='" << key << "'; expected one Tensor per key.");
    }
    entries_.emplace_back(std::string(key), std::move(value));
}

torch::Tensor ForwardCapture::Get(std::string_view key) const
{
    for (const auto& [name, value] : entries_) if (name == key) return value;
    return {};
}

std::vector<ForwardCapture::Entry> ForwardCapture::GetAll(std::string_view prefix) const
{
    // 順序は実行順のままとし、mapの辞書順へ並べ替えない。
    std::vector<Entry> result;
    for (const auto& entry : entries_) if (entry.first.starts_with(prefix)) result.push_back(entry);
    return result;
}

std::vector<std::string> ForwardCapture::Unfulfilled() const
{
    std::vector<std::string> result;
    for (const auto& request : requests_) {
        if (!std::ranges::any_of(entries_, [&](const auto& entry) { return MatchesCaptureRequest(request, entry.first); })) {
            result.push_back(request);
        }
    }
    return result;
}

void ForwardCapture::Reset()
{
    entries_.clear();
}

constexpr char kFrameStackType[] = "FrameStack";
constexpr char kDropKeysType[] = "DropKeys";
constexpr char kNormalizeType[] = "Normalize";
constexpr char kSetSpecType[] = "SetSpec";
constexpr char kNoneMode[] = "none";
constexpr char kConstantMode[] = "constant";
constexpr char kRunningMode[] = "running";
constexpr char kAllKeys[] = "all";
constexpr char kListedKeys[] = "listed";

namespace {

    enum class BlockConfigRead { Keys, Values };

    class ForwarderBlockConfig : public Config {
    protected:
        ForwarderBlockConfig(const ConfigData& data, BlockConfigRead read) : Config(data, ""), read_(read) {}

        template<typename T>
        void ReadConfig(const ConfigData& data, const std::string& key, T& value)
        {
            if (ReadsValues()) Config::ReadConfig(data, key, value);
            else (void)data.Get(key);
        }

        bool ReadsValues() const { return read_ == BlockConfigRead::Values; }
    private:
        BlockConfigRead read_;
    };

    struct ForwarderBuilderConfig final : Config {
        std::string structure;
        ForwarderBuilderConfig(const ConfigData& config_data, const std::string& prefix) : Config(config_data, prefix)
        {
            ANET_READ_CONFIG(config_data, structure);
        }
    };

    struct DropKeysBlockConfig final : ForwarderBlockConfig {
        std::vector<std::string> keys;
        DropKeysBlockConfig(const ConfigData& data, BlockConfigRead read) : ForwarderBlockConfig(data, read)
        {
            ANET_READ_CONFIG(data, keys);
        }
    };

    struct NormalizeBlockConfig final : ForwarderBlockConfig {
        std::string mode = kRunningMode;
        struct {
            std::vector<float> mean;
            std::vector<float> std;
            bool use_clipping = true;
            float clip_range = 10.0f;
        } constant;
        struct {
            bool use_clipping = true;
            float clip_range = 10.0f;
            bool use_centering = false;
            float epsilon = 1.0f;
            bool use_robust_update = true;
            int robust_warmup_count = 200;
            float robust_std_threshold = 5.0f;
            int post_process_type = rl::kPostProcessSymLog;
            float post_process_threshold = 10.0f;
        } running;

        NormalizeBlockConfig(const ConfigData& data, BlockConfigRead read) : ForwarderBlockConfig(data, read)
        {
            ANET_READ_CONFIG(data, mode);
            ANET_READ_CONFIG(data, constant.mean);
            ANET_READ_CONFIG(data, constant.std);
            ANET_READ_CONFIG(data, constant.use_clipping);
            ANET_READ_CONFIG(data, constant.clip_range);
            ANET_READ_CONFIG(data, running.use_clipping);
            ANET_READ_CONFIG(data, running.clip_range);
            ANET_READ_CONFIG(data, running.use_centering);
            ANET_READ_CONFIG(data, running.epsilon);
            ANET_READ_CONFIG(data, running.use_robust_update);
            ANET_READ_CONFIG(data, running.robust_warmup_count);
            ANET_READ_CONFIG(data, running.robust_std_threshold);
            ANET_READ_CONFIG(data, running.post_process_type);
            ANET_READ_CONFIG(data, running.post_process_threshold);
        }
    };

    struct NamedBlockConfig final : Config {
        NamedBlockConfig(std::string block_name, ConfigData scope)
            : Config(scope, ""), name(std::move(block_name)), config(std::move(scope))
        {
            ANET_READ_CONFIG(config, type);
        }

        std::string name;
        std::string type;
        ConfigData config;
    };

}

static std::vector<std::string> SortedSpecKeys(const TensorSpecMap& specs)
{
    std::vector<std::string> keys;
    for (const auto& [key, spec] : specs) keys.push_back(key);
    std::ranges::sort(keys);
    return keys;
}

static rl::ObservationNormalizerConfig MakeConstantNormalizationConfig(const NormalizeBlockConfig& config)
{
    rl::ObservationNormalizerConfig result{
        .use_clipping = config.constant.use_clipping,
        .clip_range = config.constant.clip_range,
        .constant_mean = config.constant.mean,
        .constant_std = config.constant.std,
        .use_dynamic_scaling = false,
        .full_keys = config.GetFullKeys().Extract("constant"),
    };
    result.full_keys.CopyFieldFrom("constant_mean", config.GetFullKeys(), "constant.mean");
    result.full_keys.CopyFieldFrom("constant_std", config.GetFullKeys(), "constant.std");
    return result;
}

static rl::ObservationNormalizerConfig MakeRunningNormalizationConfig(const NormalizeBlockConfig& config)
{
    return {
        .use_clipping = config.running.use_clipping,
        .clip_range = config.running.clip_range,
        .use_centering = config.running.use_centering,
        .epsilon = config.running.epsilon,
        .use_robust_update = config.running.use_robust_update,
        .robust_warmup_count = config.running.robust_warmup_count,
        .robust_std_threshold = config.running.robust_std_threshold,
        .post_process_type = config.running.post_process_type,
        .post_process_threshold = config.running.post_process_threshold,
        .full_keys = config.GetFullKeys().Extract("running"),
    };
}

static void ValidateNormalizationConstants(const std::vector<float>& values, const std::string& field,
    const TensorSpecMap& frame_specs)
{
    if (values.empty()) return;
    for (const auto& [key, spec] : frame_specs) {
        const auto expected = spec.CalcFlattenDim();
        if (std::ssize(values) != expected) {
            ANET_SYSTEM_ERROR("Invalid " << field << " for Frame key='" << key << "': got " << values.size()
                << " values; expected " << expected << ".");
        }
    }
}

static std::vector<NamedBlockConfig> ReadNamedBlocks(const ConfigData& config_data,
    const std::string& block_prefix, const std::vector<std::string>& names)
{
    // structure に列挙された名前だけを解決し、選ばれた各ブロックの型を検査する。
    const auto blocks = config_data.MakeSubConfigData(block_prefix);
    std::vector<std::string> selected_keys;
    std::vector<NamedBlockConfig> result;
    for (const auto& name : names) {
        auto selected_key = block_prefix + ".[" + name + "]";
        const auto found = blocks.find(name);
        if (found == blocks.end()) {
            ANET_SYSTEM_ERROR("Missing forwarder block '" << name << "'; expected "
                << config_data.GetFullKey(selected_key + ".type") << ".");
        }
        result.emplace_back(name, found->second.CreateReadScope());
        selected_keys.push_back(std::move(selected_key));
    }
    // 未使用の定義を区別できるよう、実体化した項目を設定カタログに記録する。
    config_data.MarkInventory(block_prefix, selected_keys);
    return result;
}

namespace {

    struct ForwarderFactories {
        static ForwarderFactories& Get()
        {
            static ForwarderFactories factories;
            return factories;
        }

        std::mutex mutex;
        std::unordered_map<std::string, std::shared_ptr<ForwarderModuleFactory>> by_type;
    };

    class PassThroughForwarder final : public SpecForwarder {
    public:
        explicit PassThroughForwarder(const TensorSpecMap& specs) : specs_(specs) {}
    public:
        const TensorSpecMap& GetInputSpecs() const override { return specs_; }
        const TensorSpecMap& GetOutputSpecs() const override { return specs_; }
        TensorDict Forward(const TensorDict& input, ForwardCapture*) override { return input; }
    private:
        TensorSpecMap specs_;
    };

    struct BuiltForwarderBlock {
        std::string name;
        std::string type;
        seed_t seed;
        std::shared_ptr<SpecForwarder> forwarder;
    };

    class BuiltForwarder final : public SpecForwarder {
    public:
        BuiltForwarder(ForwardKind kind, const TensorSpecMap& input_specs, std::vector<BuiltForwarderBlock> blocks)
            : kind_(std::move(kind)), input_specs_(input_specs), blocks_(std::move(blocks)) {}
    public:
        const TensorSpecMap& GetInputSpecs() const override { return input_specs_; }
        const TensorSpecMap& GetOutputSpecs() const override
        {
            return blocks_.empty() ? input_specs_ : blocks_.back().forwarder->GetOutputSpecs();
        }
        TensorDict Forward(const TensorDict& input, ForwardCapture* capture) override
        {
            ANET_PROFILE_FUNC();
            if (blocks_.empty()) return input;
            auto output = blocks_.front().forwarder->Forward(input, capture);
            for (std::size_t index = 1; index < blocks_.size(); ++index) {
                output = blocks_[index].forwarder->Forward(output, capture);
            }
            return output;
        }
        std::unique_ptr<graphviz::GraphViz> MakeGraphViz(bool detail) const override
        {
            auto graph = std::make_unique<graphviz::Tree>(kind_, graphviz::LayoutDirection::LeftToRight);
            std::string previous = "input";
            AddSpecs(graph->AddNode(previous).label.SetTitle(previous), input_specs_, detail);
            for (std::size_t index = 0; index < blocks_.size(); ++index) {
                const auto& block = blocks_[index];
                const auto id = "block_" + std::to_string(index);
                auto& label = graph->AddNode(id).label.SetTitle(block.name).AddAttr("type", block.type);
                if (detail) {
                    label.AddAttr("seed", block.seed);
                    for (const auto& [key, value] : block.forwarder->GetGraphVizDetails()) label.AddAttr(key, value);
                    AddSpecs(label, block.forwarder->GetOutputSpecs(), true);
                }
                graph->AddEdge(previous, id);
                previous = id;
            }
            constexpr char kOutputNode[] = "output";
            AddSpecs(graph->AddNode(kOutputNode).label.SetTitle(kOutputNode), GetOutputSpecs(), detail);
            graph->AddEdge(previous, kOutputNode);
            return graph;
        }
    private:
        static void AddSpecs(graphviz::LabelData& label, const TensorSpecMap& specs, bool detail)
        {
            const auto keys = SortedSpecKeys(specs);
            label.AddAttr("keys", json(keys).dump());
            if (!detail) return;
            for (const auto& key : keys) {
                const auto spec = specs.at(key).ToJson();
                json summary;
                for (const auto* field : { TensorSpec::kShapeField, TensorSpec::kDtypeField,
                    TensorSpec::kScaleField, TensorSpec::kStackAxisField }) {
                    summary[field] = spec.at(field);
                }
                label.AddAttr(key, summary.dump());
            }
        }
    private:
        ForwardKind kind_;
        TensorSpecMap input_specs_;
        std::vector<BuiltForwarderBlock> blocks_;
    };

}

struct ForwarderBuilder::Definition {
    struct Block {
        std::string name;
        std::string type;
        seed_t seed;
        std::shared_ptr<ForwarderModule> module;
    };

    std::optional<std::vector<ForwardKind>> forward_kinds;
    std::vector<Block> blocks;
};

std::unique_ptr<graphviz::GraphViz> SpecForwarder::MakeGraphViz(bool) const
{
    return {};
}

ForwarderRequest::ForwarderRequest(ForwardKind forward_kind, ConfigData options)
    : forward_kind_(std::move(forward_kind)), options_(std::move(options))
{
}

std::shared_ptr<SpecForwarder> anet::MakePassThroughForwarder(const TensorSpecMap& specs)
{
    return std::make_shared<PassThroughForwarder>(specs);
}

void anet::RegisterForwarderModuleFactory(const std::string& type_name,
    std::shared_ptr<ForwarderModuleFactory> factory)
{
    if (!factory) {
        ANET_SYSTEM_ERROR("Forwarder type='" << type_name << "' requires a factory.");
    }
    auto& factories = ForwarderFactories::Get();
    const std::lock_guard lock(factories.mutex);
    if (!factories.by_type.emplace(type_name, std::move(factory)).second) {
        ANET_SYSTEM_ERROR("Forwarder type='" << type_name << "' is already registered; expected unique type names.");
    }
}

ForwarderBuilder::ForwarderBuilder(const ConfigData& config_data, const std::string& config_prefix, seed_t seed)
    : definition_(std::make_unique<Definition>())
{
    const ForwarderBuilderConfig config(config_data, config_prefix);
    const auto names = SplitPipelineString(config.structure);
    const auto block_prefix = config_prefix.empty() ? "block" : config_prefix + ".block";
    const auto blocks = ReadNamedBlocks(config_data, block_prefix, names);
    std::vector<std::shared_ptr<ForwarderModuleFactory>> block_factories;
    for (const auto& block : blocks) {
        auto& factories = ForwarderFactories::Get();
        const std::lock_guard lock(factories.mutex);
        const auto factory = factories.by_type.find(block.type);
        if (factory == factories.by_type.end()) {
            ANET_SYSTEM_ERROR("Invalid " << block.config.GetFullKey("type") << "='" << block.type
                << "'; expected a registered Forwarder type.");
        }
        block_factories.push_back(factory->second);
    }
    for (const auto& factory : block_factories) {
        const auto kinds = factory->GetForwardKinds();
        if (!kinds) continue;
        auto& common_kinds = definition_->forward_kinds;
        if (common_kinds) {
            std::erase_if(*common_kinds, [&](const auto& kind) {
                return std::ranges::find(*kinds, kind) == kinds->end();
            });
        } else {
            common_kinds = kinds;
        }
    }
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        const auto& block = blocks[index];
        block_factories[index]->ReadConfigKeys(block.config);
        block.config.RequireAllKeysRead("type=" + block.type);
    }
    const SeedMaker seed_maker(seed);
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        const auto& block = blocks[index];
        const ForwarderModuleContext context{
            .seed = seed_maker.MakeNamedSeed(block.name.c_str()),
            .forward_kinds = definition_->forward_kinds,
        };
        auto module = block_factories[index]->CreateModule(block.config, context);
        if (!module) {
            ANET_SYSTEM_ERROR("Forwarder factory type='" << block.type << "' returned an empty module for block '" << block.name << "'.");
        }
        definition_->blocks.push_back({ .name = block.name, .type = block.type, .seed = context.seed, .module = std::move(module) });
    }
    for (const auto& block : definition_->blocks) {
        block.module->ValidateForwardKinds(definition_->forward_kinds);
    }
}

ForwarderBuilder::~ForwarderBuilder() = default;

std::shared_ptr<SpecForwarder> ForwarderBuilder::BuildForwarder(const ForwarderRequest& request,
    const TensorSpecMap& input_specs) const
{
    if (input_specs.empty()) {
        ANET_SYSTEM_ERROR("ForwarderBuilder expected at least one input key, got 0.");
    }
    const auto& forward_kind = request.GetForwardKind();
    const auto& supported = definition_->forward_kinds;
    if (supported && std::ranges::find(*supported, forward_kind) == supported->end()) {
        ANET_SYSTEM_ERROR("ForwarderBuilder does not support ForwardKind='" << forward_kind
            << "'; supported kinds=" << json(*supported).dump() << ".");
    }
    std::vector<BuiltForwarderBlock> blocks;
    const TensorSpecMap* specs = &input_specs;
    for (const auto& block : definition_->blocks) {
        auto forwarder = block.module->CreateForwarder(request, *specs);
        if (!forwarder) {
            ANET_SYSTEM_ERROR("Forwarder block='" << block.name << "' returned an empty Forwarder for kind='" << forward_kind << "'.");
        }
        specs = &forwarder->GetOutputSpecs();
        blocks.push_back({ .name = block.name, .type = block.type, .seed = block.seed, .forwarder = std::move(forwarder) });
    }
    return std::make_shared<BuiltForwarder>(forward_kind, input_specs, std::move(blocks));
}

std::optional<float> ForwarderBuilder::GetScalar(const std::string& key, int64_t index) const
{
    for (const auto& block : definition_->blocks) {
        if (auto value = block.module->GetScalar(key, index)) return value;
    }
    return std::nullopt;
}

std::optional<torch::Tensor> ForwarderBuilder::GetTensor(const std::string& key, int64_t index) const
{
    for (const auto& block : definition_->blocks) {
        if (auto value = block.module->GetTensor(key, index)) return value;
    }
    return std::nullopt;
}

constexpr char kStackCountOption[] = "stack_count";
constexpr char kStackKeysOption[] = "stack_keys";
constexpr char kRowCountOption[] = "row_count";
constexpr char kStartKeyOption[] = "start_key";
constexpr char kSeedOption[] = "seed";

static ConfigData MakeFrameStackOptions(int count, const std::vector<std::string>& keys)
{
    ConfigData options;
    options.Set(kStackCountOption, count);
    options.Set(kStackKeysOption, keys);
    return options;
}

static std::vector<ForwardKind> GetPerceptionForwardKinds()
{
    return { rl::kForwardKindFrameWithStatsUpdate, rl::kForwardKindPerceptionFromStack,
        rl::kForwardKindPerceptionFromFrame, rl::kForwardKindPerceptionFromSequence };
}

using GraphVizDetails = std::vector<std::pair<std::string, std::string>>;

static GraphVizDetails MakeConfigDetails(const Config& config)
{
    const auto fields = config.ToJson();
    GraphVizDetails details;
    for (const auto& [key, value] : fields.items()) details.emplace_back(key, value.dump());
    return details;
}

static rl::StateSpec MakeNormalizationFrameSpecs(const TensorSpecMap& input_specs)
{
    rl::StateSpec frames;
    for (const auto& [key, spec] : input_specs) {
        if (spec.scale != ValueScale::Unscaled || !torch::isFloatingType(spec.dtype) || spec.IsDiscrete()) continue;
        auto frame = spec;
        if (frame.has_stack_axis) {
            if (frame.shape.empty()) ANET_SYSTEM_ERROR("Normalize key='" << key << "' has no dimension for its stack axis.");
            frame.shape.erase(frame.shape.begin());
            frame.has_stack_axis = false;
        }
        frames.obs_spec.emplace(key, std::move(frame));
    }
    return frames;
}

namespace {

    class ConfiguredForwarder : public SpecForwarder {
    public:
        ConfiguredForwarder(const TensorSpecMap& input_specs, TensorSpecMap output_specs, GraphVizDetails details)
            : input_specs_(input_specs), output_specs_(std::move(output_specs)), details_(std::move(details)) {}
    public:
        const TensorSpecMap& GetInputSpecs() const override { return input_specs_; }
        const TensorSpecMap& GetOutputSpecs() const override { return output_specs_; }
        GraphVizDetails GetGraphVizDetails() const override { return details_; }
    private:
        TensorSpecMap input_specs_;
        TensorSpecMap output_specs_;
        GraphVizDetails details_;
    };

    class TransformForwarder final : public ConfiguredForwarder {
    public:
        TransformForwarder(const TensorSpecMap& input_specs, TensorSpecMap output_specs,
            GraphVizDetails details, TensorDictFunction transform = {})
            : ConfiguredForwarder(input_specs, std::move(output_specs), std::move(details)),
              transform_(std::move(transform)) {}
    public:
        TensorDict Forward(const TensorDict& input, ForwardCapture*) override
        {
            return transform_ ? transform_(input) : input;
        }
    private:
        TensorDictFunction transform_;
    };

    class SequenceStackForwarder final : public ConfiguredForwarder {
    public:
        SequenceStackForwarder(const TensorSpecMap& input_specs, TensorSpecMap output_specs, GraphVizDetails details,
            int count, int row_count, std::string start_key, std::vector<std::string> keys)
            : ConfiguredForwarder(input_specs, std::move(output_specs), std::move(details)),
              count_(count), row_count_(row_count), start_key_(std::move(start_key)), keys_(std::move(keys)) {}
    public:
        TensorDict Forward(const TensorDict& input, ForwardCapture*) override
        {
            ANET_PROFILE_FUNC();
            const auto starts = input.Get(start_key_);
            if (!starts || !starts->defined() || !starts->device().is_cpu() || starts->scalar_type() != torch::kBool
                || starts->dim() != 1 || starts->size(0) != row_count_) {
                ANET_SYSTEM_ERROR("FrameStack key='" << start_key_ << "'; expected CPU bool with shape [" << row_count_ << "].");
            }
            if (groups_.empty()) {
                for (const auto& key : keys_) {
                    const auto device = input.At(key).device();
                    auto group = std::ranges::find(groups_, device, &StackGroup::device);
                    if (group == groups_.end()) {
                        groups_.push_back({ .device = device, .keys = { key } });
                    } else {
                        group->keys.push_back(key);
                    }
                }
                for (auto& group : groups_) {
                    group.stacker.emplace(count_, row_count_, group.device, group.keys);
                }
            }
            TensorDict output;
            for (const auto& [key, value] : input) {
                if (key != start_key_) output.Set(key, value);
            }
            for (auto& group : groups_) {
                TensorDict frames;
                for (const auto& key : group.keys) frames.Set(key, input.At(key));
                const auto stacked = group.stacker->Stack(frames, *starts);
                for (const auto& [key, value] : stacked) output.Set(key, value);
            }
            return output;
        }
    private:
        struct StackGroup {
            torch::Device device;
            std::vector<std::string> keys;
            std::optional<rl::DictFrameStacker> stacker;
        };
    private:
        int count_;
        int row_count_;
        std::string start_key_;
        std::vector<std::string> keys_;
        std::vector<StackGroup> groups_;
    };

    class FrameStackModule final : public ForwarderModule {
    public:
        std::shared_ptr<SpecForwarder> CreateForwarder(const ForwarderRequest& request,
            const TensorSpecMap& input_specs) const override
        {
            const auto& kind = request.GetForwardKind();
            if (kind == rl::kForwardKindFrameWithStatsUpdate) return MakePassThroughForwarder(input_specs);
            const auto& options = request.GetOptions();
            const auto count = options.Get<int>(kStackCountOption);
            if (count < 1) {
                ANET_SYSTEM_ERROR("FrameStack " << kStackCountOption << "=" << count << "; expected >= 1.");
            }
            const bool from_stack = kind == rl::kForwardKindPerceptionFromStack;
            auto output_specs = input_specs;
            std::vector<std::string> keys_to_stack;
            for (const auto& key : options.Get<std::vector<std::string>>(kStackKeysOption)) {
                const auto found = output_specs.find(key);
                if (found == output_specs.end()) continue;
                if (std::ranges::find(keys_to_stack, key) != keys_to_stack.end()) continue;
                auto& spec = found->second;
                if (from_stack && spec.has_stack_axis) continue;
                if (spec.has_stack_axis || (from_stack && count > 1)) {
                    ANET_SYSTEM_ERROR("FrameStack kind='" << kind << "' key='" << key << "' stack_axis="
                        << spec.has_stack_axis << "; expected " << (from_stack && count > 1) << ".");
                }
                spec.shape.insert(spec.shape.begin(), count);
                spec.has_stack_axis = true;
                keys_to_stack.push_back(key);
            }
            GraphVizDetails details{ { "count", std::to_string(count) }, { "keys", json(keys_to_stack).dump() } };
            if (kind == rl::kForwardKindPerceptionFromSequence) {
                auto start_key = options.Get(kStartKeyOption);
                if (!output_specs.erase(start_key)) {
                    ANET_SYSTEM_ERROR("FrameStack key='" << start_key << "'; expected a start key in the input specs.");
                }
                const auto row_count = options.Get<int>(kRowCountOption);
                if (row_count < 1) {
                    ANET_SYSTEM_ERROR("FrameStack " << kRowCountOption << "=" << row_count << "; expected >= 1.");
                }
                return std::make_shared<SequenceStackForwarder>(input_specs, std::move(output_specs), std::move(details),
                    count, row_count, std::move(start_key), std::move(keys_to_stack));
            }
            return std::make_shared<TransformForwarder>(input_specs, std::move(output_specs), std::move(details),
                [count, keys = std::move(keys_to_stack)](const TensorDict& input) {
                    ANET_PROFILE_SCOPE(ExpandFrameStack);
                    auto output = input;
                    for (const auto& key : keys) {
                        auto frame = input.At(key).unsqueeze(1);
                        auto shape = frame.sizes().vec();
                        shape[1] = count;
                        output.Set(key, frame.expand(shape));
                    }
                    return output;
                });
        }
    };

    class FrameStackFactory final : public ForwarderModuleFactory {
    public:
        std::optional<std::vector<ForwardKind>> GetForwardKinds() const override { return GetPerceptionForwardKinds(); }
        std::shared_ptr<ForwarderModule> CreateModule(const ConfigData&, const ForwarderModuleContext&) const override
        {
            return std::make_shared<FrameStackModule>();
        }
    };

    class NormalizeModule final : public ForwarderModule {
    public:
        explicit NormalizeModule(const NormalizeBlockConfig& config)
            : config_(config), constant_(MakeConstantNormalizationConfig(config)),
              running_(MakeRunningNormalizationConfig(config)), details_(MakeConfigDetails(config))
        {
            rl::ValidateObservationNormalizerConfig(constant_);
            rl::ValidateObservationNormalizerConfig(running_);
        }
    public:
        std::shared_ptr<SpecForwarder> CreateForwarder(const ForwarderRequest& request,
            const TensorSpecMap& input_specs) const override
        {
            const bool intake = request.GetForwardKind() == rl::kForwardKindFrameWithStatsUpdate;
            const bool running = config_.mode == kRunningMode;
            const bool passthrough = config_.mode == kNoneMode;
            const auto frames = MakeNormalizationFrameSpecs(input_specs);
            std::shared_ptr<rl::ObservationNormalizer> normalizer;
            {
                const std::lock_guard lock(mutex_);
                if (running) {
                    if (intake && !running_specs_) {
                        metrics_normalizer_ = rl::ObservationNormalizerFactory(running_).CreateObservationNormalizer(frames);
                        running_specs_ = frames.obs_spec;
                    }
                    CheckRunningSpecs(frames.obs_spec, intake);
                    normalizer = metrics_normalizer_;
                } else if (!frames.obs_spec.empty() && (intake || !passthrough)) {
                    if (!passthrough) {
                        ValidateNormalizationConstants(config_.constant.mean, constant_.full_keys.GetFullKey("constant_mean"), frames.obs_spec);
                        ValidateNormalizationConstants(config_.constant.std, constant_.full_keys.GetFullKey("constant_std"), frames.obs_spec);
                    }
                    const rl::ObservationNormalizerConfig none{ .pass_through = true, .use_clipping = false };
                    normalizer = rl::ObservationNormalizerFactory(passthrough ? none : constant_).CreateObservationNormalizer(frames);
                    if (intake) metrics_normalizer_ = normalizer;
                }
            }
            if (passthrough || frames.obs_spec.empty()) {
                return std::make_shared<TransformForwarder>(input_specs, input_specs, details_);
            }

            auto output_specs = input_specs;
            std::vector<std::string> keys;
            std::vector<std::string> stacked_intake_keys;
            for (const auto& [key, frame_spec] : frames.obs_spec) {
                keys.push_back(key);
                auto& output_spec = output_specs.at(key);
                if (intake) {
                    if (output_spec.has_stack_axis) stacked_intake_keys.push_back(key);
                } else {
                    output_spec.scale = ValueScale::Scaled;
                }
            }
            return std::make_shared<TransformForwarder>(input_specs, std::move(output_specs), details_,
                [normalizer, keys = std::move(keys), stacked_intake_keys = std::move(stacked_intake_keys), intake]
                (const TensorDict& input) {
                    ANET_PROFILE_SCOPE(NormalizeObservation);
                    TensorDict selected;
                    for (const auto& key : keys) {
                        const auto value = input.At(key);
                        selected.Set(key, std::ranges::find(stacked_intake_keys, key) == stacked_intake_keys.end()
                            ? value : value.flatten(0, 1));
                    }
                    if (intake) {
                        // クリップ率も更新し、保存するデータはそのまま通す。
                        (void)normalizer->NormalizeAndUpdateStats(selected);
                        return input;
                    }
                    auto result = input;
                    const auto normalized = normalizer->Normalize(selected);
                    for (const auto& [key, tensor] : normalized) result.Set(key, tensor);
                    return result;
                });
        }

        std::optional<float> GetScalar(const std::string& key, int64_t index) const override
        {
            const std::lock_guard lock(mutex_);
            return metrics_normalizer_ ? metrics_normalizer_->GetScalar(key, index) : std::nullopt;
        }

        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override
        {
            const std::lock_guard lock(mutex_);
            return metrics_normalizer_ ? metrics_normalizer_->GetTensor(key, index) : std::nullopt;
        }
    private:
        void CheckRunningSpecs(const TensorSpecMap& frame_specs, bool intake) const
        {
            for (const auto& [key, spec] : frame_specs) {
                if (!running_specs_ || !running_specs_->contains(key)) {
                    ANET_SYSTEM_ERROR("Normalize key='" << key << "' has no running statistics; expected "
                        << rl::kForwardKindFrameWithStatsUpdate << " to supply the key first.");
                }
                const auto& stored_shape = running_specs_->at(key).shape;
                if (stored_shape != spec.shape) {
                    ANET_SYSTEM_ERROR("Normalize key='" << key << "' frame shape=" << json(spec.shape).dump()
                        << "; expected " << json(stored_shape).dump() << " for running statistics.");
                }
            }
            if (intake && running_specs_->size() != frame_specs.size()) {
                ANET_SYSTEM_ERROR("Normalize intake keys=" << json(SortedSpecKeys(frame_specs)).dump()
                    << "; expected " << json(SortedSpecKeys(*running_specs_)).dump() << " for running statistics.");
            }
        }
    private:
        NormalizeBlockConfig config_;
        rl::ObservationNormalizerConfig constant_;
        rl::ObservationNormalizerConfig running_;
        GraphVizDetails details_;
        mutable std::mutex mutex_;
        mutable std::shared_ptr<rl::ObservationNormalizer> metrics_normalizer_;
        mutable std::optional<TensorSpecMap> running_specs_;
    };

    class NormalizeFactory final : public ForwarderModuleFactory {
    public:
        std::optional<std::vector<ForwardKind>> GetForwardKinds() const override { return GetPerceptionForwardKinds(); }
        void ReadConfigKeys(const ConfigData& data) const override { (void)NormalizeBlockConfig(data, BlockConfigRead::Keys); }
        std::shared_ptr<ForwarderModule> CreateModule(const ConfigData& data, const ForwarderModuleContext&) const override
        {
            const NormalizeBlockConfig config(data, BlockConfigRead::Values);
            if (config.mode != kNoneMode && config.mode != kConstantMode && config.mode != kRunningMode) {
                ANET_SYSTEM_ERROR("Invalid " << data.GetFullKey("mode") << "='" << config.mode
                    << "'; expected " << kNoneMode << ", " << kConstantMode << " or " << kRunningMode << ".");
            }
            return std::make_shared<NormalizeModule>(config);
        }
    };

    template<typename T>
    struct SetSpecAttribute {
        std::string mode = "keep";
        T set;
    };

    struct SetSpecBlockConfig final : ForwarderBlockConfig {
        struct Range { std::vector<double> values; };
        struct Labels { std::vector<std::string> names; };
        std::vector<std::string> keys;
        std::string scale = "keep";
        SetSpecAttribute<Range> min_values;
        SetSpecAttribute<Range> max_values;
        SetSpecAttribute<Labels> labels;
        struct {
            std::string mode = kAllKeys;
            struct {
                std::vector<std::string> names;
            } listed;
        } forward_kinds;

        SetSpecBlockConfig(const ConfigData& data, BlockConfigRead read) : ForwarderBlockConfig(data, read)
        {
            ANET_READ_CONFIG(data, keys);
            ANET_READ_CONFIG(data, scale);
            ANET_READ_CONFIG(data, min_values.mode);
            ANET_READ_CONFIG(data, max_values.mode);
            ANET_READ_CONFIG(data, labels.mode);
            ANET_READ_CONFIG(data, forward_kinds.mode);
            ANET_READ_CONFIG(data, forward_kinds.listed.names);
            ReadNumbers(data, "min_values.set.values", min_values.set.values);
            ReadNumbers(data, "max_values.set.values", max_values.set.values);
            ANET_READ_CONFIG(data, labels.set.names);
        }

        void ReadNumbers(const ConfigData& data, const std::string& field, std::vector<double>& values)
        {
            const auto tokens = data.Get<std::vector<std::string>>(field);
            if (!ReadsValues()) return;
            const auto full_key = data.GetFullKey(field);
            ConfigData scalar;
            for (const auto& token : tokens) {
                // 各要素を設定の double と同じ文法で読み、診断には指定元の完全キーを使う。
                scalar.Set(full_key, token);
                values.push_back(scalar.Get<double>(full_key));
            }
            RecordFullKey(field, full_key);
            RecordConfigValue(field, values);
        }
    };

    class SetSpecModule final : public ForwarderModule {
    public:
        explicit SetSpecModule(const SetSpecBlockConfig& config)
            : config_(config), details_(MakeConfigDetails(config)) {}
    public:
        void ValidateForwardKinds(const std::optional<std::vector<ForwardKind>>& supported) const override
        {
            if (!supported || config_.forward_kinds.mode != kListedKeys) return;
            for (const auto& name : config_.forward_kinds.listed.names) {
                if (std::ranges::find(*supported, name) == supported->end()) {
                    ANET_SYSTEM_ERROR("Invalid " << config_.GetFullKey("forward_kinds.listed.names") << "='" << name
                        << "'; supported kinds=" << json(*supported).dump() << ".");
                }
            }
        }

        std::shared_ptr<SpecForwarder> CreateForwarder(const ForwarderRequest& request,
            const TensorSpecMap& input_specs) const override
        {
            if (config_.forward_kinds.mode == kListedKeys
                && std::ranges::find(config_.forward_kinds.listed.names, request.GetForwardKind()) == config_.forward_kinds.listed.names.end()) {
                return std::make_shared<TransformForwarder>(input_specs, input_specs, details_);
            }
            auto output_specs = input_specs;
            for (const auto& key : config_.keys) {
                const auto found = output_specs.find(key);
                if (found == output_specs.end()) {
                    ANET_SYSTEM_ERROR("Unknown key '" << key << "' in " << config_.GetFullKey("keys")
                        << "; available keys=" << json(SortedSpecKeys(input_specs)).dump() << ".");
                }
                auto& spec = found->second;
                const auto frame_spec = FrameSpec(spec);
                if (config_.scale == "Unscaled") {
                    if (spec.IsDiscrete() || c10::isIntegralType(spec.dtype, /*includeBool=*/false)) {
                        ANET_SYSTEM_ERROR("Invalid " << config_.GetFullKey("scale") << "=Unscaled for key='" << key
                            << "' dtype=" << spec.dtype << " num_classes=" << spec.num_classes
                            << "; expected a non-integral dtype with no discrete classes.");
                    }
                    spec.scale = ValueScale::Unscaled;
                } else if (config_.scale == "Scaled") {
                    spec.scale = ValueScale::Scaled;
                }
                if (config_.min_values.mode == "set") {
                    CheckRangeCount(key, "min_values.set.values", config_.min_values.set.values, frame_spec.CalcFlattenDim());
                    spec.min_values = config_.min_values.set.values;
                }
                if (config_.max_values.mode == "set") {
                    CheckRangeCount(key, "max_values.set.values", config_.max_values.set.values, frame_spec.CalcFlattenDim());
                    spec.max_values = config_.max_values.set.values;
                }
                if (config_.labels.mode == "set") {
                    auto labeled = frame_spec;
                    labeled.labels = config_.labels.set.names;
                    if (!labeled.HasValidLabels()) {
                        ANET_SYSTEM_ERROR("Invalid " << config_.GetFullKey("labels.set.names") << " for key='" << key
                            << "': got " << labeled.labels.size() << " labels; expected labels matching one frame, shape="
                            << json(labeled.shape).dump() << ".");
                    }
                    spec.labels = config_.labels.set.names;
                }
            }
            return std::make_shared<TransformForwarder>(input_specs, std::move(output_specs), details_);
        }
    private:
        static TensorSpec FrameSpec(TensorSpec spec)
        {
            if (spec.has_stack_axis && !spec.shape.empty()) spec.shape.erase(spec.shape.begin());
            return spec;
        }

        void CheckRangeCount(const std::string& key, const std::string& field,
            const std::vector<double>& values, int64_t expected) const
        {
            if (values.size() != 1 && std::ssize(values) != expected) {
                ANET_SYSTEM_ERROR("Invalid " << config_.GetFullKey(field) << " for key='" << key << "': got " << values.size()
                    << " values; expected 1 or " << expected << ".");
            }
        }
    private:
        SetSpecBlockConfig config_;
        GraphVizDetails details_;
    };

    class SetSpecFactory final : public ForwarderModuleFactory {
    public:
        std::optional<std::vector<ForwardKind>> GetForwardKinds() const override { return std::nullopt; }
        void ReadConfigKeys(const ConfigData& data) const override { (void)SetSpecBlockConfig(data, BlockConfigRead::Keys); }
        std::shared_ptr<ForwarderModule> CreateModule(const ConfigData& data, const ForwarderModuleContext&) const override
        {
            const SetSpecBlockConfig config(data, BlockConfigRead::Values);
            if (config.keys.empty()) {
                ANET_SYSTEM_ERROR("Invalid " << data.GetFullKey("keys") << "=[]; expected at least one key.");
            }
            if (config.scale != "keep" && config.scale != "Scaled" && config.scale != "Unscaled") {
                ANET_SYSTEM_ERROR("Invalid " << data.GetFullKey("scale") << "='" << config.scale << "'; expected keep, Scaled or Unscaled.");
            }
            CheckAttribute(data, "min_values", "set.values", config.min_values.mode, config.min_values.set.values.empty());
            CheckAttribute(data, "max_values", "set.values", config.max_values.mode, config.max_values.set.values.empty());
            CheckAttribute(data, "labels", "set.names", config.labels.mode, config.labels.set.names.empty());
            if (config.scale == "keep" && config.min_values.mode == "keep"
                && config.max_values.mode == "keep" && config.labels.mode == "keep") {
                ANET_SYSTEM_ERROR("Invalid " << data.GetFullKey("type") << "=" << kSetSpecType
                    << "; expected at least one attribute to change.");
            }
            if (config.forward_kinds.mode != kAllKeys && config.forward_kinds.mode != kListedKeys) {
                ANET_SYSTEM_ERROR("Invalid " << data.GetFullKey("forward_kinds.mode") << "='" << config.forward_kinds.mode
                    << "'; expected " << kAllKeys << " or " << kListedKeys << ".");
            }
            if (config.forward_kinds.mode == kListedKeys && config.forward_kinds.listed.names.empty()) {
                ANET_SYSTEM_ERROR("Invalid " << data.GetFullKey("forward_kinds.listed.names")
                    << "=[]; expected at least one ForwardKind.");
            }
            return std::make_shared<SetSpecModule>(config);
        }
    private:
        static void CheckAttribute(const ConfigData& data, const std::string& field, const std::string& value_field,
            const std::string& mode, bool empty)
        {
            if (mode != "keep" && mode != "set") {
                ANET_SYSTEM_ERROR("Invalid " << data.GetFullKey(field + ".mode") << "='" << mode
                    << "'; expected keep or set.");
            }
            if (mode == "set" && empty) {
                ANET_SYSTEM_ERROR("Invalid " << data.GetFullKey(field + "." + value_field)
                    << "=[]; expected at least one value.");
            }
        }
    };

    class DropKeysModule final : public ForwarderModule {
    public:
        explicit DropKeysModule(const DropKeysBlockConfig& config) : config_(config), details_(MakeConfigDetails(config)) {}
    public:
        std::shared_ptr<SpecForwarder> CreateForwarder(const ForwarderRequest& request,
            const TensorSpecMap& input_specs) const override
        {
            if (request.GetForwardKind() == rl::kForwardKindFrameWithStatsUpdate) {
                return std::make_shared<TransformForwarder>(input_specs, input_specs, details_);
            }
            auto output_specs = input_specs;
            for (const auto& key : config_.keys) {
                if (!input_specs.contains(key)) {
                    ANET_SYSTEM_ERROR("Unknown key '" << key << "' in " << config_.GetFullKey("keys")
                        << "; available keys=" << json(SortedSpecKeys(input_specs)).dump() << ".");
                }
                output_specs.erase(key);
            }
            if (output_specs.empty()) {
                ANET_SYSTEM_ERROR("Invalid " << config_.GetFullKey("keys") << "=" << json(config_.keys).dump()
                    << "; expected at least one output key to remain.");
            }
            return std::make_shared<TransformForwarder>(input_specs, std::move(output_specs), details_,
                [keys = config_.keys](const TensorDict& input) {
                    TensorDict output;
                    for (const auto& [key, value] : input) {
                        if (std::ranges::find(keys, key) == keys.end()) output.Set(key, value);
                    }
                    return output;
                });
        }
    private:
        DropKeysBlockConfig config_;
        GraphVizDetails details_;
    };

    class DropKeysFactory final : public ForwarderModuleFactory {
    public:
        std::optional<std::vector<ForwardKind>> GetForwardKinds() const override { return GetPerceptionForwardKinds(); }
        void ReadConfigKeys(const ConfigData& data) const override { (void)DropKeysBlockConfig(data, BlockConfigRead::Keys); }
        std::shared_ptr<ForwarderModule> CreateModule(const ConfigData& data, const ForwarderModuleContext&) const override
        {
            const DropKeysBlockConfig config(data, BlockConfigRead::Values);
            if (config.keys.empty()) {
                ANET_SYSTEM_ERROR("Invalid " << data.GetFullKey("keys") << "=[]; expected at least one key.");
            }
            return std::make_shared<DropKeysModule>(config);
        }
    };

}

ForwarderRequest rl::MakeFrameWithStatsUpdateRequest()
{
    return ForwarderRequest(kForwardKindFrameWithStatsUpdate);
}

ForwarderRequest rl::MakePerceptionFromStackRequest(int stack_count, std::vector<std::string> stack_keys)
{
    return ForwarderRequest(kForwardKindPerceptionFromStack, MakeFrameStackOptions(stack_count, stack_keys));
}

ForwarderRequest rl::MakePerceptionFromFrameRequest(int stack_count, std::vector<std::string> stack_keys)
{
    return ForwarderRequest(kForwardKindPerceptionFromFrame, MakeFrameStackOptions(stack_count, stack_keys));
}

ForwarderRequest rl::MakePerceptionFromSequenceRequest(int row_count, seed_t seed, std::string start_key,
    int stack_count, std::vector<std::string> stack_keys)
{
    auto options = MakeFrameStackOptions(stack_count, stack_keys);
    options.Set(kRowCountOption, row_count);
    options.Set(kStartKeyOption, start_key);
    options.Set(kSeedOption, seed);
    return ForwarderRequest(kForwardKindPerceptionFromSequence, std::move(options));
}

namespace anet::rl::detail {

void InitForwarderModules()
{
    static const auto failure = CaptureRegistrationFailure([] {
        RegisterForwarderModuleFactory(kFrameStackType, std::make_shared<FrameStackFactory>());
        RegisterForwarderModuleFactory(kNormalizeType, std::make_shared<NormalizeFactory>());
        RegisterForwarderModuleFactory(kDropKeysType, std::make_shared<DropKeysFactory>());
        RegisterForwarderModuleFactory(kSetSpecType, std::make_shared<SetSpecFactory>());
    });
    if (failure) std::rethrow_exception(failure);
}

}
