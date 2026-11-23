#include "anet/forwarder.hpp"

#include <algorithm>
#include <array>
#include <iterator>
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

namespace {

    struct ForwarderChainConfig final : Config {
        std::string structure;

        ForwarderChainConfig(const ConfigData& config_data, const std::string& prefix) : Config(config_data, prefix)
        {
            ANET_READ_CONFIG(config_data, structure);
        }
    };

    struct FrameStackBlockConfig final : Config {
        int count = 1;
        struct {
            std::string mode = "all";
            struct {
                std::vector<std::string> names;
            } listed;
        } keys;

        explicit FrameStackBlockConfig(const ConfigData& config_data) : Config(config_data, "")
        {
            ANET_READ_CONFIG(config_data, count);
            ANET_READ_CONFIG(config_data, keys.mode);
            ANET_READ_CONFIG(config_data, keys.listed.names);
        }
    };

    class IdentityForwarder final : public Forwarder {
    public:
        TensorDict Forward(const TensorDict& input, ForwardCapture*) override { return input; }
    };

    class FrameStackExecutor final : public ForwardExecutor {
    public:
        FrameStackExecutor(int count, const std::vector<std::string>& keys, int lane_count, torch::Device device)
            : stacker_(count, lane_count, device, keys) {}
    public:
        TensorDict Forward(const TensorDict& input, const torch::Tensor& episode_start, ForwardCapture*) override
        {
            return stacker_.Stack(input, episode_start);
        }
    private:
        rl::DictFrameStacker stacker_;
    };

    class FrameStackForwarder final : public SpecForwarder, public ForwarderProvider {
    public:
        FrameStackForwarder(int count, std::vector<std::string> keys)
            : count_(count), keys_(std::move(keys)) {}
    public:
        TensorDict Forward(const TensorDict& input, ForwardCapture*) override
        {
            ANET_PROFILE_FUNC();
            TensorDict stacked;
            for (const auto& [key, tensor] : input) {
                if (IsStacked(key)) {
                    // 可視化用の 1 Frame を全スロットで共有する。
                    auto frame = tensor.unsqueeze(1);
                    auto shape = frame.sizes().vec();
                    shape[1] = count_;
                    stacked.Set(key, frame.expand(shape));
                } else {
                    stacked.Set(key, tensor);
                }
            }
            return stacked;
        }

        TensorSpecMap ForwardSpec(const TensorSpecMap& input_specs) const override
        {
            auto stacked_specs = input_specs;
            for (auto& [key, spec] : stacked_specs) {
                if (!IsStacked(key)) continue;
                spec.shape.insert(spec.shape.begin(), count_);
                spec.has_stack_axis = true;
            }
            return stacked_specs;
        }

        std::shared_ptr<Forwarder> MakeForwarder(ForwardRequest request) const override
        {
            switch (request) {
            case ForwardRequest::FrameWithStatsUpdate:
                return nullptr;
            case ForwardRequest::PerceptionFromStack:
                // ReplayBuffer は 1 枚のときだけ stack 軸を付けずに返す。
                if (count_ > 1) return nullptr;
                return std::make_shared<FrameStackForwarder>(count_, keys_);
            case ForwardRequest::PerceptionFromFrame:
                return std::make_shared<FrameStackForwarder>(count_, keys_);
            }
            ANET_SYSTEM_ERROR("Unknown Forwarder request=" << static_cast<int>(request));
        }

        std::unique_ptr<ForwardExecutor> CreateExecutor(int lane_count, torch::Device device, seed_t) const override
        {
            return std::make_unique<FrameStackExecutor>(count_, keys_, lane_count, device);
        }
    public:
        int GetCount() const { return count_; }
        const std::vector<std::string>& GetKeys() const { return keys_; }
    private:
        bool IsStacked(const std::string& key) const
        {
            return std::ranges::find(keys_, key) != keys_.end();
        }
    private:
        const int count_;
        const std::vector<std::string> keys_;
    };

    struct DropKeysBlockConfig final : Config {
        std::vector<std::string> keys;

        explicit DropKeysBlockConfig(const ConfigData& config_data) : Config(config_data, "")
        {
            ANET_READ_CONFIG(config_data, keys);
        }
    };

    struct NormalizeBlockConfig final : Config {
        std::string mode = "running";
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

        explicit NormalizeBlockConfig(const ConfigData& config_data) : Config(config_data, "")
        {
            ANET_READ_CONFIG(config_data, mode);
            ANET_READ_CONFIG(config_data, constant.mean);
            ANET_READ_CONFIG(config_data, constant.std);
            ANET_READ_CONFIG(config_data, constant.use_clipping);
            ANET_READ_CONFIG(config_data, constant.clip_range);
            ANET_READ_CONFIG(config_data, running.use_clipping);
            ANET_READ_CONFIG(config_data, running.clip_range);
            ANET_READ_CONFIG(config_data, running.use_centering);
            ANET_READ_CONFIG(config_data, running.epsilon);
            ANET_READ_CONFIG(config_data, running.use_robust_update);
            ANET_READ_CONFIG(config_data, running.robust_warmup_count);
            ANET_READ_CONFIG(config_data, running.robust_std_threshold);
            ANET_READ_CONFIG(config_data, running.post_process_type);
            ANET_READ_CONFIG(config_data, running.post_process_threshold);
        }
    };

    class DropKeysForwarder final : public SpecForwarder {
    public:
        explicit DropKeysForwarder(std::vector<std::string> keys) : keys_(std::move(keys)) {}
    public:
        TensorDict Forward(const TensorDict& input, ForwardCapture*) override
        {
            ANET_PROFILE_FUNC();
            TensorDict output;
            for (const auto& [key, value] : input) {
                if (std::ranges::find(keys_, key) == keys_.end()) output.Set(key, value);
            }
            return output;
        }

        TensorSpecMap ForwardSpec(const TensorSpecMap& input_specs) const override
        {
            auto output_specs = input_specs;
            for (const auto& key : keys_) output_specs.erase(key);
            return output_specs;
        }
    private:
        const std::vector<std::string> keys_;
    };

    class PipelineForwarder final : public Forwarder {
    public:
        explicit PipelineForwarder(std::vector<std::shared_ptr<Forwarder>> blocks) : blocks_(std::move(blocks)) {}
    public:
        TensorDict Forward(const TensorDict& input, ForwardCapture* capture) override
        {
            ANET_PROFILE_FUNC();
            auto output = blocks_.front()->Forward(input, capture);
            for (size_t index = 1; index < blocks_.size(); ++index) {
                output = blocks_[index]->Forward(output, capture);
            }
            return output;
        }
    private:
        const std::vector<std::shared_ptr<Forwarder>> blocks_;
    };

    class StatelessExecutor final : public ForwardExecutor {
    public:
        explicit StatelessExecutor(std::shared_ptr<Forwarder> forwarder) : forwarder_(std::move(forwarder)) {}
    public:
        TensorDict Forward(const TensorDict& input, const torch::Tensor&, ForwardCapture* capture) override
        {
            return forwarder_->Forward(input, capture);
        }
    private:
        const std::shared_ptr<Forwarder> forwarder_;
    };

    class UpdateNormalizationStatsForwarder final : public Forwarder {
    public:
        explicit UpdateNormalizationStatsForwarder(std::shared_ptr<rl::ObservationNormalizer> normalizer)
            : normalizer_(std::move(normalizer)) {}
    public:
        TensorDict Forward(const TensorDict& input, ForwardCapture*) override
        {
            // クリップ率の計測も経験の受入れで行い、保存する Frame には元の値を返す。
            (void)normalizer_->NormalizeAndUpdateStats(input);
            return input;
        }
    private:
        const std::shared_ptr<rl::ObservationNormalizer> normalizer_;
    };

    class NormalizeForwarder final : public SpecForwarder, public ForwarderProvider {
    public:
        NormalizeForwarder(std::shared_ptr<rl::ObservationNormalizer> normalizer, bool apply)
            : normalizer_(std::move(normalizer)), apply_(apply) {}
    public:
        TensorDict Forward(const TensorDict& input, ForwardCapture*) override
        {
            return apply_ ? normalizer_->Normalize(input) : input;
        }

        TensorSpecMap ForwardSpec(const TensorSpecMap& input_specs) const override
        {
            auto output_specs = input_specs;
            if (apply_) {
                for (auto& [key, spec] : output_specs) {
                    if (spec.scale == ValueScale::Unscaled && torch::isFloatingType(spec.dtype)) {
                        spec.scale = ValueScale::Scaled;
                    }
                }
            }
            return output_specs;
        }

        std::shared_ptr<Forwarder> MakeForwarder(ForwardRequest request) const override
        {
            if (!apply_) return nullptr;
            if (request == ForwardRequest::FrameWithStatsUpdate) {
                return std::make_shared<UpdateNormalizationStatsForwarder>(normalizer_);
            }
            return std::make_shared<NormalizeForwarder>(normalizer_, true);
        }

        std::unique_ptr<ForwardExecutor> CreateExecutor(int, torch::Device, seed_t) const override
        {
            auto forwarder = MakeForwarder(ForwardRequest::PerceptionFromStack);
            if (!forwarder) return nullptr;
            return std::make_unique<StatelessExecutor>(std::move(forwarder));
        }
    public:
        std::optional<float> GetScalar(const std::string& key) const
        {
            return normalizer_ ? normalizer_->GetScalar(key) : std::nullopt;
        }

        std::optional<torch::Tensor> GetTensor(const std::string& key) const
        {
            return normalizer_ ? normalizer_->GetTensor(key) : std::nullopt;
        }
    private:
        const std::shared_ptr<rl::ObservationNormalizer> normalizer_;
        const bool apply_;
    };

    class PipelineExecutor final : public ForwardExecutor {
    public:
        explicit PipelineExecutor(std::vector<std::unique_ptr<ForwardExecutor>> blocks) : blocks_(std::move(blocks)) {}
    public:
        TensorDict Forward(const TensorDict& input, const torch::Tensor& episode_start, ForwardCapture* capture) override
        {
            ANET_PROFILE_FUNC();
            auto output = blocks_.front()->Forward(input, episode_start, capture);
            for (size_t index = 1; index < blocks_.size(); ++index) {
                output = blocks_[index]->Forward(output, episode_start, capture);
            }
            return output;
        }
    private:
        const std::vector<std::unique_ptr<ForwardExecutor>> blocks_;
    };

    struct NamedBlockConfig final : Config {
        NamedBlockConfig(std::string block_name, ConfigData scope, std::string default_type = {})
            : Config(scope, ""), name(std::move(block_name)), type(std::move(default_type)), config(std::move(scope))
        {
            ANET_READ_CONFIG(config, type);
        }

        std::string name;
        std::string type;
        ConfigData config;
    };

    struct ForwarderBlock {
        std::string name;
        std::string type;
        seed_t seed;
        json settings;
        std::shared_ptr<SpecForwarder> forwarder;
        std::shared_ptr<ForwarderProvider> provider;
    };

}

struct ForwarderChain::Definition {
    TensorSpecMap input_specs;
    std::shared_ptr<FrameStackForwarder> stack;
    std::shared_ptr<NormalizeForwarder> normalization;
    std::vector<ForwarderBlock> blocks;
    std::array<std::shared_ptr<Forwarder>, 3> requests;
};

static constexpr char kFrameStackType[] = "FrameStack";
static constexpr char kDropKeysType[] = "DropKeys";
static constexpr char kNormalizeType[] = "Normalize";
static constexpr char kNoneMode[] = "none";
static constexpr char kConstantMode[] = "constant";
static constexpr char kRunningMode[] = "running";
static constexpr char kAllKeys[] = "all";
static constexpr char kListedKeys[] = "listed";

static void CheckFrameSpecs(const TensorSpecMap& specs)
{
    if (specs.empty()) ANET_SYSTEM_ERROR("ForwarderChain expected at least one input key, got 0.");
    for (const auto& [key, spec] : specs) {
        if (spec.has_stack_axis) {
            ANET_SYSTEM_ERROR("ForwarderChain input key='" << key << "' has stack_axis=true; expected false.");
        }
    }
}

static std::vector<std::string> SortedSpecKeys(const TensorSpecMap& specs)
{
    std::vector<std::string> keys;
    for (const auto& [key, spec] : specs) keys.push_back(key);
    std::ranges::sort(keys);
    return keys;
}

static std::shared_ptr<FrameStackForwarder> MakeFrameStack(
    const ConfigData& block, const TensorSpecMap& input_specs, seed_t, json& settings)
{
    const FrameStackBlockConfig config(block);
    settings = config.ToJson();
    if (config.count < 1) {
        ANET_SYSTEM_ERROR("Invalid " << block.GetFullKey("count") << "=" << config.count << "; expected >= 1.");
    }
    if (config.keys.mode != kAllKeys && config.keys.mode != kListedKeys) {
        ANET_SYSTEM_ERROR("Invalid " << block.GetFullKey("keys.mode") << "='" << config.keys.mode
            << "'; expected " << kAllKeys << " or " << kListedKeys << ".");
    }
    if (config.keys.mode == kListedKeys) {
        if (config.keys.listed.names.empty()) {
            ANET_SYSTEM_ERROR("Invalid " << block.GetFullKey("keys.listed.names") << "=[]; expected at least one key.");
        }
        for (const auto& key : config.keys.listed.names) {
            if (!input_specs.contains(key)) {
                ANET_SYSTEM_ERROR("Unknown key '" << key << "' in " << block.GetFullKey("keys.listed.names")
                    << "; available keys=" << json(SortedSpecKeys(input_specs)).dump() << ".");
            }
        }
        return std::make_shared<FrameStackForwarder>(config.count, config.keys.listed.names);
    }
    return std::make_shared<FrameStackForwarder>(config.count, SortedSpecKeys(input_specs));
}

static std::shared_ptr<DropKeysForwarder> MakeDropKeys(
    const ConfigData& block, const TensorSpecMap& input_specs, seed_t, json& settings)
{
    const DropKeysBlockConfig config(block);
    settings = config.ToJson();
    if (config.keys.empty()) {
        ANET_SYSTEM_ERROR("Invalid " << block.GetFullKey("keys") << "=[]; expected at least one key.");
    }
    auto remaining = input_specs;
    for (const auto& key : config.keys) {
        if (!input_specs.contains(key)) {
            ANET_SYSTEM_ERROR("Unknown key '" << key << "' in " << block.GetFullKey("keys")
                << "; available keys=" << json(SortedSpecKeys(input_specs)).dump() << ".");
        }
        remaining.erase(key);
    }
    if (remaining.empty()) {
        ANET_SYSTEM_ERROR("Invalid " << block.GetFullKey("keys") << "=" << json(config.keys).dump()
            << "; expected at least one output key to remain.");
    }
    return std::make_shared<DropKeysForwarder>(config.keys);
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

static std::shared_ptr<NormalizeForwarder> MakeNormalize(
    const ConfigData& block, const TensorSpecMap& input_specs, seed_t, json& settings)
{
    const NormalizeBlockConfig config(block);
    settings = config.ToJson();
    if (config.mode != kNoneMode && config.mode != kConstantMode && config.mode != kRunningMode) {
        ANET_SYSTEM_ERROR("Invalid " << block.GetFullKey("mode") << "='" << config.mode
            << "'; expected " << kNoneMode << ", " << kConstantMode << " or " << kRunningMode << ".");
    }

    // モードにかかわらず各欄の値域を検査し、形との照合は定数モードで行う。
    const auto constant = MakeConstantNormalizationConfig(config);
    const auto running = MakeRunningNormalizationConfig(config);
    rl::ValidateObservationNormalizerConfig(constant);
    rl::ValidateObservationNormalizerConfig(running);
    rl::StateSpec frames;
    for (const auto& [key, spec] : input_specs) {
        if (spec.scale != ValueScale::Unscaled || !torch::isFloatingType(spec.dtype)) continue;
        auto frame = spec;
        if (frame.has_stack_axis) {
            frame.has_stack_axis = false;
            frame.shape.erase(frame.shape.begin());
        }
        frames.obs_spec.emplace(key, std::move(frame));
    }
    if (frames.obs_spec.empty()) return std::make_shared<NormalizeForwarder>(nullptr, false);
    if (config.mode == kConstantMode) {
        ValidateNormalizationConstants(config.constant.mean, block.GetFullKey("constant.mean"), frames.obs_spec);
        ValidateNormalizationConstants(config.constant.std, block.GetFullKey("constant.std"), frames.obs_spec);
    }
    const rl::ObservationNormalizerConfig passthrough{ .pass_through = true, .use_clipping = false };
    const auto& selected = config.mode == kNoneMode ? passthrough : config.mode == kConstantMode ? constant : running;
    auto normalizer = rl::ObservationNormalizerFactory(selected).CreateObservationNormalizer(frames);
    return std::make_shared<NormalizeForwarder>(std::move(normalizer), config.mode != kNoneMode);
}

static std::vector<NamedBlockConfig> ReadNamedBlocks(const ConfigData& config_data,
    const std::string& block_prefix, const std::vector<std::string>& names)
{
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
        const auto& selected = result.emplace_back(name, found->second.CreateReadScope());
        const auto& type = selected.type;
        if (type != kFrameStackType && type != kDropKeysType && type != kNormalizeType) {
            ANET_SYSTEM_ERROR("Invalid forwarder block '" << name << "': " << selected.config.GetFullKey("type")
                << "='" << type << "'; expected " << kFrameStackType << ", " << kNormalizeType << " or " << kDropKeysType << ".");
        }
        selected_keys.push_back(std::move(selected_key));
    }
    config_data.MarkInventory(block_prefix, selected_keys);
    return result;
}

static std::shared_ptr<Forwarder> MakePipeline(const std::vector<ForwarderBlock>& blocks, ForwardRequest request)
{
    std::vector<std::shared_ptr<Forwarder>> selected;
    for (const auto& block : blocks) {
        if (block.provider) {
            if (auto forwarder = block.provider->MakeForwarder(request)) selected.push_back(std::move(forwarder));
        } else if (request != ForwardRequest::FrameWithStatsUpdate) {
            selected.push_back(block.forwarder);
        }
    }
    if (selected.empty()) return std::make_shared<IdentityForwarder>();
    if (selected.size() == 1) return std::move(selected.front());
    return std::make_shared<PipelineForwarder>(std::move(selected));
}

ForwarderChain::ForwarderChain(const ConfigData& config_data, const std::string& config_prefix,
    const TensorSpecMap& input_specs, seed_t seed)
    : definition_(std::make_unique<Definition>())
{
    const ForwarderChainConfig config(config_data, config_prefix);
    const auto names = SplitPipelineString(config.structure);
    const auto block_prefix = config_prefix.empty() ? "block" : config_prefix + ".block";
    auto blocks = ReadNamedBlocks(config_data, block_prefix, names);
    const auto first_stack = std::ranges::find(blocks, kFrameStackType, &NamedBlockConfig::type);
    if (first_stack != blocks.end() && first_stack != blocks.begin()) {
        ANET_SYSTEM_ERROR("Invalid forwarder structure '" << config.structure << "': block '"
            << first_stack->name << "' is " << kFrameStackType << "; expected it first.");
    }
    if (std::ranges::count(blocks, kFrameStackType, &NamedBlockConfig::type) > 1) {
        ANET_SYSTEM_ERROR("Invalid forwarder structure '" << config.structure << "'; expected one " << kFrameStackType << ".");
    }
    if (std::ranges::count(blocks, kNormalizeType, &NamedBlockConfig::type) > 1) {
        ANET_SYSTEM_ERROR("Invalid forwarder structure '" << config.structure << "'; expected at most one " << kNormalizeType << ".");
    }
    CheckFrameSpecs(input_specs);
    definition_->input_specs = input_specs;
    if (first_stack == blocks.end()) {
        blocks.emplace(blocks.begin(), kFrameStackType, ConfigData{}, kFrameStackType);
    }

    const SeedMaker seed_maker(seed);
    auto specs = input_specs;
    for (const auto& block : blocks) {
        ForwarderBlock entry{ .name = block.name, .type = block.type, .seed = seed_maker.MakeNamedSeed(block.name.c_str()) };
        if (block.type == kFrameStackType) {
            definition_->stack = MakeFrameStack(block.config, specs, entry.seed, entry.settings);
            entry.forwarder = definition_->stack;
            entry.provider = definition_->stack;
        } else if (block.type == kNormalizeType) {
            definition_->normalization = MakeNormalize(block.config, specs, entry.seed, entry.settings);
            entry.forwarder = definition_->normalization;
            entry.provider = definition_->normalization;
        } else {
            entry.forwarder = MakeDropKeys(block.config, specs, entry.seed, entry.settings);
        }
        block.config.RequireAllKeysRead("type=" + block.type);
        specs = entry.forwarder->ForwardSpec(specs);
        definition_->blocks.push_back(std::move(entry));
    }
    for (const auto request : { ForwardRequest::FrameWithStatsUpdate, ForwardRequest::PerceptionFromStack,
        ForwardRequest::PerceptionFromFrame }) {
        definition_->requests[static_cast<size_t>(request)] = MakePipeline(definition_->blocks, request);
    }
}

ForwarderChain::~ForwarderChain() = default;

TensorDict ForwarderChain::Forward(const TensorDict& input, ForwardCapture* capture)
{
    return GetForwarder(ForwardRequest::PerceptionFromFrame)->Forward(input, capture);
}

TensorSpecMap ForwarderChain::ForwardSpec(const TensorSpecMap& input_specs) const
{
    CheckFrameSpecs(input_specs);
    auto specs = input_specs;
    for (const auto& block : definition_->blocks) specs = block.forwarder->ForwardSpec(specs);
    return specs;
}

std::shared_ptr<Forwarder> ForwarderChain::GetForwarder(ForwardRequest request) const
{
    return definition_->requests.at(static_cast<size_t>(request));
}

std::unique_ptr<ForwardExecutor> ForwarderChain::CreateExecutor(int lane_count, torch::Device device, seed_t seed) const
{
    const SeedMaker seed_maker(seed);
    std::vector<std::unique_ptr<ForwardExecutor>> executors;
    for (const auto& block : definition_->blocks) {
        if (block.provider) {
            auto executor = block.provider->CreateExecutor(lane_count, device, seed_maker.MakeNamedSeed(block.name.c_str()));
            if (executor) executors.push_back(std::move(executor));
        } else {
            executors.push_back(std::make_unique<StatelessExecutor>(block.forwarder));
        }
    }
    if (executors.size() == 1) return std::move(executors.front());
    return std::make_unique<PipelineExecutor>(std::move(executors));
}

int ForwarderChain::GetStackCount() const
{
    return definition_->stack->GetCount();
}

const std::vector<std::string>& ForwarderChain::GetStackKeys() const
{
    return definition_->stack->GetKeys();
}

std::optional<float> ForwarderChain::GetScalar(const std::string& key) const
{
    return definition_->normalization ? definition_->normalization->GetScalar(key) : std::nullopt;
}

std::optional<torch::Tensor> ForwarderChain::GetTensor(const std::string& key) const
{
    return definition_->normalization ? definition_->normalization->GetTensor(key) : std::nullopt;
}

std::unique_ptr<graphviz::GraphViz> ForwarderChain::MakeGraphViz(bool detail) const
{
    auto graph = std::make_unique<graphviz::Tree>("perception", graphviz::LayoutDirection::LeftToRight);
    auto specs = definition_->input_specs;
    const auto add_specs = [](graphviz::LabelData& label, const TensorSpecMap& tensor_specs, bool include_details) {
        const auto keys = SortedSpecKeys(tensor_specs);
        label.AddAttr("keys", json(keys).dump());
        if (include_details) {
            for (const auto& key : keys) {
                const auto full_spec = tensor_specs.at(key).ToJson();
                const json summary = {{"shape", full_spec.at("shape")}, {"dtype", full_spec.at("dtype")},
                    {"scale", full_spec.at("scale")}, {"stack_axis", full_spec.at("stack_axis")}};
                label.AddAttr(key, summary.dump());
            }
        }
    };

    std::string previous = "frame";
    add_specs(graph->AddNode(previous).label.SetTitle("Frame"), specs, detail);
    for (size_t index = 0; index < definition_->blocks.size(); ++index) {
        const auto& block = definition_->blocks[index];
        const auto id = "block_" + std::to_string(index);
        auto& label = graph->AddNode(id).label.SetTitle(block.name).AddAttr("type", block.type);
        if (detail) {
            label.AddAttr("seed", block.seed);
            for (const auto& [key, value] : block.settings.items()) label.AddAttr(key, value.dump());
        }
        specs = block.forwarder->ForwardSpec(specs);
        if (detail) add_specs(label, specs, true);
        graph->AddEdge(previous, id);
        previous = id;
    }
    add_specs(graph->AddNode("output").label.SetTitle("Perception"), specs, detail);
    graph->AddEdge(previous, "output");
    return graph;
}
