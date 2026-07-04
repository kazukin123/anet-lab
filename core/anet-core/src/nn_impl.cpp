// nn_impl.cpp

#include "nn_impl.hpp"

#include <charconv>
#include <iostream>
#include <sstream>
#include <regex>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <format>
#include <iomanip>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <ATen/ops/_foreach_lerp.h>
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/tensor_util.hpp"
#include "anet/nn_util.hpp"


using namespace anet::nn;
namespace LOG = anet::log;


static constexpr const char* kNetBlockConfigKeyPrefix = "net.block.[";
static constexpr const char* kNetBlockConfigKeySuffix = "]";


// ===========================================================================
// Helper Functions (Internal)
// ===========================================================================

// 文字列のトリム
static std::string Trim(const std::string& s)
{
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) start++;
    auto end = s.end();
    do { end--; } while (std::distance(start, end) > 0 && std::isspace(*end));
    return std::string(start, end + 1);
}

// パイプライン文字列の分割 (カッコ内のスペースは無視する)
static std::vector<std::string> SplitPipelineString(const std::string& s)
{
    std::string normalized = s;
    // '>' をスペースに置換
    std::replace(normalized.begin(), normalized.end(), '>', ' ');

    std::vector<std::string> tokens;
    std::string current_token;
    int paren_depth = 0;

    for (char c : normalized) {
        if (c == '(') {
            paren_depth++;
        } else if (c == ')') {
            if (paren_depth > 0) paren_depth--;
        }

        if (std::isspace(c) && paren_depth == 0) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else {
            current_token += c;
        }
    }
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }

    return tokens;
}

// 文字列から整数への変換
static int ParseInt(const std::string& s, const std::string& param_name)
{
    try {
        return std::stoi(s);
    } catch (...) {
        throw std::runtime_error("Invalid integer parameter: " + param_name + " = " + s);
    }
}

static double ParseDoubleStrict(const std::string& s, const std::string& param_name)
{
    try {
        size_t pos = 0;
        const double value = std::stod(s, &pos);
        if (pos != s.size()) {
            ANET_SYSTEM_ERROR(
                "Invalid double parameter: " << param_name
                << " value=\"" << s << "\" expected=double.");
        }
        return value;
    } catch (const std::exception&) {
        ANET_SYSTEM_ERROR(
            "Invalid double parameter: " << param_name
            << " value=\"" << s << "\" expected=double.");
    }
    return 0.0;
}

static std::string FormatDoubleForConfig(double value)
{
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

static std::string MakeDotNodeId(const std::string& prefix, const std::string& key)
{
    std::string id = prefix + "_";
    id.reserve(prefix.size() + key.size() + 1);
    for (unsigned char c : key) {
        if (std::isalnum(c) || c == '_' || c == '-' || c == '.') {
            id.push_back(static_cast<char>(c));
        } else {
            id.push_back('_');
        }
    }
    return id;
}

static std::string FormatInt64Vector(const std::vector<int64_t>& values)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

static std::string FormatStringVector(const std::vector<std::string>& values)
{
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << values[i];
    }
    return oss.str();
}

static std::optional<double> TryParseFloatingLiteral(const std::string& value)
{
    if (value.find_first_of(".eE") == std::string::npos) {
        return std::nullopt;
    }

    double number = 0.0;
    const char* first = value.data();
    const char* last = first + value.size();
    const auto [ptr, ec] = std::from_chars(first, last, number, std::chars_format::general);
    if (ec == std::errc{} && ptr == last) {
        return number;
    }

    return std::nullopt;
}

static void AddConfigAttr(anet::graphviz::LabelData& label, const std::string& key, const std::string& value, int precision)
{
    if (const auto number = TryParseFloatingLiteral(value)) {
        label.AddAttr(key, *number, precision);
        return;
    }

    label.AddAttr(key, value);
}

static anet::graphviz::LayoutDirection ToGraphVizLayoutDirection(const NetworkGraphVizConfig& config)
{
    if (config.layout == "LR") return anet::graphviz::LayoutDirection::LeftToRight;
    if (config.layout == "TB") return anet::graphviz::LayoutDirection::TopToBottom;

    ANET_SYSTEM_ERROR("Invalid NetworkGraphVizConfig.layout: " << config.layout << ". Expected LR or TB.");
    return anet::graphviz::LayoutDirection::LeftToRight;
}

static void RegisterNetworkGraphVizStyles(anet::graphviz::Tree& tree)
{
    anet::graphviz::NodeStyle input_style;
    input_style.shape = anet::graphviz::NodeShape::Ellipse;
    input_style.border_style = anet::graphviz::BorderStyle::Filled;
    input_style.color = "#2E86AB";
    input_style.fillcolor = "#E8F4FA";
    tree.RegisterNodeStyle("input", input_style);

    anet::graphviz::NodeStyle branch_style;
    branch_style.shape = anet::graphviz::NodeShape::Box;
    branch_style.border_style = anet::graphviz::BorderStyle::Filled;
    branch_style.color = "#7D6608";
    branch_style.fillcolor = "#FCF3CF";
    tree.RegisterNodeStyle("branch", branch_style);

    anet::graphviz::NodeStyle block_style;
    block_style.shape = anet::graphviz::NodeShape::Plain;
    tree.RegisterNodeStyle("block", block_style);

    anet::graphviz::NodeStyle output_style;
    output_style.shape = anet::graphviz::NodeShape::Ellipse;
    output_style.border_style = anet::graphviz::BorderStyle::Filled;
    output_style.color = "#1E8449";
    output_style.fillcolor = "#E9F7EF";
    tree.RegisterNodeStyle("output", output_style);

    anet::graphviz::NodeStyle head_output_style;
    head_output_style.shape = anet::graphviz::NodeShape::Ellipse;
    head_output_style.border_style = anet::graphviz::BorderStyle::Filled;
    head_output_style.color = "#C0392B";
    head_output_style.fillcolor = "#FDEDEC";
    tree.RegisterNodeStyle("head_output", head_output_style);

    anet::graphviz::NodeStyle head_style;
    head_style.shape = anet::graphviz::NodeShape::Diamond;
    head_style.border_style = anet::graphviz::BorderStyle::Filled;
    head_style.color = "#6C3483";
    head_style.fillcolor = "#F4ECF7";
    tree.RegisterNodeStyle("head", head_style);
}

static void AddTensorSpecAttrs(anet::graphviz::LabelData& label, const anet::TensorSpec& spec)
{
    label.AddAttr("shape", FormatInt64Vector(spec.shape));
    label.AddAttr("dtype", std::string(c10::toString(spec.dtype)));
    if (spec.num_classes > 0) {
        label.AddAttr("num_classes", spec.num_classes);
    }
}

static void AddHeadGraphVizInfo(anet::graphviz::LabelData& label, const HeadGraphVizInfo& info, const NetworkGraphVizConfig& config)
{
    if (!info.type.empty()) {
        label.SetTitle(info.type);
    }

    if (!config.show_head_info) return;

    for (const auto& [key, value] : info.details) {
        label.AddAttr(key, value);
    }
}

void anet::nn::ValidateNetworkGraphVizConfig(const NetworkGraphVizConfig& config, const std::string& owner)
{
    if (config.layout != "LR" && config.layout != "TB") {
        ANET_SYSTEM_ERROR("Invalid " << owner << ".layout: " << config.layout << ". Expected LR or TB.");
    }
    if (config.float_precision < 0) {
        ANET_SYSTEM_ERROR("Invalid " << owner << ".float_precision: " << config.float_precision << ". Expected non-negative integer.");
    }
}


// ===========================================================================
// NetworkConfig
// ===========================================================================

static std::map<std::string, NetworkBlockConfig> ReadBlockConfig(const anet::ConfigData& config_data, const std::string& config_prefix)
{
    std::map<std::string, NetworkBlockConfig> block_configs;

    // グローバルとローカルのタグ別設定を一時保持
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> global_tag_block_map;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> local_tag_block_map;

    // プレフィックスのピリオドを正規表現エスケープする ("net.dyn" -> "net\.dyn")
    std::string escaped_prefix = std::regex_replace(config_prefix, std::regex(R"(\.)"), R"(\.)");

    // グローバル(net) と ローカル(config_prefix) の正規表現
    std::regex re_global(R"(net\.block\.\[([^\]]+)\]\.(.+))");
    std::regex re_local(escaped_prefix + R"(\.block\.\[([^\]]+)\]\.(.+))");

    auto config_map = config_data.Map();
    for (const auto& [key, value] : config_map) {
        std::smatch m;
        // まずローカルマッチを試す
        if (std::regex_match(key, m, re_local)) {
            local_tag_block_map[m[1].str()][m[2].str()] = value;
            continue;
        }
        // 次にグローバルマッチを試す
        if (std::regex_match(key, m, re_global)) {
            global_tag_block_map[m[1].str()][m[2].str()] = value;
        }
    }

    // グローバルとローカルをマージ (ローカルが優先上書きされる)
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> merged_tag_block_map = std::move(global_tag_block_map);
    for (const auto& [tag, config_map] : local_tag_block_map) {
        for (const auto& [sub_key, val] : config_map) {
            merged_tag_block_map[tag][sub_key] = val;
        }
    }

    for (const auto& [tag, config_map] : merged_tag_block_map) {
        // NetworkBlockConfig を作る
        NetworkBlockConfig block_config;
        for (const auto& [config_sub_key, config_value] : config_map) {
            if (config_sub_key == "type") {
                block_config.type = config_value;
            } else {
                block_config.config_data.Set(config_sub_key, config_value);
            }
        }
        ANET_LOG_DEBUG("block_config type=" << block_config.type << "\n config=" << block_config.config_data.ToString());

        // チェック
        if (block_config.type.empty()) {
            ANET_SYSTEM_ERROR("Block type not specified for block: " + tag);
        }

        // 作ったNetworkBlockConfigを保存
        block_configs[tag] = std::move(block_config);
    }

    return block_configs;
}

class ConfigProfile {
public:
    ConfigProfile(std::string tag, ConfigProfileConfig config)
        : tag_(std::move(tag)), config_(std::move(config))
    {
    }
    virtual void ValidateConfig() const = 0;
    virtual double GenerateValue(size_t index, size_t count) const = 0;
    virtual ~ConfigProfile() = default;
protected:
    const std::string& tag() const { return tag_; }
    const ConfigProfileConfig& config() const { return config_; }
private:
    std::string tag_;
    ConfigProfileConfig config_;
};

class LinearConfigProfile final : public ConfigProfile {
public:
    using ConfigProfile::ConfigProfile;

    void ValidateConfig() const override
    {
        if (config().type != "linear") {
            ANET_SYSTEM_ERROR(
                "Invalid net.config_profile.[" << tag() << "].type: " << config().type
                << ". Expected linear.");
        }
        if (!config().has_end) {
            ANET_SYSTEM_ERROR(
                "Missing required net.config_profile.[" << tag() << "].end. "
                "Expected a double value.");
        }
    }

    double GenerateValue(size_t index, size_t count) const override
    {
        if (count <= 1) {
            return config().start;
        }

        const double t = static_cast<double>(index) / static_cast<double>(count - 1);
        return config().start + (config().end - config().start) * t;
    }
};

using ConfigProfilePtrMap = std::map<std::string, std::unique_ptr<ConfigProfile>>;

static std::unique_ptr<ConfigProfile> CreateConfigProfile(
    const std::string& tag, const ConfigProfileConfig& config)
{
    // profile定義をtypeに応じた実装へ変換する。
    if (config.type == "linear") {
        return std::make_unique<LinearConfigProfile>(tag, config);
    }
    ANET_SYSTEM_ERROR(
        "Invalid net.config_profile.[" << tag << "].type: " << config.type
        << ". Expected linear.");
    return nullptr;
}

static ConfigProfilePtrMap CreateConfigProfileMap(
    const std::map<std::string, ConfigProfileConfig>& configs)
{
    ConfigProfilePtrMap profiles;
    for (const auto& [tag, config] : configs) {
        auto profile = CreateConfigProfile(tag, config);
        profile->ValidateConfig();
        profiles.emplace(tag, std::move(profile));
    }
    return profiles;
}

static void ApplyConfigProfileConfigField(
    ConfigProfileConfig& config,
    const std::string& full_key,
    const std::string& sub_key,
    const std::string& value)
{
    if (sub_key == "type") {
        config.type = value;
        config.has_type = true;
    } else if (sub_key == "start") {
        config.start = ParseDoubleStrict(value, full_key);
        config.has_start = true;
    } else if (sub_key == "end") {
        config.end = ParseDoubleStrict(value, full_key);
        config.has_end = true;
    } else {
        ANET_SYSTEM_ERROR(
            "Unknown config profile key: " << full_key
            << ". Expected one of: type, start, end.");
    }
}

static std::map<std::string, ConfigProfileConfig> MergeConfigProfileConfig(
    const std::map<std::string, ConfigProfileConfig>& base_profiles,
    const std::map<std::string, ConfigProfileConfig>& override_profiles)
{
    auto merged_profiles = base_profiles;
    for (const auto& [tag, override_config] : override_profiles) {
        auto& config = merged_profiles[tag];
        if (override_config.has_type) {
            config.type = override_config.type;
            config.has_type = true;
        }
        if (override_config.has_start) {
            config.start = override_config.start;
            config.has_start = true;
        }
        if (override_config.has_end) {
            config.end = override_config.end;
            config.has_end = true;
        }
    }
    return merged_profiles;
}

static std::map<std::string, ConfigProfileConfig> ReadConfigProfileConfig(
    const anet::ConfigData& config_data, const std::string& config_prefix)
{
    std::map<std::string, ConfigProfileConfig> config_profiles;

    // net.block と同じく、グローバル net とローカル config_prefix を merge し、ローカルを優先する。
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> global_profile_map;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> local_profile_map;

    std::string escaped_prefix = std::regex_replace(config_prefix, std::regex(R"(\.)"), R"(\.)");
    std::regex re_global(R"(net\.config_profile\.\[([^\]]+)\]\.(.+))");
    std::regex re_local(escaped_prefix + R"(\.config_profile\.\[([^\]]+)\]\.(.+))");

    for (const auto& [key, value] : config_data.Map()) {
        std::smatch m;
        if (std::regex_match(key, m, re_local)) {
            local_profile_map[m[1].str()][m[2].str()] = value;
            continue;
        }
        if (std::regex_match(key, m, re_global)) {
            global_profile_map[m[1].str()][m[2].str()] = value;
        }
    }

    auto merged_profile_map = std::move(global_profile_map);
    for (const auto& [tag, profile_map] : local_profile_map) {
        for (const auto& [sub_key, value] : profile_map) {
            merged_profile_map[tag][sub_key] = value;
        }
    }

    for (const auto& [tag, profile_map] : merged_profile_map) {
        ConfigProfileConfig config;
        // profile定義を読み取り、未知キーは後方互換せずに設定エラーとして扱う。
        for (const auto& [sub_key, value] : profile_map) {
            const std::string full_key = "net.config_profile.[" + tag + "]." + sub_key;
            ApplyConfigProfileConfigField(config, full_key, sub_key, value);
        }
        // 読み取ったprofileを実装objectへ変換し、型固有の必須項目を検証する。
        auto profile = CreateConfigProfile(tag, config);
        profile->ValidateConfig();
        config_profiles[tag] = config;
    }

    return config_profiles;
}

static std::map<std::string, NetworkBranchConfig> ReadBranchConfig(const anet::ConfigData& config_data, const std::string& config_prefix)
{
    std::map<std::string, NetworkBranchConfig> branches;
    std::map<std::string, std::map<std::string, ConfigProfileConfig>> branch_config_profiles;

    std::string escaped_prefix = std::regex_replace(config_prefix, std::regex(R"(\.)"), R"(\.)");
    std::regex re_branch(escaped_prefix + R"(\.branch\.\[([^\]]+)\]\.(bind|structure|auto_format))");
    std::regex re_branch_config_profile(
        escaped_prefix + R"(\.branch\.\[([^\]]+)\]\.config_profile\.\[([^\]]+)\]\.(.+))");
    std::regex re_raw(R"(\(raw\))");

    for (const auto& [key, value] : config_data.Map()) {
        std::smatch m;
        if (std::regex_match(key, m, re_branch_config_profile)) {
            const std::string b_name = m[1].str();
            const std::string tag = m[2].str();
            const std::string sub_key = m[3].str();
            auto& config = branch_config_profiles[b_name][tag];
            ApplyConfigProfileConfigField(config, key, sub_key, value);
            continue;
        }

        if (std::regex_match(key, m, re_branch)) {
            std::string b_name = m[1].str();
            std::string b_prop = m[2].str();

            branches[b_name].name = b_name;

            if (b_prop == "bind") {
                std::stringstream ss(value);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    item = Trim(item);
                    if (item.empty()) continue;

                    bool is_raw = false;
                    if (std::regex_search(item, re_raw)) {
                        is_raw = true;
                        item = std::regex_replace(item, re_raw, "");
                        item = Trim(item);
                    }
                    branches[b_name].bind_keys.push_back(item);
                    if (is_raw) {
                        branches[b_name].raw_keys.push_back(item);
                    }
                }
            } else if (b_prop == "structure") {
                branches[b_name].structure_str = value;
            } else if (b_prop == "auto_format") {
                branches[b_name].auto_format = (anet::ToLower(value) == "true" || value == "1");
            }
        }
    }

    for (auto& [b_name, config_profiles] : branch_config_profiles) {
        auto it = branches.find(b_name);
        if (it == branches.end()) {
            ANET_SYSTEM_ERROR(
                "Branch config_profile is specified for undefined branch '" << b_name
                << "'. Define '" << config_prefix << ".branch.[" << b_name << "].bind' and 'structure' first.");
        }
        it->second.config_profiles = std::move(config_profiles);
    }

    // 「auto_format=false」の場合、全ての bind_keysを raw_keys を追加
    for (auto& [b_name, branch_cfg] : branches) {    // 全ブランチでチェック
        if (!branch_cfg.auto_format) {
            for (const auto& k : branch_cfg.bind_keys) {
                if (!anet::Contains(branch_cfg.raw_keys, k)) {
                    branch_cfg.raw_keys.push_back(k);
                }
            }
        }
    }

    // ブランチ無しだとエラー
    if (branches.empty()) {
        ANET_SYSTEM_ERROR(
            "No branches defined in NetworkConfig for prefix '" << config_prefix << "'. "
            "Please define at least one branch using '" << config_prefix << ".branch.[name].bind' and 'structure'."
        );
    }

    return branches;
}

NetworkConfig::NetworkConfig(const anet::ConfigData& config_data, const std::string& config_prefix)
{
    block_configs = ReadBlockConfig(config_data, config_prefix);
    config_profiles = ReadConfigProfileConfig(config_data, config_prefix);
    branches = ReadBranchConfig(config_data, config_prefix);

    // [config_prefix].body.output.[Head期待キー] = Bodyブランチ名 を読み取る
    std::string escaped_prefix = std::regex_replace(config_prefix, std::regex(R"(\.)"), R"(\.)");
    std::regex re_output(escaped_prefix + R"(\.body\.output\.\[([^\]]+)\])");
    for (const auto& [key, value] : config_data.Map()) {
        std::smatch m;
        if (std::regex_match(key, m, re_output)) {
            output_keys[m[1].str()] = value;
        }
    }
}

anet::json NetworkConfig::ToJson() const
{
    anet::json j;
    j["blocks"] = anet::json::object();
    j["branches"] = anet::json::object();

    // 各ブランチのstructure_strをパースし、使用されているブロック名を抽出
    std::set<std::string> used_blocks;
    std::regex re_rep(R"(\(\*(\d+)\))");

    for (const auto& [b_name, b_cfg] : branches) {
        // nn_impl.cpp上部で定義されている SplitPipelineString と Trim を利用
        auto tokens = SplitPipelineString(b_cfg.structure_str);

        for (const auto& token_raw : tokens) {
            std::string current_token = token_raw;

            // "ResBlock(*4)" のようなリピート指定部分を削ぎ落とす
            std::smatch m_rep;
            if (std::regex_search(current_token, m_rep, re_rep)) {
                current_token = std::regex_replace(current_token, re_rep, "");
            }

            std::string block_def_name = Trim(current_token);
            if (!block_def_name.empty()) {
                used_blocks.insert(block_def_name);
            }
        }
    }

    // 使用されているブロックのみをJSONに出力
    for (const auto& [k, v] : block_configs) {
        if (used_blocks.find(k) != used_blocks.end()) {
            j["blocks"][k] = { {"type", v.type}, {"config", v.config_data.ToJson()} };
        }
    }

    // ブランチ情報はすべて出力
    for (const auto& [k, v] : branches) {
        j["branches"][k] = { {"bind_keys", v.bind_keys}, {"structure", v.structure_str}, {"raw_keys", v.raw_keys} };
        if (!v.config_profiles.empty()) {
            j["branches"][k]["config_profiles"] = anet::json::object();
            for (const auto& [profile_name, profile_config] : v.config_profiles) {
                auto profile_json = anet::json::object();
                if (profile_config.has_type) {
                    profile_json["type"] = profile_config.type;
                }
                if (profile_config.has_start) {
                    profile_json["start"] = profile_config.start;
                }
                if (profile_config.has_end) {
                    profile_json["end"] = profile_config.end;
                }
                j["branches"][k]["config_profiles"][profile_name] = std::move(profile_json);
            }
        }
    }

    // profile 定義は、展開後の per-block 実効値と合わせて dump から追えるようにする。
    j["config_profiles"] = anet::json::object();
    for (const auto& [k, v] : config_profiles) {
        j["config_profiles"][k] = { {"type", v.type}, {"start", v.start}, {"end", v.end} };
    }

    // 出力のマッピング情報を出力
    for (const auto& [head_key, branch_key] : output_keys) {
        j["output_keys"][head_key] = branch_key;
    }

    return j;
}


// ===========================================================================
// NetworkBlock (Runtime Node)
// ===========================================================================

NetworkBlock::NetworkBlock(std::string name, std::shared_ptr<NetworkModule> module)
    : name_(std::move(name)), module_(std::move(module))
{
    register_module("inner", module_);
}

torch::Tensor NetworkBlock::Forward(torch::Tensor input)
{
    return module_->Forward(input);
}


// ===========================================================================
// NetworkStruct (Execution Engine)
// ===========================================================================

NetworkStruct::NetworkStruct(std::vector<std::shared_ptr<NetworkBlock>> blocks)
    : blocks_(std::move(blocks))
{
    for (const auto& block : blocks_) {
        register_module(block->GetName(), block);
    }
}

torch::Tensor NetworkStruct::Forward(torch::Tensor input, const anet::TraceSink& sink)
{
    ANET_PROFILE_FUNC();

    if (sink) sink("00_Input", input);

    torch::Tensor x = input;
    int index = 1;
    for (const auto& block : blocks_) {
        x = block->Forward(x);
        if (sink && block->IsConv2dVisualizable() && x.dim() == 4 && x.size(2) >= 2 && x.size(3) >= 2 && x.is_floating_point()) {
            sink(std::format("{:02d}_{}", index++, block->GetName().c_str()), x);
        }
    }
    return x;
}


// ===========================================================================
// NetworkBoundaryPreprocessor
// ===========================================================================

NetworkBoundaryPreprocessor::NetworkBoundaryPreprocessor(
    const anet::TensorSpecMap& specs, const std::vector<std::string>& raw_keys)
    : specs_(specs), raw_keys_(raw_keys.begin(), raw_keys.end())
{
}

anet::TensorDict NetworkBoundaryPreprocessor::Format(const anet::TensorDict& raw_input) const
{
	ANET_PROFILE_FUNC();

    anet::TensorDict formatted;

    for (const auto& kv : raw_input) {
        const std::string& key = kv.first;
        const torch::Tensor& tensor = kv.second;
        if (!tensor.defined()) continue;

        // (raw) 指定されている場合はそのまま通す
        if (raw_keys_.count(key) > 0) {
            formatted.Set(key, tensor);
            continue;
        }

        // Specが無いもの（Agent独自のテンソル等）はそのまま通す
        auto it = specs_.find(key);
        if (it == specs_.end()) {
            formatted.Set(key, tensor);
            continue;
        }

        const TensorSpec& spec = it->second;
        torch::Tensor t = tensor;

        // 離散値は One-Hot 化する
        if (spec.IsDiscrete()) {
            t = t.to(torch::kInt64);
            t = torch::one_hot(t, spec.num_classes).to(torch::kFloat32);
            auto new_dim = t.dim();
            // Shapeの補正: PyTorchのone_hotは末尾に次元(クラス数)を足すので余計なサイズ1次元を補正

            if (spec.type == anet::SpaceType::Grid) {
                // 入力が [B, S, 1, H, W] -> one_hot -> [B, S, 1, H, W, C] (6次元: FrameStackあり)
                if (new_dim == 6 && t.size(2) == 1) {
                    t = t.squeeze(2); // [B, S, H, W, C] にする
                    new_dim = 5;
                }
                // 入力が [B, 1, H, W] -> one_hot -> [B, 1, H, W, C] (5次元: FrameStackなし)
                else if (new_dim == 5 && t.size(1) == 1) {
                    t = t.squeeze(1); // [B, H, W, C] にする
                    new_dim = 4;
                }

                // この時点で t は必ず [B, S, H, W, C] (5次元) または [B, H, W, C] (4次元) になる
                if (new_dim == 5) {
                    // FrameStackあり: CとSをまとめてConv2dに食わせるため [B, S*C, H, W] (4次元) に平坦化する
                    t = t.permute({ 0, 1, 4, 2, 3 }); // -> [B, S, C, H, W]
                    t = t.reshape({ t.size(0), t.size(1) * t.size(2), t.size(3), t.size(4) });
                } else if (new_dim == 4) {
                    // FrameStackなし: そのまま並び替え -> [B, C, H, W] (4次元)
                    t = t.permute({ 0, 3, 1, 2 });
                }
            } else if (spec.type == anet::SpaceType::Vector) {
                // Vectorの場合も念のためFrameStack(6次元/4次元)をケアしておく
                if (new_dim == 4 && t.size(2) == 1) { // [B, S, 1] -> one_hot -> [B, S, 1, C]
                    t = t.squeeze(2); // -> [B, S, C]
                } else if (new_dim == 3 && t.size(1) == 1) { // [B, 1] -> one_hot -> [B, 1, C]
                    t = t.squeeze(1); // -> [B, C]
                }
            }
        } else {
            // 連続値の正規化・キャスト
            if (t.scalar_type() == torch::kByte || t.scalar_type() == torch::kInt8 || t.scalar_type() == torch::kUInt8) {
                t = t.to(torch::kFloat32) / 255.0f;
            } else if (t.scalar_type() != torch::kFloat32) {
                t = t.to(torch::kFloat32);
            }
        }
        formatted.Set(key, t);
    }
    return formatted;
}


// ===========================================================================
// NetworkBranch
// ===========================================================================

NetworkBranch::NetworkBranch(std::string name, std::vector<std::string> bind_keys, std::shared_ptr<NetworkStruct> network_struct)
    : name_(std::move(name)), bind_keys_(std::move(bind_keys)), network_struct_(std::move(network_struct))
{
    register_module("network_struct", network_struct_);
}

void NetworkBranch::Execute(anet::TensorDict& current_state, const anet::TraceSink& sink)
{
    torch::Tensor block_input;

    if (bind_keys_.empty()) {
        block_input = torch::empty({ 0 }, current_state.device());
    } else {
        std::vector<torch::Tensor> inputs;
        for (const auto& key : bind_keys_) {
            auto t_opt = current_state.Get(key);
            if (!t_opt.has_value()) {
                ANET_SYSTEM_ERROR("NetworkBranch '" << name_ << "' failed to execute: Input key '" << key << "' not found in TensorDict.");
            }
            inputs.push_back(*t_opt);
        }
        block_input = (inputs.size() == 1) ? inputs[0] : torch::cat(inputs, 1);
    }

    anet::TraceSink branch_sink;
    if (sink) {
        const std::string prefix = name_ + "/";
        branch_sink = [&sink, prefix](std::string_view key, const torch::Tensor& tensor) {
            sink(prefix + std::string(key), tensor);
        };
    }

    torch::Tensor output = network_struct_->Forward(block_input, branch_sink);
    current_state.Set(name_, output);
}


// ===========================================================================
// NetworkBody / CompositeModule
// ===========================================================================

NetworkBody::NetworkBody(
    std::vector<std::shared_ptr<NetworkBranch>> branches,
    const anet::TensorSpecMap& specs,
    const std::vector<std::string>& raw_keys,
    std::map<std::string, std::string> output_keys)
	: branches_(std::move(branches))
    , preprocessor_(specs, raw_keys)
    , output_keys_(std::move(output_keys))
{
    for (const auto& branch : branches_) {
        register_module("branch_" + branch->GetName(), branch);
    }
}

anet::TensorDict NetworkBody::Forward(const anet::TensorDict& input, const anet::TraceSink& sink)
{
    ANET_PROFILE_FUNC();

    // 境界プレプロセッサで生データをフォーマット
    auto state = preprocessor_.Format(input);

    // 入力順ソートされた順序でブランチ群を実行
    for (const auto& branch : branches_) {
        branch->Execute(state, sink);
    }

    // 指定されたマッピングに従って Head用の TensorDict を構築
    anet::TensorDict out;
    for (const auto& [head_key, branch_key] : output_keys_) {
        if (auto t = state.Get(branch_key)) {
            out.Set(head_key, *t);
        } else {
            ANET_SYSTEM_ERROR("NetworkBody output mapping failed: branch '" << branch_key << "' not found in DAG state.");
        }
    }
    return out;
}


// ===========================================================================
// NetworkStructBuilder
// ===========================================================================

namespace {

struct ConfigProfileMarkerUse {
    size_t instance_seq = 0;
    std::string block_def_name;
    std::string field_key;
};

struct ConfigProfileMarker {
    std::string field_key;
    std::string group_name;
    size_t group_index = 0;
};

struct ConfigProfileBlockInstance {
    std::string block_def_name;
    std::vector<ConfigProfileMarker> markers;
};

struct ConfigProfileExpansionPlan {
    std::vector<ConfigProfileBlockInstance> instances;
    std::map<std::string, std::vector<ConfigProfileMarkerUse>> group_uses;
    std::set<std::string> used_groups;
};

static std::optional<std::string> ExtractConfigProfileMarkerGroup(const std::string& value)
{
    if (value.empty() || value[0] != '@') {
        return std::nullopt;
    }
    return value.substr(1);
}

static ConfigProfileExpansionPlan MakeConfigProfileExpansionPlan(
    const NetworkConfig& root_config, const std::string& structure_str)
{
    ConfigProfileExpansionPlan plan;
    if (structure_str.empty()) return plan;

    auto tokens = SplitPipelineString(structure_str);
    std::regex re_rep(R"(\(\*(\d+)\))");

    size_t instance_seq = 0;
    for (const auto& token_raw : tokens) {
        std::string current_token = token_raw;

        int repeat_count = 1;
        std::smatch m_rep;
        if (std::regex_search(current_token, m_rep, re_rep)) {
            repeat_count = std::stoi(m_rep[1].str());
            current_token = std::regex_replace(current_token, re_rep, "");
        }

        for (int r = 0; r < repeat_count; ++r) {
            std::string block_def_name = Trim(current_token);
            if (block_def_name.empty()) continue;

            if (root_config.block_configs.find(block_def_name) == root_config.block_configs.end()) {
                throw std::runtime_error("Block definition not found: " + block_def_name);
            }
            const auto& block_cfg = root_config.block_configs.at(block_def_name);

            ConfigProfileBlockInstance instance;
            instance.block_def_name = block_def_name;

            // branch内の展開順に沿って、instanceごとのprofile markerを集計する。
            for (const auto& [field_key, value] : block_cfg.config_data.Map()) {
                const auto group_name = ExtractConfigProfileMarkerGroup(value);
                if (!group_name.has_value()) continue;
                if (group_name->empty()) {
                    ANET_SYSTEM_ERROR(
                        "Invalid config_profile marker in block '" << block_def_name
                        << "' field '" << field_key << "': value=\""
                        << value << "\". Expected @<group>.");
                }

                const size_t group_index = plan.group_uses[*group_name].size();
                plan.group_uses[*group_name].push_back(ConfigProfileMarkerUse{
                    .instance_seq = instance_seq,
                    .block_def_name = block_def_name,
                    .field_key = field_key,
                });
                plan.used_groups.insert(*group_name);
                instance.markers.push_back(ConfigProfileMarker{
                    .field_key = field_key,
                    .group_name = *group_name,
                    .group_index = group_index,
                });
            }

            plan.instances.push_back(std::move(instance));
            ++instance_seq;
        }
    }

    return plan;
}

static void ValidateConfigProfileExpansionPlan(
    const std::map<std::string, ConfigProfileConfig>& config_profiles,
    const ConfigProfileExpansionPlan& plan)
{
    for (const auto& [group_name, uses] : plan.group_uses) {
        if (config_profiles.find(group_name) == config_profiles.end()) {
            const auto& use = uses.front();
            ANET_SYSTEM_ERROR(
                "Undefined config profile group: @" << group_name
                << " used by block '" << use.block_def_name
                << "' field '" << use.field_key
                << "'. Expected net.config_profile.[" << group_name << "].");
        }
    }
}

} // namespace

std::shared_ptr<NetworkStruct> NetworkStructBuilder::Build(
    const NetworkConfig& root_config, const std::string& structure_str)
{
    return Build(root_config, structure_str, root_config.config_profiles);
}

std::shared_ptr<NetworkStruct> NetworkStructBuilder::Build(
    const NetworkConfig& root_config,
    const std::string& structure_str,
    const std::map<std::string, ConfigProfileConfig>& config_profiles)
{
    if (structure_str.empty()) return std::make_shared<NetworkStruct>();

    std::vector<std::shared_ptr<NetworkBlock>> blocks;
    const auto expansion_plan = MakeConfigProfileExpansionPlan(root_config, structure_str);
    ValidateConfigProfileExpansionPlan(config_profiles, expansion_plan);
    const auto profile_objects = CreateConfigProfileMap(config_profiles);

    std::map<std::string, int> type_counters;

    for (const auto& instance : expansion_plan.instances) {
        const auto& block_def_name = instance.block_def_name;
        const auto& block_cfg = root_config.block_configs.at(block_def_name);

        auto factory = NetworkModuleRepository::Instance().GetFactory(block_cfg.type);
        ModuleContext ctx;

        std::shared_ptr<NetworkModule> inner_module;
        if (instance.markers.empty()) {
            inner_module = factory->CreateModule(block_cfg.config_data, ctx);
        } else {
            // 補間対象のinstanceだけConfigDataをコピーし、profile objectで実効値へ置換する。
            anet::ConfigData effective_config = block_cfg.config_data;
            for (const auto& marker : instance.markers) {
                const auto& profile = *profile_objects.at(marker.group_name);
                const size_t count = expansion_plan.group_uses.at(marker.group_name).size();
                const double value = profile.GenerateValue(marker.group_index, count);
                effective_config.Set(marker.field_key, FormatDoubleForConfig(value));
            }
            inner_module = factory->CreateModule(effective_config, ctx);
        }

        int idx = type_counters[block_def_name]++;
        std::string instance_name = block_def_name + "_" + std::to_string(idx);

        blocks.push_back(std::make_shared<NetworkBlock>(instance_name, inner_module));
    }
    return std::make_shared<NetworkStruct>(std::move(blocks));
}


// ===========================================================================
// NetworkBodyBuilder
// ===========================================================================

std::shared_ptr<NetworkBody> NetworkBodyBuilder::Build(const NetworkConfig& config, const anet::TensorSpecMap& input_specs)
{
    std::map<std::string, std::shared_ptr<NetworkBranch>> all_branches;
    std::map<std::string, int> in_degree;
    std::map<std::string, std::vector<std::string>> adj;
    std::vector<std::string> all_raw_keys;
    std::set<std::string> used_config_profile_groups;

    // config_profileはbranch内で展開し、branch-local overrideをglobal既定へ重ねる。
    for (const auto& [b_name, b_cfg] : config.branches) {
        const auto effective_profiles = MergeConfigProfileConfig(config.config_profiles, b_cfg.config_profiles);
        const auto expansion_plan = MakeConfigProfileExpansionPlan(config, b_cfg.structure_str);
        ValidateConfigProfileExpansionPlan(effective_profiles, expansion_plan);
        for (const auto& group_name : expansion_plan.used_groups) {
            used_config_profile_groups.insert(group_name);
        }
        for (const auto& [group_name, profile] : b_cfg.config_profiles) {
            (void)profile;
            if (expansion_plan.used_groups.find(group_name) == expansion_plan.used_groups.end()) {
                LOG::warn() << "net.branch.[" << b_name << "].config_profile.["
                    << group_name << "] is defined but not used by that branch.";
            }
        }
    }

    for (const auto& [group_name, profile] : config.config_profiles) {
        (void)profile;
        if (used_config_profile_groups.find(group_name) == used_config_profile_groups.end()) {
            LOG::warn() << "net.config_profile.[" << group_name << "] is defined but not used by any branch.";
        }
    }

    // 各ブランチの生成と初期化
    for (const auto& [b_name, b_cfg] : config.branches) {
        const auto effective_profiles = MergeConfigProfileConfig(config.config_profiles, b_cfg.config_profiles);
        auto engine = NetworkStructBuilder::Build(config, b_cfg.structure_str, effective_profiles);
        all_branches[b_name] = std::make_shared<NetworkBranch>(b_name, b_cfg.bind_keys, engine);
        in_degree[b_name] = 0;

        for (const auto& rk : b_cfg.raw_keys) {
            all_raw_keys.push_back(rk);
        }
    }

    // 依存関係（エッジ）の構築
    for (const auto& [b_name, b_cfg] : config.branches) {
        for (const auto& bind_key : b_cfg.bind_keys) {
            if (input_specs.find(bind_key) != input_specs.end()) {
                // Envからの入力（依存なし）
                continue;
            } else if (all_branches.find(bind_key) != all_branches.end()) {
                // 他のブランチからの入力（依存あり）
                adj[bind_key].push_back(b_name);
                in_degree[b_name]++;
            } else {
                ANET_SYSTEM_ERROR("NetworkBodyBuilder: Branch '" << b_name << "' requires unknown input key '" << bind_key << "'. Check your 'bind' configuration.");
            }
        }
    }

    // トポロジカルソート (Kahn's algorithm)
    std::queue<std::string> q;
    for (const auto& [b_name, deg] : in_degree) {
        if (deg == 0) q.push(b_name);
    }

    std::vector<std::shared_ptr<NetworkBranch>> sorted_branches;
    while (!q.empty()) {
        std::string u = q.front();
        q.pop();
        sorted_branches.push_back(all_branches[u]);

        for (const auto& v : adj[u]) {
            in_degree[v]--;
            if (in_degree[v] == 0) {
                q.push(v);
            }
        }
    }

    // 循環参照のチェック
    if (sorted_branches.size() != all_branches.size()) {
        ANET_SYSTEM_ERROR("NetworkBodyBuilder: Cycle detected in branch bindings! Check for circular dependencies in configuration.");
    }

    return std::make_shared<NetworkBody>(sorted_branches, input_specs, all_raw_keys, config.output_keys);
}


// ===========================================================================
// Network
// ===========================================================================

Network::Network(
    const NetworkConfig& config, const anet::TensorSpecMap& input_specs, std::shared_ptr<NetworkHeadFactory> head_factory,
    std::shared_ptr<NetworkBody> body, std::shared_ptr<NetworkHead> head)
    : config_(config)
    , input_specs_(input_specs)
    , head_factory_(head_factory)
    , body_(std::move(body))
    , head_(std::move(head))
{
    register_module("body", body_);
    if (head_) register_module("head", head_);
}

anet::TensorDict Network::Forward(const anet::TensorDict& input, const anet::TraceSink& sink)
{
    ANET_PROFILE_FUNC();

    //  Body部を実行 (ここはAMPが有効ならFP16で高速処理される)
    auto features = body_->Forward(input, sink);

    // Head部を実行
    if (head_) {
        // Head部ではAMPを強制OFF（外側の設定を無効化）にする
        anet::Autocast disable_amp(torch::kCUDA, false, torch::kFloat32);

        // Bodyから出てきたTensorはBF16になっているかもしれないのでキャストしてHeadに流し込む
        return head_->Forward(features.To(torch::kFloat32));
    } else {
		// Headが無い場合はBodyの出力をそのまま返す
        return features;
    }
}

std::optional<anet::TensorDictFunction> Network::GetTensorDictFunction(const std::string& key)
{
    // Headが無い Network では Function key として forward だけを公開する
    if (!head_) {
        if (key != "forward") return std::nullopt;

        return [this](const anet::TensorDict& state_input) -> anet::TensorDict {
            torch::NoGradGuard grad_guard;

            // Bodyのみ実行し、Network::Forward の Headなし経路と同じ出力を返す
            return this->body_->Forward(state_input);
        };
    }

    // Headが指定されたキーの機能(関数)を持ってい無ければ無し
    auto head_func = head_->GetTensorDictFunction(key);
    if (!head_func) {
        return std::nullopt;
    }

    //  Bodyの実行を内包したクロージャ(ラムダ)を返す
    return [this, h_func = *head_func](const anet::TensorDict& state_input) -> anet::TensorDict {
        torch::NoGradGuard grad_guard;

        // Bodyを実行 (ここはAMPが有効ならFP16で高速処理される)
        anet::TensorDict features = this->body_->Forward(state_input);

        // 関数実行時もHead扱いの処理なので、確実にAMPを切る
        anet::Autocast disable_amp(torch::kCUDA, false, torch::kFloat32);

        // 抽出された特徴量DictをHeadの関数に渡して結果を返す
        return h_func(features.To(torch::kFloat32));
        };
}

std::unique_ptr<anet::graphviz::GraphViz> Network::MakeGraphViz(const NetworkGraphVizConfig& viz_config) const
{
    ValidateNetworkGraphVizConfig(viz_config, "NetworkGraphVizConfig");

    auto tree = std::make_unique<anet::graphviz::Tree>("Network", ToGraphVizLayoutDirection(viz_config));
    RegisterNetworkGraphVizStyles(*tree);

    std::unordered_map<std::string, std::string> input_nodes;
    for (const auto& [key, spec] : input_specs_) {
        const std::string node_id = MakeDotNodeId("input", key);
        auto& node = tree->AddNode(node_id, "input");
        node.label.SetTitle("input: " + key);
        if (viz_config.show_tensor_specs) {
            AddTensorSpecAttrs(node.label, spec);
        }
        input_nodes[key] = node_id;
    }

    std::unordered_map<std::string, std::string> branch_output_nodes;
    for (const auto& branch : body_->GetBranches()) {
        const std::string branch_name = branch->GetName();
        const std::string branch_id = MakeDotNodeId("branch", branch_name);
        auto& branch_node = tree->AddNode(branch_id, viz_config.cluster_branches ? "branch" : "default");
        branch_node.label.SetTitle("branch: " + branch_name);
        if (viz_config.show_branch_config) {
            auto branch_config_it = config_.branches.find(branch_name);
            if (branch_config_it != config_.branches.end()) {
                const auto& branch_config = branch_config_it->second;
                branch_node.label.AddAttr("auto_format", branch_config.auto_format ? "true" : "false");
                branch_node.label.AddAttr("raw_keys", FormatStringVector(branch_config.raw_keys));
            }
        }

        for (const auto& bind_key : branch->GetBindKeys()) {
            auto input_it = input_nodes.find(bind_key);
            auto branch_it = branch_output_nodes.find(bind_key);
            if (input_it != input_nodes.end()) {
                auto& edge = tree->AddEdge(input_it->second, branch_id);
                edge.label.SetText(bind_key);
            } else if (branch_it != branch_output_nodes.end()) {
                auto& edge = tree->AddEdge(branch_it->second, branch_id);
                edge.label.SetText(bind_key);
            }
        }

        std::string prev_id = branch_id;
        if (auto network_struct = branch->GetNetworkStruct()) {
            int block_index = 0;
            for (const auto& block : network_struct->GetBlocks()) {
                const std::string block_id = MakeDotNodeId(
                    "block",
                    branch_name + "." + block->GetName() + "." + std::to_string(block_index++));
                auto& block_node = tree->AddNode(block_id, "block");
                block_node.label.SetTitle(block->GetName());

                if (viz_config.show_param_shapes) {
                    if (auto module = block->GetModule()) {
                        const auto current_config = module->GetCurrentConfigData();
                        for (const auto& [key, value] : current_config.Map()) {
                            AddConfigAttr(block_node.label, key, value, viz_config.float_precision);
                        }
                    }
                }

                if (viz_config.show_param_count) {
                    int64_t param_count = 0;
                    for (const auto& param : block->parameters(true)) {
                        param_count += param.numel();
                    }
                    block_node.label.AddAttr("params", param_count);
                }

                tree->AddEdge(prev_id, block_id);
                prev_id = block_id;
            }
        }
        branch_output_nodes[branch_name] = prev_id;
    }

    std::vector<std::string> body_output_nodes;
    for (const auto& [head_key, branch_key] : body_->GetOutputKeys()) {
        const std::string output_id = MakeDotNodeId("output", head_key);
        auto& output_node = tree->AddNode(output_id, "output");
        output_node.label.SetTitle("body output: " + head_key);
        output_node.label.AddAttr("from", branch_key);

        auto branch_it = branch_output_nodes.find(branch_key);
        auto input_it = input_nodes.find(branch_key);
        if (branch_it != branch_output_nodes.end()) {
            tree->AddEdge(branch_it->second, output_id);
        } else if (input_it != input_nodes.end()) {
            tree->AddEdge(input_it->second, output_id);
        }
        body_output_nodes.push_back(output_id);
    }

    if (head_) {
        const std::string head_id = "head";
        auto& head_node = tree->AddNode(head_id, "head");
        head_node.label.SetTitle("Head");
        const auto head_info = head_->GetGraphVizInfo();
        AddHeadGraphVizInfo(head_node.label, head_info, viz_config);

        if (!body_output_nodes.empty()) {
            for (const auto& output_id : body_output_nodes) {
                tree->AddEdge(output_id, head_id);
            }
        } else {
            for (const auto& [_, branch_output_id] : branch_output_nodes) {
                tree->AddEdge(branch_output_id, head_id);
            }
        }

        for (const auto& output : head_info.outputs) {
            const std::string output_id = MakeDotNodeId("head_output", output.name);
            auto& output_node = tree->AddNode(output_id, "head_output");
            output_node.label.SetTitle("output: " + output.name);
            if (viz_config.show_head_info && !output.shape.empty()) {
                output_node.label.AddAttr("shape", FormatInt64Vector(output.shape));
            }
            tree->AddEdge(head_id, output_id);
        }
    }

    return tree;
}

std::shared_ptr<Network> Network::Clone(std::optional<torch::Device> device) const
{
    ANET_PROFILE_FUNC();

    // デバイス指定がない場合は、自身のパラメータが乗っているデバイスに合わせる
    torch::Device target_device = torch::kCPU;
    if (device.has_value()) {
        target_device = *device;
    } else {
        auto params = this->parameters();
        if (!params.empty()) {
            target_device = params[0].device();
        }
    }

    // config と input_specs を元に新しいインスタンスを再構築する
    auto cloned_net = NetworkBuilder::BuildNetwork(config_, input_specs_, head_factory_, target_device);

    // デバイス移動
    cloned_net->to(target_device);

    // 現在の重みとバッファを完全にコピー
    this->CopyTo(*cloned_net);
    return cloned_net;
}

void Network::CopyTo(Network& target) const
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;

    // パラメータ (Weight, Bias等) のコピー
    auto src_params = this->named_parameters(true);
    auto dst_params = target.named_parameters(true);
    ANET_ASSERT(src_params.size() == dst_params.size());
    for (const auto& kv : src_params) {
        dst_params[kv.key()].copy_(kv.value());
    }

    // バッファ (BatchNormの移動平均等) のコピー
    auto src_buffers = this->named_buffers(true);
    auto dst_buffers = target.named_buffers(true);
    ANET_ASSERT(src_buffers.size() == dst_buffers.size());
    for (const auto& kv : src_buffers) {
        dst_buffers[kv.key()].copy_(kv.value());
    }
}

void Network::SoftCopyTo(Network& target, double tau) const
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;

    // パラメータのブレンド: target = tau * src + (1 - tau) * target
    auto src_params = this->named_parameters(true);
    auto dst_params = target.named_parameters(true);
    ANET_ASSERT(src_params.size() == dst_params.size());
    std::vector<torch::Tensor> src_param_tensors;
    std::vector<torch::Tensor> dst_param_tensors;
    src_param_tensors.reserve(src_params.size());
    dst_param_tensors.reserve(dst_params.size());
    for (const auto& kv : src_params) {
        src_param_tensors.push_back(kv.value());
        dst_param_tensors.push_back(dst_params[kv.key()]);
    }
    if (!dst_param_tensors.empty()) {
        torch::_foreach_lerp_(dst_param_tensors, src_param_tensors, tau);
    }

    // バッファのブレンド
    auto src_buffers = this->named_buffers(true);
    auto dst_buffers = target.named_buffers(true);
    ANET_ASSERT(src_buffers.size() == dst_buffers.size());
    std::vector<torch::Tensor> src_float_buffers;
    std::vector<torch::Tensor> dst_float_buffers;
    src_float_buffers.reserve(src_buffers.size());
    dst_float_buffers.reserve(dst_buffers.size());
    for (const auto& kv : src_buffers) {
        // float系のテンソル以外(intのカウンタ等)はlerpできないためそのままcopy_
        if (kv.value().is_floating_point()) {
            src_float_buffers.push_back(kv.value());
            dst_float_buffers.push_back(dst_buffers[kv.key()]);
        } else {
            dst_buffers[kv.key()].copy_(kv.value());
        }
    }
    if (!dst_float_buffers.empty()) {
        torch::_foreach_lerp_(dst_float_buffers, src_float_buffers, tau);
    }
}

// ===========================================================================
// NetworkModuleRepository & Standard Factories
// ===========================================================================

NetworkModuleRepository& NetworkModuleRepository::Instance()
{
    static NetworkModuleRepository instance;
    return instance;
}

void NetworkModuleRepository::Register(const std::string& type_name, std::shared_ptr<NetworkModuleFactory> factory)
{
    std::lock_guard<std::mutex> lock(mtx_);

    registry_[type_name] = std::move(factory);
}

std::shared_ptr<NetworkModuleFactory> NetworkModuleRepository::GetFactory(const std::string& type_name) const
{
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = registry_.find(type_name);
    if (it == registry_.end()) {
        ANET_SYSTEM_ERROR("Unknown module type: " << type_name);
    }
    return it->second;
}


// ===========================================================================
// NetworkBuilder
// ===========================================================================

std::shared_ptr<Network> NetworkBuilder::BuildNetwork(
    const NetworkConfig& network_config, const anet::TensorSpecMap& input_specs, std::shared_ptr<NetworkHeadFactory> head_factory, std::optional<torch::Device> device)
{
	ANET_LOG_DEBUG("network_config=" << network_config.ToJson().dump());

    // Body (DAG) の構築
    auto body = NetworkBodyBuilder::Build(network_config, input_specs);

    // ダミー入力を作成 (Lazyモジュールの初期化 兼 Shape推論用)
    anet::TensorDict dummy_input;
    for (const auto& [key, spec] : input_specs) {
        auto shape = spec.shape;
        shape.insert(shape.begin(), 1); // Batch次元(1)を先頭に追加

        // Float型等のダミーテンソルを生成
        auto dtype = (spec.IsDiscrete() || spec.dtype == torch::kUInt8) ? torch::kInt64 : spec.dtype;
        auto options = torch::TensorOptions().dtype(dtype);
        if (device.has_value()) {
            options = options.device(*device);
        }

        auto t = torch::zeros(shape, options);
        dummy_input.Set(key, t);
    }
	LOG::info() << "NetworkBuilder: input_shape=" << dummy_input.ToDefString();

    // 初回ダミー実行
    // これにより、Body内の全Lazy層(Linear等)が初期化され、Head構築用の出力Shapeが確定する
    anet::TensorDict dummy_feature;
    {
        torch::NoGradGuard grad_guard;

        dummy_feature = body->Forward(dummy_input);
    }   // あえて例外キャッチしない

    // Headを構築 (Bodyの出力TensorDictをそのまま渡す)
    std::shared_ptr<anet::nn::NetworkHead> head = head_factory ? head_factory->CreateHead(dummy_feature) : nullptr;

    // Network作って終わり
    return std::make_shared<Network>(network_config, input_specs, head_factory, body, head);
}
