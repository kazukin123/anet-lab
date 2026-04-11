// nn_impl.cpp

#include "nn_impl.hpp"

#include <iostream>
#include <sstream>
#include <regex>
#include <stdexcept>
#include <algorithm>
#include <format>
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

static std::map<std::string, NetworkBranchConfig> ReadBranchConfig(const anet::ConfigData& config_data, const std::string& config_prefix)
{
    std::map<std::string, NetworkBranchConfig> branches;

    std::string escaped_prefix = std::regex_replace(config_prefix, std::regex(R"(\.)"), R"(\.)");
    std::regex re_branch(escaped_prefix + R"(\.branch\.\[([^\]]+)\]\.(bind|structure|auto_format))");
    std::regex re_raw(R"(\(raw\))");

    for (const auto& [key, value] : config_data.Map()) {
        std::smatch m;
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

torch::Tensor NetworkStruct::Forward(torch::Tensor input)
{
    anet::ProfileRange r("NetworkStruct::Forward");

    torch::Tensor x = input;
    for (const auto& block : blocks_) {
        x = block->Forward(x);
    }
    return x;
}

anet::TensorDict NetworkStruct::GetConv2dOutputs(torch::Tensor input)
{
    anet::ProfileRange r("NetworkStruct::GetConv2dOutputs");
    anet::TensorDict outputs;
    outputs.Set("00_Input", input);

    torch::Tensor x = input;
    int index = 1;
    for (const auto& block : blocks_) {
        x = block->Forward(x);
        if (block->IsConv2dVisualizable() && x.dim() == 4 && x.size(2) >= 2 && x.size(3) >= 2 && x.is_floating_point()) {
            std::string key = std::format("{:02d}_{}", index++, block->GetName().c_str());
            outputs.Set(key, x);
        }
    }
    return outputs;
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
	anet::ProfileRange r("NetworkBoundaryPreprocessor::Format");

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
            if (t.scalar_type() == torch::kByte) { // uint8
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

void NetworkBranch::Execute(anet::TensorDict& current_state)
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

    torch::Tensor output = network_struct_->Forward(block_input);
    current_state.Set(name_, output);
}

void NetworkBranch::ExtractConv2dOutputs(const anet::TensorDict& current_state, anet::TensorDict& outputs) const
{
    torch::Tensor block_input;

    // Executeと同じロジックで入力を結合
    if (bind_keys_.empty()) {
        block_input = torch::empty({ 0 }, current_state.device());
    } else {
        std::vector<torch::Tensor> inputs;
        for (const auto& key : bind_keys_) {
            auto t_opt = current_state.Get(key);
            if (t_opt.has_value()) {
                inputs.push_back(*t_opt);
            }
        }
        if (inputs.empty()) return;
        block_input = (inputs.size() == 1) ? inputs[0] : torch::cat(inputs, 1);
    }

    // 直列エンジンの Conv2d出力を取得
    anet::TensorDict engine_outs = network_struct_->GetConv2dOutputs(block_input);

    // ブランチ名をプレフィックスとして付けて全体出力にマージ
    for (const auto& [key, tensor] : engine_outs) {
        outputs.Set(name_ + "/" + key, tensor);
    }
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

anet::TensorDict NetworkBody::Forward(const anet::TensorDict& input)
{
    anet::ProfileRange r("NetworkBody::Forward");

    // 境界プレプロセッサで生データをフォーマット
    auto state = preprocessor_.Format(input);

    // 入力順ソートされた順序でブランチ群を実行
    for (const auto& branch : branches_) {
        branch->Execute(state);
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

anet::TensorDict NetworkBody::GetConv2dOutputs(const anet::TensorDict& input) const
{
    anet::TensorDict visual_outputs;

    // フォーマット（Forward時と同じく関所を通す）
    auto state = preprocessor_.Format(input);

    // DAGを順に実行しつつ、各ブランチの出力を記録
    for (const auto& branch : branches_) {
        branch->ExtractConv2dOutputs(state, visual_outputs);
        // 次のブランチの入力のために、通常のForwardも行ってstateを更新する必要がある
        // ※GetConv2dOutputs は本番で毎ステップ呼ばれるものではないため、再計算のコストは許容する
        branch->Execute(state);
    }

    return visual_outputs;
}


// ===========================================================================
// NetworkStructBuilder
// ===========================================================================

std::shared_ptr<NetworkStruct> NetworkStructBuilder::Build(
    const NetworkConfig& root_config, const std::string& structure_str)
{
    if (structure_str.empty()) return std::make_shared<NetworkStruct>();

    std::vector<std::shared_ptr<NetworkBlock>> blocks;
    auto tokens = SplitPipelineString(structure_str);

    std::regex re_rep(R"(\(\*(\d+)\))");
    std::map<std::string, int> type_counters;

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

            auto factory = NetworkModuleRepository::Instance().GetFactory(block_cfg.type);
            ModuleContext ctx;
            auto inner_module = factory->CreateModule(block_cfg.config_data, ctx);

            int idx = type_counters[block_def_name]++;
            std::string instance_name = block_def_name + "_" + std::to_string(idx);

            blocks.push_back(std::make_shared<NetworkBlock>(instance_name, inner_module));
        }
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

    // 各ブランチの生成と初期化
    for (const auto& [b_name, b_cfg] : config.branches) {
        auto engine = NetworkStructBuilder::Build(config, b_cfg.structure_str);
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
    register_module("head", head_);
}

anet::TensorDict Network::Forward(const anet::TensorDict& input)
{
    anet::ProfileRange r("Network::Forward");

    //  Body部を実行 (ここはAMPが有効ならFP16で高速処理される)
    auto features = body_->Forward(input);

    // Head部を実行
    {
        // Head部ではAMPを強制OFF（外側の設定を無効化）にする
        anet::Autocast disable_amp(torch::kCUDA, false, torch::kFloat32);

        // Bodyから出てきたTensorはBF16になっているかもしれないのでキャストしてHeadに流し込む
        return head_->Forward(features.To(torch::kFloat32));
    }
}

anet::TensorDict Network::GetConv2dOutputs(const anet::TensorDict& input) const
{
    anet::ProfileRange r("Network::GetConv2dOutputs");

    torch::NoGradGuard no_grad;

    // Bodyへ委譲
    return body_->GetConv2dOutputs(input);
}

std::optional<anet::TensorDictFunction> Network::GetTensorDictFunction(const std::string& key)
{
    // Headが指定されたキーの機能(関数)を持っているか確認
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

std::shared_ptr<Network> Network::Clone(std::optional<torch::Device> device) const
{
    anet::ProfileRange r("Network::Clone");

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
    anet::ProfileRange r("Network::CopyTo");
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
    anet::ProfileRange r("Network::SoftCopyTo");
    torch::NoGradGuard no_grad;

    // パラメータのブレンド: target = tau * src + (1 - tau) * target
    auto src_params = this->named_parameters(true);
    auto dst_params = target.named_parameters(true);
    ANET_ASSERT(src_params.size() == dst_params.size());
    for (const auto& kv : src_params) {
        dst_params[kv.key()].lerp_(kv.value(), tau);
    }

    // バッファのブレンド
    auto src_buffers = this->named_buffers(true);
    auto dst_buffers = target.named_buffers(true);
    ANET_ASSERT(src_buffers.size() == dst_buffers.size());
    for (const auto& kv : src_buffers) {
        // float系のテンソル以外(intのカウンタ等)はlerpできないためそのままcopy_
        if (kv.value().is_floating_point()) {
            dst_buffers[kv.key()].lerp_(kv.value(), tau);
        } else {
            dst_buffers[kv.key()].copy_(kv.value());
        }
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
    ANET_CHECK(head_factory != nullptr);

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

    // 初回ダミー実行
    // これにより、Body内の全Lazy層(Linear等)が初期化され、Head構築用の出力Shapeが確定する
    anet::TensorDict dummy_feature;
    {
        torch::NoGradGuard grad_guard;

        dummy_feature = body->Forward(dummy_input);
    }   // あえて例外キャッチしない

    // Headを構築 (Bodyの出力TensorDictをそのまま渡す)
    auto head = head_factory->CreateHead(dummy_feature);

    // Network作って終わり
    return std::make_shared<Network>(network_config, input_specs, head_factory, body, head);
}
