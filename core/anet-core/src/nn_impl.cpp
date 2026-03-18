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
static constexpr const char* kNetBodyStructureConfigKey = "net.body.structure";


// ===========================================================================
// Helper Functions (Internal)
// ===========================================================================

// 文字列のトリム
std::string Trim(const std::string& s)
{
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) {
        start++;
    }
    auto end = s.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));
    return std::string(start, end + 1);
}

// パイプライン文字列の分割 (カッコ内のスペースは無視する)
std::vector<std::string> SplitPipelineString(const std::string& s)
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

// 簡易的な文字列分割 (スペース区切り)
static std::vector<std::string> SplitSpace(const std::string& s)
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (ss >> item) {
        if (!item.empty()) result.push_back(item);
    }
    return result;
}

// 文字列から整数への変換
int anet::nn::ParseInt(const std::string& s, const std::string& param_name)
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

static std::map<std::string, NetworkBlockConfig> ReadBlockConfig(const anet::ConfigData& config_data)
{
    std::map<std::string, NetworkBlockConfig> block_configs;

    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> tag_block_map;
    auto config_map = config_data.Map();
    for (const auto& kv : config_map) {
        const std::string& config_key = kv.first;
        const std::string& config_value = kv.second;

        // 設定Keyからtagを抽出
        auto config_tag = anet::ExtractBetween(config_key, kNetBlockConfigKeyPrefix, kNetBlockConfigKeySuffix);
        if (config_tag.empty()) {
            continue;
        }

        // サブキーを抽出
        auto pos = config_key.find(kNetBlockConfigKeySuffix);
        auto size = config_key.size();
        if ((pos + 2) >= config_key.size()) continue;
        auto config_sub_key = config_key.substr(pos + 2);
        if (config_sub_key.empty()) continue;

        // 集約Mapに保存
        tag_block_map[config_tag][config_sub_key] = config_value;
    }

    for (const auto& kv : tag_block_map) {
        const auto& tag = kv.first;
        const auto& config_map = kv.second;

        // NetworkBlockConfig を作る
        NetworkBlockConfig block_config;;
        for (const auto& kv2 : config_map) {
            const auto& config_sub_key = kv2.first;
            const auto& config_value = kv2.second;
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

        block_configs[tag] = std::move(block_config);
    }

    return block_configs;
}

NetworkConfig::NetworkConfig(const anet::ConfigData& config_data)
{
    config_data.Read(kNetBodyStructureConfigKey, structure_str, structure_str);
    block_configs = ReadBlockConfig(config_data);
}

anet::json NetworkConfig::ToJson() const
{
	/// @todo 設定ではなく実際に適用されている構造を出力するようにする

    anet::json j;

    // ルート構造文字列を保存
    j["structure"] = this->structure_str;

    // ブロック定義を格納するオブジェクトを用意
    j["blocks"] = anet::json::object();

    // 使用されているブロック名を収集するためのキューとセット
    std::vector<std::string> structure_queue;
    std::set<std::string> processed_blocks;

    // 初期状態: ルート構造をキューに入れる
    structure_queue.push_back(this->structure_str);

    // 汎用Regex: カッコとその中身 (...) を全て除去する
    std::regex re_options(R"(\([^)]*\))");

    // 幅優先探索で依存ブロックを収集
    size_t head = 0;
    while (head < structure_queue.size()) {
        std::string current_structure = structure_queue[head++];

        auto pipeline_tokens = SplitPipelineString(current_structure);

        for (const auto& token_raw : pipeline_tokens) {
            std::string temp_token = token_raw;

            // (...) を全て除去して純粋なブロック名にする
            temp_token = std::regex_replace(temp_token, re_options, "");

            // トリム
            std::string block_name = Trim(temp_token);
            if (block_name.empty()) continue;

            // 既に処理済みならスキップ
            if (processed_blocks.count(block_name) > 0) continue;

            // 未処理ブロックとして登録
            processed_blocks.insert(block_name);

            // Config検索 & JSON追加
            auto it = this->block_configs.find(block_name);
            if (it != this->block_configs.end()) {
                const auto& block_config = it->second;

                anet::json block_j;
                block_j["type"] = block_config.type;
                block_j["config"] = block_config.config_data.ToJson();
                j["blocks"][block_name] = block_j;

                // Compositeの場合、その内部構造も探索対象に追加
                if (block_config.type == "Composite") {
                    std::string sub_structure = block_config.config_data.Get("structure", "");
                    if (!sub_structure.empty()) {
                        structure_queue.push_back(sub_structure);
                    }
                }
            }
        }
    }

    return j;
}


// ===========================================================================
// NetworkBlock (Runtime Node)
// ===========================================================================

NetworkBlock::NetworkBlock(
    std::string name, std::shared_ptr<NetworkModule> module,
    std::vector<std::string> input_tags, std::string output_tag)
    : name_(std::move(name))
    , module_(std::move(module))
    , input_tags_(std::move(input_tags))
    , output_tag_(std::move(output_tag))
{
    // Torchのサブモジュールとして登録
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

    // 実行時キャッシュ (Tag -> Tensor)
    std::map<std::string, torch::Tensor> tensor_cache;

    // 直前のブロックの出力 (デフォルト入力)
    torch::Tensor last_output = input;

    for (const auto& block : blocks_) {
        anet::ProfileRange r1("NetworkStruct::Forward.block");

        torch::Tensor block_input;
        const auto& in_tags = block->GetInputTags();

        // --- 1. 入力解決 (Wiring) ---
        anet::ProfileRange r2("NetworkStruct::Forward.wireing");
        if (in_tags.empty()) {
            // 指定なし: 直前の出力を使う (Sequential動作)
            block_input = last_output;
        } else {
            // 指定あり: タグ解決
            std::vector<torch::Tensor> inputs;

            for (const auto& tag : in_tags) {
                if (tag == kReservedTagInput) {
                    // @input : Graph全体への入力
                    inputs.push_back(input);
                } else if (tag == kReservedTagPrev) {
                    // @prev : 直前の出力
                    inputs.push_back(last_output);
                } else {
                    // 通常タグ : キャッシュから検索
                    if (tensor_cache.find(tag) == tensor_cache.end()) {
                        throw std::runtime_error("Input tag not found in cache: " + tag + " (at block " + block->GetName() + ")");
                    }
                    inputs.push_back(tensor_cache.at(tag));
                }
            }

            if (inputs.size() == 1) {
                block_input = inputs[0];
            } else {
                // Dim=1 (Channel) で結合
                block_input = torch::cat(inputs, 1);
            }
        }

        // --- 2. 実行 ---
        anet::ProfileRange r3("NetworkStruct::Forward.execute", r2);
        torch::Tensor block_output = block->Forward(block_input);

        // --- 3. 出力キャッシュ (Tagging) ---
        anet::ProfileRange r4("NetworkStruct::Forward.tagging",r3);
        const auto& out_tag = block->GetOutputTag();
        if (!out_tag.empty()) {
            if (out_tag == kReservedTagInput || out_tag == kReservedTagPrev) {
                throw std::runtime_error("Cannot use reserved tag name for output: " + out_tag);
            }
            tensor_cache[out_tag] = block_output;
        }

        last_output = block_output;
    }

    return last_output;
}

int64_t NetworkStruct::InferFeatureDim(const std::vector<int64_t>& input_shape)
{
    torch::NoGradGuard no_grad;
    // Batch次元(1)を追加してダミー作成: [1, C, H, W]
    std::vector<int64_t> shape_with_batch = { 1 };
    shape_with_batch.insert(shape_with_batch.end(), input_shape.begin(), input_shape.end());

    auto dummy_in = torch::zeros(shape_with_batch);
    this->eval();   // ダミーデータによる統計汚染防止
    auto dummy_out = this->Forward(dummy_in);
    this->train();  // 元に戻す

    return dummy_out.numel();
}

anet::TensorDict NetworkStruct::GetConv2dOutputs(torch::Tensor input)
{
    anet::ProfileRange r("NetworkStruct::GetConv2dOutputs");
    anet::TensorDict outputs;

    // 入力画像を "00_Input" として保存
    outputs.Set("00_Input", input);

    // Forwardと同じ実行時キャッシュと結線(Wiring)ロジックを利用
    std::map<std::string, torch::Tensor> tensor_cache;
    torch::Tensor last_output = input;
    int index = 1;

    for (const auto& block : blocks_) {
        torch::Tensor block_input;
        const auto& in_tags = block->GetInputTags();
        ANET_LOG_DEBUG("block.GetName()=" << block->GetName());

        // --- 入力解決 (Wiring) ---
        if (in_tags.empty()) {
            block_input = last_output;
        } else {
            std::vector<torch::Tensor> inputs;
            for (const auto& tag : in_tags) {
                if (tag == kReservedTagInput) {
                    inputs.push_back(input);
                } else if (tag == kReservedTagPrev) {
                    inputs.push_back(last_output);
                } else {
                    if (tensor_cache.find(tag) == tensor_cache.end()) {
                        ANET_SYSTEM_ERROR("Input tag not found in cache: " << tag);
                    }
                    inputs.push_back(tensor_cache.at(tag));
                }
            }
            if (inputs.size() == 1) {
                block_input = inputs[0];
            } else {
                block_input = torch::cat(inputs, 1);
            }
        }

        // ---  実行 ---
        torch::Tensor block_output = block->Forward(block_input);

        // --- 出力キャッシュ (Tagging) ---
        const auto& out_tag = block->GetOutputTag();
        if (!out_tag.empty()) {
            tensor_cache[out_tag] = block_output;
        }

        last_output = block_output;

        // --- 抽出ロジック (可視化用) ---
        if (block->IsConv2dVisualizable()) {
            // 画像ぽい様式かチェック
            if (block_output.dim() == 4 && block_output.size(2) >= 2 && block_output.size(3) >= 2 && block_output.is_floating_point()) {
                ANET_LOG_DEBUG("Image bloc. GetName()=" << block->GetName());

                // "01_Embed4064_0", "02_ConvInit32_0" のようなキーで保存
                std::string name = block->GetName();
                std::string key = std::format("{:02d}_{}", index++, name.c_str());
                outputs.Set(key, block_output);
            } else {
                ANET_LOG_DEBUG("Not image block. GetName()=" << block->GetName());
            }
        }
    }

    return outputs;
}

// ===========================================================================
// CompositeModule
// ===========================================================================

CompositeModule::CompositeModule(std::shared_ptr<NetworkStruct> graph)
    : graph_(std::move(graph))
{
    register_module("composite_graph", graph_);
}

torch::Tensor CompositeModule::Forward(torch::Tensor input) {
    return graph_->Forward(input);
}


// ===========================================================================
// NetworkBody / CompositeModule
// ===========================================================================

NetworkBody::NetworkBody(std::shared_ptr<NetworkStruct> graph)
    : graph_(std::move(graph))
{
    register_module("graph", graph_);
}

int64_t NetworkBody::InferFeatureDim(const std::vector<int64_t>& input_shape) {
    return graph_->InferFeatureDim(input_shape);
}

torch::Tensor NetworkBody::Forward(torch::Tensor input) {
    return graph_->Forward(input);
}

anet::TensorDict NetworkBody::GetConv2dOutputs(torch::Tensor input)
{
    // グラフ(NetworkStruct)へ委譲
    return graph_->GetConv2dOutputs(input);
}


// ===========================================================================
// NetworkStructBuilder (Recursive Logic)
// ===========================================================================
std::shared_ptr<NetworkStruct> NetworkStructBuilder::Build(
    const NetworkConfig& root_config,
    const std::string& structure_str)
{
    std::vector<std::shared_ptr<NetworkBlock>> blocks;

    auto tokens = SplitPipelineString(structure_str);

    // 正規表現
    std::regex re_rep(R"(\(\*(\d+)\))");      // (*3)
    std::regex re_tag(R"(\(=([^)]+)\))");     // (=tag) - Output
    std::regex re_src(R"(\(@([^)]+)\))");     // (@tag) - Input

    // 同一Type内での連番用カウンタ
    std::map<std::string, int> type_counters;

    for (const auto& token_raw : tokens) {
        std::string current_token = token_raw;

        // A. 繰り返し (*N) の検出
        int repeat_count = 1;
        std::smatch m_rep;
        if (std::regex_search(current_token, m_rep, re_rep)) {
            repeat_count = std::stoi(m_rep[1].str());
            current_token = std::regex_replace(current_token, re_rep, "");
        }

        // B. 繰り返し展開
        for (int r = 0; r < repeat_count; ++r) {
            std::string temp_token = current_token;
            std::string output_tag = "";
            std::vector<std::string> input_tags;

            // B-1. Output Tag (=tag)
            std::smatch m_tag;
            if (std::regex_search(temp_token, m_tag, re_tag)) {
                output_tag = m_tag[1].str();
                temp_token = std::regex_replace(temp_token, re_tag, "");
            }

            // B-2. Input Tags (@tag)
            std::smatch m_src;
            while (std::regex_search(temp_token, m_src, re_src)) {
                std::string content = m_src[1].str();
                // スペースで分割して複数タグに対応
                auto split_tags = SplitSpace(content);

                for (auto& raw_tag : split_tags) {
                    // 先頭の '@' を除去 ("@tag" -> "tag")
                    if (!raw_tag.empty() && raw_tag[0] == '@') {
                        raw_tag = raw_tag.substr(1);
                    }
                    if (!raw_tag.empty()) {
                        input_tags.push_back(raw_tag);
                    }
                }

                temp_token = std::regex_replace(temp_token, re_src, "", std::regex_constants::format_first_only);
            }

            // B-3. BlockDef名
            std::string block_def_name = Trim(temp_token);
            if (block_def_name.empty()) continue;

            // 定義取得
            if (root_config.block_configs.find(block_def_name) == root_config.block_configs.end()) {
                throw std::runtime_error("Block definition not found: " + block_def_name);
            }
            const auto& block_cfg = root_config.block_configs.at(block_def_name);

            // モジュール生成
            std::shared_ptr<NetworkModule> inner_module;

            if (block_cfg.type == "Composite") {
                // === Composite Block (Recursive) ===
                std::string sub_structure = block_cfg.config_data.Get("structure", "");
                if (sub_structure.empty()) {
                    throw std::runtime_error("Composite block requires 'structure' config: " + block_def_name);
                }

                auto sub_graph = NetworkStructBuilder::Build(root_config, sub_structure);
                inner_module = std::make_shared<CompositeModule>(sub_graph);

            } else {
                // === Leaf Block (Factory) ===
                auto factory = NetworkModuleRepository::Instance().GetFactory(block_cfg.type);

                ModuleContext ctx;
                ctx.input_tags = input_tags;

                inner_module = factory->CreateModule(block_cfg.config_data, ctx);
            }

            // C. 名前生成 (Name): BlockDef_N
            int idx = type_counters[block_def_name]++;
            std::string instance_name = block_def_name + "_" + std::to_string(idx);

            // D. Block生成
            auto block = std::make_shared<NetworkBlock>(
                instance_name,
                inner_module,
                input_tags,
                output_tag
            );
            blocks.push_back(block);
        }
    }

    return std::make_shared<NetworkStruct>(std::move(blocks));
}


// ===========================================================================
// Network
// ===========================================================================

Network::Network(
    const NetworkConfig& config, const std::vector<int64_t>& input_shape, std::shared_ptr<NetworkHeadFactory> head_factory,
    std::shared_ptr<NetworkBody> body, std::shared_ptr<NetworkHead> head)
    : config_(config)
    , input_shape_(input_shape)
    , head_factory_(head_factory)
    , body_(std::move(body))
    , head_(std::move(head))
{
    register_module("body", body_);
    register_module("head", head_);
}

anet::TensorDict Network::Forward(const torch::Tensor& input)
{
	anet::ProfileRange r("Network::Forward");

    auto features = body_->Forward(input);

    {
        // Head部ではAMPを強制OFF（外側の設定を無効化）にする
        anet::Autocast disable_amp(torch::kCUDA, false, torch::kFloat32);

        // Bodyから出てきたTensorはBF16になっているかもしれないのでキャストしてHeadに流し込む
        return head_->Forward(features.to(torch::kFloat32));
    }
}

std::optional<anet::TensorFunction> Network::GetTensorFunction(const std::string& key)
{
    // Head優先
    auto head_func = head_->GetTensorFunction(key);
    if (head_func) {
        return [this, func = *head_func](const torch::Tensor& input) {
            auto features = body_->Forward(input);
            return func(features);
            };
    }
    return std::nullopt;
}

anet::TensorDict Network::GetConv2dOutputs(const torch::Tensor& input) const
{
    anet::ProfileRange r("Network::GetConv2dOutputs");
    torch::NoGradGuard no_grad;

    // Bodyへ委譲
    return body_->GetConv2dOutputs(input);
}

std::shared_ptr<Network> Network::Clone(std::optional<torch::Device> device) const
{
    anet::ProfileRange r("Network::Clone");

    // 保存した情報を使って新しいインスタンスを生成
    auto cloned_net = NetworkBuilder::BuildNetwork(config_, input_shape_, head_factory_);

    // デバイスを合わせる
    if (device.has_value()) {
        cloned_net->to(device.value());
    } else {
        // device指定が無い場合、既存と同じ(直接取れないのでnamed_parameters経由）
        auto named_params = this->named_parameters(false); // false = 直下のパラメータのみ
        if (!named_params.is_empty()) {
            cloned_net->to(named_params.begin()->value().device());
        }
    }

    // 自身(this)の重みをcloned_netへ完全上書き
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
// NetworkBuilder (Facade)
// ===========================================================================

// net.block.[Conv1d_S].type = Conv1d
// net.block.[Conv1d_S].out_channels = 32
// net.block.[Conv1d_S].kernel_size = 8
// net.block.[Conv1d_S].stride = 4
// net.block.[Conv1d_S].padding = 0
//
// net.body.structure = Conv1d_S > ReLU > Conv1d_M > ReLU > Flatten

std::shared_ptr<Network> NetworkBuilder::BuildNetwork(
    const NetworkConfig& network_config,
    const std::vector<int64_t>& input_shape,
    std::shared_ptr<NetworkHeadFactory> head_factory)
{
    // Body (Struct) の構築
    auto graph = NetworkStructBuilder::Build(network_config, network_config.structure_str);
    auto body = std::make_shared<NetworkBody>(graph);

    // Shape Inference
    int64_t feature_dim = body->InferFeatureDim(input_shape);

    // Head
    auto head = head_factory->CreateHead(feature_dim);

    // Networkを生成して返す
    return std::make_shared<Network>(network_config, input_shape, head_factory, body, head);
}

