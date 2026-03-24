// muzero_based_agent.cpp

#include "muzero_based_agent.hpp"
#include <algorithm>
#include "anet/log.hpp"
#include "anet/profile.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/tensor_check.hpp"
#include "nn_heads.hpp"

using namespace anet::rl::muzero_proto;


// ======================================================
// MuZeroDynamicsHead & Factory
// ======================================================

class MuZeroDynamicsHead : public anet::nn::NetworkHead {
public:
    MuZeroDynamicsHead(
        int64_t feature_dim,
        const anet::nn::NetworkConfig& base_config,
        const std::string& reward_branch_struct,
        const anet::nn::WeightInitConfig& reward_init_config)
    {
        // Reward用の分岐ブランチを動的構築
        reward_branch_ = anet::nn::NetworkStructBuilder::Build(base_config, reward_branch_struct);
        register_module("reward_branch", reward_branch_);

        // 構築したブランチの出力次元を推論
        int64_t branch_out_dim = reward_branch_->InferFeatureDim({ feature_dim });

        // 最終出力のスカラー層を固定構築
        reward_layer_ = register_module("reward_layer", torch::nn::Linear(branch_out_dim, 1));
        anet::nn::WeightInitializer::Initialize(reward_layer_, reward_init_config);
    }

    anet::TensorDict Forward(torch::Tensor feature_vector) override
    {
        anet::ProfileRange r("MuZeroDynamicsHead::Forward");

        //ANET_ASSERT_SHAPE(feature_vector, { ANET_SHAPE_ANY, feature_dim_ });
        ANET_ASSERT_NAN(feature_vector);

        anet::TensorDict dict;

        // 次の隠れ状態としてそのままパススルー
        dict["hidden_state"] = feature_vector;

        // ブランチを通して特徴を抽出し、最終層でスカラー報酬にする
        auto branch_out = reward_branch_->Forward(feature_vector);
        auto reward = reward_layer_->forward(branch_out);
        ANET_ASSERT_SHAPE(reward, { ANET_SHAPE_ANY, 1 });
        ANET_ASSERT_NAN(reward);

        // dictに結果を詰める
        dict["reward"] = reward_layer_->forward(branch_out);

        return dict;
    }

    std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override
    {
        return std::nullopt;
    }
private:
    std::shared_ptr<anet::nn::NetworkStruct> reward_branch_;
    torch::nn::Linear reward_layer_{ nullptr };
};

class MuZeroDynamicsHeadFactory : public anet::nn::NetworkHeadFactory {
public:
    MuZeroDynamicsHeadFactory(
        const anet::nn::NetworkConfig& base_config,
        const std::string& reward_branch_struct,
        const anet::nn::WeightInitConfig& reward_init_config)
        : base_config_(base_config)
        , reward_branch_struct_(reward_branch_struct)
        , reward_init_config_(reward_init_config) {
    }

    std::shared_ptr<anet::nn::NetworkHead> CreateHead(int64_t feature_dim) const override
    {
        return std::make_shared<MuZeroDynamicsHead>(
            feature_dim, base_config_, reward_branch_struct_, reward_init_config_);
    }

private:
    anet::nn::NetworkConfig base_config_;
    std::string reward_branch_struct_;
    anet::nn::WeightInitConfig reward_init_config_;
};


// ======================================================
// MuZeroPredictionHead & Factory
// ======================================================

class MuZeroPredictionHead : public anet::nn::NetworkHead {
public:
    MuZeroPredictionHead(int64_t feature_dim, int64_t num_actions,
        const anet::nn::NetworkConfig& base_config,
        const std::string& value_branch_struct,
        const std::string& policy_branch_struct,
        const anet::nn::WeightInitConfig& value_init_config,
        const anet::nn::WeightInitConfig& policy_init_config)
        : feature_dim_(feature_dim)
        , num_actions_(num_actions)
    {
        // Value Stream
        value_branch_ = anet::nn::NetworkStructBuilder::Build(base_config, value_branch_struct);
        register_module("value_branch", value_branch_);
        int64_t v_dim = value_branch_->InferFeatureDim({ feature_dim });
        value_layer_ = register_module("value_layer", torch::nn::Linear(v_dim, 1));
        anet::nn::WeightInitializer::Initialize(value_layer_, value_init_config);

        // Policy Stream
        policy_branch_ = anet::nn::NetworkStructBuilder::Build(base_config, policy_branch_struct);
        register_module("policy_branch", policy_branch_);
        int64_t p_dim = policy_branch_->InferFeatureDim({ feature_dim });
        policy_layer_ = register_module("policy_layer", torch::nn::Linear(p_dim, num_actions));
        anet::nn::WeightInitializer::Initialize(policy_layer_, policy_init_config);
    }

    anet::TensorDict Forward(torch::Tensor feature_vector) override
    {
        anet::ProfileRange r("MuZeroPredictionHead::Forward");

        ANET_ASSERT_SHAPE(feature_vector, { ANET_SHAPE_ANY, feature_dim_ });
        ANET_ASSERT_NAN(feature_vector);

        anet::TensorDict dict;

        // Valueを取る
        auto v_branch_out = value_branch_->Forward(feature_vector);
        auto value = value_layer_->forward(v_branch_out);
        ANET_ASSERT_SHAPE(value, { ANET_SHAPE_ANY, 1 });
        ANET_ASSERT_NAN(value);
        dict["value"] = value;

        // Policyを取る
        auto p_branch_out = policy_branch_->Forward(feature_vector);
        auto policy_logits = policy_layer_->forward(p_branch_out);
        ANET_ASSERT_SHAPE(policy_logits, { ANET_SHAPE_ANY, num_actions_ });
        ANET_ASSERT_NAN(policy_logits);
        dict["policy_logits"] = policy_logits;

        return dict;
    }

    std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override
    {
        return std::nullopt;
    }
private:
    const int64_t feature_dim_;
    const int64_t num_actions_;
    std::shared_ptr<anet::nn::NetworkStruct> value_branch_;
    std::shared_ptr<anet::nn::NetworkStruct> policy_branch_;
    torch::nn::Linear value_layer_{ nullptr };
    torch::nn::Linear policy_layer_{ nullptr };
};

class MuZeroPredictionHeadFactory : public anet::nn::NetworkHeadFactory {
public:
    MuZeroPredictionHeadFactory(
        int64_t num_actions,
        const anet::nn::NetworkConfig& base_config,
        const std::string& value_branch_struct,
        const std::string& policy_branch_struct,
        const anet::nn::WeightInitConfig& value_init_config,
        const anet::nn::WeightInitConfig& policy_init_config)
        : num_actions_(num_actions)
        , base_config_(base_config)
        , value_branch_struct_(value_branch_struct)
        , policy_branch_struct_(policy_branch_struct)
        , value_init_config_(value_init_config)
        , policy_init_config_(policy_init_config) {
    }

    std::shared_ptr<anet::nn::NetworkHead> CreateHead(int64_t feature_dim) const override
    {
        return std::make_shared<MuZeroPredictionHead>(
            feature_dim, num_actions_, base_config_,
            value_branch_struct_, policy_branch_struct_,
            value_init_config_, policy_init_config_);
    }
private:
    int64_t num_actions_;
    anet::nn::NetworkConfig base_config_;
    std::string value_branch_struct_;
    std::string policy_branch_struct_;
    anet::nn::WeightInitConfig value_init_config_;
    anet::nn::WeightInitConfig policy_init_config_;
};


// ======================================================
// MuZeroRepresentationHeadFactory
// ======================================================

class MuZeroRepresentationHeadFactory : public anet::nn::NetworkHeadFactory {
public:
    explicit MuZeroRepresentationHeadFactory(int64_t expected_input_dim)
        : expected_input_dim_(expected_input_dim) {
    }

    std::shared_ptr<anet::nn::NetworkHead> CreateHead(int64_t input_dim) const override
    {
        // Body側の設定で指定された最終出力次元が
        // Agentの全体設定である hidden_state_dim と一致しているかを検証する
        ANET_ASSERT_MSG(input_dim == expected_input_dim_,
            "Representation Body's output dim does not match config.hidden_state_dim!");

        return std::make_shared<anet::nn::PassThroughHead>("hidden_state");
    }
private:
    int64_t expected_input_dim_;
};


// ======================================================
// MuZeroNetworkSuite
// ======================================================

MuZeroNetworkSuite::MuZeroNetworkSuite(
    std::shared_ptr<anet::nn::Network> representation_net,
    std::shared_ptr<anet::nn::Network> dynamics_net,
    std::shared_ptr<anet::nn::Network> prediction_net)
    : representation_net_(std::move(representation_net))
    , dynamics_net_(std::move(dynamics_net))
    , prediction_net_(std::move(prediction_net))
{
    ANET_ASSERT(representation_net_ != nullptr);
    ANET_ASSERT(dynamics_net_ != nullptr);
    ANET_ASSERT(prediction_net_ != nullptr);

    // LibTorchのOptimizerが一括でパラメータを取得できるようモジュール登録
    register_module("representation", representation_net_);
    register_module("dynamics", dynamics_net_);
    register_module("prediction", prediction_net_);
}


// ======================================================
// MuZeroNetworkModel
// ======================================================

MuZeroNetworkModel::MuZeroNetworkModel(
    const ModelConfig& config, const ConfigData& config_data, const anet::rl::StateSpec& state_spec, const anet::rl::ActionSpec& action_spec)
    : config_(config)
    , num_actions_(action_spec.GetNumActions())
{
    // Representation Network を構築
    anet::nn::NetworkConfig rep_config{ config_data, config_.structure.rep };
    anet::MetricsLogger::Instance()->Log("net.rep", rep_config.ToJson());
    auto rep_head_factory = std::make_shared<MuZeroRepresentationHeadFactory>(config_.hidden_state_dim);
    auto rep_net = anet::nn::NetworkBuilder::BuildNetwork(rep_config, state_spec.shape, rep_head_factory);

    // Dynamics Network を構築
    /// @todo MuZero試作制約：隠れ状態を1次元固定ではなく多次元表現に対応する。アクション用のプレーン(チャンネル)を考慮する必要がある
    anet::nn::NetworkConfig dyn_config{ config_data, config_.structure.dyn };
    anet::MetricsLogger::Instance()->Log("net.dyn", dyn_config.ToJson());
    int64_t action_dim = action_spec.GetNumActions();   // アクションをOne-Hot表現として結合すると想定し、アクション数をそのまま次元として足す
    std::vector<int64_t> dyn_input_shape = { config_.hidden_state_dim + action_dim };
    auto dyn_head_factory = std::make_shared<MuZeroDynamicsHeadFactory>(
        dyn_config, config_.structure.reward_branch, config_.head_init_weight.reward);
    auto dyn_net = anet::nn::NetworkBuilder::BuildNetwork(dyn_config, dyn_input_shape, dyn_head_factory);

    //  Prediction Network を構築
    /// @todo MuZero試作制約：価値(Value)と報酬(Reward)をスカラー値ではなく、Two - Hotエンコーディング等を用いたカテゴリカル分布で予測・学習する
    anet::nn::NetworkConfig pred_config{ config_data, config_.structure.pred };
    anet::MetricsLogger::Instance()->Log("net.pred", pred_config.ToJson());
    std::vector<int64_t> pred_input_shape = { config_.hidden_state_dim };
    auto pred_head_factory = std::make_shared<MuZeroPredictionHeadFactory>(
        num_actions_, pred_config,
        config_.structure.value_branch, config_.structure.policy_branch,
        config_.head_init_weight.value, config_.head_init_weight.policy);
    auto pred_net = anet::nn::NetworkBuilder::BuildNetwork(pred_config, pred_input_shape, pred_head_factory);

    // 自身が所有するSuiteを生成
    suite_ = std::make_shared<MuZeroNetworkSuite>(rep_net, dyn_net, pred_net);
}

namespace {

    /// @brief 隠れ状態のMin-Max正規化ヘルパー (MuZero特有の安定化処理)
    /// バッチごとに状態ベクトルの値を [0, 1] の範囲にスケールします。
    torch::Tensor ScaleHiddenState(const torch::Tensor& hidden_state)
    {
        anet::ProfileRange r("ScaleHiddenState");

        ANET_ASSERT_NAN(hidden_state);

        // (B, D) または (B, C, H, W) の場合を考慮し、バッチ次元(dim=0)以外をflattenして最大/最小を計算
        auto flat_h = hidden_state.view({ hidden_state.size(0), -1 });

        auto max_val = std::get<0>(flat_h.max(/*dim=*/1, /*keepdim=*/true));
        auto min_val = std::get<0>(flat_h.min(/*dim=*/1, /*keepdim=*/true));

        // 元の形状にbroadcastできるようにview (例: 2Dなら [B, 1, 1, 1] に戻す)
        auto view_shape = std::vector<int64_t>(hidden_state.dim(), 1);
        view_shape[0] = hidden_state.size(0);

        max_val = max_val.view(view_shape);
        min_val = min_val.view(view_shape);

        // ゼロ除算回避のイプシロン(1e-5)を加算
        return (hidden_state - min_val) / (max_val - min_val + 1e-5f);
    }
}

anet::TensorDict MuZeroNetworkModel::InitialInference(const torch::Tensor& obs) const
{
    anet::ProfileRange r("MuZeroNetworkModel::InitialInference");

    // obsの形状は環境に依存するため、バッチ次元があることのみを確認 (末尾ANYを許容)
    ANET_LOG_DEBUG("obs=" << anet::ToDefString(obs));
    ANET_ASSERT_SHAPE(obs, { ANET_SHAPE_ANY, ANET_SHAPE_ENDANY });

    //  Representation: 観測 -> 隠れ状態
    auto rep_out = suite_->GetRepresentationNet()->Forward(obs);
    torch::Tensor hidden_state = rep_out.At("hidden_state");
    ANET_LOG_DEBUG("hidden_state=" << anet::ToDefString(hidden_state));
    ANET_ASSERT_SHAPE(hidden_state, { ANET_SHAPE_ANY, config_.hidden_state_dim });

    // 隠れ状態を [0, 1] に正規化
    hidden_state = ::ScaleHiddenState(hidden_state);

    //  Prediction: 隠れ状態 -> 価値, 方策
    auto pred_out = suite_->GetPredictionNet()->Forward(hidden_state);
    torch::Tensor value = pred_out.At("value");
    torch::Tensor policy_logits = pred_out.At("policy_logits");
    ANET_LOG_DEBUG("value=" << anet::ToDefString(value));
    ANET_LOG_DEBUG("policy_logits=" << anet::ToDefString(policy_logits));

    anet::TensorDict result;
    result["hidden_state"] = hidden_state;
    result["value"] = value;
    result["policy_logits"] = policy_logits;

    return result;
}

anet::TensorDict MuZeroNetworkModel::RecurrentInference(const torch::Tensor& hidden_state, const torch::Tensor& action) const
{
    anet::ProfileRange r("MuZeroNetworkModel::RecurrentInference");

    ANET_LOG_DEBUG("hidden_state=" << anet::ToDefString(hidden_state));
    ANET_ASSERT_SHAPE(hidden_state, { ANET_SHAPE_ANY, config_.hidden_state_dim });
    ANET_ASSERT_NAN(hidden_state);
    ANET_LOG_DEBUG("action=" << anet::ToDefString(action));

    // MuZero特有の勾配スケーリング・トリック
    // Forwardでは元の値をそのまま通し、Backwardの時だけ勾配を0.5倍にする
    auto scaled_hidden_state = hidden_state * 0.5f + hidden_state.detach() * 0.5f;

    // アクションが [B, 1] などの2次元で渡された場合を考慮し、[B] に平坦化する
    auto flat_action = action.dim() > 1 ? action.squeeze(1) : action;
    ANET_ASSERT_SHAPE(flat_action, { ANET_SHAPE_ANY }); // Squeeze後は1D(Batch)になっているはず

    // One-Hotベクトルに変換し、隠れ状態のデータ型 (通常はFloat32) に合わせる
    // (one_hotはデフォルトでint64を返すため、型合わせが必須)
    auto action_one_hot = torch::nn::functional::one_hot(flat_action, num_actions_).to(hidden_state.dtype());
    ANET_ASSERT_SHAPE(action_one_hot, { ANET_SHAPE_ANY, num_actions_ });

    // 隠れ状態とOne-Hot化されたアクションを結合
    auto dyn_input = torch::cat({ scaled_hidden_state, action_one_hot }, /*dim=*/1);   // (B, hidden_state_dim + num_actions)
    ANET_ASSERT_SHAPE(dyn_input, { ANET_SHAPE_ANY, config_.hidden_state_dim + num_actions_ });   

    // Dynamics: 隠れ状態 + アクション -> 次の隠れ状態, 報酬
    auto dyn_out = suite_->GetDynamicsNet()->Forward(dyn_input);
    torch::Tensor next_hidden_state = dyn_out.At("hidden_state");
    torch::Tensor reward = dyn_out.At("reward");
    ANET_ASSERT_SHAPE(next_hidden_state, { ANET_SHAPE_ANY, config_.hidden_state_dim });
    ANET_ASSERT_SHAPE(reward, { ANET_SHAPE_ANY, 1 });

    // 隠れ状態を [0, 1] に正規化
    next_hidden_state = ::ScaleHiddenState(next_hidden_state);

    // Prediction: 次の隠れ状態 -> 価値, 方策
    auto pred_out = suite_->GetPredictionNet()->Forward(next_hidden_state);
    torch::Tensor value = pred_out.At("value");
    torch::Tensor policy_logits = pred_out.At("policy_logits");

    anet::TensorDict result;
    result["hidden_state"] = next_hidden_state;
    result["reward"] = reward;
    result["value"] = value;
    result["policy_logits"] = policy_logits;

    return result;
}

void MuZeroNetworkModel::To(torch::Device device)
{
    suite_->to(device);
}


// ======================================================
// MCTS Data Structures
// ======================================================

void MinMaxStats::Update(float value)
{
    minimum_ = std::min(minimum_, value);
    maximum_ = std::max(maximum_, value);
}

float MinMaxStats::Normalize(float value) const
{
    if (maximum_ > minimum_) {
        float norm = (value - minimum_) / (maximum_ - minimum_);
        return std::clamp(norm, 0.0f, 1.0f);
    }
    return 0.5f;
}

Edge::Edge(int64_t action, float prior_prob)
    : action(action), prior_prob(prior_prob)
{
    ;
}

Node::Node()
{
    ;
}

MCTSTree::MCTSTree(const MCTSConfig& config)
    : config_(config)
{
    ;
}


// ======================================================
// MCTSEngine
// ======================================================

MCTSEngine::MCTSEngine(const MCTSConfig& config, std::shared_ptr<MuZeroNetworkModel> model, const anet::rl::ActionSpec& action_spec, std::optional<seed_t> seed)
    : anet::RandomHolder(seed)
    , config_(config), model_(std::move(model)), num_actions_(action_spec.GetNumActions())
{
    ANET_ASSERT(model_ != nullptr);
    ANET_ASSERT(rnd_ != nullptr);
}

void MCTSEngine::ExpandNode(std::shared_ptr<Node> node, const torch::Tensor& policy_logits)
{
    anet::ProfileRange r("MCTSEngine::ExpandNode");

    // ロジットをソフトマックスで確率分布に変換
    torch::NoGradGuard no_grad;
    auto probs = torch::softmax(policy_logits, /*dim=*/-1).squeeze(0); // [1, A] -> [A]

    // CPUへ転送してアクセス (GPUの場合は同期が発生する)
    probs = probs.cpu();    // 同期発生

    node->edges.resize(num_actions_);
    for (int64_t a = 0; a < num_actions_; ++a) {
        node->edges[a] = std::make_shared<Edge>(a, probs[a].item<float>());
    }
}

void MCTSEngine::AddExplorationNoise(std::shared_ptr<Node> node)
{
    anet::ProfileRange r("MCTSEngine::AddExplorationNoise");

    // ディリクレノイズの生成
    /// @todo 試作版では std::gamma_distribution を用いてシンプルなディリクレノイズを生成
    std::vector<float> noise(num_actions_);
    float sum = 0.0f;
    for (int64_t i = 0; i < num_actions_; ++i) {
        noise[i] = rnd_->Gamma(config_.root_dirichlet_alpha, 1.0f);
        sum += noise[i];
    }
    if (sum < 1e-8f) sum = 1e-8f; // ゼロ除算防止の安全策

    // ノイズのブレンド
    float frac = config_.root_exploration_fraction;
    for (int64_t i = 0; i < num_actions_; ++i) {
        auto edge = node->edges[i];
        edge->prior_prob = edge->prior_prob * (1.0f - frac) + (noise[i] / sum) * frac;
    }
}

int64_t MCTSEngine::SelectAction(std::shared_ptr<Node> node, const MinMaxStats& min_max_stats)
{
    anet::ProfileRange r("MCTSEngine::SelectAction");

    int64_t best_action = -1;
    float best_ucb = std::numeric_limits<float>::lowest();

    for (const auto& edge : node->edges) {
        float ucb = CalculateUCBScore(node, edge, min_max_stats);
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_action = edge->action;
        }
    }

    ANET_ASSERT(best_action != -1);
    return best_action;
}

float MCTSEngine::CalculateUCBScore(std::shared_ptr<Node> parent, std::shared_ptr<Edge> edge, const MinMaxStats& min_max_stats)
{
    // PUCTアルゴリズムによるスコア計算
    float pb_c = std::log((parent->visit_count + config_.pb_c_base + 1.0f) / config_.pb_c_base) + config_.pb_c_init;
    pb_c *= std::sqrt(static_cast<float>(parent->visit_count)) / (edge->visit_count + 1.0f);

    float prior_score = pb_c * edge->prior_prob;

    if (edge->visit_count > 0) {
        float normalized_q = min_max_stats.Normalize(edge->q_value);
        return normalized_q + prior_score;
    } else {
        // 未訪問の場合は、親ノードの予測Valueを使って正規化スコアをゲタ履きさせる
        float normalized_parent_v = min_max_stats.Normalize(parent->value);
        return normalized_parent_v + prior_score;
    }
}

void MCTSEngine::Backpropagate(const std::vector<std::shared_ptr<Edge>>& search_path, float value, MinMaxStats& min_max_stats)
{
    anet::ProfileRange r("MCTSEngine::Backpropagate");

    // 割引報酬和を計算しながら木を遡る
    // 本来は value = r + gamma * v だが、経路を下から遡るために逆順で更新する

    float current_value = value;
    for (auto it = search_path.rbegin(); it != search_path.rend(); ++it) {
        auto edge = *it;

        // 価値の更新
        current_value = edge->reward + config_.discount * current_value;

        // Edgeの統計情報を更新
        edge->q_value = (edge->visit_count * edge->q_value + current_value) / (edge->visit_count + 1);
        edge->visit_count += 1;

        // MinMaxStatsの更新 (正規化用)
        min_max_stats.Update(edge->q_value);
    }
}

std::shared_ptr<MCTSTree> MCTSEngine::Search(
    const torch::Tensor& initial_hidden_state, const torch::Tensor& initial_policy_logits, float initial_value, bool add_noise)
{
    anet::ProfileRange r("MCTSEngine::Search");

    /// @todo MuZero試作制約：現在は1スレッド/1状態の探索。将来的にBatched MCTSへの対応が必要

    auto tree = std::make_shared<MCTSTree>(config_);
    auto root = std::make_shared<Node>();

    // ①ルートノードの初期化
    root->hidden_state = initial_hidden_state;
    root->value = initial_value;
    ExpandNode(root, initial_policy_logits);
    if (add_noise) {
        AddExplorationNoise(root);
    }

    tree->SetRoot(root);

    // ②シミュレーション(探索)ループ
    for (int i = 0; i < config_.num_simulations; ++i) {
        auto current_node = root;
        std::vector<std::shared_ptr<Edge>> search_path;

        // ②-1 Select (末端ノードに到達するまで木を降下)
        while (current_node->IsExpanded()) {
            int64_t action = SelectAction(current_node, tree->GetMinMaxStats());
            auto edge = current_node->edges[action];
            search_path.push_back(edge);

            if (edge->child == nullptr) {
                break; // 未展開のノード(葉)に到達
            }
            current_node = edge->child;
        }

        // search_pathの最後は必ず展開対象のエッジになる
        auto leaf_edge = search_path.back();
        auto parent_node = current_node; // 降下しきった時のノードが親

        // ②-2 Expand & Evaluate (Dynamics と Prediction を用いて展開)
        // テンソルを [B=1, ...] の形式に整える
        auto action_tensor = torch::tensor({ leaf_edge->action }, torch::dtype(torch::kInt64).device(parent_node->hidden_state.device())).unsqueeze(0);

        /// @todo MuZero試作制約：カテゴリカル分布予測から期待値への変換を省略し、直接スカラー値を使用している
        auto infer_out = model_->RecurrentInference(parent_node->hidden_state, action_tensor);
        anet::ProfileRange r_sync("MCTSEngine::Search.sync");
        float reward = infer_out.At("reward").item<float>();    // GPU同期
        float value = infer_out.At("value").item<float>();      // GPU同期
        r_sync.End();

        // 新しい子ノードを構築してエッジに接続
        auto child_node = std::make_shared<Node>();
        child_node->hidden_state = infer_out.At("hidden_state");
        child_node->value = value;
        leaf_edge->reward = reward;
        leaf_edge->child = child_node;

        ExpandNode(child_node, infer_out.At("policy_logits"));

        // ②-3 Backpropagate (経路に沿ってQ値と訪問回数を更新)
        Backpropagate(search_path, value, tree->GetMinMaxStats());
        root->visit_count += 1;
    }

    return tree;
}


// ======================================================
// MuZeroActor
// ======================================================

MuZeroActor::MuZeroActor(
    const ActorConfig& actor_config, const MCTSConfig& mcts_config,
    std::shared_ptr<std::shared_mutex> mutex, std::shared_ptr<float> latest_tau,
    std::shared_ptr<MuZeroNetworkModel> model, const anet::rl::ActionSpec& action_spec, anet::rl::RunMode run_mode, torch::Device device, std::optional<seed_t> seed)
    : anet::RandomHolder(seed)
    , mutex_(mutex), latest_tau_(latest_tau)
    , actor_config_(actor_config), model_(model), num_actions_(action_spec.GetNumActions()), run_mode_(run_mode), device_(device)
{
    ANET_ASSERT(model_ != nullptr);
    ANET_ASSERT(rnd_ != nullptr);

    // MCTSエンジンには MCTSConfig だけを渡す (関心の分離が完了)
    mcts_engine_ = std::make_shared<MCTSEngine>(mcts_config, model_, action_spec, seed);
}

void MuZeroActor::Sync()
{
    /// @todo MuZero試作制約：Actor向けのModelのClone非対応  
}

anet::rl::BatchActionInfo MuZeroActor::MakeAction(const anet::rl::StepCounts& step, const anet::rl::BatchState& state) const
{
    anet::ProfileRange r("MuZeroActor::MakeAction");

    torch::NoGradGuard grad_guard;

    int64_t batch_size = state.obs.size(0);

    // デバイス転送
    torch::Tensor batch_obs_dev = state.obs.to(device_, /*non_blocking=*/true);

    // バッチ分の結果を格納するテンソルを準備
    auto out_actions = torch::empty({ batch_size }, torch::TensorOptions().dtype(torch::kInt64));
    auto out_target_policies = torch::empty({ batch_size, num_actions_ }, torch::TensorOptions().dtype(torch::kFloat32));
    auto out_root_values = torch::empty({ batch_size, 1 }, torch::TensorOptions().dtype(torch::kFloat32));

    // τ（温度係数）をアニーリングで決定 （train_step を使用 ）
    float temperature = 0.0f; // 評価モード(Eval)の場合はデフォルト0
    if (!anet::rl::IsEval(run_mode_)) {
        //float progress = static_cast<float>(step.train_step) / std::max<int64_t>(1, actor_config_.temp_decay_steps);
        float progress = static_cast<float>(step.exp_step) / static_cast<float>(std::max<int64_t>(1, actor_config_.temp_decay_steps));
        progress = std::clamp(progress, 0.0f, 1.0f);
        temperature = actor_config_.temp_start - progress * (actor_config_.temp_start - actor_config_.temp_end);
        if (latest_tau_) {
            *latest_tau_ = temperature;
        }
    }

    // @todo MuZero試作制約：現在はMCTSがバッチ化されていないため、各環境(バッチ)ごとにループしてMCTSを回す

    bool is_eval = anet::rl::IsEval(run_mode_);

    // 読込排他開始
    std::shared_lock<std::shared_mutex> lock(*mutex_);

    for (int64_t b = 0; b < batch_size; ++b) {

        // Initial Inference (観測から初期の隠れ状態・価値・方策を得る)
        auto single_obs = batch_obs_dev[b].unsqueeze(0); // [1, obs_dim...]
        auto infer_out = model_->InitialInference(single_obs);
        auto hidden_state = infer_out.At("hidden_state");
        auto policy_logits = infer_out.At("policy_logits");

        float value = 0.0f;
        {
            anet::ProfileRange r_sync("MuZeroActor::MakeAction.syncInitialValue");
            value = infer_out.At("value").item<float>();       // GPU同期
        }

        // MCTSの実行
        auto tree = mcts_engine_->Search(hidden_state, policy_logits, value, !is_eval);
        auto root = tree->GetRoot();

        // 訪問回数 (visit_count) から Target Policy と 行動 を決定する
        std::vector<float> target_policy(num_actions_, 0.0f);
        int64_t selected_action = -1;

        if (temperature < 1e-3f) {
            // ArgMax (搾取)
            int max_visit = -1;
            for (const auto& edge : root->edges) {
                if (edge->visit_count > max_visit) {
                    max_visit = edge->visit_count;
                    selected_action = edge->action;
                }
            }
            target_policy[selected_action] = 1.0f;
        } else {
            // Softmax Sampling (探索)
            float inv_temp = 1.0f / temperature;
            float sum_probs = 0.0f;
            std::vector<float> adjusted_counts(num_actions_, 0.0f);

            for (int i = 0; i < num_actions_; ++i) {
                int visits = root->edges[i]->visit_count;
                if (visits > 0) {
                    adjusted_counts[i] = std::pow(static_cast<float>(visits), inv_temp);
                    sum_probs += adjusted_counts[i];
                }
            }
            if (sum_probs < 1e-8f) sum_probs = 1e-8f;

            float rand_val = rnd_->Uniform01() * sum_probs;
            float current_sum = 0.0f;
            selected_action = num_actions_ - 1; // フォールバック

            for (int i = 0; i < num_actions_; ++i) {
                target_policy[i] = adjusted_counts[i] / sum_probs;
                current_sum += adjusted_counts[i];
                if (rand_val <= current_sum && selected_action == num_actions_ - 1) {
                    selected_action = i;
                }
            }
        }

        ANET_ASSERT(selected_action >= 0 && selected_action < num_actions_);

        // バッチ出力用のテンソルに書き込み
        out_actions[b] = selected_action;
        out_target_policies[b] = torch::tensor(target_policy, torch::dtype(torch::kFloat32));
        out_root_values[b] = root->value;
    }

    // 学習に必要な MCTSの結果(ターゲット) を info ディクショナリに格納
    // これらは ReplayBuffer に保存され、Learner で Loss の計算に使われる
    anet::TensorDict info;
    info["target_policy"] = out_target_policies;
    info["root_value"] = out_root_values;

    // BatchActionInfo を返却
    return anet::rl::BatchActionInfo(out_actions, info);
}


// ======================================================
// MuZeroLearner
// ======================================================

MuZeroLearner::MuZeroLearner(
    const LearnerConfig& config,
    std::shared_ptr<std::shared_mutex> mutex,
    std::shared_ptr<MuZeroNetworkModel> model, std::shared_ptr<MuZeroReplayBuffer> replay_buffer, torch::Device device)
    : config_(config)
    , mutex_(mutex)
    , model_(std::move(model))
    , replay_buffer_(std::move(replay_buffer))
    , device_(device)
{
    ANET_ASSERT(model_ != nullptr);
    ANET_ASSERT(replay_buffer_ != nullptr);

    // Optimizerの初期化 (AdamW推奨)
    auto optim_opts = torch::optim::AdamWOptions(config_.learning_rate).weight_decay(config_.weight_decay);
    optimizer_ = std::make_unique<torch::optim::AdamW>(model_->GetSuite()->parameters(), optim_opts);
}

anet::rl::BatchUpdateResultList MuZeroLearner::UpdateFromBatch(
    const anet::rl::StepCounts& step, const anet::rl::BatchExperience& experiences)
{
    anet::ProfileRange r("MuZeroLearner::UpdateFromBatch");

    // Runnerから渡された最新の経験をReplayBufferに蓄積する
    replay_buffer_->Push(experiences);

    // バッファ量不十分なら何もしない
    if (replay_buffer_->Size() < config_.batch_size) {
        return {};
    }

    // Kステップ展開(Unroll)されたバッチデータをサンプリング
    auto sample = replay_buffer_->Sample(config_.batch_size, device_);

    // [ASSERT] サンプリング直後のバッチ形状チェック
    int64_t B = sample.obs.size(0);
    int64_t K = sample.actions.size(1);
    int64_t num_actions = sample.target_policies.size(2);
    ANET_ASSERT(B == config_.batch_size);
    ANET_ASSERT_SHAPE(sample.obs, { B, ANET_SHAPE_ENDANY }); // obsは環境によって次元が変わる
    ANET_ASSERT_SHAPE(sample.actions, { B, K });
    ANET_ASSERT_SHAPE(sample.target_rewards, { B, K });
    ANET_ASSERT_SHAPE(sample.target_values, { B, K + 1 });
    ANET_ASSERT_SHAPE(sample.target_policies, { B, K + 1, num_actions });
    ANET_ASSERT_SHAPE(sample.masks, { B, K + 1 });

    // ここから先はモデルの推論と重み更新が走るため、書込排他を取得する
    std::unique_lock<std::shared_mutex> lock(*mutex_);

    // 更新準備
    model_->GetSuite()->train();
    optimizer_->zero_grad();

    // loss値計算用
    torch::Tensor total_loss = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    torch::Tensor total_v_loss = torch::zeros_like(total_loss);
    torch::Tensor total_p_loss = torch::zeros_like(total_loss);
    torch::Tensor total_r_loss = torch::zeros_like(total_loss);


    // =================================================================
    // Initial Inference (t = 0)
    // =================================================================
    
    auto initial_out = model_->InitialInference(sample.obs);
    torch::Tensor hidden_state = initial_out.At("hidden_state");
    torch::Tensor value = initial_out.At("value");
    torch::Tensor policy_logits = initial_out.At("policy_logits");
    torch::Tensor reward;
    ANET_ASSERT_SHAPE(hidden_state, { B, ANET_SHAPE_ANY });
    ANET_ASSERT_SHAPE(value, { B, 1 });
    ANET_ASSERT_SHAPE(policy_logits, { B, num_actions });
    ANET_ASSERT_NAN(hidden_state);
    ANET_ASSERT_NAN(value);
    ANET_ASSERT_NAN(policy_logits);

    float gradient_scale = 1.0f / static_cast<float>(K);

    // メトリクス用のアキュムレータ
    torch::Tensor total_v_mae = torch::zeros_like(total_loss);
    torch::Tensor total_r_mae = torch::zeros_like(total_loss);
    torch::Tensor total_entropy = torch::zeros_like(total_loss);
    torch::Tensor total_target_v = torch::zeros_like(total_loss);


    // =================================================================
    // BPTT Unroll Loop (t = 0 to K)
    // =================================================================
    for (int k = 0; k <= K; ++k) {

        auto target_value = sample.target_values.select(1, k);
        auto target_policy = sample.target_policies.select(1, k);
        auto mask = sample.masks.select(1, k);
        auto mask_sum = mask.sum() + 1e-8f; // ゼロ除算防止

        ANET_ASSERT_SHAPE(target_value, { B });
        ANET_ASSERT_SHAPE(target_policy, { B, num_actions });
        ANET_ASSERT_SHAPE(mask, { B });
        ANET_ASSERT_SHAPE(mask_sum, { });


        // =================================================================
        // Value Loss
        // =================================================================

        /// @todo MuZero試作制約: 本来はValueをカテゴリカル分布で予測し、CrossEntropyでLossを計算する
        /// @todo MuZero試作制約: target_value に対して Invertible Transform を適用して分散を抑えるべき
        auto v_loss_raw = torch::nn::functional::mse_loss(
            value.squeeze(-1), target_value,
            torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone));
        auto v_loss = (v_loss_raw * mask).sum() / mask_sum;

        // メトリクス：Value MAE & Target Value Mean
        auto v_diff = torch::abs(value.squeeze(-1) - target_value);
        total_v_mae += (v_diff * mask).sum() / mask_sum;
        total_target_v += (target_value * mask).sum() / mask_sum;

        // =================================================================
        // Policy Loss
        // =================================================================

        auto log_probs = torch::log_softmax(policy_logits, /*dim=*/-1);
        auto p_loss_raw = -torch::sum(target_policy * log_probs, /*dim=*/-1);
        auto p_loss = (p_loss_raw * mask).sum() / mask_sum;

        // メトリクス: Policy Entropy (予測分布のエントロピー)
        auto probs = torch::exp(log_probs);
        auto entropy_raw = -torch::sum(probs * log_probs, /*dim=*/-1);
        total_entropy += (entropy_raw * mask).sum() / mask_sum;

        // =================================================================
        // Reward Loss
        // =================================================================

        torch::Tensor r_loss = torch::zeros_like(v_loss);
        if (k > 0) {
            auto target_reward = sample.target_rewards.select(1, k - 1);
            ANET_ASSERT_SHAPE(target_reward, { B });

            auto prev_mask = sample.masks.select(1, k - 1);
            auto prev_mask_sum = prev_mask.sum() + 1e-8f;

            auto r_loss_raw = torch::nn::functional::mse_loss(
                reward.squeeze(-1), target_reward,
                torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone));
            r_loss = (r_loss_raw * prev_mask).sum() / prev_mask_sum;

            // メトリクス: Reward MAE
            auto r_diff = torch::abs(reward.squeeze(-1) - target_reward);
            total_r_mae += (r_diff * prev_mask).sum() / prev_mask_sum;
        }

        // [ASSERT] 各LossのNaNチェック
        ANET_ASSERT_NAN(v_loss);
        ANET_ASSERT_NAN(p_loss);
        ANET_ASSERT_NAN(r_loss);

        total_v_loss += v_loss;
        total_p_loss += p_loss;
        total_r_loss += r_loss;

        auto step_loss =
            (v_loss * config_.value_loss_weight) +
            (p_loss * config_.policy_loss_weight) +
            (r_loss * config_.reward_loss_weight);

        total_loss += step_loss * gradient_scale;

        //  次のステップへ (Dynamicsによる未来予測)
        if (k < K) {
            auto action = sample.actions.select(1, k);
            ANET_ASSERT_SHAPE(action, { B });

            auto dyn_out = model_->RecurrentInference(hidden_state, action);
            hidden_state = dyn_out.At("hidden_state");
            reward = dyn_out.At("reward");
            value = dyn_out.At("value");
            policy_logits = dyn_out.At("policy_logits");

            // [ASSERT] Recurrent推論結果の形状とNaNチェック
            ANET_ASSERT_SHAPE(hidden_state, { B, ANET_SHAPE_ANY });
            ANET_ASSERT_SHAPE(reward, { B, 1 });
            ANET_ASSERT_SHAPE(value, { B, 1 });
            ANET_ASSERT_SHAPE(policy_logits, { B, num_actions });
            ANET_ASSERT_NAN(hidden_state);
            ANET_ASSERT_NAN(reward);
            ANET_ASSERT_NAN(value);
            ANET_ASSERT_NAN(policy_logits);
        }
    }


    // =================================================================
    // Backpropagation & Optimization
    // =================================================================

    ANET_ASSERT_NAN(total_loss); // Backward直前の最終確認
    {
        anet::ProfileRange r_back("MuZeroLearner::Backward");
        total_loss.backward();
    }
    torch::nn::utils::clip_grad_norm_(model_->GetSuite()->parameters(), config_.max_grad_norm);
    {
        anet::ProfileRange r_opt("MuZeroLearner::OptimizerStep");
        optimizer_->step();
    }

    /// @todo MuZero試作制約: PER (Prioritized Experience Replay) を実装した場合、ここで計算した誤差をPriorityとして返却する必要がある

    auto result = std::make_shared<MuZeroUpdateResult>();
    result->total_loss = total_loss;
    result->value_loss = total_v_loss * gradient_scale;
    result->policy_loss = total_p_loss * gradient_scale;
    result->reward_loss = total_r_loss * gradient_scale;
    result->value_mae = total_v_mae * gradient_scale;
    result->reward_mae = total_r_mae * gradient_scale;
    result->policy_entropy = total_entropy * gradient_scale;
    result->target_value_mean = total_target_v * gradient_scale;

    return { result };
}
