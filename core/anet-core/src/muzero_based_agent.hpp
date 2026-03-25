// muzero_based_agent.hpp

#pragma once

#include <memory>
#include <limits>
#include <torch/torch.h>
#include "anet/nn.hpp"
#include "anet/tensor_util.hpp"
#include "anet/random.hpp"
#include "anet/rl.hpp"
#include "anet/graphviz.hpp"
#include "muzero_rb.hpp"
#include "anet/muzero_proto_agent.hpp"


namespace anet::rl::muzero_proto {  // MuZero試作版


    // ======================================================
    // MuZeroNetworkSuite
    // ======================================================

    /// 3つの独立したネットワークを保持するコンテナ
    class MuZeroNetworkSuite : public torch::nn::Module {
    public:
        MuZeroNetworkSuite(
            std::shared_ptr<anet::nn::Network> representation_net,
            std::shared_ptr<anet::nn::Network> dynamics_net,
            std::shared_ptr<anet::nn::Network> prediction_net);

        std::shared_ptr<anet::nn::Network> GetRepresentationNet() const { return representation_net_; }
        std::shared_ptr<anet::nn::Network> GetDynamicsNet() const { return dynamics_net_; }
        std::shared_ptr<anet::nn::Network> GetPredictionNet() const { return prediction_net_; }
    private:
        std::shared_ptr<anet::nn::Network> representation_net_;
        std::shared_ptr<anet::nn::Network> dynamics_net_;
        std::shared_ptr<anet::nn::Network> prediction_net_;
    };


    // ======================================================
    // MuZeroNetworkModel
    // ======================================================

    /// MCTSやLearner向けに、意味のある推論インターフェースを提供する高位ラッパー
    class MuZeroNetworkModel {      // : public anet::Serializable
    public:
        explicit MuZeroNetworkModel(
            const ModelConfig& config, const ConfigData& config_data,
            const anet::rl::StateSpec& state_spec, const anet::rl::ActionSpec& action_spec);

        /// @brief 初期推論: 観測(Observation)から、隠れ状態(s0)、価値(v0)、方策(p0)を算出
        /// @param obs 生の観測テンソル (B, state_dim...)
        /// @return TensorDict containing: "hidden_state", "value", "policy_logits"
        anet::TensorDict InitialInference(const torch::Tensor& obs) const;

        /// @brief 再帰推論: 隠れ状態(s)と行動(a)から、次の隠れ状態(s')、即時報酬(r)、価値(v')、方策(p')を算出
        /// @param hidden_state 現在の隠れ状態テンソル (B, hidden_dim...)
        /// @param action 選択された行動テンソル (B, 1) または (B, action_dim)
        /// @return TensorDict containing: "next_hidden_state", "reward", "value", "policy_logits"
        anet::TensorDict RecurrentInference(const torch::Tensor& hidden_state, const torch::Tensor& action) const;

        /// 内部のSuiteへのアクセス（LearnerがLoss計算等で直接叩く必要がある場合用）
        std::shared_ptr<MuZeroNetworkSuite> GetSuite() const { return suite_; }

        /// ネットワーク全体を特定のデバイスに転送
        void To(torch::Device device);
    private:
        ModelConfig config_;
        std::shared_ptr<MuZeroNetworkSuite> suite_;
        int64_t num_actions_;           ///< One-Hot化のためにアクション数を保持
    };


    // ======================================================
    // MCTS Data Structures
    // ======================================================

    /// 価値(Value)の正規化を行うための統計情報クラス
    class MinMaxStats {
    public:
        void Update(float value);
        float Normalize(float value) const;
    private:
        float minimum_ = std::numeric_limits<float>::max();
        float maximum_ = std::numeric_limits<float>::lowest();
    };

    struct Node;

    /// 状態(Node)間の遷移(行動)を表すエッジ
    /// @todo MuZero試作制約：環境のカオス性(Chance Node)は考慮せず、次のNodeと1:1で結合する
    struct Edge {
        int64_t action;
        float prior_prob;        ///< ネットワークが出力した事前確率 P(s, a)
        float reward = 0.0f;     ///< 遷移によって得られた予測報酬 R(s, a)
        int visit_count = 0;     ///< 訪問回数 N(s, a)
        float q_value = 0.0f;    ///< 行動価値 Q(s, a)
        std::shared_ptr<Node> child{ nullptr }; ///< この行動によって遷移する次の状態

        Edge(int64_t action, float prior_prob);
    };

    /// MCTS上の1つの状態(隠れ状態)を表すノード
    struct Node {
        torch::Tensor hidden_state;  ///< ネットワークの潜在表現 s
        float value = 0.0f;          ///< 予測された価値 V(s)
        int visit_count = 0;         ///< 訪問回数 N(s)
        std::vector<std::shared_ptr<Edge>> edges; ///< 可能な行動へのエッジリスト

        Node();
        bool IsExpanded() const { return !edges.empty(); }
    };

    /// 構築された探索木を保持・管理するコンテナ
    class MCTSTree {
    public:
        explicit MCTSTree(const MCTSConfig& config);

        void SetRoot(std::shared_ptr<Node> root) { root_ = root; }
        std::shared_ptr<Node> GetRoot() const { return root_; }
        MinMaxStats& GetMinMaxStats() { return min_max_stats_; }

        std::unique_ptr<anet::graphviz::GraphViz> CreateGraph(std::optional<int64_t> selected_action = std::nullopt) const;
    private:
        const MCTSConfig config_;
        std::shared_ptr<Node> root_;
        MinMaxStats min_max_stats_;
    };


    // ======================================================
    // MCTSEngine
    // ======================================================

    /// 1つのルート状態から指定回数のシミュレーションを回すロジック本体
    class MCTSEngine : public anet::RandomHolder {
    public:
        MCTSEngine(const MCTSConfig& config, std::shared_ptr<MuZeroNetworkModel> model, const anet::rl::ActionSpec& action_spec, std::optional<seed_t> seed);

        /// ルートノードの初期推論結果を受け取り、探索木を構築して返す
        std::shared_ptr<MCTSTree> Search(
            const torch::Tensor& initial_hidden_state,
            const torch::Tensor& initial_policy_logits, float initial_value, bool add_nois);
    private:
        void ExpandNode(std::shared_ptr<Node> node, const torch::Tensor& policy_logits);
        void AddExplorationNoise(std::shared_ptr<Node> node);
        int64_t SelectAction(std::shared_ptr<Node> node, const MinMaxStats& min_max_stats);
        void Backpropagate(const std::vector<std::shared_ptr<Edge>>& search_path, float value, MinMaxStats& min_max_stats);

        float CalculateUCBScore(std::shared_ptr<Node> parent, std::shared_ptr<Edge> edge, const MinMaxStats& min_max_stats);
    private:
        const MCTSConfig config_;
        const std::shared_ptr<MuZeroNetworkModel> model_;
        const int64_t num_actions_;
    };


    // ======================================================
    // MuZeroActor
    // ======================================================

    /// 環境とインタラクトし、MCTSを駆動して行動を決定するエージェント
    class MuZeroActor : public anet::rl::Actor, public anet::RandomHolder {
    public:
        MuZeroActor(
            const ActorConfig& actor_config, const MCTSConfig& mcts_config,
            std::shared_ptr<std::shared_mutex> mutex, std::shared_ptr<float> latest_tau,
            std::shared_ptr<MuZeroNetworkModel> model, const anet::rl::ActionSpec& action_spec, anet::rl::RunMode run_mode, torch::Device device, std::optional<seed_t> seed);

        /// 現在の観測からMCTSを回し、行動と学習用情報を返す
        std::shared_ptr<anet::rl::BatchActionInfo> MakeAction(const anet::rl::StepCounts& step, const anet::rl::BatchState& state) const override;

        void Sync() override;
    private:
        const ActorConfig actor_config_;
        const std::shared_ptr<MuZeroNetworkModel> model_;
        const int64_t num_actions_;
        const anet::rl::RunMode run_mode_;
        const std::shared_ptr<std::shared_mutex> mutex_;
        const torch::Device device_;
        const std::shared_ptr<float> latest_tau_;

        std::shared_ptr<MCTSEngine> mcts_engine_;
    };


    // ======================================================
    // MuZeroLearner
    // ======================================================

    /// 学習結果をRunner等に返すための情報コンテナ
    class MuZeroUpdateResult : public anet::rl::BatchUpdateResult {
    public:
        torch::Tensor total_loss;
        torch::Tensor value_loss;
        torch::Tensor policy_loss;
        torch::Tensor reward_loss;

        torch::Tensor value_mae;
        torch::Tensor reward_mae;
        torch::Tensor policy_entropy;
        torch::Tensor target_value_mean;
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override
        {
            // 既知のキーであれば、値の有無に関わらずラムダの結果(値 or NaN)を返す
            if (key == "loss.total") return anet::ItemOrNaN<float>(total_loss);
            if (key == "loss.value") return anet::ItemOrNaN<float>(value_loss);
            if (key == "loss.policy") return anet::ItemOrNaN<float>(policy_loss);
            if (key == "loss.reward") return anet::ItemOrNaN<float>(reward_loss);
            if (key == "metric.value_mae") return anet::ItemOrNaN<float>(value_mae);
            if (key == "metric.reward_mae") return anet::ItemOrNaN<float>(reward_mae);
            if (key == "metric.policy_entropy") return anet::ItemOrNaN<float>(policy_entropy);
            if (key == "metric.target_value_mean") return anet::ItemOrNaN<float>(target_value_mean);

            // 全く知らないキーの場合のみ nullopt を返す
            return std::nullopt;
        }

        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override
        {
            return std::nullopt;
        }

        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index) const override
        {
            return std::nullopt;
        }
    };

    /// ReplayBufferからデータをサンプリングし、BPTTでネットワークを更新するクラス
    class MuZeroLearner : public anet::rl::Learner {
    public:
        MuZeroLearner(
            const LearnerConfig& config,
            std::shared_ptr<std::shared_mutex> mutex,
            std::shared_ptr<MuZeroNetworkModel> model, std::shared_ptr<MuZeroReplayBuffer> replay_buffer, torch::Device device);

        /// Runnerから最新の経験を受け取り、RBにPushした上で学習を回す
        anet::rl::BatchUpdateResultList UpdateFromBatch(
            const anet::rl::StepCounts& step, const anet::rl::BatchExperience& experiences) override;

    private:
        const LearnerConfig config_;
        const std::shared_ptr<std::shared_mutex> mutex_;
        std::shared_ptr<MuZeroNetworkModel> model_;
        std::shared_ptr<MuZeroReplayBuffer> replay_buffer_;
        torch::Device device_;

        std::unique_ptr<torch::optim::Optimizer> optimizer_;
    };


} // namespace anet::rl::muzero_proto

