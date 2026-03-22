// muzero_proto_agent.hpp
#pragma once

#include <memory>
#include <string>
#include <optional>
#include <torch/torch.h>
#include "anet/rl.hpp"
#include "anet/config.hpp"
#include "anet/nn.hpp"
#include "anet/agent.hpp"

namespace anet::rl::muzero_proto {  // MuZero試作版


    // ======================================================
    // MuZero Config Structs
    // ======================================================

    struct ModelConfig {
        struct {
            std::string rep;
            std::string dyn;
            std::string pred;
            std::string reward_branch;
            std::string value_branch;
            std::string policy_branch;
        } structure;
        int64_t hidden_state_dim = 256;
        struct {
            anet::nn::WeightInitConfig reward = { 4, 0.0, "linear", 0.0 };
            anet::nn::WeightInitConfig value = { 4, 0.0, "linear", 0.0 };
            anet::nn::WeightInitConfig policy = { 4, 0.0, "linear", 0.0 };
        } head_init_weight;
    };

    struct MCTSConfig {
        int num_simulations = 50;
        float discount = 0.997f;
        float pb_c_init = 1.25f;
        float pb_c_base = 19652.0f;
        float root_dirichlet_alpha = 0.25f;
        float root_exploration_fraction = 0.25f;
    };

    struct ActorConfig {
        float temp_start = 1.0f;
        float temp_end = 0.05f;
        int64_t temp_decay_steps = 100000;
    };

    struct ReplayBufferConfig {
        int64_t capacity = 100000;
        int n_step = 5;
        float gamma = 0.997f;
        int unroll_steps = 5;
    };

    struct LearnerConfig {
        float learning_rate = 1e-3f;
        float weight_decay = 1e-4f;
        float max_grad_norm = 1.0f;
        int64_t batch_size = 128;
        float value_loss_weight = 1.0f;
        float policy_loss_weight = 1.0f;
        float reward_loss_weight = 1.0f;
    };


    // ======================================================
    // MuZeroAgentConfig 
    // ======================================================

    struct MuZeroAgentConfig : public anet::Config {
        ModelConfig model;
        MCTSConfig mcts;
        ActorConfig actor;
        ReplayBufferConfig buffer;
        LearnerConfig learner;

        explicit MuZeroAgentConfig(const anet::ConfigData& config_data = anet::EmptyConfigData)
            : anet::Config(config_data, "MuZeroAgent")
        {
            // --- Model ---
            ANET_READ_CONFIG(config_data, model.hidden_state_dim);
            ANET_READ_CONFIG(config_data, model.structure.rep);
            ANET_READ_CONFIG(config_data, model.structure.dyn);
            ANET_READ_CONFIG(config_data, model.structure.pred);
            ANET_READ_CONFIG(config_data, model.structure.reward_branch);
            ANET_READ_CONFIG(config_data, model.structure.value_branch);
            ANET_READ_CONFIG(config_data, model.structure.policy_branch);

            // --- MCTS ---
            ANET_READ_CONFIG(config_data, mcts.num_simulations);
            ANET_READ_CONFIG(config_data, mcts.discount);
            ANET_READ_CONFIG(config_data, mcts.pb_c_init);
            ANET_READ_CONFIG(config_data, mcts.pb_c_base);
            ANET_READ_CONFIG(config_data, mcts.root_dirichlet_alpha);
            ANET_READ_CONFIG(config_data, mcts.root_exploration_fraction);

            // --- Actor ---
            ANET_READ_CONFIG(config_data, actor.temp_start);
            ANET_READ_CONFIG(config_data, actor.temp_end);
            ANET_READ_CONFIG(config_data, actor.temp_decay_steps);

            // --- Replay ---
            ANET_READ_CONFIG(config_data, buffer.capacity);
            ANET_READ_CONFIG(config_data, buffer.n_step);
            ANET_READ_CONFIG(config_data, buffer.gamma);
            ANET_READ_CONFIG(config_data, buffer.unroll_steps);

            // --- Learner ---
            ANET_READ_CONFIG(config_data, learner.learning_rate);
            ANET_READ_CONFIG(config_data, learner.weight_decay);
            ANET_READ_CONFIG(config_data, learner.max_grad_norm);
            ANET_READ_CONFIG(config_data, learner.batch_size);
            ANET_READ_CONFIG(config_data, learner.value_loss_weight);
            ANET_READ_CONFIG(config_data, learner.policy_loss_weight);
            ANET_READ_CONFIG(config_data, learner.reward_loss_weight);
        }
    };


    // ======================================================
    // MuZeroAgent
    // ======================================================

    // 内部実装クラスの前方宣言
    class MuZeroNetworkModel;
    class MuZeroReplayBuffer;

    class MuZeroAgent : public anet::rl::AgentBase {
    public:
        MuZeroAgent(
            const MuZeroAgentConfig& config, const anet::ConfigData& config_data,
            const anet::rl::BatchEnvSpec& batch_env_spec, const anet::rl::EnvSpec& env_spec, const torch::Device device,
            std::optional<anet::seed_t> seed = std::nullopt);

        std::shared_ptr<anet::rl::Actor> CreateActor(const anet::rl::BatchEnvSpec& batch_env_spec, anet::rl::RunMode run_mode, bool clone_model, std::optional<torch::Device> device = std::nullopt) const override;
        std::shared_ptr<anet::rl::Learner> CreateLearner() override;

    public: // anet::Module
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
    public:

        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override { return std::nullopt; }
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override { return std::nullopt; }
    public:
        int64_t Save(anet::OutputArchive& archive) const override { return 0; }
        int64_t Load(anet::InputArchive& archive) override { return 0; }
    private:
        const MuZeroAgentConfig config_;
        const anet::rl::EnvSpec env_spec_;
        const anet::rl::BatchEnvSpec batch_env_spec_;
        const torch::Device device_;
        //seed_t actor_seed_;
        std::shared_ptr<MuZeroNetworkModel> model_;
        std::shared_ptr<MuZeroReplayBuffer> replay_buffer_;
        std::shared_ptr<float> latest_tau_;
    };


    // ======================================================
    // MuZeroAgentFactory
    // ======================================================

    class MuZeroAgentFactory : public anet::rl::AgentFactory {
    public:
        std::shared_ptr<anet::rl::Agent> CreateAgent(
            const anet::rl::EnvSpec& env_spec, const anet::rl::BatchEnvSpec& batch_env_spec, const torch::Device& device,
            const anet::ConfigData& config_data = anet::EmptyConfigData,
            std::shared_ptr<anet::rl::Notifier> notifier = nullptr,
            std::optional<anet::seed_t> seed = std::nullopt) const override;

        std::string GetTargetAgentClassId() const override { return "MuZeroAgent"; }
    };

}   // namespace anet::rl::muzero_proto

