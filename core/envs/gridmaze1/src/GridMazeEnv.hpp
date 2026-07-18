// GridMazeEnv.hpp
#pragma once

#include <vector>
#include <torch/torch.h>
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/env.hpp"

namespace anet::rl::env {

    struct GridMazeEnvConfig : public anet::Config {
        int width = 5;
        int height = 5;
        seed_t seed = 0;
        bool randomize_seed = false;

        float wall_density = 0.2f;
        float hole_density = 0.1f;

        int init_pos_x = 0;
        int init_pos_y = -1;
        int goal_pos_x = -1;
        int goal_pos_y = -1;

        float reward_goal = 1.0f;
        float reward_hole = -1.0f;
        float reward_wall = -0.1f;
        float reward_step = -0.01f;

        int max_steps = 100;

        GridMazeEnvConfig(const anet::ConfigData& config_data = anet::EmptyConfigData, const std::string& config_prefix = "")
            : anet::Config(config_data, "GridMazeEnv", config_prefix)
        {
            ANET_READ_CONFIG(config_data, width);
            ANET_READ_CONFIG(config_data, height);
            ANET_READ_CONFIG(config_data, seed);
            ANET_READ_CONFIG(config_data, randomize_seed);
            ANET_READ_CONFIG(config_data, wall_density);
            ANET_READ_CONFIG(config_data, hole_density);
            ANET_READ_CONFIG(config_data, init_pos_x);
            ANET_READ_CONFIG(config_data, init_pos_y);
            ANET_READ_CONFIG(config_data, goal_pos_x);
            ANET_READ_CONFIG(config_data, goal_pos_y);
            ANET_READ_CONFIG(config_data, reward_goal);
            ANET_READ_CONFIG(config_data, reward_hole);
            ANET_READ_CONFIG(config_data, reward_wall);
            ANET_READ_CONFIG(config_data, reward_step);
            ANET_READ_CONFIG(config_data, max_steps);
        }
    };

    class GridMazeEnv : public anet::rl::SingleDiscreteEnvBase, public anet::RandomHolder {
    public:
        enum class Cell { Empty = 0, Wall = 1, Hole = 2, Goal = 3 };

        GridMazeEnv(
            const GridMazeEnvConfig& config,
            const torch::Device& device,
            const std::string& name,
            const std::optional<anet::seed_t> seed = std::nullopt);

        anet::rl::EnvSpec GetSpec() const override;
        std::shared_ptr<const anet::rl::SingleResetResult> Reset(anet::rl::RunMode mode = anet::rl::RunMode::Train) override;
        std::shared_ptr<const anet::rl::SingleStepResult> Step(int64_t action, anet::rl::RunMode mode = anet::rl::RunMode::Train) override;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
    private:
        void GenerateGridMap();
        std::vector<float> GetObservation() const;
        anet::rl::AuxData MakeAuxData() const;
    private:
        GridMazeEnvConfig config_;
        torch::TensorOptions obs_opt_;

        std::vector<std::vector<Cell>> map_;
        seed_t current_seed_;
        int pos_x_, pos_y_;
        int step_count_;
        int wall_collisions_;
        float ep_reward_sum_;
        bool done_, truncated_, episode_start_;
        bool episode_just_ended_ = false;

        // メトリクス提供用の直近エピソード結果保持
        float last_episode_len_ = 0.0f;
        float last_reward_sum_ = 0.0f;
        float last_is_success_ = 0.0f;
        float last_is_hole_ = 0.0f;
        float last_wall_collisions_ = 0.0f;
    };

    class GridMazeEnvFactory : public anet::rl::SingleDiscreteEnvFactory {
    public:
        GridMazeEnvFactory() = default;
        std::string GetTargetEnvClassId() const override { return "GridMazeEnv"; }
        std::shared_ptr<anet::rl::SingleDiscreteEnv> CreateSingleEnv(
            const anet::ConfigData& config_data,
            const torch::Device& device, const std::string& name,
            std::optional<anet::seed_t> seed = std::nullopt,
            const std::string& config_prefix = "") override;
    };

} // namespace anet::rl::env
