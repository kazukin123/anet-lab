// GridMazeEnv.cpp

#include "GridMazeEnv.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/profile.hpp"
#include "anet/env.hpp"

using namespace anet::rl::env;
namespace LOG = anet::log;


// ----------------------------------------------------
// GridMazeEnv Results
// ----------------------------------------------------

// AuxData を保持するための拡張 Result クラス
class GridMazeResetResult : public anet::rl::SingleResetResult {
public:
    GridMazeResetResult(anet::rl::SingleState state, anet::rl::AuxData aux)
        : anet::rl::SingleResetResult(std::move(state)), aux_(std::move(aux)) {}
    anet::rl::AuxData GetAuxData() const override { return aux_; }
private:
    anet::rl::AuxData aux_;
};

class GridMazeStepResult : public anet::rl::SingleStepResult {
public:
    GridMazeStepResult(float reward, anet::rl::SingleState next_state, anet::rl::AuxData aux)
        : anet::rl::SingleStepResult(reward, std::move(next_state)), aux_(std::move(aux)) {}
    anet::rl::AuxData GetAuxData() const override { return aux_; }
private:
    anet::rl::AuxData aux_;
};


// ----------------------------------------------------
// GridMazeEnv
// ----------------------------------------------------

GridMazeEnv::GridMazeEnv(const GridMazeEnvConfig& config,const torch::Device& device, std::optional<anet::seed_t> seed)
    : RandomHolder(seed), config_(config)
{
    anet::MetricsLogger::Instance()->Log("GridMazeEnvConfig", config_.ToJson());

    obs_opt_ = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    Reset();
}

anet::rl::EnvSpec GridMazeEnv::GetSpec() const
{
    // --- StateSpec ---
    anet::rl::StateSpec state_spec;
    state_spec.obs_spec[anet::rl::ObsKeys::kVector] = anet::TensorSpec {
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
        .num_classes = 0,   // 連続値
        .labels = { "x", "y" },
        .min_values = { 0.0f, 0.0f },
        .max_values = { 1.0f, 1.0f }
    };

    // --- ActionSpec ---
    anet::rl::ActionSpec action_spec {
        .is_discrete = true,
        .value_labels = { "Up", "Left", "Down", "Right" },
        .dims = {
            { 0, 3, "move" }
        }
    };

    // --- EnvSpec ---
    float min_r = std::min({ config_.reward_hole, config_.reward_wall, config_.reward_step });
    float max_r = config_.reward_goal;
    anet::rl::EnvSpec env_spec {
        .state_spec = state_spec,
        .action_spec = action_spec,
        .reward_range = { min_r, max_r }
    };

    return env_spec;
}

std::vector<float> GridMazeEnv::GetObservation() const
{
    float norm_x = (config_.width > 1) ? static_cast<float>(pos_x_) / (config_.width - 1) : 0.0f;
    float norm_y = (config_.height > 1) ? static_cast<float>(pos_y_) / (config_.height - 1) : 0.0f;
    return { norm_x, norm_y };
}

anet::rl::AuxData GridMazeEnv::MakeAuxData() const
{
    anet::rl::AuxData aux;
    aux["seed"] = torch::tensor(static_cast<int64_t>(current_seed_), torch::kInt64);
    aux["width"] = torch::tensor(static_cast<float>(config_.width));
    aux["height"] = torch::tensor(static_cast<float>(config_.height));
    aux["pos_x"] = torch::tensor(static_cast<float>(pos_x_));
    aux["pos_y"] = torch::tensor(static_cast<float>(pos_y_));
    aux["ep_reward_sum"] = torch::tensor(ep_reward_sum_);

    auto map_tensor = torch::zeros({ config_.height, config_.width }, torch::kFloat32);
    for (int y = 0; y < config_.height; ++y) {
        for (int x = 0; x < config_.width; ++x) {
            map_tensor[y][x] = static_cast<float>(map_[y][x]);
        }
    }
    aux["map"] = map_tensor;
    return aux;
}

void GridMazeEnv::GenerateGridMap()
{
    map_.assign(config_.height, std::vector<Cell>(config_.width, Cell::Empty));
    wall_collisions_ = 0;

    rnd_->SetSeed(current_seed_);

    for (int y = 0; y < config_.height; ++y) {
        for (int x = 0; x < config_.width; ++x) {
            float p = rnd_->Uniform01();
            if (p < config_.wall_density) {
                map_[y][x] = Cell::Wall;
            } else if (p < config_.wall_density + config_.hole_density) {
                map_[y][x] = Cell::Hole;
            }
        }
    }

    int gx = config_.goal_pos_x;
    int gy = config_.goal_pos_y;
    if (gx == -1 || gy == -1) {
        gx = (gx == -1) ? rnd_->RandIndex(config_.width) : gx;
        gy = (gy == -1) ? rnd_->RandIndex(config_.height) : gy;
    }
    map_[gy][gx] = Cell::Goal;

    pos_x_ = config_.init_pos_x;
    pos_y_ = config_.init_pos_y;

    if (pos_x_ == -1 || pos_y_ == -1) {
        std::vector<std::pair<int, int>> candidates;
        for (int y = 0; y < config_.height; ++y) {
            for (int x = 0; x < config_.width; ++x) {
                if (map_[y][x] != Cell::Goal) {
                    if ((pos_x_ == -1 || pos_x_ == x) && (pos_y_ == -1 || pos_y_ == y)) {
                        candidates.push_back({ x, y });
                    }
                }
            }
        }
        if (!candidates.empty()) {
            auto idx = rnd_->RandIndex(candidates.size());
            pos_x_ = candidates[idx].first;
            pos_y_ = candidates[idx].second;
        } else {
            pos_x_ = pos_x_ == -1 ? 0 : pos_x_;
            pos_y_ = pos_y_ == -1 ? 0 : pos_y_;
        }
    }

    if (map_[pos_y_][pos_x_] != Cell::Goal) {
        map_[pos_y_][pos_x_] = Cell::Empty;
    }
}

std::shared_ptr<const anet::rl::SingleResetResult> GridMazeEnv::Reset(anet::rl::RunMode mode)
{
    anet::ProfileRange r("GridMazeEnv::Reset");

    if (config_.randomize_seed) {
        current_seed_ = anet::SeedMaker::MakeAutoSeed();
        //LOG::info() << "GridMazeEnv: current_seed=" << current_seed_;
    } else {
        current_seed_ = config_.seed;
    }

    GenerateGridMap();

    step_count_ = 0;
    ep_reward_sum_ = 0.0f;
    done_ = false;
    truncated_ = false;
    episode_start_ = true;

    anet::rl::SingleState state {
        .obs = { anet::rl::ObsKeys::kVector, torch::tensor(GetObservation(), obs_opt_) },
        .done = done_,
        .truncated = truncated_,
        .episode_start = episode_start_
    };

    return std::make_shared<GridMazeResetResult>(std::move(state), MakeAuxData());
}

std::shared_ptr<const anet::rl::SingleStepResult> GridMazeEnv::Step(int64_t action, anet::rl::RunMode mode)
{
    anet::ProfileRange r("GridMazeEnv::Step");
    episode_start_ = false;
    step_count_++;
    episode_just_ended_ = false;

    float reward = config_.reward_step;

    int next_x = pos_x_;
    int next_y = pos_y_;
    if (action == 0) next_y -= 1;      // Up
    else if (action == 2) next_y += 1; // Down
    else if (action == 1) next_x -= 1; // Left
    else if (action == 3) next_x += 1; // Right

    if (next_x < 0 || next_x >= config_.width || next_y < 0 || next_y >= config_.height ||
        map_[next_y][next_x] == Cell::Wall) {
        reward = config_.reward_wall;
        wall_collisions_++;
    } else {
        pos_x_ = next_x;
        pos_y_ = next_y;
        if (map_[pos_y_][pos_x_] == Cell::Hole) {
            reward = config_.reward_hole;
            done_ = true;
        } else if (map_[pos_y_][pos_x_] == Cell::Goal) {
            reward = config_.reward_goal;
            done_ = true;
        }
    }

    if (!done_ && config_.max_steps > 0 && step_count_ >= config_.max_steps) {
        truncated_ = true;
    }

    ep_reward_sum_ += reward;

    if (done_ || truncated_) {
        episode_just_ended_ = true;
 
        last_episode_len_ = static_cast<float>(step_count_);
        last_reward_sum_ = ep_reward_sum_;
        last_is_success_ = (reward == config_.reward_goal) ? 1.0f : 0.0f;
        last_is_hole_ = (reward == config_.reward_hole) ? 1.0f : 0.0f;
        last_wall_collisions_ = static_cast<float>(wall_collisions_);
    }

    anet::rl::SingleState next_state {
        .obs = { anet::rl::ObsKeys::kVector, torch::tensor(GetObservation(), obs_opt_) },
        .done = done_,
        .truncated = truncated_,
        .episode_start = episode_start_
    };

    return std::make_shared<GridMazeStepResult>(reward, std::move(next_state), MakeAuxData());
}

std::optional<float> GridMazeEnv::GetScalar(const std::string& key, int64_t index) const
{
    const float nan = std::numeric_limits<float>::quiet_NaN();

    if (key == "episode_len") {
        if (!episode_just_ended_) return nan;
        return last_episode_len_;
    }
    if (key == "reward_sum") {
        if (!episode_just_ended_) return nan;
        return last_reward_sum_;
    }
    if (key == "is_success") {
        if (!episode_just_ended_) return nan;
        return last_is_success_;
    }
    if (key == "is_hole") {
        if (!episode_just_ended_) return nan;
        return last_is_hole_;
    }
    if (key == "wall_collisions") {
        if (!episode_just_ended_) return nan;
        return last_wall_collisions_;
    }
    return std::nullopt;
}


// ----------------------------------------------------
// GridMazeEnvFactory
// ----------------------------------------------------

std::shared_ptr<anet::rl::SingleDiscreteEnv> GridMazeEnvFactory::CreateSingleEnv(
    const anet::ConfigData& config_data,
    const torch::Device& device, std::optional<anet::seed_t> seed,
    const std::string& config_prefix)
{
    GridMazeEnvConfig config(config_data, config_prefix);
    return std::make_shared<GridMazeEnv>(config, device, seed);
}

ANET_REGISTER_ENV_FACTORY(GridMazeEnvFactory);

