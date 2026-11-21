// GridMazeEnv.cpp

#include "GridMazeEnv.hpp"
#include <array>
#include <exception>
#include <mutex>
#include <unordered_set>
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/env.hpp"

using namespace anet::rl::env;
namespace {

// PRD102の限定例外。共有するのは採番済み入力seedと初期化結果だけ。
class GridMazeProcessSeedState {
public:
    anet::seed_t Get(const anet::log::Logger& logger)
    {
        ANET_PROFILE_FUNC();
        // 例外も結果として保存し、call_once自体の自動retryを防ぐ。
        std::call_once(once_, [&] {
            try {
                Initialize(logger);
            } catch (...) {
                failure_ = std::current_exception();
            }
        });
        if (failure_) {
            std::rethrow_exception(failure_);
        }
        return seed_;
    }

private:
    void Initialize(const anet::log::Logger& logger)
    {
        // 自動採番に失敗した場合は他の戦略へ代替しない。
        try {
            seed_ = anet::SeedMaker::MakeAutoSeed();
        } catch (const std::exception& error) {
            ANET_SYSTEM_ERROR("GridMaze process seed initialization failed: expected an automatic seed; cause=" << error.what());
        }
        logger.info() << "GridMaze process seed initialized: seed=" << seed_;
    }

    std::once_flag once_;
    anet::seed_t seed_ = 0;
    std::exception_ptr failure_;
};

GridMazeProcessSeedState process_seed_state;

// PRD100で許容した限定例外。共有するのは通知済みseedだけで、生成状態は個体が持つ。
std::mutex regeneration_notice_mutex;
std::unordered_set<anet::seed_t> regeneration_notice_seeds;

} // namespace


void GridMazeEnvConfig::ValidateSeedMode() const
{
    // 設定読込と直接構築で、同じモードの値域を検証する。
    if (seed_mode != kSeedModeEpisode && seed_mode != kSeedModeProcess) {
        ANET_SYSTEM_ERROR("GridMazeEnv.seed_mode='" << seed_mode << "': expected "
            << kSeedModeEpisode << ", " << kSeedModeProcess);
    }
}

// ----------------------------------------------------
// GridMazeEnv Results
// ----------------------------------------------------

// AuxData を保持するための拡張 Result クラス
class GridMazeResetResult : public anet::rl::SingleResetResult {
public:
    GridMazeResetResult(anet::rl::SingleState state, anet::rl::AuxData aux)
        : anet::rl::SingleEnvResult(std::move(aux)), anet::rl::SingleResetResult(std::move(state)) {}
};

class GridMazeStepResult : public anet::rl::SingleStepResult {
public:
    GridMazeStepResult(float reward, anet::rl::SingleState state, anet::rl::AuxData aux,
        float reward_sum, int wall_collisions, float reward_goal, float reward_hole)
        : anet::rl::SingleEnvResult(std::move(aux)), anet::rl::SingleStepResult(reward, std::move(state)),
          reward_sum_(reward_sum), wall_collisions_(wall_collisions),
          success_(reward == reward_goal), hole_(reward == reward_hole) {}

    std::optional<float> GetScalar(const std::string& key) const override
    {
        // 累積量は途中でも速報値を返し、結果の成否は終端のstepだけで確定させる。
        if (key == "reward_sum") return reward_sum_;
        if (key == "wall_collisions") return static_cast<float>(wall_collisions_);
        if (key == "is_success" || key == "is_hole") {
            if (!next_state.done && !next_state.truncated) return std::numeric_limits<float>::quiet_NaN();
            return (key == "is_success" ? success_ : hole_) ? 1.0f : 0.0f;
        }
        return anet::rl::SingleEnvResult::GetScalar(key);
    }
private:
    float reward_sum_;
    int wall_collisions_;
    bool success_;
    bool hole_;
};


// ----------------------------------------------------
// GridMazeEnv
// ----------------------------------------------------

GridMazeEnv::GridMazeEnv(const GridMazeEnvConfig& config, const torch::Device& device,
    const std::string& name, std::optional<anet::seed_t> seed, anet::rl::RunMode run_mode)
    : SingleDiscreteEnvBase(name, run_mode, config.GetScopedConfigData())
    , RandomHolder(seed)
    , config_(config)
    , initial_seed_(config.seed.value_or(0))
{
    // 直接構築のDTOにも、設定読込と同じモード検証を適用する。
    config_.ValidateSeedMode();
    // 盤面の確保と座標選択に先立ち、正の寸法を保証する。
    if (config_.width <= 0 || config_.height <= 0) {
        ANET_SYSTEM_ERROR("GridMazeEnv.width=" << config_.width << ", GridMazeEnv.height=" << config_.height
            << ": expected positive board dimensions");
    }
    // -1以外の座標を、その軸の範囲へ明示的に制限する。
    const auto validate_coordinate = [](const char* key, int value, int extent) {
        if (value < -1 || value >= extent) {
            ANET_SYSTEM_ERROR(key << "=" << value << ": expected -1 or [0, " << extent << ")");
        }
    };
    validate_coordinate("GridMazeEnv.init_pos_x", config_.init_pos_x, config_.width);
    validate_coordinate("GridMazeEnv.init_pos_y", config_.init_pos_y, config_.height);
    validate_coordinate("GridMazeEnv.goal_pos_x", config_.goal_pos_x, config_.width);
    validate_coordinate("GridMazeEnv.goal_pos_y", config_.goal_pos_y, config_.height);
    // 固定した開始とゴールの重複は生成を試す前に設定エラーにする。
    if (config_.init_pos_x >= 0 && config_.init_pos_y >= 0
        && config_.init_pos_x == config_.goal_pos_x && config_.init_pos_y == config_.goal_pos_y) {
        ANET_SYSTEM_ERROR("GridMazeEnv.init_pos_x=" << config_.init_pos_x
            << ", GridMazeEnv.init_pos_y=" << config_.init_pos_y
            << ", GridMazeEnv.goal_pos_x=" << config_.goal_pos_x
            << ", GridMazeEnv.goal_pos_y=" << config_.goal_pos_y
            << ": expected distinct fixed start and goal coordinates");
    }
    // プロセス共通の入力seedは構築時に取得し、Resetでは個体内の値を使う。
    if (!config_.seed.has_value() && config_.seed_mode == GridMazeEnvConfig::kSeedModeProcess) {
        initial_seed_ = process_seed_state.Get(log);
    }
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
    ANET_PROFILE_FUNC();
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
            ANET_SYSTEM_ERROR("GridMazeEnv.init_pos_x=" << config_.init_pos_x
                << ", GridMazeEnv.init_pos_y=" << config_.init_pos_y
                << ": no start candidates for width=" << config_.width << ", height=" << config_.height
                << ", goal_pos=(" << gx << "," << gy << "); expected at least one start cell distinct from goal");
        }
    }

    if (map_[pos_y_][pos_x_] != Cell::Goal) {
        map_[pos_y_][pos_x_] = Cell::Empty;
    }
}

bool GridMazeEnv::HasReachableGoal() const
{
    ANET_PROFILE_FUNC();
    // 偶然に開始とゴールが重なった候補は採用しない。
    if (map_[pos_y_][pos_x_] == Cell::Goal) return false;

    // 各セルを一度だけ調べる。判定では乱数を消費しない。
    std::vector<std::vector<bool>> visited(config_.height, std::vector<bool>(config_.width, false));
    std::vector<std::pair<int, int>> pending{{pos_x_, pos_y_}};
    visited[pos_y_][pos_x_] = true;
    constexpr std::array<int, 4> dx{0, -1, 0, 1};
    constexpr std::array<int, 4> dy{-1, 0, 1, 0};
    for (size_t i = 0; i < pending.size(); ++i) {
        const auto [x, y] = pending[i];
        for (int direction = 0; direction < 4; ++direction) {
            const int nx = x + dx[direction];
            const int ny = y + dy[direction];
            if (nx < 0 || ny < 0 || nx >= config_.width || ny >= config_.height) continue;
            if (visited[ny][nx] || map_[ny][nx] == Cell::Wall || map_[ny][nx] == Cell::Hole) continue;
            if (map_[ny][nx] == Cell::Goal) return true;
            visited[ny][nx] = true;
            pending.push_back({nx, ny});
        }
    }
    return false;
}

std::shared_ptr<const anet::rl::SingleResetResult> GridMazeEnv::Reset()
{
    ANET_PROFILE_FUNC();

    // 各Resetの入力seedを固定し、解けない候補だけを派生seedで引き直す。
    const auto initial_seed = !config_.seed.has_value() && config_.seed_mode == GridMazeEnvConfig::kSeedModeEpisode
        ? anet::SeedMaker::MakeAutoSeed() : initial_seed_;
    constexpr int max_retries = 1000;
    bool reachable = false;
    int retry = 0;
    for (; retry <= max_retries; ++retry) {
        current_seed_ = retry == 0 ? initial_seed : anet::splitmix64(initial_seed + static_cast<anet::seed_t>(retry));
        GenerateGridMap();
        if (HasReachableGoal()) {
            reachable = true;
            break;
        }
    }
    if (!reachable) {
        ANET_SYSTEM_ERROR("GridMaze maze generation failed: seed=" << initial_seed
            << ", width=" << config_.width << ", height=" << config_.height
            << ", wall_density=" << config_.wall_density << ", hole_density=" << config_.hole_density
            << ", init_pos=(" << config_.init_pos_x << "," << config_.init_pos_y << ")"
            << ", goal_pos=(" << config_.goal_pos_x << "," << config_.goal_pos_y << ")"
            << ", max_retries=" << max_retries);
    }

    // 同じ固定seedの通知はプロセス全体で一度にし、個体の2回目以降は共有ロックを取らない。
    if (config_.seed.has_value() && retry > 0 && !regeneration_notice_checked_) {
        bool emit_notice = false;
        {
            std::lock_guard<std::mutex> lock(regeneration_notice_mutex);
            emit_notice = regeneration_notice_seeds.insert(*config_.seed).second;
        }
        regeneration_notice_checked_ = true;
        if (emit_notice) {
            log.info() << "GridMaze maze regenerated: seed=" << *config_.seed
                << ", retries=" << retry << ", actual_seed=" << current_seed_;
        }
    }

    step_count_ = 0;
    ep_reward_sum_ = 0.0f;
    // 終端フラグを戻し、新しいエピソードの開始状態にする。
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

std::shared_ptr<const anet::rl::SingleStepResult> GridMazeEnv::Step(int64_t action)
{
    ANET_PROFILE_FUNC();
    episode_start_ = false;
    step_count_++;

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

    // この時点の累積報酬と盤面の結果を、新しいStep結果へ保存する。

    anet::rl::SingleState next_state {
        .obs = { anet::rl::ObsKeys::kVector, torch::tensor(GetObservation(), obs_opt_) },
        .done = done_,
        .truncated = truncated_,
        .episode_start = episode_start_
    };

    return std::make_shared<GridMazeStepResult>(reward, std::move(next_state), MakeAuxData(),
        ep_reward_sum_, wall_collisions_, config_.reward_goal, config_.reward_hole);
}

std::optional<float> GridMazeEnv::GetScalar(const std::string& key, int64_t index) const
{
    // 遷移の統計はStepの結果だけが所有し、env本体に持ち越して公開しない。
    return std::nullopt;
}


// ----------------------------------------------------
// GridMazeEnvFactory
// ----------------------------------------------------

std::shared_ptr<anet::rl::SingleDiscreteEnv> GridMazeEnvFactory::CreateSingleEnv(
    const anet::ConfigData& config_data,
    const torch::Device& device, const std::string& name, std::optional<anet::seed_t> seed,
    anet::rl::RunMode run_mode,
    const std::string& config_prefix)
{
    GridMazeEnvConfig config(config_data, config_prefix);
    return std::make_shared<GridMazeEnv>(config, device, name, seed, run_mode);
}
