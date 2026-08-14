#include "AtariEnv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <mutex>
#include <regex>
#include <sstream>
#include <string_view>

#include <ale/ale_interface.hpp>

#include "AtariPreprocess.hpp"
#include "anet/diag.hpp"
#include "anet/profile.hpp"

using namespace anet::rl;
using namespace anet::rl::env;
using namespace anet::rl::env::atari;

constexpr int kScreenHeight = 210;
constexpr int kScreenWidth = 160;

class AtariResetResult final : public anet::rl::SingleResetResult {
public:
    AtariResetResult(anet::rl::SingleState state, anet::rl::AuxData aux)
        : SingleResetResult(std::move(state)), aux_(std::move(aux)) {}

    anet::rl::AuxData GetAuxData() const override { return aux_; }

private:
    anet::rl::AuxData aux_;
};

class AtariStepResult final : public anet::rl::SingleStepResult {
public:
    AtariStepResult(float reward, anet::rl::SingleState state, anet::rl::AuxData aux)
        : SingleStepResult(reward, std::move(state)), aux_(std::move(aux)) {}

    anet::rl::AuxData GetAuxData() const override { return aux_; }

private:
    anet::rl::AuxData aux_;
};

static std::string JoinValues(const std::vector<unsigned>& values)
{
    std::ostringstream stream;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) stream << ',';
        stream << values[i];
    }
    return stream.str();
}

static bool ContainsAction(const std::vector<int>& actions, int action)
{
    return std::find(actions.begin(), actions.end(), action) != actions.end();
}

// WASAPI は ALE SoundSDL の AUDIO_U8 前提を満たさず無音になるため
// (docs/design/220_atari_env.jp.md §3.1)、未設定時のみ DirectSound を既定にする。
// 明示設定された環境変数は尊重する。
static void EnsureSdlAudioDriverDefault()
{
    static std::once_flag once;
    std::call_once(once, [] {
#ifdef _WIN32
        if (std::getenv("SDL_AUDIODRIVER") == nullptr) {
            _putenv_s("SDL_AUDIODRIVER", "directsound");
        }
#endif
    });
}

AtariEnvConfig::AtariEnvConfig(
    const anet::ConfigData& config_data, const std::string& config_prefix)
    : Config(config_data, "AtariEnv", config_prefix)
{
    ANET_READ_CONFIG(config_data, game);
    ANET_READ_CONFIG(config_data, rom_dir);
    ANET_READ_CONFIG(config_data, screen_size);
    ANET_READ_CONFIG(config_data, frame_skip);
    ANET_READ_CONFIG(config_data, max_pool);
    ANET_READ_CONFIG(config_data, repeat_action_probability);
    ANET_READ_CONFIG(config_data, noop_max);
    ANET_READ_CONFIG(config_data, fire_reset);
    ANET_READ_CONFIG(config_data, episodic_life);
    ANET_READ_CONFIG(config_data, reward_clip);
    ANET_READ_CONFIG(config_data, full_action_space);
    ANET_READ_CONFIG(config_data, mode);
    ANET_READ_CONFIG(config_data, difficulty);
    ANET_READ_CONFIG(config_data, max_episode_frames);
    ANET_READ_CONFIG(config_data, retain_rgb_frame);
    ANET_READ_CONFIG(config_data, display_screen);
    ANET_READ_CONFIG(config_data, sound);
}

AtariEnv::AtariEnv(
    const AtariEnvConfig& config,
    const torch::Device& device,
    const std::string& name,
    std::optional<anet::seed_t> seed,
    RunMode run_mode)
    : SingleDiscreteEnvBase(name, run_mode, config.GetScopedConfigData())
    , RandomHolder(seed)
    , config_(config)
    , device_(device)
{
    // 設定とROMを先に確定し、不完全なEnvを公開しない。
    ValidateConfig();
    if (config_.sound) {
        EnsureSdlAudioDriverDefault();
    }
    const auto rom_path = ResolveRomPath();
    ale_ = std::make_unique<ale::ALEInterface>();
    ConfigureAle(rom_path);
    SelectActions();

    if (config_.display_screen) {
        log.warn() << "AtariEnv.display_screen=true creates one SDL window per ALE instance and is unsupported with worker threads; use AtariView for Runner observation.";
    }

    Reset();
}

AtariEnv::~AtariEnv() = default;

void AtariEnv::ValidateConfig() const
{
    if (!std::regex_match(config_.game, std::regex("[a-z0-9]+(?:_[a-z0-9]+)*"))) {
        ANET_SYSTEM_ERROR("AtariEnv.game must be a non-empty snake_case ROM stem, but was '" << config_.game << "'.");
    }
    if (config_.screen_size <= 0) {
        ANET_SYSTEM_ERROR("AtariEnv.screen_size must be > 0, but was " << config_.screen_size << ".");
    }
    if (config_.frame_skip < 1) {
        ANET_SYSTEM_ERROR("AtariEnv.frame_skip must be >= 1, but was " << config_.frame_skip << ".");
    }
    if (!std::isfinite(config_.repeat_action_probability)
        || config_.repeat_action_probability < 0.0f
        || config_.repeat_action_probability > 1.0f) {
        ANET_SYSTEM_ERROR("AtariEnv.repeat_action_probability must be in [0,1], but was "
            << config_.repeat_action_probability << ".");
    }
    if (config_.noop_max < 0) {
        ANET_SYSTEM_ERROR("AtariEnv.noop_max must be >= 0, but was " << config_.noop_max << ".");
    }
    if (config_.mode < -1) {
        ANET_SYSTEM_ERROR("AtariEnv.mode must be >= -1, but was " << config_.mode << ".");
    }
    if (config_.difficulty < -1) {
        ANET_SYSTEM_ERROR("AtariEnv.difficulty must be >= -1, but was " << config_.difficulty << ".");
    }
    if (config_.max_episode_frames < 0) {
        ANET_SYSTEM_ERROR("AtariEnv.max_episode_frames must be >= 0, but was "
            << config_.max_episode_frames << ".");
    }
}

std::string AtariEnv::ResolveRomPath() const
{
    std::filesystem::path rom_dir;
    std::string source;
    if (!config_.rom_dir.empty()) {
        rom_dir = std::filesystem::path(config_.rom_dir);
        source = "AtariEnv.rom_dir";
    } else {
        const char* env_rom_dir = std::getenv("ATARI_ROM_DIR");
        if (env_rom_dir != nullptr && std::string(env_rom_dir).empty() == false) {
            rom_dir = std::filesystem::path(env_rom_dir);
            source = "ATARI_ROM_DIR";
        }
    }

    const auto rom_path = rom_dir.empty()
        ? std::filesystem::path(config_.game + ".bin")
        : rom_dir / (config_.game + ".bin");
    if (source.empty() || !std::filesystem::is_regular_file(rom_path)) {
        ANET_SYSTEM_ERROR("Atari ROM not found. path='" << rom_path.string()
            << "'. Set AtariEnv.rom_dir or ATARI_ROM_DIR to a directory containing '"
            << config_.game << ".bin'.");
    }
    return rom_path.string();
}

void AtariEnv::ConfigureAle(const std::string& rom_path)
{
    // ALE内部skipを無効化し、中間フレームをEnv側で所有する。
    ale_->setInt("random_seed", static_cast<int>(GetSeed() & 0x7FFFFFFF));
    ale_->setFloat("repeat_action_probability", config_.repeat_action_probability);
    ale_->setInt("frame_skip", 1);
    ale_->setInt("max_num_frames_per_episode", 0);
    ale_->setBool("truncate_on_loss_of_life", false);
    ale_->setBool("color_averaging", false);
    ale_->setBool("display_screen", config_.display_screen);
    ale_->setBool("sound", config_.sound);
    ale_->loadROM(std::filesystem::path(rom_path));

    if (config_.mode >= 0) {
        const auto available = ale_->getAvailableModes();
        if (std::find(available.begin(), available.end(), static_cast<unsigned>(config_.mode)) == available.end()) {
            ANET_SYSTEM_ERROR("AtariEnv.mode=" << config_.mode
                << " is unavailable for game='" << config_.game
                << "'. available=" << JoinValues(available) << ".");
        }
        ale_->setMode(static_cast<unsigned>(config_.mode));
    }
    if (config_.difficulty >= 0) {
        const auto available = ale_->getAvailableDifficulties();
        if (std::find(available.begin(), available.end(), static_cast<unsigned>(config_.difficulty)) == available.end()) {
            ANET_SYSTEM_ERROR("AtariEnv.difficulty=" << config_.difficulty
                << " is unavailable for game='" << config_.game
                << "'. available=" << JoinValues(available) << ".");
        }
        ale_->setDifficulty(static_cast<unsigned>(config_.difficulty));
    }
}

void AtariEnv::SelectActions()
{
    const auto actions = config_.full_action_space
        ? ale_->getLegalActionSet()
        : ale_->getMinimalActionSet();
    action_set_.reserve(actions.size());
    action_labels_.reserve(actions.size());
    for (const auto action : actions) {
        action_set_.push_back(static_cast<int>(action));
        auto label = ale::action_to_string(action);
        constexpr std::string_view prefix = "PLAYER_A_";
        if (label.starts_with(prefix)) {
            label.erase(0, prefix.size());
        }
        action_labels_.push_back(std::move(label));
    }
}

EnvSpec AtariEnv::GetSpec() const
{
    StateSpec state_spec;
    state_spec.obs_spec[ObsKeys::kGrid] = anet::TensorSpec{
        .type = anet::SpaceType::Grid,
        .shape = { 1, config_.screen_size, config_.screen_size },
        .dtype = torch::kUInt8,
        .num_classes = 0,
        .min_values = { 0.0 },
        .max_values = { 255.0 },
    };
    return EnvSpec{
        .state_spec = std::move(state_spec),
        .action_spec = ActionSpec{
            .is_discrete = true,
            .value_labels = action_labels_,
        },
        .reward_range = config_.reward_clip
            ? std::pair<float, float>{ -1.0f, 1.0f }
            : std::pair<float, float>{ -std::numeric_limits<float>::max(), std::numeric_limits<float>::max() },
    };
}

void AtariEnv::ApplyResetActions()
{
    // NOOP回数はEnv RNGから決定し、sticky action用ALE RNGと分離する。
    if (config_.noop_max > 0) {
        const int noops = rnd_->RandInt(1, config_.noop_max);
        for (int i = 0; i < noops; ++i) {
            ale_->act(ale::PLAYER_A_NOOP);
            if (ale_->game_over(false)) {
                ale_->reset_game();
            }
        }
    }
    if (config_.fire_reset && ContainsAction(action_set_, ale::PLAYER_A_FIRE)) {
        ale_->act(ale::PLAYER_A_FIRE);
        if (ale_->game_over(false)) {
            ale_->reset_game();
        }
    }
}

torch::Tensor AtariEnv::CaptureObservation()
{
    std::vector<uint8_t> frame;
    ale_->getScreenGrayscale(frame);
    return ResizeGrayscale(
        frame.data(), kScreenHeight, kScreenWidth, config_.screen_size).to(device_);
}

void AtariEnv::CaptureRgbFrame()
{
    if (!config_.retain_rgb_frame) {
        rgb_frame_ = torch::Tensor();
        return;
    }
    std::vector<uint8_t> rgb;
    ale_->getScreenRGB(rgb);
    rgb_frame_ = InterleavedRgbToChw(
        rgb.data(), kScreenHeight, kScreenWidth);
}

SingleState AtariEnv::MakeState(
    torch::Tensor grid, bool done, bool truncated, bool episode_start) const
{
    anet::TensorDict obs;
    obs.Set(ObsKeys::kGrid, std::move(grid));
    return SingleState{
        .obs = std::move(obs),
        .done = done,
        .truncated = truncated,
        .episode_start = episode_start,
    };
}

AuxData AtariEnv::MakeAuxData() const
{
    return AuxData{
        { "episode_score", torch::tensor(episode_score_, torch::kFloat32) },
        { "episode_len", torch::tensor(episode_len_, torch::kInt64) },
        { "episode_frames", torch::tensor(static_cast<int64_t>(ale_->getEpisodeFrameNumber()), torch::kInt64) },
        { "lives", torch::tensor(static_cast<int64_t>(current_lives_), torch::kInt64) },
    };
}

std::shared_ptr<const SingleResetResult> AtariEnv::Reset()
{
    ANET_PROFILE_FUNC();

    // life-loss doneではALEを維持し、実ゲーム境界だけhard resetする。
    if (life_loss_pending_ && !ale_->game_over(false)) {
        life_loss_pending_ = false;
        episode_score_ += static_cast<float>(ale_->act(ale::PLAYER_A_NOOP));
        if (ale_->game_over(false)) {
            completion_available_ = true;
            completed_episode_score_ = episode_score_;
            completed_episode_len_ = episode_len_;
            completed_episode_frames_ = ale_->getEpisodeFrameNumber();
            ale_->reset_game();
            episode_score_ = 0.0f;
            episode_len_ = 0;
            ApplyResetActions();
        }
    } else {
        ale_->reset_game();
        episode_score_ = 0.0f;
        episode_len_ = 0;
        life_loss_pending_ = false;
        ApplyResetActions();
    }

    current_lives_ = ale_->lives();
    CaptureRgbFrame();
    auto state = MakeState(CaptureObservation(), false, false, true);
    return std::make_shared<const AtariResetResult>(std::move(state), MakeAuxData());
}

std::shared_ptr<const SingleStepResult> AtariEnv::Step(int64_t action)
{
    ANET_PROFILE_FUNC();
    completion_available_ = false;
    if (action < 0 || action >= static_cast<int64_t>(action_set_.size())) {
        ANET_SYSTEM_ERROR("AtariEnv action index out of range. action=" << action
            << ", expected=[0," << action_set_.size() << ").");
    }

    // skip窓全体の報酬を集約し、real game overだけを早期終了理由にする。
    const auto ale_action = static_cast<ale::Action>(action_set_[static_cast<size_t>(action)]);
    float reward_raw = 0.0f;
    RollingMaxPool rolling_max_pool;
    for (int i = 0; i < config_.frame_skip; ++i) {
        reward_raw += static_cast<float>(ale_->act(ale_action));
        if (config_.max_pool) {
            std::vector<uint8_t> frame;
            ale_->getScreenGrayscale(frame);
            rolling_max_pool.Push(std::move(frame));
        }
        if (ale_->game_over(false)) break;
    }

    // 観測はこのStepで実行できた最後の最大2フレームだけから作る。
    if (config_.max_pool) {
        rolling_max_pool.Finish(pooled_frame_);
    } else {
        ale_->getScreenGrayscale(pooled_frame_);
    }

    auto grid = ResizeGrayscale(
        pooled_frame_.data(), kScreenHeight, kScreenWidth, config_.screen_size).to(device_);

    episode_score_ += reward_raw;
    episode_len_++;
    const bool real_done = ale_->game_over(false);
    const int new_lives = ale_->lives();
    const bool life_done = config_.episodic_life && !real_done && new_lives < current_lives_;
    const bool done = real_done || life_done;
    const bool truncated = !done && config_.max_episode_frames > 0
        && ale_->getEpisodeFrameNumber() >= config_.max_episode_frames;
    current_lives_ = new_lives;
    life_loss_pending_ = life_done;

    if (real_done || truncated) {
        completion_available_ = true;
        completed_episode_score_ = episode_score_;
        completed_episode_len_ = episode_len_;
        completed_episode_frames_ = ale_->getEpisodeFrameNumber();
    }

    CaptureRgbFrame();
    auto state = MakeState(std::move(grid), done, truncated, false);
    const float reward = config_.reward_clip
        ? (reward_raw > 0.0f ? 1.0f : reward_raw < 0.0f ? -1.0f : 0.0f)
        : reward_raw;
    return std::make_shared<const AtariStepResult>(reward, std::move(state), MakeAuxData());
}

std::optional<float> AtariEnv::GetScalar(const std::string& key, int64_t index) const
{
    if (key == "lives") return static_cast<float>(current_lives_);
    if (key == "episode_score") {
        return completion_available_ ? completed_episode_score_ : std::numeric_limits<float>::quiet_NaN();
    }
    if (key == "episode_len") {
        return completion_available_ ? static_cast<float>(completed_episode_len_) : std::numeric_limits<float>::quiet_NaN();
    }
    if (key == "episode_frames") {
        return completion_available_ ? static_cast<float>(completed_episode_frames_) : std::numeric_limits<float>::quiet_NaN();
    }
    return std::nullopt;
}

std::optional<torch::Tensor> AtariEnv::GetTensor(const std::string& key, int64_t index) const
{
    if (key == "rgb_frame" && config_.retain_rgb_frame && rgb_frame_.defined()) {
        return rgb_frame_;
    }
    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> AtariEnv::GetTensorVector(
    const std::string& key, int64_t index) const
{
    return std::nullopt;
}

std::shared_ptr<SingleDiscreteEnv> AtariEnvFactory::CreateSingleEnv(
    const anet::ConfigData& config_data,
    const torch::Device& device,
    const std::string& name,
    std::optional<anet::seed_t> seed,
    RunMode run_mode,
    const std::string& config_prefix)
{
    const AtariEnvConfig config(config_data, config_prefix);
    return std::make_shared<AtariEnv>(config, device, name, seed, run_mode);
}
