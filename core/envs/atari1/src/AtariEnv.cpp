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
#include <unordered_map>

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

// ---------------------------------------------------------------------------
// 人間正規化スコア（HNS）の基準表
//
// 2 系統の表が併存しており、同じゲームでも値が実質的に異なる（Pong の human は
// 57 表 14.6 / 49 表 9.3）。取り違えると文献比較が狂うため両方を保持する。
// ゲーム名は ALE の rom() 表記（snake_case）。
// ---------------------------------------------------------------------------

struct AtariHnsEntry {
    float random;
    float human;
};

// Wang et al. 2016 (Dueling Network Architectures) 系の 57 ゲーム表。
// 値は DeepMind dqn_zoo/atari_data.py の _ATARI_DATA と同一。
// 評価条件は 30 noop starts / 108,000 frames 上限で、本 env の max_episode_frames と一致する。
// Rainbow / IQN / Agent57 / BBF がこの表を使う。
static const std::unordered_map<std::string, AtariHnsEntry>& HnsTableDqn57()
{
    static const std::unordered_map<std::string, AtariHnsEntry> table = {
        { "alien",              {   227.8f,   7127.7f } },
        { "amidar",             {     5.8f,   1719.5f } },
        { "assault",            {   222.4f,    742.0f } },
        { "asterix",            {   210.0f,   8503.3f } },
        { "asteroids",          {   719.1f,  47388.7f } },
        { "atlantis",           { 12850.0f,  29028.1f } },
        { "bank_heist",         {    14.2f,    753.1f } },
        { "battle_zone",        {  2360.0f,  37187.5f } },
        { "beam_rider",         {   363.9f,  16926.5f } },
        { "berzerk",            {   123.7f,   2630.4f } },
        { "bowling",            {    23.1f,    160.7f } },
        { "boxing",             {     0.1f,     12.1f } },
        { "breakout",           {     1.7f,     30.5f } },
        { "centipede",          {  2090.9f,  12017.0f } },
        { "chopper_command",    {   811.0f,   7387.8f } },
        { "crazy_climber",      { 10780.5f,  35829.4f } },
        { "defender",           {  2874.5f,  18688.9f } },
        { "demon_attack",       {   152.1f,   1971.0f } },
        { "double_dunk",        {   -18.6f,    -16.4f } },
        { "enduro",             {     0.0f,    860.5f } },
        { "fishing_derby",      {   -91.7f,    -38.7f } },
        { "freeway",            {     0.0f,     29.6f } },
        { "frostbite",          {    65.2f,   4334.7f } },
        { "gopher",             {   257.6f,   2412.5f } },
        { "gravitar",           {   173.0f,   3351.4f } },
        { "hero",               {  1027.0f,  30826.4f } },
        { "ice_hockey",         {   -11.2f,      0.9f } },
        { "jamesbond",          {    29.0f,    302.8f } },
        { "kangaroo",           {    52.0f,   3035.0f } },
        { "krull",              {  1598.0f,   2665.5f } },
        { "kung_fu_master",     {   258.5f,  22736.3f } },
        { "montezuma_revenge",  {     0.0f,   4753.3f } },
        { "ms_pacman",          {   307.3f,   6951.6f } },
        { "name_this_game",     {  2292.3f,   8049.0f } },
        { "phoenix",            {   761.4f,   7242.6f } },
        { "pitfall",            {  -229.4f,   6463.7f } },
        { "pong",               {   -20.7f,     14.6f } },
        { "private_eye",        {    24.9f,  69571.3f } },
        { "qbert",              {   163.9f,  13455.0f } },
        { "riverraid",          {  1338.5f,  17118.0f } },
        { "road_runner",        {    11.5f,   7845.0f } },
        { "robotank",           {     2.2f,     11.9f } },
        { "seaquest",           {    68.4f,  42054.7f } },
        { "skiing",             {-17098.1f,  -4336.9f } },
        { "solaris",            {  1236.3f,  12326.7f } },
        { "space_invaders",     {   148.0f,   1668.7f } },
        { "star_gunner",        {   664.0f,  10250.0f } },
        { "surround",           {   -10.0f,      6.5f } },
        { "tennis",             {   -23.8f,     -8.3f } },
        { "time_pilot",         {  3568.0f,   5229.2f } },
        { "tutankham",          {    11.4f,    167.6f } },
        { "up_n_down",          {   533.4f,  11693.2f } },
        { "venture",            {     0.0f,   1187.5f } },
        { "video_pinball",      { 16256.9f,  17667.9f } },
        { "wizard_of_wor",      {   563.5f,   4756.5f } },
        { "yars_revenge",       {  3092.9f,  54576.9f } },
        { "zaxxon",             {    32.5f,   9173.3f } },
    };
    return table;
}

// Mnih et al. 2015 (Nature) Extended Data Table 2 の 49 ゲーム表。
// 同表の "Normalized DQN (% Human)" 列で全件検算済み（AtariEnv_test.cpp）。
// 57 表に対し berzerk / defender / phoenix / pitfall / skiing / solaris /
// surround / yars_revenge の 8 ゲームを欠く。
static const std::unordered_map<std::string, AtariHnsEntry>& HnsTableNature49()
{
    static const std::unordered_map<std::string, AtariHnsEntry> table = {
        { "alien",              {   227.8f,   6875.0f } },
        { "amidar",             {     5.8f,   1676.0f } },
        { "assault",            {   222.4f,   1496.0f } },
        { "asterix",            {   210.0f,   8503.0f } },
        { "asteroids",          {   719.1f,  13157.0f } },
        { "atlantis",           { 12850.0f,  29028.0f } },
        { "bank_heist",         {    14.2f,    734.4f } },
        { "battle_zone",        {  2360.0f,  37800.0f } },
        { "beam_rider",         {   363.9f,   5775.0f } },
        { "bowling",            {    23.1f,    154.8f } },
        { "boxing",             {     0.1f,      4.3f } },
        { "breakout",           {     1.7f,     31.8f } },
        { "centipede",          {  2091.0f,  11963.0f } },
        { "chopper_command",    {   811.0f,   9882.0f } },
        { "crazy_climber",      { 10781.0f,  35411.0f } },
        { "demon_attack",       {   152.1f,   3401.0f } },
        { "double_dunk",        {   -18.6f,    -15.5f } },
        { "enduro",             {     0.0f,    309.6f } },
        { "fishing_derby",      {   -91.7f,      5.5f } },
        { "freeway",            {     0.0f,     29.6f } },
        { "frostbite",          {    65.2f,   4335.0f } },
        { "gopher",             {   257.6f,   2321.0f } },
        { "gravitar",           {   173.0f,   2672.0f } },
        { "hero",               {  1027.0f,  25763.0f } },
        { "ice_hockey",         {   -11.2f,      0.9f } },
        { "jamesbond",          {    29.0f,    406.7f } },
        { "kangaroo",           {    52.0f,   3035.0f } },
        { "krull",              {  1598.0f,   2395.0f } },
        { "kung_fu_master",     {   258.5f,  22736.0f } },
        { "montezuma_revenge",  {     0.0f,   4367.0f } },
        { "ms_pacman",          {   307.3f,  15693.0f } },
        { "name_this_game",     {  2292.0f,   4076.0f } },
        { "pong",               {   -20.7f,      9.3f } },
        { "private_eye",        {    24.9f,  69571.0f } },
        { "qbert",              {   163.9f,  13455.0f } },
        { "riverraid",          {  1339.0f,  13513.0f } },
        { "road_runner",        {    11.5f,   7845.0f } },
        { "robotank",           {     2.2f,     11.9f } },
        { "seaquest",           {    68.4f,  20182.0f } },
        { "space_invaders",     {   148.0f,   1652.0f } },
        { "star_gunner",        {   664.0f,  10250.0f } },
        { "tennis",             {   -23.8f,     -8.9f } },
        { "time_pilot",         {  3568.0f,   5925.0f } },
        { "tutankham",          {    11.4f,    167.6f } },
        { "up_n_down",          {   533.4f,   9082.0f } },
        { "venture",            {     0.0f,   1188.0f } },
        { "video_pinball",      { 16257.0f,  17298.0f } },
        { "wizard_of_wor",      {   563.5f,   4757.0f } },
        { "zaxxon",             {    32.5f,   9173.0f } },
    };
    return table;
}

std::optional<float> anet::rl::env::HumanNormalizedScore(
    const std::string& game, float raw_score, HnsBaseline baseline)
{
    const auto& table = (baseline == HnsBaseline::Dqn57) ? HnsTableDqn57() : HnsTableNature49();
    const auto it = table.find(game);
    if (it == table.end()) return std::nullopt;
    return 100.0f * (raw_score - it->second.random) / (it->second.human - it->second.random);
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

float AtariEnv::ApplyFireReset()
{
    // Breakout系はライフ喪失でボールが消え、FIREを押すまで再投入されない。
    // 標準のwrapper構成はFireResetEnvをEpisodicLifeEnvの外側に置くため、
    // 実game overだけでなくlife-loss後のresetでもFIREが入る。ここもそれに揃える。
    //
    // FIREの後にaction set 3番目も打つのはFireResetEnvと同じ手順（step(1)の後にstep(2)）。
    // FIRE単独では始まらないゲームへの手当てで、意味は問わずindexで指定する。
    // ALEのaction setはAction enum順に並ぶため、FIREを含むなら必ずindex 1に来る。
    if (!config_.fire_reset) return 0.0f;
    if (action_set_.size() < 3) return 0.0f;
    if (action_set_[1] != ale::PLAYER_A_FIRE) return 0.0f;

    float reward = static_cast<float>(ale_->act(ale::PLAYER_A_FIRE));
    if (ale_->game_over(false)) {
        ale_->reset_game();
    }
    reward += static_cast<float>(ale_->act(static_cast<ale::Action>(action_set_[2])));
    if (ale_->game_over(false)) {
        ale_->reset_game();
    }
    return reward;
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
    // hard reset直後は game_score_ を 0 にするため、ここでの報酬は捨ててよい。
    ApplyFireReset();
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
        { "game_score", torch::tensor(game_score_, torch::kFloat32) },
        { "game_len", torch::tensor(game_len_, torch::kInt64) },
        { "game_frames", torch::tensor(static_cast<int64_t>(ale_->getEpisodeFrameNumber()), torch::kInt64) },
        { "lives", torch::tensor(static_cast<int64_t>(current_lives_), torch::kInt64) },
    };
}

std::shared_ptr<const SingleResetResult> AtariEnv::Reset()
{
    ANET_PROFILE_FUNC();

    // life-loss doneではALEを維持し、実ゲーム境界だけhard resetする。
    if (life_loss_pending_ && !ale_->game_over(false)) {
        life_loss_pending_ = false;
        game_score_ += static_cast<float>(ale_->act(ale::PLAYER_A_NOOP));
        if (ale_->game_over(false)) {
            completion_available_ = true;
            completed_game_score_ = game_score_;
            completed_game_len_ = game_len_;
            completed_game_frames_ = ale_->getEpisodeFrameNumber();
            ale_->reset_game();
            game_score_ = 0.0f;
            game_len_ = 0;
            ApplyResetActions();
        } else {
            // ゲームは継続しているのでスコアも継続。NOOPと同じく報酬を累積へ加える。
            // noop_max はここでは打たない（NoopResetEnv 相当は実 reset のみに効く）。
            game_score_ += ApplyFireReset();
        }
    } else {
        ale_->reset_game();
        game_score_ = 0.0f;
        game_len_ = 0;
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

    game_score_ += reward_raw;
    game_len_++;
    const bool real_done = ale_->game_over(false);
    const int new_lives = ale_->lives();
    // lives > 0 ガードは baselines 由来(SB3/CleanRL/rlpyt と同一)。Qbert 系は game over の
    // 数フレーム前に lives=0 を報告するため、ガードが無いと偽の life-loss done が先に出る。
    const bool life_done = config_.episodic_life && !real_done
        && new_lives < current_lives_ && new_lives > 0;
    const bool done = real_done || life_done;
    const bool truncated = !done && config_.max_episode_frames > 0
        && ale_->getEpisodeFrameNumber() >= config_.max_episode_frames;
    current_lives_ = new_lives;
    life_loss_pending_ = life_done;

    if (real_done || truncated) {
        completion_available_ = true;
        completed_game_score_ = game_score_;
        completed_game_len_ = game_len_;
        completed_game_frames_ = ale_->getEpisodeFrameNumber();
    }

    CaptureRgbFrame();
    auto state = MakeState(std::move(grid), done, truncated, false);
    const float reward = config_.reward_clip
        ? (reward_raw > 0.0f ? 1.0f : reward_raw < 0.0f ? -1.0f : 0.0f)
        : reward_raw;
    return std::make_shared<const AtariStepResult>(reward, std::move(state), MakeAuxData());
}

// `game_score.ge.[N]` の N を返す。閾値はゲーム定数ではなく判定基準（Breakout の 1 画面 = 432）
// なので、hns の静的表（§4.8）ではなくキー側のパラメータとして持つ。1 Run で複数の閾値を同時に
// 測るため、ゲーム名で引く表では表現できない。前方一致しないキーは nullopt で呼び出し側へ返す。
static std::optional<float> ParseGameScoreThreshold(const std::string& key)
{
    static constexpr std::string_view kPrefix = "game_score.ge.[";
    if (!key.starts_with(kPrefix) || !key.ends_with(']')) return std::nullopt;

    const std::string threshold_text = key.substr(kPrefix.size(), key.size() - kPrefix.size() - 1);
    if (threshold_text.empty()) {
        ANET_SYSTEM_ERROR("AtariEnv: threshold is empty in scalar key: " << key);
    }

    float threshold = 0.0f;
    size_t parsed_len = 0;
    try {
        threshold = std::stof(threshold_text, &parsed_len);
    } catch (const std::exception&) {
        ANET_SYSTEM_ERROR("AtariEnv: invalid threshold in scalar key: " << key);
    }
    // 末尾に余りがあるキー（`[1x]` 等）を黙って受理しない。
    if (parsed_len != threshold_text.size()) {
        ANET_SYSTEM_ERROR("AtariEnv: invalid threshold in scalar key: " << key);
    }
    return threshold;
}

std::optional<float> AtariEnv::GetScalar(const std::string& key, int64_t index) const
{
    if (key == "lives") return static_cast<float>(current_lives_);
    if (key == "game_score") {
        return completion_available_ ? completed_game_score_ : std::numeric_limits<float>::quiet_NaN();
    }
    // 閾値越えを 0/1 で返す。`mean.` 集約がそのまま「越えたゲームの割合」になる。確定タイミングと
    // NaN 契約は game_score と揃える。未確定 step で 0 を返すと分母が完了 env 数ではなく num_envs
    // になり割合が壊れるため、ここは必ず NaN（§4.7）。
    if (const auto threshold = ParseGameScoreThreshold(key)) {
        if (!completion_available_) return std::numeric_limits<float>::quiet_NaN();
        return completed_game_score_ >= *threshold ? 1.0f : 0.0f;
    }
    if (key == "game_len") {
        return completion_available_ ? static_cast<float>(completed_game_len_) : std::numeric_limits<float>::quiet_NaN();
    }
    if (key == "game_frames") {
        return completion_available_ ? static_cast<float>(completed_game_frames_) : std::numeric_limits<float>::quiet_NaN();
    }
    // 基準表に載らないゲームでも nullopt ではなく NaN を返す。nullopt は
    // DiscreteBatchEnvBase の集約(env.cpp)でバッチ全体を打ち切ってしまうため。
    if (key == "hns57" || key == "hns49") {
        if (!completion_available_) return std::numeric_limits<float>::quiet_NaN();
        const auto baseline = (key == "hns57") ? HnsBaseline::Dqn57 : HnsBaseline::Nature49;
        const auto hns = HumanNormalizedScore(config_.game, completed_game_score_, baseline);
        return hns.value_or(std::numeric_limits<float>::quiet_NaN());
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
