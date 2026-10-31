#include "AtariEnv.hpp"

#include <algorithm>
#include <charconv>
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
#include <ale/environment/ale_ram.hpp>

#include "AtariPreprocess.hpp"
#include "anet/diag.hpp"
#include "anet/profile.hpp"

using namespace anet::rl;
using namespace anet::rl::env;
using namespace anet::rl::env::atari;

// ゲームと番号の意味は不変なので、Stepごとに設定を複製せず結果と共有する。
struct anet::rl::env::AtariScalarContext {
    std::string game;
    std::set<int64_t> known_numbers;
};

constexpr int kScreenHeight = 210;
constexpr int kScreenWidth = 160;

static void ValidateResizeMode(const std::string& key, const std::string& value)
{
    if (value != "adaptive_average" && value != "pixel_area") {
        ANET_SYSTEM_ERROR(key << "='" << value << "'. Expected adaptive_average or pixel_area.");
    }
}

static void ValidateScreenSize(const std::string& key, int size, const std::string& resize_mode)
{
    // 設定読取と直接組み立てた設定の構築で、同じネイティブ解像度の条件を使う。
    if (size <= 0) {
        ANET_SYSTEM_ERROR(key << " must be > 0, but was " << size << ".");
    }
    if (resize_mode == "pixel_area" && size > std::min(kScreenHeight, kScreenWidth)) {
        ANET_SYSTEM_ERROR(key << "=" << size
            << " with resize_mode='pixel_area'. Expected an integer in [1,160].");
    }
}

class AtariResetResult final : public anet::rl::SingleResetResult {
public:
    AtariResetResult(anet::rl::SingleState state, anet::rl::AuxData aux)
        : SingleEnvResult(std::move(aux)), SingleResetResult(std::move(state)) {}
};

class AtariStepResult final : public anet::rl::SingleStepResult {
public:
    AtariStepResult(float reward, anet::rl::SingleState state, anet::rl::AuxData aux,
        bool game_ended, float game_score, int64_t game_frames, const torch::Tensor& rgb,
        std::shared_ptr<const AtariScalarContext> context, std::map<int64_t, float> ram_values)
        : SingleEnvResult(std::move(aux)), SingleStepResult(reward, std::move(state)),
          game_ended_(game_ended), game_score_(game_score), game_frames_(game_frames),
          scalar_context_(std::move(context)), ram_values_(std::move(ram_values))
    {
        // 画面のstorageはその遷移が所有する。次のResetでenvが別の画面へ移っても参照は有効。
        if (rgb.defined()) aux_.emplace("rgb_frame", rgb);
    }

    std::optional<float> GetScalar(const std::string& key) const override;
private:
    bool game_ended_;
    float game_score_;
    int64_t game_frames_;
    std::shared_ptr<const AtariScalarContext> scalar_context_;
    std::map<int64_t, float> ram_values_;
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

void RamMetricState::Begin(uint8_t initial)
{
    // 到達値は v0 を含め、回数系は最初のフレーム遷移まで 0 とする。
    previous = initial;
    value = (definition.reducer == RamMetricReducer::MaxSeen
        || definition.reducer == RamMetricReducer::MinSeen) ? initial : 0;
}

void RamMetricState::Observe(uint8_t current)
{
    // 毎フレームの生バイトを前回値と比較し、遷移回数または到達値を更新する。
    switch (definition.reducer) {
    case RamMetricReducer::MaxSeen: value = std::max<int64_t>(value, current); break;
    case RamMetricReducer::MinSeen: value = std::min<int64_t>(value, current); break;
    case RamMetricReducer::IncCount: value += current > previous; break;
    case RamMetricReducer::DecCount: value += current < previous; break;
    case RamMetricReducer::ReachCount: value += current == definition.target && previous != definition.target; break;
    }
    previous = current;
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

static RamMetricDefinition ParseRamMetricDefinition(
    const std::string& key, const std::string& label, const std::string& raw_value)
{
    // 定義 1 行の番地と畳み方を、ROM に依存せず確定する。
    std::istringstream value_stream(raw_value);
    std::string address_text;
    std::string reducer_text;
    std::string extra;
    static const std::regex address_pattern(R"(0x[0-9A-Fa-f]{2})");
    if (!(value_stream >> address_text >> reducer_text) || (value_stream >> extra)
        || !std::regex_match(address_text, address_pattern)) {
        ANET_SYSTEM_ERROR(key << " must be '<0x80..0xFF> <reducer>', but was '"
            << raw_value << "'.");
    }
    size_t parsed = 0;
    unsigned long address = 0;
    try {
        address = std::stoul(address_text, &parsed, 16);
    } catch (const std::exception&) {
        ANET_SYSTEM_ERROR(key << " has invalid address '" << address_text
            << "' in '" << raw_value << "'; expected 0x80..0xFF.");
    }
    if (parsed != address_text.size() || address < 0x80 || address > 0xFF) {
        ANET_SYSTEM_ERROR(key << " has invalid address '" << address_text
            << "' in '" << raw_value << "'; expected 0x80..0xFF.");
    }
    RamMetricReducer reducer;
    uint8_t target = 0;
    if (reducer_text == "max_seen") reducer = RamMetricReducer::MaxSeen;
    else if (reducer_text == "min_seen") reducer = RamMetricReducer::MinSeen;
    else if (reducer_text == "inc_count") reducer = RamMetricReducer::IncCount;
    else if (reducer_text == "dec_count") reducer = RamMetricReducer::DecCount;
    else if (reducer_text.starts_with("reach_count:")) {
        reducer = RamMetricReducer::ReachCount;
        const auto target_text = reducer_text.substr(std::string_view("reach_count:").size());
        unsigned target_value = 0;
        const auto [end, error] = std::from_chars(
            target_text.data(), target_text.data() + target_text.size(), target_value);
        if (error != std::errc{} || end != target_text.data() + target_text.size()
            || target_text.empty() || target_value > 255) {
            ANET_SYSTEM_ERROR(key << " has invalid reducer '" << reducer_text
                << "' in '" << raw_value << "'; expected reach_count:N with N in 0..255.");
        }
        target = static_cast<uint8_t>(target_value);
    } else {
        ANET_SYSTEM_ERROR(key << " has invalid reducer '" << reducer_text
            << "' in '" << raw_value
            << "'; expected max_seen, min_seen, inc_count, dec_count, or reach_count:N.");
    }
    return RamMetricDefinition{
        .label = label,
        .address = static_cast<uint8_t>(address),
        .reducer = reducer,
        .target = target,
    };
}

static std::map<int64_t, RamMetricDefinition> ParseRamMetricBindings(
    const std::string& key, const std::string& raw_value,
    const std::map<std::string, RamMetricDefinition>& definitions)
{
    // 番号と宣言済みラベルの対応を検証し、番号順の実行定義に変換する。
    std::map<int64_t, RamMetricDefinition> bindings;
    std::istringstream metrics_stream(raw_value);
    std::string token;
    bool any_binding = false;
    while (metrics_stream >> token) {
        any_binding = true;
        const auto colon = token.find(':');
        if (colon == std::string::npos || colon == 0 || token.find(':', colon + 1) != std::string::npos) {
            ANET_SYSTEM_ERROR(key << " has invalid binding '" << token
                << "' in '" << raw_value << "'; expected n:label.");
        }
        const auto number_text = token.substr(0, colon);
        int64_t number = 0;
        const auto [end, error] = std::from_chars(
            number_text.data(), number_text.data() + number_text.size(), number);
        if (error != std::errc{} || end != number_text.data() + number_text.size()
            || number < 1) {
            ANET_SYSTEM_ERROR(key << " has invalid number '" << number_text
                << "' in '" << raw_value << "'; expected integer >= 1.");
        }
        const auto label = token.substr(colon + 1);
        const auto definition = definitions.find(label);
        if (definition == definitions.end()) {
            ANET_SYSTEM_ERROR(key << " references unknown label '" << label
                << "' in '" << raw_value << "'; expected a declared label in the same game block.");
        }
        if (bindings.contains(number)) {
            ANET_SYSTEM_ERROR(key << " repeats number " << number
                << " in '" << raw_value << "'; expected unique numbers per game.");
        }
        bindings.emplace(number, definition->second);
    }
    if (!any_binding) {
        ANET_SYSTEM_ERROR("Atari RAM metric bindings must not be empty: " << key << ".");
    }
    return bindings;
}

AtariEnvConfig::AtariEnvConfig(
    const anet::ConfigData& config_data, const std::string& config_prefix)
    : Config(config_data, "AtariEnv", config_prefix)
{
    ANET_READ_CONFIG(config_data, game);
    ANET_READ_CONFIG(config_data, rom_dir);
    ANET_READ_CONFIG(config_data, screen_size);
    ANET_READ_CONFIG(config_data, resize_mode);
    // 読み取る宣言を両方検証し、個別上書きで不正な共通値を隠さない。
    for (const auto& key : {MakeDefaultConfigKey("resize_mode"), MakeOverrideConfigKey("resize_mode")}) {
        if (config_data.Has(key)) ValidateResizeMode(key, config_data.Get(key));
    }
    const auto override_size_key = MakeOverrideConfigKey("screen_size");
    const auto size_key = config_data.Has(override_size_key) ? override_size_key : MakeDefaultConfigKey("screen_size");
    ValidateScreenSize(size_key, screen_size, resize_mode);
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

    // RAM 定義はゲーム別に検証し、番号付きの定義だけを実行対象として保持する。
    const std::string ram_prefix = MakeDefaultConfigKey("ram_metric") + ".";
    const std::regex default_key_pattern(
        "^" + MakeDefaultConfigKey("ram_metric")
        + R"(\.\[([a-z0-9]+(?:_[a-z0-9]+)*)\]\.(metrics|\[[A-Za-z0-9_]+\])$)");
    const std::regex eval_override_pattern(R"(^run\.eval\.\[[^\]]+\]\.env\.ram_metric\..+$)");
    for (const auto& [key, value] : config_data.Map()) {
        if (key.starts_with("run.eval.") && key.find(".env.ram_metric.") != std::string::npos
            && std::regex_match(key, eval_override_pattern)) {
            ANET_SYSTEM_ERROR("Atari RAM metric override is unsupported: " << key << "='" << value << "'.");
        }
        if (key.starts_with(ram_prefix) && !std::regex_match(key, default_key_pattern)) {
            ANET_SYSTEM_ERROR("Invalid Atari RAM metric key: " << key << "='" << value << "'.");
        }
    }
    for (const auto& [game_name, block] : config_data.MakeSubConfigData(MakeDefaultConfigKey("ram_metric"))) {
        std::map<std::string, RamMetricDefinition> definitions;
        for (const auto& [sub_key, raw_value] : block.Map()) {
            if (sub_key == "metrics") continue;
            if (sub_key.size() < 3 || sub_key.front() != '[' || sub_key.back() != ']') {
                ANET_SYSTEM_ERROR("AtariEnv.ram_metric.[" << game_name << "]." << sub_key
                    << " is not a RAM definition label.");
            }
            const std::string label = sub_key.substr(1, sub_key.size() - 2);
            const std::string definition_key = ram_prefix + "[" + game_name + "].[" + label + "]";
            definitions.emplace(label, ParseRamMetricDefinition(definition_key, label, raw_value));
        }
        // 検証済みの宣言を Module Config にも収め、Run の実効設定へ反映する。
        for (const auto& [sub_key, raw_value] : block.Map()) {
            const std::string relative_key = "ram_metric.[" + game_name + "]." + sub_key;
            my_config_data_.Set(relative_key, raw_value);
            my_config_json_[relative_key] = raw_value;
        }
        if (!block.Has("metrics")) continue;
        const std::string metrics_key = ram_prefix + "[" + game_name + "].metrics";
        auto bindings = ParseRamMetricBindings(metrics_key, block.Get("metrics"), definitions);
        for (const auto& [number, definition] : bindings) {
            known_ram_metric_numbers.insert(number);
        }
        ram_metrics.emplace(game_name, std::move(bindings));
    }
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
    , scalar_context_(std::make_shared<const AtariScalarContext>(AtariScalarContext{
        .game = config.game, .known_numbers = config.known_ram_metric_numbers}))
    , device_(device)
{
    if (const auto game_metrics = config_.ram_metrics.find(config_.game);
        game_metrics != config_.ram_metrics.end()) {
        for (const auto& [number, definition] : game_metrics->second) {
            ram_metrics_.emplace(number, RamMetricState{.definition = definition});
        }
    }
    // 設定とROMを先に確定し、不完全なEnvを公開しない。
    ValidateConfig();
    resizer_ = std::make_unique<GrayscaleResizer>(
        kScreenHeight, kScreenWidth, config_.screen_size, config_.resize_mode);
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

    log.verbose() << "AtariEnv resize_mode=" << config_.resize_mode;
    Reset();
}

void AtariEnv::BeginRamMetrics()
{
    if (ram_metrics_.empty()) return;
    // reset_game() の直後だけ v0 を採り、soft reset ではこの初期化を呼ばない。
    const auto& ram = ale_->getRAM();
    for (auto& [number, state] : ram_metrics_) state.Begin(ram.get(state.definition.address & 0x7F));
}

void AtariEnv::ObserveRamMetrics()
{
    if (ram_metrics_.empty()) return;
    // ALE の RAM キャッシュは act() 後に更新済みなので、現在フレームの値を読む。
    const auto& ram = ale_->getRAM();
    for (auto& [number, state] : ram_metrics_) state.Observe(ram.get(state.definition.address & 0x7F));
}

float AtariEnv::ActAndObserve(int action)
{
    const float reward = static_cast<float>(ale_->act(static_cast<ale::Action>(action)));
    ObserveRamMetrics();
    return reward;
}

void AtariEnv::ResetAleGame()
{
    ale_->reset_game();
    BeginRamMetrics();
}

AtariEnv::~AtariEnv() = default;

void AtariEnv::ValidateConfig() const
{
    // 設定オブジェクトを直接組み立てた呼び出しも同じ方式契約を守る。
    const auto mode_prefix = config_.GetOverridePrefix().empty() ? "AtariEnv" : config_.GetOverridePrefix();
    ValidateResizeMode(mode_prefix + ".resize_mode", config_.resize_mode);
    ValidateScreenSize(mode_prefix + ".screen_size", config_.screen_size, config_.resize_mode);
    if (!std::regex_match(config_.game, std::regex("[a-z0-9]+(?:_[a-z0-9]+)*"))) {
        ANET_SYSTEM_ERROR("AtariEnv.game must be a non-empty snake_case ROM stem, but was '" << config_.game << "'.");
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

    float reward = ActAndObserve(ale::PLAYER_A_FIRE);
    if (ale_->game_over(false)) {
        ResetAleGame();
    }
    reward += ActAndObserve(action_set_[2]);
    if (ale_->game_over(false)) {
        ResetAleGame();
    }
    return reward;
}

void AtariEnv::ApplyResetActions()
{
    // NOOP回数はEnv RNGから決定し、sticky action用ALE RNGと分離する。
    if (config_.noop_max > 0) {
        const int noops = rnd_->RandInt(1, config_.noop_max);
        for (int i = 0; i < noops; ++i) {
            ActAndObserve(ale::PLAYER_A_NOOP);
            if (ale_->game_over(false)) {
                ResetAleGame();
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
    return resizer_->Resize(frame.data()).to(device_);
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
    AuxData aux{
        { "game_score", torch::tensor(game_score_, torch::kFloat32) },
        { "game_len", torch::tensor(game_len_, torch::kInt64) },
        { "game_frames", torch::tensor(static_cast<int64_t>(ale_->getEpisodeFrameNumber()), torch::kInt64) },
        { "lives", torch::tensor(static_cast<int64_t>(current_lives_), torch::kInt64) },
    };
    for (const auto& [number, state] : ram_metrics_) {
        aux.emplace("ram_metric." + state.definition.label, torch::tensor(state.Value(), torch::kInt64));
    }
    return aux;
}

void AtariEnv::RecordGameCompletion(bool truncated)
{

    // metrics / trace を持たない実行（EvalPanel 等）のゲームも、時刻付きで Run ログから追えるようにする。
    // スコアは ALE の整数報酬の累積なので整数で出す（float の既定書式は 7 桁以上を指数表記にする）。
    std::ostringstream ram_suffix;
    if (!ram_metrics_.empty()) {
        ram_suffix << " ram_metric:";
        for (const auto& [number, state] : ram_metrics_) {
            ram_suffix << ' ' << state.definition.label << '=' << state.Value();
        }
    }
    log.verbose() << (truncated ? "Game truncated by max_episode_frames." : "Game over.")
        << " game_score=" << static_cast<int64_t>(game_score_)
        << " game_len=" << game_len_
        << " game_frames=" << ale_->getEpisodeFrameNumber() << ram_suffix.str();
}

std::shared_ptr<const SingleResetResult> AtariEnv::Reset()
{
    ANET_PROFILE_FUNC();

    // life-lossだけなら同じゲームを維持し、実ゲーム終了後は作り直す。
    if (life_loss_pending_ && !ale_->game_over(false)) {
        ANET_PROFILE_SCOPE(soft_reset);
        life_loss_pending_ = false;
        // NOOPはStep側で済んでいるため、ここではFIRE系列だけを打つ。
        // noop_maxは実ゲームを作り直すときだけ適用する。
        game_score_ += ApplyFireReset();
    } else {
        ANET_PROFILE_SCOPE(hard_reset);
        ResetAleGame();
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
    if (action < 0 || action >= static_cast<int64_t>(action_set_.size())) {
        ANET_SYSTEM_ERROR("AtariEnv action index out of range. action=" << action
            << ", expected=[0," << action_set_.size() << ").");
    }

    // skip窓全体の報酬を集約し、real game overだけを早期終了理由にする。
    const auto ale_action = static_cast<ale::Action>(action_set_[static_cast<size_t>(action)]);
    float reward_raw = 0.0f;
    RollingMaxPool rolling_max_pool;
    for (int i = 0; i < config_.frame_skip; ++i) {
        reward_raw += ActAndObserve(ale_action);
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

    auto grid = resizer_->Resize(pooled_frame_.data()).to(device_);

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

    CaptureRgbFrame();
    auto state = MakeState(std::move(grid), done, truncated, false);
    const float reward = config_.reward_clip
        ? (reward_raw > 0.0f ? 1.0f : reward_raw < 0.0f ? -1.0f : 0.0f)
        : reward_raw;
    // 学習用の観測・RGB・auxを確定してから、life-loss後のNOOPを進める。
    auto aux = MakeAuxData();
    if (life_done) {
        ANET_PROFILE_SCOPE(life_loss_noop);
        game_score_ += ActAndObserve(ale::PLAYER_A_NOOP);
    }
    // 補助NOOPで終わった場合も、Stepの結果へ入れるゲームの確定値をここで記録する。
    const bool game_ended = real_done || truncated || (life_done && ale_->game_over(false));
    if (game_ended) RecordGameCompletion(truncated);
    // RAMも確定値の時点で所有する。途中のstepではNOOPが無く、速報値と同じ時点になる。
    std::map<int64_t, float> ram_values;
    for (const auto& [number, metric] : ram_metrics_) {
        ram_values.emplace(number, static_cast<float>(metric.Value()));
    }
    return std::make_shared<const AtariStepResult>(reward, std::move(state), std::move(aux),
        game_ended, game_score_, ale_->getEpisodeFrameNumber(), rgb_frame_, scalar_context_, std::move(ram_values));
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

std::optional<float> AtariStepResult::GetScalar(const std::string& key) const
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const bool life_only = (next_state.done || next_state.truncated) && !game_ended_;
    // ゲーム単位の鍵は、途中なら速報値、残機だけの終端ならNaN、ゲーム終了なら確定値。
    if (key == "game_score" || key == "game_len" || key == "game_frames") {
        if (life_only) return nan;
        if (game_ended_ && key == "game_score") return game_score_;
        if (game_ended_ && key == "game_frames") return static_cast<float>(game_frames_);
        return SingleEnvResult::GetScalar(key);
    }
    if (const auto threshold = ParseGameScoreThreshold(key)) {
        if (life_only) return nan;
        return *GetScalar("game_score") >= *threshold ? 1.0f : 0.0f;
    }
    if (key == "hns57" || key == "hns49") {
        if (life_only) return nan;
        const auto baseline = key == "hns57" ? HnsBaseline::Dqn57 : HnsBaseline::Nature49;
        return HumanNormalizedScore(scalar_context_->game, *GetScalar("game_score"), baseline).value_or(nan);
    }
    if (key.starts_with("ram_metric.")) {
        constexpr std::string_view prefix = "ram_metric.[";
        if (!key.starts_with(prefix) || !key.ends_with(']') || key.size() <= prefix.size() + 1) {
            ANET_SYSTEM_ERROR("Invalid AtariEnv scalar key: " << key
                << ". Expected ram_metric.[n] with n >= 1.");
        }
        const std::string_view number_text(key.data() + prefix.size(), key.size() - prefix.size() - 1);
        if (!std::all_of(number_text.begin(), number_text.end(), [](char c) { return c >= '0' && c <= '9'; })) {
            ANET_SYSTEM_ERROR("Invalid AtariEnv scalar key: " << key
                << ". Expected ram_metric.[n] with n >= 1.");
        }
        int64_t number = 0;
        const auto [end, error] = std::from_chars(number_text.data(), number_text.data() + number_text.size(), number);
        if (error != std::errc{} || end != number_text.data() + number_text.size() || number < 1) {
            ANET_SYSTEM_ERROR("Invalid AtariEnv scalar key: " << key
                << ". Expected ram_metric.[n] with n in int64 range.");
        }
        if (!scalar_context_->known_numbers.contains(number)) return std::nullopt;
        const auto metric = ram_values_.find(number);
        return life_only || metric == ram_values_.end() ? nan : metric->second;
    }
    // livesなどのauxはNOOP前のまま読み、非スカラーの画面はscalarとして公開しない。
    return SingleEnvResult::GetScalar(key);
}

std::optional<float> AtariEnv::GetScalar(const std::string& key, int64_t index) const
{
    // env本体への問い合わせは、今の残機だけを返す。遷移の値はStepの結果へ移した。
    if (key == "lives") return static_cast<float>(current_lives_);
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
