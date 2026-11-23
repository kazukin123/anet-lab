#include "anet/test/catch.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <wx/cmdline.h>
#include <wx/utils.h>
#include "anet/config.hpp"
#include "anet/observers.hpp"
#include "anet/test/util.hpp"
#include "nn_impl.hpp"



TEST_CASE("RFD0117 captures current Env defaults for key comparison", "[.][rfd0117-capture]")
{
    // 履歴の固定入力ではなく、今回変更する出荷設定の全キーを採取する。
    constexpr const char* kCaptureOutput = "ANET_RFD0117_CAPTURE_OUTPUT";
    wxString output_directory;
    REQUIRE(wxGetEnv(kCaptureOutput, &output_directory));
    const auto root = std::filesystem::path(output_directory.ToStdWstring());
    const auto config_root = std::filesystem::current_path() / "apps/runner/config";
    std::filesystem::create_directories(root);
    for (const auto* name : {"Atari", "CartPole", "DropMerge", "GridMaze", "GridMaze_muzero", "ImageCls", "LunarLander"}) {
        DYNAMIC_SECTION(name) {
            const auto output_path = root / (std::string(name) + ".json");
            REQUIRE_FALSE(std::filesystem::exists(output_path));
            const auto input_path = root / (std::string(name) + ".txt");
            {
                std::ofstream input(input_path);
                input << "$include <" << (config_root / "_main.txt").generic_string() << ">\n"
                    << "$include <" << (config_root / (std::string(name) + ".txt")).generic_string() << ">\n";
            }
            const anet::ConfigManager manager(input_path);
            std::ofstream output(output_path);
            REQUIRE(output.good());
            output << manager.GetConfigData().ToJson().dump(2) << '\n';
        }
    }
}

TEST_CASE("RFD0117 resolves an acceptance Run with scalar definitions", "[.][rfd0117-resolve]")
{
    constexpr char kResolveInput[] = "ANET_RFD0117_RESOLVE_INPUT";
    constexpr char kResolveOutput[] = "ANET_RFD0117_RESOLVE_OUTPUT";
    wxString input;
    wxString output;
    REQUIRE(wxGetEnv(kResolveInput, &input));
    REQUIRE(wxGetEnv(kResolveOutput, &output));
    const anet::ConfigManager manager(std::filesystem::path(input.ToStdWstring()));
    const auto data = manager.GetConfigData();
    const anet::rl::ObserverFactory factory(data);
    std::ofstream file(std::filesystem::path(output.ToStdWstring()));
    REQUIRE(file.good());
    const anet::json result = {{"config", data.ToJson()}, {"resolution", manager.GetResolutionJson()},
        {"scalar_defs", anet::rl::ScalarMetricDefsToJson(factory.GetScalarMetricDefs())}};
    file << result.dump(2) << '\n';
}











TEST_CASE("Atari.txt presets and RAM metric groups follow the shipped rules", "[config][shipped_config][atari]")
{
    // 出荷している Atari.txt の検査。設定の仕組みは config_test と AtariEnv_test の fixture で確かめる（docs/quality/impl.md「テスト」の規約）。
    const auto config_dir = std::filesystem::current_path() / "apps/runner/config";
    const auto load_preset = [&](const std::string& preset) {
        // Atari.txt の既定 run.$ は Run 層で AtariEnv.$ を選ぶので、プリセットはそれより強い CLI 層で選ぶ。
        const wxCmdLineEntryDesc description[] = {
            { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
                wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE }, {wxCMD_LINE_NONE},
        };
        wxCmdLineParser cli(description, wxString::FromUTF8("AtariEnv.$=AtariEnv.@" + preset));
        REQUIRE(cli.Parse(false) == 0);
        anet::ConfigManagerOptions options;
        options.config_search_dirs = { config_dir };
        options.overwrite_config_paths = { config_dir / "Atari.txt" };
        return anet::ConfigManager(config_dir / "_main.txt", &cli, options).GetConfigData();
    };

    // プロトコルプリセットは名前が外部の手順（ALE v5、classic）を指すので、名前と値の対応を固定する。
    const auto v5 = load_preset("v5_noop30");
    CHECK(v5.Get<float>("AtariEnv.repeat_action_probability") == Catch::Approx(0.25f));
    CHECK(v5.Get<int>("AtariEnv.noop_max") == 30);
    CHECK_FALSE(v5.Get<bool>("AtariEnv.episodic_life"));
    CHECK_FALSE(v5.Get<bool>("AtariEnv.fire_reset"));

    const auto classic = load_preset("classic");
    CHECK(classic.Get<float>("AtariEnv.repeat_action_probability") == Catch::Approx(0.0f));
    CHECK(classic.Get<int>("AtariEnv.noop_max") == 30);
    CHECK(classic.Get<bool>("AtariEnv.episodic_life"));
    CHECK(classic.Get<bool>("AtariEnv.fire_reset"));

    // RAM の指標の群の規則（RFD0082 D19）。得点の平均を持つ群は番号 1〜4 の mean と max を持ち、得点の EMA を持つ群は EMA も持つ。
    // 軸・event・eval 名・interval は同じ群の得点の指標に揃える。tag の番号は mean 50/52/54/59、EMA 51/53/55/59、max 56/57/58/59。
    // 4 の 59 は指標の番号の整理（RFD0108）までの暫定。
    const anet::rl::ObserverFactory metric_factory(v5);
    const auto& scalar_defs = metric_factory.GetScalarMetricDefs();
    const auto find_scalar = [&](const std::string& tag) {
        return std::find_if(scalar_defs.begin(), scalar_defs.end(), [&](const auto& def) {
            return def.tag == tag;
        });
    };
    const auto check_ram_scalar = [&](const std::string& tag, const std::string& source_key,
        bool has_ema, float ema_alpha, const std::string& reference_tag) {
        INFO(tag);
        const auto scalar = find_scalar(tag);
        const auto reference = find_scalar(reference_tag);
        REQUIRE(scalar != scalar_defs.end());
        REQUIRE(reference != scalar_defs.end());
        CHECK(scalar->source_key == source_key);
        CHECK(scalar->has_ema == has_ema);
        if (has_ema) CHECK(scalar->ema_alpha == Catch::Approx(ema_alpha));
        CHECK(scalar->step_axis == reference->step_axis);
        CHECK(scalar->runner == reference->runner);
        CHECK(scalar->event == reference->event);
        CHECK(scalar->target == reference->target);
        CHECK(scalar->scope == reference->scope);
        CHECK(scalar->eval_name == reference->eval_name);
        CHECK(scalar->interval == reference->interval);
    };
    // 群は設定から拾う。群を足してもテストは直さずに済み、書き忘れた群は落ちる。
    const std::string score_suffix = "/10_game_score_mean";
    std::vector<std::string> groups;
    for (const auto& def : scalar_defs) {
        if (def.tag.ends_with(score_suffix)) groups.push_back(def.tag.substr(0, def.tag.size() - score_suffix.size()));
    }
    REQUIRE_FALSE(groups.empty());
    std::set<std::string> expected_ram_tags;
    for (const auto& group : groups) {
        const std::string stem = group + "/";
        // EMA は得点の平均の EMA を持つ群だけが持ち、係数もそれに揃える。
        const auto score_ema = find_scalar(stem + "11_game_score_mean_ema");
        struct RamTagNumbers { int number; int mean; int ema; int max; };
        for (const auto& [number, mean_no, ema_no, max_no] : {RamTagNumbers{1, 50, 51, 56}, RamTagNumbers{2, 52, 53, 57},
                 RamTagNumbers{3, 54, 55, 58}, RamTagNumbers{4, 59, 59, 59}}) {
            const std::string key = "ram_metric.[" + std::to_string(number) + "]";
            const std::string suffix = "ram_metric_" + std::to_string(number);
            const std::string mean_tag = stem + std::to_string(mean_no) + "_" + suffix + "_mean";
            const std::string max_tag = stem + std::to_string(max_no) + "_" + suffix + "_max";
            check_ram_scalar(mean_tag, "mean." + key, false, 0.0f, stem + "10_game_score_mean");
            check_ram_scalar(max_tag, "max." + key, false, 0.0f, stem + "16_game_score_max");
            expected_ram_tags.insert({mean_tag, max_tag});
            if (score_ema != scalar_defs.end()) {
                const std::string ema_tag = stem + std::to_string(ema_no) + "_" + suffix + "_mean_ema";
                check_ram_scalar(ema_tag, "mean." + key, true, score_ema->ema_alpha,
                    stem + "11_game_score_mean_ema");
                expected_ram_tags.insert(ema_tag);
            }
        }
    }
    // RAM の tag は、どれかの群の規則どおりのものだけにする。
    for (const auto& def : scalar_defs) {
        if (def.tag.find("_ram_metric_") == std::string::npos) continue;
        CAPTURE(def.tag);
        CHECK(expected_ram_tags.contains(def.tag));
    }
}

TEST_CASE("ConfigManager compares frozen RFD0072 configuration inputs", "[.][rfd0072-baseline]")
{
    // 入力定義とgoldenを版管理し、固定設定の展開と比較結果だけをローカルへ置く。
    const auto repo = std::filesystem::current_path();
    wxString prepared_root;
    constexpr const char* kRfd0072RootEnv = "ANET_RFD0072_ROOT";
    REQUIRE(wxGetEnv(kRfd0072RootEnv, &prepared_root));
    const auto root = std::filesystem::path(prepared_root.ToStdWstring());
    std::ifstream manifest_file(repo / "src/core/base/testdata/rfd0072/manifest.json");
    REQUIRE(manifest_file.good());
    const auto manifest = anet::json::parse(manifest_file);
    std::ifstream prepared_file(root / "manifest.json");
    REQUIRE(prepared_file.good());
    REQUIRE(anet::json::parse(prepared_file) == manifest);
    const bool capture = wxGetEnv("ANET_RFD0072_CAPTURE", nullptr);
    for (const auto& input : manifest["inputs"]) {
        const auto id = input["id"].get<std::string>();
        INFO(id);
        const auto baseline_path = repo / "src/core/base/testdata/rfd0072/baseline" / (id + ".json");
        if (capture) {
            REQUIRE_FALSE(std::filesystem::exists(baseline_path));
        }
        const wxCmdLineEntryDesc description[] = {
            { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
                wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },
            { wxCMD_LINE_NONE },
        };
        auto cli = input["cli"].get<std::string>();
        wxCmdLineParser command_line(description, wxString::FromUTF8(cli));
        REQUIRE(command_line.Parse(false) == 0);
        anet::ConfigData::MapType injected;
        for (const auto& [key, value] : manifest["injected"].items()) {
            injected.Set(key, value.get<std::string>());
        }
        const anet::ConfigManager manager(root / (capture ? "inputs-original" : "inputs") / (id + ".txt"), &command_line,
            anet::ConfigManagerOptions{ .injected_config = anet::ConfigData(injected) });
        const auto config_data = manager.GetConfigData();
        anet::json values = anet::json::object();
        for (const auto& [key, value] : config_data.Map()) {
            values[key] = value;
        }
        REQUIRE(values.size() > 100);
        auto resolution = manager.GetResolutionJson();
        const auto overrides = resolution.value("overrides", anet::json::array());
        resolution.erase("overrides");
        const anet::json actual = {
            { "values", values }, { "resolution", resolution },
            { "overrides", overrides },
            { "map_order", config_data.Map().Order() }, { "baseline_commit", manifest["commit"] },
        };
        const auto output_path = capture ? baseline_path : root / "actual" / (id + ".json");
        if (capture) {
            REQUIRE_FALSE(std::filesystem::exists(output_path));
        }
        std::filesystem::create_directories(output_path.parent_path());
        // Windowsでもgoldenの改行をLFのまま保存する。
        std::ofstream output(output_path, std::ios::binary);
        output << actual.dump(2);
        REQUIRE(output.good());
        if (!capture) {
            std::ifstream baseline_file(baseline_path);
            REQUIRE(baseline_file.good());
            const auto baseline = anet::json::parse(baseline_file);
            CHECK(baseline["baseline_commit"] == manifest["commit"]);
            // json objectはキー順に保持される。文字列表記は正規化しない。
            CHECK(actual["values"] == baseline["values"]);
            std::ifstream expected_file(repo / "src/core/base/testdata/rfd0072/resolution" / (id + ".json"));
            CHECK(expected_file.good());
            if (expected_file.good()) {
                auto expected = anet::json::parse(expected_file);
                const auto expected_overrides = expected.at("overrides");
                expected.erase("overrides");
                CHECK(actual["resolution"] == expected);
                if (expected_overrides.empty()) {
                    CHECK(overrides.empty());
                } else {
                    CHECK(overrides == expected_overrides);
                }
            }

        }
    }
}



TEST_CASE("RFD0093 snapshots current configuration inputs", "[.][rfd0093-config-snapshot]")
{
    // 現用設定を全キー・文字列値で採取し、過去の固定入力には触れない。
    wxString directory;
    REQUIRE(wxGetEnv("ANET_RFD0093_OUTPUT", &directory));
    const auto output = std::filesystem::path(directory.ToStdWstring());
    std::filesystem::create_directories(output);
    wxString config_root;
    const auto config = wxGetEnv("ANET_RFD0093_CONFIG_ROOT", &config_root)
        ? std::filesystem::path(config_root.ToStdWstring()).generic_string()
        : (std::filesystem::current_path() / "apps/runner/config").generic_string();
    for (const auto& env : {"Atari", "DropMerge", "LunarLander", "ImageCls", "GridMaze", "GridMaze_muzero", "CartPole"}) {
        const auto path = output / (std::string(env) + ".txt");
        { std::ofstream file(path); file << "$include <" << config << "/_main.txt>\n$include <" << config << "/" << env << ".txt>\n"; }
        const anet::ConfigManager manager(path);
        const auto destination = output / (std::string(env) + ".json");
        REQUIRE_FALSE(std::filesystem::exists(destination));
        std::ofstream file(destination);
        file << manager.GetConfigData().ToJson().dump(2) << '\n';
        REQUIRE(file.good());
    }
}

constexpr const char* kAtariConfigName = "Atari";
constexpr const char* kLunarLanderConfigName = "LunarLander";
constexpr const char* kGridMazeConfigName = "GridMaze";
constexpr const char* kDropMergeConfigName = "DropMerge";
constexpr const char* kSkipRunProfile = "run.@skip";



TEST_CASE("RFD0158 captures and checks migrated evaluation schedules", "[.][rfd0158-config]")
{
    constexpr const char* kOutputEnv = "ANET_RFD0158_OUTPUT";
    constexpr const char* kEvalTargetTag = "eval_target";
    constexpr const char* kEvalTag = "eval";
    constexpr const char* kGreedyTag = "greedy";
    constexpr const char* kGreedySingleTag = "greedy_single";
    wxString destination;
    REQUIRE(wxGetEnv(kOutputEnv, &destination));
    const auto output = std::filesystem::path(destination.ToStdWstring());
    std::filesystem::create_directories(output);
    const auto source = std::filesystem::current_path() / "apps/runner/config";
    for (const std::string env : {kAtariConfigName, kLunarLanderConfigName, kGridMazeConfigName, kDropMergeConfigName}) {
        CAPTURE(env);
        // 本体の選択を保ち、skip を持たない既定だけ明示的に重ねる。
        std::string chain, line;
        std::ifstream original(source / (env + ".txt"));
        const std::regex selection(R"(^run\.\$\s*=\s*([^#]+))");
        while (std::getline(original, line)) {
            std::smatch match;
            if (std::regex_search(line, match, selection)) chain = match[1].str();
        }
        if (env != kAtariConfigName && chain.find(kSkipRunProfile) == std::string::npos) {
            chain += (chain.empty() ? "" : " > ") + std::string(kSkipRunProfile);
        }
        REQUIRE_FALSE(chain.empty());
        const auto path = output / (env + ".txt");
        const auto includes = "$include <" + (source / "_main.txt").generic_string() + ">\n$include <"
            + (source / (env + ".txt")).generic_string() + ">\n";
        {
            std::ofstream file(path);
            file << includes << "run.$ = " << chain << '\n';
        }
        const auto data = anet::ConfigManager(path).GetConfigData();
        { std::ofstream file(output / (env + ".json")); file << data.ToJson().dump(2) << '\n'; }
        const auto check_schedule = [](const anet::ConfigData& config, const std::string& tag,
            int interval, int offset, int warmup_count) {
            const auto prefix = "run.eval_schedule.[" + tag + "].";
            CAPTURE(tag);
            CHECK(config.Get<int>(prefix + "interval") == interval);
            CHECK(config.Get<int>(prefix + "offset", 0) == offset);
            CHECK(config.Get<int>(prefix + "warmup_count", 1) == warmup_count);
        };
        if (env == kAtariConfigName) {
            check_schedule(data, kEvalTargetTag, 0, 500, 1);
            check_schedule(data, kEvalTag, 1954, 0, 1);
            check_schedule(data, kGreedyTag, 39453, 0, 1);
            check_schedule(data, kGreedySingleTag, 39453, 0, 1);
            {
                std::ofstream file(path);
                file << includes << "run.$ = " << chain << " > run.@evalonly\n";
            }
            const auto eval_only = anet::ConfigManager(path).GetConfigData();
            { std::ofstream file(output / "Atari-evalonly.json"); file << eval_only.ToJson().dump(2) << '\n'; }
            for (const auto* tag : {kEvalTargetTag, kEvalTag, kGreedyTag, kGreedySingleTag}) {
                check_schedule(eval_only, tag, 10, 0, 0);
            }
        } else {
            const int interval = env == kLunarLanderConfigName ? 1000 : env == kGridMazeConfigName ? 50 : 500;
            const int offset = env == kLunarLanderConfigName ? 50 : env == kGridMazeConfigName ? 3 : 25;
            check_schedule(data, kGreedyTag, interval, offset, 1);
        }
    }
}







TEST_CASE("RFD0103 resolves the actual validation input", "[.][rfd0103-resolve]")
{
    wxString input, output;
    REQUIRE(wxGetEnv("ANET_RFD0103_INPUT", &input));
    REQUIRE(wxGetEnv("ANET_RFD0103_OUTPUT", &output));
    const anet::ConfigManager manager(std::filesystem::path(input.ToStdWstring()));
    std::ofstream file(std::filesystem::path(output.ToStdWstring()));
    REQUIRE(file.good());
    file << manager.GetConfigData().ToJson().dump(2) << '\n';
    REQUIRE(file.good());
}

TEST_CASE("RFD0093 validation conditions survive the Atari Run chain", "[.][rfd0093-validation-config]")
{
    // 実際のPython採取器が生成した入力をConfigManagerの公開経路で解決する。
    wxString input;
    REQUIRE(wxGetEnv("ANET_RFD0093_VALIDATION_CONFIG", &input));
    const anet::ConfigManager manager(std::filesystem::path(input.ToStdWstring()));
    const auto data = manager.GetConfigData();
    CHECK(data.Get("app.run_name") == "run_{t}_rfd0093_contract_atari");
    CHECK(data.Get("app.batchrun.exp_exit_step") == "478528");
    CHECK(data.Get("app.exp_exit_step") == "478528");
    for (const auto& [key, value] : data.Map()) {
        if (key.starts_with("run.eval_schedule.[") && key.ends_with(".interval")) {
            CAPTURE(key);
            CHECK(value == "0");
        }
    }
}


static std::set<std::string> NetworkCatalogPrefixes(const anet::ConfigData& data)
{
    std::set<std::string> prefixes;
    for (const auto& [key, value] : data.Map()) {
        for (const std::string marker : {".block.[", ".config_profile.["}) {
            const auto pos = key.find(marker);
            if (pos == std::string::npos) continue;
            auto prefix = key.substr(0, pos);
            const auto branch = prefix.find(".branch.[");
            if (branch != std::string::npos) prefix.resize(branch);
            prefixes.insert(std::move(prefix));
        }
    }
    return prefixes;
}

static void ValidateCatalogProfile(const std::string& name, const anet::nn::ConfigProfileConfig& profile)
{
    anet::nn::NetworkConfig probe;
    probe.config_profiles.emplace(name, profile);
    anet::nn::NetworkBlockConfig block;
    block.type = "Linear";
    block.config_data.Set("linear.out_features", 1);
    block.config_data.Set("init.mode", "constant");
    block.config_data.Set("init.constant_val", "@" + name);
    probe.block_configs.emplace("ValidationLinear", std::move(block));
    (void)anet::nn::NetworkStructBuilder::Build(probe, "ValidationLinear",
        std::make_shared<anet::nn::ModuleRandomSource>(117));
}

static void ValidateUnusedCatalogs(const anet::ConfigData& data)
{
    anet::nn::InitNN();
    for (const auto& prefix : NetworkCatalogPrefixes(data)) {
        auto catalog_data = data;
        if (data.MakeSubConfigData(prefix + ".branch").empty()) {
            catalog_data.Set(prefix + ".branch.[ValidationBranch].bind", "obs");
            catalog_data.Set(prefix + ".branch.[ValidationBranch].structure", "");
        }
        const anet::nn::NetworkConfig config(catalog_data, prefix);
        for (const auto& [name, block] : config.block_configs) {
            const auto scope = block.config_data.CreateReadScope();
            // A9の仮値。補間profileの参照関係をblock単体の検査へ持ち込まない。
            auto parameters = scope;
            for (const auto& [key, value] : scope.Map()) {
                if (value.starts_with('@')) parameters.Set(key, "0");
            }
            const auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory(block.type);
            (void)factory->CreateModule(parameters,
                {.random_source = std::make_shared<anet::nn::ModuleRandomSource>(117)});
            parameters.RequireAllKeysRead("type=" + block.type);
        }
        for (const auto& [name, profile] : config.config_profiles) ValidateCatalogProfile(name, profile);
        for (const auto& [branch_name, branch] : config.branches) {
            for (const auto& [name, local] : branch.config_profiles) {
                auto effective = config.config_profiles.contains(name)
                    ? config.config_profiles.at(name) : anet::nn::ConfigProfileConfig{};
                effective.config_data.OverwriteFrom(local.config_data);
                ValidateCatalogProfile(name, effective);
            }
        }
    }
    std::set<std::string> forward_prefixes;
    for (const auto& [key, value] : data.Map()) {
        const auto pos = key.find(".nn_forward.[");
        if (pos != std::string::npos) forward_prefixes.insert(key.substr(0, pos) + ".nn_forward");
    }
    if (forward_prefixes.empty()) return;
    anet::ConfigData probe;
    probe.Set("net.block.[Linear].type", "Linear");
    probe.Set("net.block.[Linear].linear.out_features", 2);
    probe.Set("net.block.[Linear].noisy.enabled", false);
    probe.Set("net.branch.[features].bind", "obs");
    probe.Set("net.branch.[features].structure", "Linear");
    probe.Set("net.body.output.[features]", "features");
    const anet::TensorSpec spec{.type = anet::SpaceType::Vector, .shape = {2}, .dtype = torch::kFloat32};
    const auto network = anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(probe),
        anet::TensorSpecMap{{"obs", spec}}, nullptr, 117);
    for (const auto& prefix : forward_prefixes) {
        for (const auto& [name, entry] : data.MakeSubConfigData(prefix)) {
            const anet::nn::NetworkForwardConfig config(prefix + ".[" + name + "]", entry);
            (void)network->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows, 117);
        }
    }
}

TEST_CASE("Shipped configuration reads every unused NN catalog entry", "[rfd0117][shipped_config]")
{
    const auto config_root = std::filesystem::current_path() / "apps/runner/config";
    const auto output = std::filesystem::current_path() / "out/test-tmp/rfd0117-shipped-catalogs";
    std::filesystem::create_directories(output);
    for (const auto* env : {"Atari", "CartPole", "DropMerge", "GridMaze", "GridMaze_muzero", "ImageCls", "LunarLander"}) {
        CAPTURE(env);
        const auto path = output / (std::string(env) + ".txt");
        {
            std::ofstream file(path);
            REQUIRE(file.good());
            file << "$include <" << (config_root / "_main.txt").generic_string() << ">\n"
                << "$include <" << (config_root / (std::string(env) + ".txt")).generic_string() << ">\n";
        }
        CHECK_NOTHROW(ValidateUnusedCatalogs(anet::ConfigManager(path).GetConfigData()));
    }
}

TEST_CASE("Shipped catalog checker rejects unread fields in unused definitions", "[rfd0117][shipped_config]")
{
    anet::test::LogCaptureGuard logs;
    const std::string kind = GENERATE("block", "profile", "forward");
    anet::ConfigData data;
    std::string unknown;
    if (kind == "block") {
        data.Set("net.block.[unused].type", "Conv2d");
        data.Set("net.block.[unused].conv.out_channels", 4);
        data.Set("net.block.[unused].conv.kernel_size", 1);
        unknown = "net.block.[unused].out_channels";
    } else if (kind == "profile") {
        data.Set("net.config_profile.[unused].end", 0.2);
        unknown = "net.config_profile.[unused].typo";
    } else {
        data.Set("AnyAgent.nn_forward.[unused].noisy.mode", "mu_only");
        unknown = "AnyAgent.nn_forward.[unused].typo";
    }
    data.Set(unknown, 7);
    (void)data.Get(unknown);
    CHECK_THROWS_WITH(ValidateUnusedCatalogs(data), Catch::Matchers::ContainsSubstring(kind == "block" ? "out_channels" : "typo"));
}

TEST_CASE("Shipped catalog checker accepts branch-only profiles and dormant forward modes", "[rfd0117][shipped_config]")
{
    anet::ConfigData data;
    data.Set("net.block.[Drop].type", "Dropout");
    data.Set("net.block.[Drop].dropout_rate", "@only_branch");
    data.Set("net.branch.[features].bind", "obs");
    data.Set("net.branch.[features].structure", "Drop");
    data.Set("net.branch.[features].config_profile.[only_branch].end", 0.2);
    data.Set("AnyAgent.nn_forward.[unused].noisy.mode", "sample");
    CHECK_NOTHROW(ValidateUnusedCatalogs(data));
}
