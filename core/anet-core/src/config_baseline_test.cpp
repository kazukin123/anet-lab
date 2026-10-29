#include "anet/catch_test.hpp"
#include "anet/config.hpp"
#include "anet/default_dqn_agent.hpp"
#include "anet/image_cls_agent.hpp"
#include "anet/muzero_proto_agent.hpp"
#include "anet/rainbow_agent.hpp"

#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <wx/cmdline.h>
#include <wx/utils.h>

TEST_CASE("PRD104 defaults and auxiliary arms preserve metrics and propagate Actor limits", "[config][prd104]")
{
    const auto repo = std::filesystem::current_path();
    const auto config_root = (repo / "apps/runner/config").generic_string();
    const auto output = repo / "out/test-tmp/prd104-config";
    std::filesystem::create_directories(output);
    for (const std::string env : {"GridMaze", "LunarLander", "Atari", "DropMerge"}) {
        CAPTURE(env);
        const auto input = output / (env + ".txt");
        const auto includes = "$include <" + config_root + "/_main.txt>\n$include <" + config_root + "/" + env + ".txt>\n"
            + "$include <" + (repo / "core/anet-core/testdata/prd104/arms.txt").generic_string() + ">\n";
        { std::ofstream file(input); file << includes; }
        const auto defaults = anet::ConfigManager(input).GetConfigData();
        const anet::rl::dqn::DefaultDQNAgentConfig baseline(defaults);
        CHECK(baseline.learner.skip.target_window == "decision");
        for (const auto& [key, actor] : baseline.actor) {
            CAPTURE(key);
            CHECK(actor.skip.max_length_mode == "full");
            CHECK_FALSE(actor.skip.max_length.has_value());
        }
        CHECK_FALSE(defaults.Has("metrics.scalar.[50_agent_skip/07_max_length]"));
        std::string chain, line;
        std::ifstream source(config_root + "/" + env + ".txt");
        const std::regex selection(R"(^run\.\$\s*=\s*([^#]+))");
        while (std::getline(source, line)) {
            std::smatch match;
            if (std::regex_search(line, match, selection)) chain = match[1].str();
        }
        if (chain.find("run.@skip") == std::string::npos) chain += (chain.empty() ? "" : " > ") + std::string("run.@skip");
        { std::ofstream file(input); file << includes << "run.$ = " << chain << '\n'; }
        const auto before = anet::ConfigManager(input).GetConfigData();
        { std::ofstream file(input); file << includes << "run.$ = " << chain << " > run.@skip_aux\n"; }
        const auto after = anet::ConfigManager(input).GetConfigData();
        const anet::rl::dqn::DefaultDQNAgentConfig auxiliary(after);
        CHECK(auxiliary.learner.skip.target_window == "action_change");
        for (const auto& [key, actor] : auxiliary.actor) {
            CAPTURE(key);
            CHECK(actor.skip.max_length_mode == "single");
        }
        // 補助タスクの腕はmetricsの選択を書き換えず、既存の診断をすべて保つ。
        for (const auto& [key, value] : before.Map()) {
            if (key.starts_with("metrics.scalar.[")) CHECK(after.Get(key) == value);
        }
    }
}

TEST_CASE("PRD061 captures resolved and typed configuration", "[.][prd061-capture]")
{
    // 採取先を明示し、変更前の記録を後段の実行で上書きしない。
    wxString output_directory;
    REQUIRE(wxGetEnv("ANET_PRD061_OUTPUT", &output_directory));
    const auto root = std::filesystem::path(output_directory.ToStdWstring());
    const auto repo = std::filesystem::current_path();
    std::ifstream manifest_file(repo / "core/anet-core/testdata/prd072/manifest.json");
    REQUIRE(manifest_file.good());
    const auto manifest = anet::json::parse(manifest_file);
    std::filesystem::create_directories(root);
    for (const auto& input : manifest["inputs"]) {
        const auto id = input["id"].get<std::string>();
        INFO(id);
        const auto output_path = root / (id + ".json");
        REQUIRE_FALSE(std::filesystem::exists(output_path));
        const auto input_path = root / (id + ".txt");
        const auto config_path = (repo / "apps/runner/config").generic_string();
        {
            std::ofstream output(input_path, std::ios::binary);
            output << "$include <" << config_path << "/_main.txt>\n"
                << "$include <" << config_path << "/" << input["env"].get<std::string>() << ".txt>\n";
        }
        const wxCmdLineEntryDesc description[] = {
            { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
                wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },
            { wxCMD_LINE_NONE },
        };
        wxCmdLineParser command_line(description, wxString::FromUTF8(input["cli"].get<std::string>()));
        REQUIRE(command_line.Parse(false) == 0);
        const anet::ConfigManager manager(input_path, &command_line);
        const auto config_data = manager.GetConfigData();
        anet::json values = anet::json::object();
        for (const auto& [key, value] : config_data.Map()) values[key] = value;

        // Envや学習資源を生成せず、実際のConfig constructorで既定値・派生値を記録する。
        const auto agent_class = config_data.Get("agent.class_id");
        anet::json typed;
        if (agent_class == "DefaultDQNAgent") {
            typed = anet::rl::dqn::DefaultDQNAgentConfig(config_data).ToJson();
        } else if (agent_class == "ImageClsAgent") {
            typed = anet::rl::img_cls::ImageClsAgentConfig(config_data).ToJson();
        } else if (agent_class == "MuZeroAgent") {
            typed = anet::rl::muzero_proto::MuZeroAgentConfig(config_data).ToJson();
        } else if (agent_class == "RainbowAgent") {
            typed = anet::rl::dqn::RainbowAgentConfig(config_data).ToJson();
        } else {
            FAIL("Unsupported baseline Agent: " << agent_class);
        }
        const anet::json actual = {{"values", values}, {"typed", typed}, {"agent", agent_class}};
        std::ofstream output(output_path, std::ios::binary);
        REQUIRE(output.good());
        output << actual.dump(2) << '\n';
    }
}

TEST_CASE("PRD061 preserves Atari evaluation Run profiles", "[config][prd061][profiles]")
{
    const auto repo = std::filesystem::current_path();
    const auto root = repo / "out/test-tmp/prd061-profiles";
    std::filesystem::create_directories(root);
    const auto input_path = root / "main.txt";
    {
        std::ofstream output(input_path, std::ios::binary);
        output << "$include <" << (repo / "apps/runner/config/_main.txt").generic_string() << ">\n"
            << "$include <" << (repo / "apps/runner/config/Atari.txt").generic_string() << ">\n";
    }
    for (const auto profile : {"evalonly", "eval2only", "pl_check"}) {
        INFO(profile);
        const wxCmdLineEntryDesc description[] = {
            { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
                wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE }, {wxCMD_LINE_NONE},
        };
        wxCmdLineParser cli(description, wxString::FromUTF8(std::string("run.$=run.@") + profile));
        REQUIRE(cli.Parse(false) == 0);
        const auto config = anet::ConfigManager(input_path, &cli).GetConfigData();
        CHECK(config.Get("run.eval_schedule.[eval_target].interval") == (std::string(profile) == "evalonly" ? "10" : "0"));
        if (std::string(profile) == "evalonly") {
            CHECK(config.Get("run.eval_schedule.[eval].interval") == "11");
            CHECK(config.Get("run.eval.[eval_target].eval_batch_size") == "16");
            CHECK(config.Get("run.eval.[eval_target].eval_episodes") == "100");
            CHECK(config.Get("run.train.num_envs") == "1");
        } else if (std::string(profile) == "pl_check") {
            CHECK(config.Get("run.eval_schedule.[eval].interval") == "0");
        }
    }
}

TEST_CASE("PRD061 target Actor inherits final eval policy with explicit override priority", "[config][prd061][actor_inheritance]")
{
    // 現用のbaselineと環境別上書きを読み、CLIの最終値がtargetへ伝播することを確認する。
    const auto repo = std::filesystem::current_path();
    const auto root = repo / "out/test-tmp/prd061-actor-inheritance";
    std::filesystem::create_directories(root);
    const auto input_path = root / "main.txt";
    {
        std::ofstream output(input_path, std::ios::binary);
        output << "$include <" << (repo / "apps/runner/config/_main.txt").generic_string() << ">\n"
            << "$include <" << (repo / "apps/runner/config/Atari.txt").generic_string() << ">\n";
    }
    for (const bool override_target : {false, true}) {
        INFO("override_target=" << override_target);
        const wxCmdLineEntryDesc description[] = {
            { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
                wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE }, {wxCMD_LINE_NONE},
        };
        std::string arguments = "DefaultDQNAgent.actor.[eval].policy.epsilon=0.123";
        if (override_target) arguments += " DefaultDQNAgent.actor.[eval_target].policy.epsilon=0.234";
        wxCmdLineParser cli(description, wxString::FromUTF8(arguments));
        REQUIRE(cli.Parse(false) == 0);
        const auto config = anet::ConfigManager(input_path, &cli).GetConfigData();
        CHECK(config.Get("DefaultDQNAgent.actor.[eval].policy.epsilon") == "0.123");
        CHECK(config.Get("DefaultDQNAgent.actor.[eval_target].policy.epsilon") == (override_target ? "0.234" : "0.123"));
        // 右側のtarget差分と、Actor自身に指定した葉の優先順位も維持する。
        CHECK(config.Get("DefaultDQNAgent.actor.[eval_target].network") == "target");
        CHECK(config.Get("DefaultDQNAgent.actor.[eval].network") == "online");
    }
}

TEST_CASE("PRD061 shipped configurations resolve every Actor reference without panel overrides", "[config][prd061][shipped_actors]")
{
    const auto repo = std::filesystem::current_path();
    const auto root = repo / "out/test-tmp/prd061-shipped-actors";
    std::filesystem::create_directories(root);
    for (const auto env : {"Atari", "LunarLander", "GridMaze_muzero", "ImageCls", "DropMerge", "GridMaze", "CartPole"}) {
        // 環境が宣言したappチェーンを読み、P1などの後続選択を保持する。
        std::ifstream env_file(repo / "apps/runner/config" / (std::string(env) + ".txt"));
        REQUIRE(env_file.good());
        std::string app_chain;
        const std::regex declaration(R"(^\s*app\.\$\s*=\s*([^#]+))");
        for (std::string line; std::getline(env_file, line);) {
            std::smatch match;
            if (std::regex_search(line, match, declaration)) app_chain = match[1].str();
        }
        REQUIRE_FALSE(app_chain.empty());
        const std::regex online_term(R"((^|>\s*)app\.online(\s*(>|$)))");
        REQUIRE(std::regex_search(app_chain, online_term));
        const bool has_p1 = std::regex_search(app_chain, std::regex(R"((^|>)\s*P1\s*(>|$))"));
        for (const auto mode : {"online", "batchrun"}) {
            INFO("env=" << env << " app=" << mode);
            const auto input_path = root / "main.txt";
            {
                // 現用の組合せを解決し、EvalPanelタグ・device・Actorをテスト側で補正しない。
                std::ofstream output(input_path, std::ios::binary);
                output << "$include <" << (repo / "apps/runner/config/_main.txt").generic_string() << ">\n"
                    << "$include <" << (repo / "apps/runner/config").generic_string() << "/" << env << ".txt>\n"
                    << "P1.prd061_chain_probe = retained\n";
                if (std::string(mode) == "batchrun") {
                    output << "app.$ = " << std::regex_replace(app_chain, online_term, "$1app.batchrun$2") << '\n';
                }
            }
            const auto config = anet::ConfigManager(input_path).GetConfigData();
            if (has_p1) CHECK(config.Get("app.prd061_chain_probe", "") == "retained");
            const auto agent_class = config.Get("agent.class_id");
            anet::json typed;
            if (agent_class == "DefaultDQNAgent") typed = anet::rl::dqn::DefaultDQNAgentConfig(config).ToJson();
            else if (agent_class == "ImageClsAgent") typed = anet::rl::img_cls::ImageClsAgentConfig(config).ToJson();
            else if (agent_class == "MuZeroAgent") typed = anet::rl::muzero_proto::MuZeroAgentConfig(config).ToJson();
            else if (agent_class == "RainbowAgent") typed = anet::rl::dqn::RainbowAgentConfig(config).ToJson();
            else FAIL("Unexpected Agent: " << agent_class);

            const auto check_actor = [&](const std::string& key) {
                INFO("actor=" << key);
                CHECK(typed.contains("actor.[" + key + "].clone_model"));
            };
            check_actor(config.Get("run.train.actor", "train"));
            const auto slots = config.MakeSubConfigData("run.eval");
            // 休眠タグも、有効化時に使用する参照先が現用カタログに存在することを固定する。
            for (const auto& [tag, slot] : slots) check_actor(slot.Get("actor", tag.c_str()));
            const auto panel_tag = config.Get("app.eval_panel.eval_config_tag");
            REQUIRE(slots.contains(panel_tag));
            const auto panel_actor = slots.at(panel_tag).Get("actor", panel_tag.c_str());
            check_actor(panel_actor);
            // 共通EvalPanelは専用Actorを使い、定期評価の探索設定を引き継がない。
            CHECK(panel_tag == "eval_panel");
            CHECK(panel_actor == "eval_panel");
            if (agent_class == "DefaultDQNAgent") {
                CHECK(typed.value("actor.[eval_panel].policy.policy_type", "") == "Greedy");
                CHECK(typed.value("actor.[eval_panel].network", "") == "target");
            } else if (agent_class == "MuZeroAgent") {
                CHECK(typed.value("actor.[eval_panel].temp_start", -1.0) == 0.0);
                CHECK(typed.value("actor.[eval_panel].temp_end", -1.0) == 0.0);
                CHECK_FALSE(typed.value("actor.[eval_panel].add_exploration_noise", true));
            }
            if (agent_class == "MuZeroAgent") {
                CHECK(config.Get("agent.device") == "cpu");
                CHECK(config.Get("run.eval_device") == "cpu");
                CHECK(config.Get("metrics.scalar.[32_agent_base/06_tau]").find("$actor tau") != std::string::npos);
            }
        }
    }
    // 共通fullプリセットも選択後の実効metricを検証する。
    const auto full_path = root / "full.txt";
    {
        std::ofstream output(full_path, std::ios::binary);
        output << "$include <" << (repo / "apps/runner/config/_main.txt").generic_string() << ">\n"
            << "metrics.scalar.$ = metrics.scalar.@full\n";
    }
    const auto full = anet::ConfigManager(full_path).GetConfigData();
    CHECK(full.Get("metrics.scalar.[32_agent_base/06_tau]").find("$actor tau") != std::string::npos);
    // 現用7環境で選ばれないRainbowも共通カタログのGreedy既定を検証する。
    const auto rainbow = anet::rl::dqn::RainbowAgentConfig(full).ToJson();
    CHECK(rainbow.at("actor.[eval_panel].network") == "target");
    CHECK(rainbow.at("actor.[eval_panel].policy.epsilon") == 0.0);
    CHECK_FALSE(rainbow.contains("actor.[eval_panel].policy.epsilon.end"));
    CHECK_FALSE(rainbow.contains("actor.[eval_panel].policy.epsilon.steps"));

}

TEST_CASE("ConfigManager compares frozen PRD072 configuration inputs", "[.][prd072-baseline]")
{
    // 入力定義とgoldenを版管理し、固定設定の展開と比較結果だけをローカルへ置く。
    const auto repo = std::filesystem::current_path();
    wxString prepared_root;
    constexpr const char* kPrd072RootEnv = "ANET_PRD072_ROOT";
    REQUIRE(wxGetEnv(kPrd072RootEnv, &prepared_root));
    const auto root = std::filesystem::path(prepared_root.ToStdWstring());
    std::ifstream manifest_file(repo / "core/anet-core/testdata/prd072/manifest.json");
    REQUIRE(manifest_file.good());
    const auto manifest = anet::json::parse(manifest_file);
    std::ifstream prepared_file(root / "manifest.json");
    REQUIRE(prepared_file.good());
    REQUIRE(anet::json::parse(prepared_file) == manifest);
    const bool capture = wxGetEnv("ANET_PRD072_CAPTURE", nullptr);
    for (const auto& input : manifest["inputs"]) {
        const auto id = input["id"].get<std::string>();
        INFO(id);
        const auto baseline_path = repo / "core/anet-core/testdata/prd072/baseline" / (id + ".json");
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
            std::ifstream expected_file(repo / "core/anet-core/testdata/prd072/resolution" / (id + ".json"));
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

TEST_CASE("Actor catalog validates networks and MuZero temperature and clone contracts", "[prd061][config]")
{
    anet::ConfigData invalid;
    invalid.Set("DefaultDQNAgent.actor.[probe].network", "missing");
    CHECK_THROWS(anet::rl::dqn::DefaultDQNAgentConfig(invalid));
    invalid.Set("RainbowAgent.actor.[probe].network", "missing");
    CHECK_THROWS(anet::rl::dqn::RainbowAgentConfig(invalid));
    anet::ConfigData data;
    data.Set("MuZeroAgent.actor.[eval].temp_start", 0.0f);
    data.Set("MuZeroAgent.actor.[eval].temp_end", 0.0f);
    data.Set("MuZeroAgent.actor.[eval].temp_decay_steps", 0);
    data.Set("MuZeroAgent.actor.[eval].add_exploration_noise", false);
    const anet::rl::muzero_proto::MuZeroAgentConfig config(data);
    CHECK(config.actor.at("eval").temp_start == 0.0f);
    CHECK_FALSE(config.actor.at("eval").add_exploration_noise);
    CHECK_FALSE(config.actor.at("eval").clone_model);
    CHECK(config.GetConfigData().Get("actor.[eval].clone_model") == "false");
    data.Set("MuZeroAgent.actor.[eval].temp_end", -1.0f);
    CHECK_THROWS(anet::rl::muzero_proto::MuZeroAgentConfig(data));
    data.Set("MuZeroAgent.actor.[eval].temp_end", 0.0f);
    data.Set("MuZeroAgent.actor.[eval].clone_model", true);
    CHECK_THROWS_WITH(anet::rl::muzero_proto::MuZeroAgentConfig(data), Catch::Matchers::ContainsSubstring("clone_model=true is unsupported"));
}

TEST_CASE("PRD093 snapshots current configuration inputs", "[.][prd093-config-snapshot]")
{
    // 現用設定を全キー・文字列値で採取し、過去の固定入力には触れない。
    wxString directory;
    REQUIRE(wxGetEnv("ANET_PRD093_OUTPUT", &directory));
    const auto output = std::filesystem::path(directory.ToStdWstring());
    std::filesystem::create_directories(output);
    wxString config_root;
    const auto config = wxGetEnv("ANET_PRD093_CONFIG_ROOT", &config_root)
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

TEST_CASE("PRD103 skip profiles expose greedy session and padded metric tags", "[.][prd103-config]")
{
    // 実際の設定を公開resolverに通し、変更前後の全キーを同時に保存する。
    wxString destination;
    REQUIRE(wxGetEnv("ANET_PRD103_OUTPUT", &destination));
    const auto output = std::filesystem::path(destination.ToStdWstring());
    std::filesystem::create_directories(output);
    const auto config = (std::filesystem::current_path() / "apps/runner/config").generic_string();
    for (const std::string env : {"GridMaze", "LunarLander", "Atari", "DropMerge"}) {
        CAPTURE(env);
        const auto path = output / (env + ".txt");
        const auto includes = "$include <" + config + "/_main.txt>\n$include <" + config + "/" + env + ".txt>\n";
        { std::ofstream file(path); file << includes; }
        const auto defaults = anet::ConfigManager(path).GetConfigData();
        { std::ofstream file(output / (env + "-default.json")); file << defaults.ToJson().dump(2) << '\n'; }
        // 選択宣言は解決済みMapには残らないため、元ファイルの有効な連鎖を読む。
        std::string chain, line;
        std::ifstream source(config + "/" + env + ".txt");
        const std::regex run_selection(R"(^run\.\$\s*=\s*([^#]+))");
        while (std::getline(source, line)) {
            std::smatch match;
            if (std::regex_search(line, match, run_selection)) chain = match[1].str();
        }
        if (chain.find("run.@skip") == std::string::npos) chain += (chain.empty() ? "" : " > ") + std::string("run.@skip");
        { std::ofstream file(path); file << includes << "run.$ = " << chain << '\n'; }
        const auto data = anet::ConfigManager(path).GetConfigData();
        { std::ofstream file(output / (env + "-skip.json")); file << data.ToJson().dump(2) << '\n'; }
        CHECK(data.Get("DefaultDQNAgent.learner.skip.sample_weight") == "uniform");
        CHECK(data.Get("DefaultDQNAgent.actor.[greedy].policy.policy_type")
            == (env == "Atari" ? defaults.Get("DefaultDQNAgent.actor.[greedy].policy.policy_type") : "Greedy"));
        CHECK(data.Get("run.eval.[greedy].actor") == "greedy");
        CHECK(data.Get("DefaultDQNAgent.actor.[greedy].network") == (env == "DropMerge" ? "target" : "online"));
        for (const auto* tag : {"17_skip_primary_grad_norm", "18_skip_primary_clip_ratio", "19_primary_clip_ratio_ema", "21_greedy_decision_rate", "22_greedy_length_mean"}) {
            CHECK_FALSE(data.Get(std::string("metrics.scalar.[50_agent_skip/") + tag + "]").empty());
        }
        const auto limit = data.Get<int>("DefaultDQNAgent.skip.max_length");
        for (int j = 1; j <= limit; ++j) {
            const auto tag_index = (j < 10 ? "0" : "") + std::to_string(j);
            const auto source_index = std::to_string(j);
            CHECK(data.Get("metrics.scalar.[50_agent_skip/03_skip_length_rate.[" + tag_index + "]]")
                == "skip_length_rate.[" + source_index + "] @train $action_info");
            CHECK(data.Get("metrics.scalar.[50_agent_skip/23_greedy_length_rate.[" + tag_index + "]]")
                == "skip_session_length_rate.[" + source_index + "] $eval.[greedy] $actor @session_end");
        }
        if (env != "Atari") {
            CHECK(data.Get("run.eval.[greedy].eval_batch_size") == "1");
            CHECK(data.Get("run.eval.[greedy].eval_episodes") == "1");
            CHECK(data.Get("run.eval_schedule.[greedy].interval") == (env == "GridMaze" ? "53" : env == "LunarLander" ? "1009" : "503"));
            CHECK(data.Get("run.eval_schedule.[greedy].use_background") == "true");
            for (const auto* tag : {"09_greedy_reward", "10_greedy_reward_ema", "11_greedy_ep_steps", "12_greedy_ep_steps_max"}) {
                CHECK_FALSE(data.Get(std::string("metrics.scalar.[21_eval/") + tag + "]").empty());
            }
        }
        if (env == "GridMaze") {
            CHECK(data.Get("DefaultDQNAgent.actor.[train].skip.policy.policy_type") == "EpsilonGreedy");
            CHECK(data.Get<float>("DefaultDQNAgent.actor.[train].skip.policy.epsilon.start") == 0.60f);
            CHECK(data.Get<float>("DefaultDQNAgent.actor.[train].skip.policy.epsilon.end") == 0.05f);
        }
        for (const auto& [key, value] : data.Map()) {
            CHECK(key.find("greedy_dist") == std::string::npos);
            CHECK(value.find("$eval.[greedy_dist]") == std::string::npos);
        }
    }
}

TEST_CASE("Final feature branch is a shared learner configuration", "[config][prd093][rename]")
{
    // 特定診断に依存しない公開設定名で境界を指定する。
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.nn_forward.[mean].noisy.mode", "mu_only");
    data.Set("DefaultDQNAgent.actor.[train].nn_forward_key", "mean");
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "mean");
    data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "mean");
    // baseline未適用のDTOは未指定を保持し、購読時の必須検証へ渡す。
    CHECK(anet::rl::dqn::DefaultDQNAgentConfig(data).learner.final_feature_branch_key.empty());
    CHECK(anet::rl::img_cls::ImageClsAgentConfig(data).final_feature_branch_key.empty());
    data.Set("DefaultDQNAgent.learner.final_feature_branch_key", "boundary");
    const anet::rl::dqn::DefaultDQNAgentConfig config(data);
    CHECK(config.learner.final_feature_branch_key == "boundary");
}

TEST_CASE("Attention probe batch size validates even without a subscription", "[config][attention][prd093]")
{
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.nn_forward.[mean].noisy.mode", "mu_only");
    data.Set("DefaultDQNAgent.actor.[train].nn_forward_key", "mean");
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "mean");
    data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "mean");
    for (const int size : { 0, -1 }) {
        data.Set("DefaultDQNAgent.learner.attention_entropy.probe.batch_size", size);
        CHECK_THROWS_WITH(anet::rl::dqn::DefaultDQNAgentConfig(data),
            Catch::Matchers::ContainsSubstring("learner.attention_entropy.probe.batch_size"));
    }
    data.Set("DefaultDQNAgent.learner.attention_entropy.probe.batch_size", 256);
    CHECK(anet::rl::dqn::DefaultDQNAgentConfig(data).learner.attention_entropy.probe.batch_size == 256);
    data.Set("ImageClsAgent.final_feature_branch_key", "boundary");
    CHECK(anet::rl::img_cls::ImageClsAgentConfig(data).final_feature_branch_key == "boundary");
}

TEST_CASE("Attention Run profiles preserve existing metric definitions", "[config][attention][prd093][profiles]")
{
    // 現用の既定Runチェーンへ末尾追加し、GPU資源を作らず最終metrics定義を検証する。
    const auto root = std::filesystem::current_path();
    const auto directory = root / "out/test-tmp/prd093-profiles";
    std::filesystem::create_directories(directory);
    for (const auto& env : { "DropMerge", "Atari" }) {
        const auto input = directory / (std::string(env) + ".txt");
        const auto config = (root / "apps/runner/config").generic_string();
        const auto write = [&](const std::string& chain) {
            std::ofstream file(input);
            file << "$include <" << config << "/_main.txt>\n$include <" << config << "/" << env << ".txt>\n";
            if (!chain.empty()) file << "run.$ = " << chain << '\n';
        };
        write("");
        const anet::ConfigManager before_manager(input);
        const auto before = before_manager.GetConfigData();
        std::ifstream source(root / "apps/runner/config" / (std::string(env) + ".txt"));
        std::string line, chain;
        const std::regex selection(R"(^\s*run\.\$\s*=\s*([^#]*))");
        while (std::getline(source, line)) {
            std::smatch match;
            if (std::regex_search(line, match, selection)) chain = match[1].str();
        }
        chain += (chain.empty() ? "" : " > ") + std::string("run.@attention");
        write(chain);
        const anet::ConfigManager after_manager(input);
        const auto after = after_manager.GetConfigData();
        for (const auto& [key, value] : before.Map()) {
            if (!key.starts_with("metrics.scalar.[")) continue;
            CAPTURE(env, key);
            CHECK(after.Get(key) == value);
        }
        int attention_rows = 0;
        for (const auto& [key, value] : after.Map()) {
            if (!key.starts_with("metrics.scalar.[48_agent_attention/")) continue;
            ++attention_rows;
            CHECK(value.find("@learn $learn_step $update_result interval:503") != std::string::npos);
        }
        CHECK(attention_rows == 4);
        CHECK(anet::rl::dqn::DefaultDQNAgentConfig(after).learner.attention_entropy.probe.batch_size == 256);
        CHECK(anet::rl::dqn::DefaultDQNAgentConfig(after).learner.final_feature_branch_key == "main_feature");
    }
}

TEST_CASE("PRD103 resolves the actual validation input", "[.][prd103-resolve]")
{
    wxString input, output;
    REQUIRE(wxGetEnv("ANET_PRD103_INPUT", &input));
    REQUIRE(wxGetEnv("ANET_PRD103_OUTPUT", &output));
    const anet::ConfigManager manager(std::filesystem::path(input.ToStdWstring()));
    std::ofstream file(std::filesystem::path(output.ToStdWstring()));
    REQUIRE(file.good());
    file << manager.GetConfigData().ToJson().dump(2) << '\n';
    REQUIRE(file.good());
}

TEST_CASE("PRD093 validation conditions survive the Atari Run chain", "[.][prd093-validation-config]")
{
    // 実際のPython採取器が生成した入力をConfigManagerの公開経路で解決する。
    wxString input;
    REQUIRE(wxGetEnv("ANET_PRD093_VALIDATION_CONFIG", &input));
    const anet::ConfigManager manager(std::filesystem::path(input.ToStdWstring()));
    const auto data = manager.GetConfigData();
    CHECK(data.Get("app.run_name") == "run_{t}_prd093_contract_atari");
    CHECK(data.Get("app.batchrun.exp_exit_step") == "478528");
    CHECK(data.Get("app.exp_exit_step") == "478528");
    for (const auto& [key, value] : data.Map()) {
        if (key.starts_with("run.eval_schedule.[") && key.ends_with(".interval")) {
            CAPTURE(key);
            CHECK(value == "0");
        }
    }
}
