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
        std::string arguments = "DefaultDQNAgent.actor.[eval].policy.eps_start=0.123";
        if (override_target) arguments += " DefaultDQNAgent.actor.[eval_target].policy.eps_start=0.234";
        wxCmdLineParser cli(description, wxString::FromUTF8(arguments));
        REQUIRE(cli.Parse(false) == 0);
        const auto config = anet::ConfigManager(input_path, &cli).GetConfigData();
        CHECK(config.Get("DefaultDQNAgent.actor.[eval].policy.eps_start") == "0.123");
        CHECK(config.Get("DefaultDQNAgent.actor.[eval_target].policy.eps_start") == (override_target ? "0.234" : "0.123"));
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
    CHECK(rainbow.at("actor.[eval_panel].policy.eps_start") == 0.0);
    CHECK(rainbow.at("actor.[eval_panel].policy.eps_end") == 0.0);
    CHECK(rainbow.at("actor.[eval_panel].policy.eps_decay_steps") == 0);

}

TEST_CASE("ConfigManager compares frozen PRD072 configuration inputs", "[.][prd072-baseline]")
{
    // 入力定義とgoldenを版管理し、固定設定の展開と比較結果だけをローカルへ置く。
    const auto repo = std::filesystem::current_path();
    const auto root = repo / ".scratch" / "prd072-differential" / "validation";
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
