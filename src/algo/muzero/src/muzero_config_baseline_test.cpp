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
#include "anet/algo/muzero.hpp"
#include "anet/observers.hpp"
#include "anet/test/util.hpp"
#include "anet/test/nn.hpp"
#include "muzero_proto_agent.hpp"


TEST_CASE("PRD061 captures resolved and typed configuration", "[.][prd061-capture]")
{
    // 採取先を明示し、変更前の記録を後段の実行で上書きしない。
    wxString output_directory;
    REQUIRE(wxGetEnv("ANET_PRD061_OUTPUT", &output_directory));
    const auto root = std::filesystem::path(output_directory.ToStdWstring());
    const auto repo = std::filesystem::current_path();
    std::ifstream manifest_file(repo / "src/core/base/testdata/prd072/manifest.json");
    REQUIRE(manifest_file.good());
    const auto manifest = anet::json::parse(manifest_file);
    std::filesystem::create_directories(root);
    for (const auto& input : manifest["inputs"]) {
        const auto id = input["id"].get<std::string>();
        if (input["env"].get<std::string>() != "GridMaze_muzero") continue;
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
        typed = anet::rl::muzero_proto::MuZeroAgentConfig(config_data).ToJson();
        const anet::json actual = {{"values", values}, {"typed", typed}, {"agent", agent_class}};
        std::ofstream output(output_path, std::ios::binary);
        REQUIRE(output.good());
        output << actual.dump(2) << '\n';
    }
}

TEST_CASE("MuZero shipped and explicit configuration covers Actor and Network visualization settings", "[config][prd061][shipped_actors][shipped_config][prd117][nn][dot]")
{
    SECTION("The public initializer registers its Agent factories") {
        anet::rl::algo::InitMuZero();
        const auto muZeroAgentFactory = anet::rl::AgentRepository::Instance().GetAgentFactory("MuZeroAgent");
        REQUIRE(muZeroAgentFactory);
        CHECK(muZeroAgentFactory->GetTargetAgentClassId() == "MuZeroAgent");
    }
    SECTION("PRD061 shipped configurations resolve every Actor reference without panel overrides") {
        const auto repo = std::filesystem::current_path();
        const auto root = repo / "out/test-tmp/prd061-shipped-actors-muzero";
        std::filesystem::create_directories(root);
        for (const auto env : {"GridMaze_muzero"}) {
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
                typed = anet::rl::muzero_proto::MuZeroAgentConfig(config).ToJson();

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
    }
    SECTION("MuZero config validates Actor and Network visualization settings") {
        SECTION("Agent Actor catalogs reject unread keys independently of prior reads") {
            const std::string owner = "MuZeroAgent";
            CAPTURE(owner);
            anet::ConfigData data;
            const auto prefix = owner + ".actor.[custom]";
            data.Set(prefix + ".clone_model", false);
            const auto read = [&] { const anet::rl::muzero_proto::MuZeroAgentConfig config(data); };
            CHECK_NOTHROW(read());
            data.Set(prefix + ".typo", 9);
            CHECK(data.Get<int>(prefix + ".typo") == 9);
            CHECK_THROWS_WITH(read(), Catch::Matchers::ContainsSubstring(prefix)
                && Catch::Matchers::ContainsSubstring("typo = 9"));
        }
        SECTION("MuZero Actor catalog validates temperature and clone contracts") {
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
        SECTION("MuZero config reads nn_viz keys and ignores old keys") {
            anet::ConfigData muzero_data;
            muzero_data.Set("MuZeroAgent.nn_viz.show_tensor_specs", "true");
            muzero_data.Set("MuZeroAgent.nn_viz.show_branch_config", "true");
            muzero_data.Set("MuZeroAgent.nn_viz.show_head_info", "true");
            muzero_data.Set("MuZeroAgent.nn_dot.show_param_count", "true");
            anet::rl::muzero_proto::MuZeroAgentConfig muzero_config(muzero_data);
            CHECK(muzero_config.nn_viz.show_tensor_specs);
            CHECK(muzero_config.nn_viz.show_branch_config);
            CHECK(muzero_config.nn_viz.show_head_info);
            CHECK_FALSE(muzero_config.nn_viz.show_param_count);
        }
    }
}
