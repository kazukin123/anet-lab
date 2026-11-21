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
#include "anet/algo/dqn.hpp"
#include "anet/observers.hpp"
#include "anet/test/util.hpp"
#include "anet/test/nn.hpp"
#include "default_dqn_agent.hpp"
#include "rainbow_agent.hpp"


constexpr const char* kAtariConfigName = "Atari";
constexpr const char* kLunarLanderConfigName = "LunarLander";
constexpr const char* kGridMazeConfigName = "GridMaze";
constexpr const char* kDropMergeConfigName = "DropMerge";
constexpr const char* kSkipRunProfile = "run.@skip";

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
        const auto env = input["env"].get<std::string>();
        if (env == "ImageCls" || env == "GridMaze_muzero") continue;
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
        if (agent_class == "DefaultDQNAgent") typed = anet::rl::dqn::DefaultDQNAgentConfig(config_data).ToJson();
        else if (agent_class == "RainbowAgent") typed = anet::rl::dqn::RainbowAgentConfig(config_data).ToJson();
        else FAIL("Unexpected DQN Agent: " << agent_class);
        const anet::json actual = {{"values", values}, {"typed", typed}, {"agent", agent_class}};
        std::ofstream output(output_path, std::ios::binary);
        REQUIRE(output.good());
        output << actual.dump(2) << '\n';
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
    for (const std::string env : {kGridMazeConfigName, kLunarLanderConfigName, kAtariConfigName, kDropMergeConfigName}) {
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
        if (chain.find(kSkipRunProfile) == std::string::npos) chain += (chain.empty() ? "" : " > ") + std::string(kSkipRunProfile);
        { std::ofstream file(path); file << includes << "run.$ = " << chain << '\n'; }
        const auto data = anet::ConfigManager(path).GetConfigData();
        { std::ofstream file(output / (env + "-skip.json")); file << data.ToJson().dump(2) << '\n'; }
        CHECK(data.Get("DefaultDQNAgent.learner.skip.sample_weight") == "uniform");
        CHECK(data.Get("DefaultDQNAgent.actor.[greedy].policy.policy_type")
            == (env == kAtariConfigName ? defaults.Get("DefaultDQNAgent.actor.[greedy].policy.policy_type") : "Greedy"));
        CHECK(data.Get("run.eval.[greedy].actor") == "greedy");
        CHECK(data.Get("DefaultDQNAgent.actor.[greedy].network") == (env == kDropMergeConfigName ? "target" : "online"));
        for (const auto* tag : {"17_skip_primary_grad_norm", "18_skip_primary_clip_ratio", "19_skip_primary_clip_ratio_ema", "21_greedy_decision_rate", "22_greedy_length_mean"}) {
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
        if (env != kAtariConfigName) {
            CHECK(data.Get("run.eval.[greedy].eval_batch_size") == "1");
            CHECK(data.Get("run.eval.[greedy].eval_episodes") == "1");
            CHECK(data.Get("run.eval_schedule.[greedy].interval") == (env == kGridMazeConfigName ? "50" : env == kLunarLanderConfigName ? "1000" : "500"));
            CHECK(data.Get("run.eval_schedule.[greedy].offset") == (env == kGridMazeConfigName ? "3" : env == kLunarLanderConfigName ? "50" : "25"));
            CHECK(data.Get("run.eval_schedule.[greedy].use_background") == "true");
            for (const auto* tag : {"09_greedy_reward", "10_greedy_reward_ema", "11_greedy_ep_steps", "12_greedy_ep_steps_max"}) {
                CHECK_FALSE(data.Get(std::string("metrics.scalar.[21_eval/") + tag + "]").empty());
            }
        }
        if (env == kGridMazeConfigName) {
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


TEST_CASE("Value decomposition ships inactive defaults and optional metrics", "[prd121][shipped_config]")
{
    const auto repo = std::filesystem::current_path();
    const auto directory = repo / "out/test-tmp/prd121-config";
    std::filesystem::create_directories(directory);
    const auto path = directory / "input.txt";
    const auto write = [&](bool value_metrics) {
        std::ofstream file(path);
        file << "$include <" << (repo / "apps/runner/config/_main.txt").generic_string() << ">\n";
        file << "$include <" << (repo / "apps/runner/config/GridMaze.txt").generic_string() << ">\n";
        if (value_metrics) file << "run.$ = run.@rnd\nDefaultDQNAgent.intrinsic.value_mode = decomposed\n"
            << "metrics.scalar.$ = metrics.scalar.@baseline > M1 > metrics.scalar.@value_decomposition\n";
    };
    write(false);
    const auto baseline = anet::ConfigManager(path).GetConfigData();
    REQUIRE(baseline.Has("DefaultDQNAgent.intrinsic.value_mode"));
    CHECK(baseline.Get<std::string>("DefaultDQNAgent.intrinsic.value_mode") == "combined");
    REQUIRE(baseline.Has("DefaultDQNAgent.learner.intrinsic.feature_gradient"));
    CHECK_FALSE(baseline.Has("DefaultDQNAgent.learner.intrinsic.gamma"));
    for (const auto& [key, value] : baseline.Map()) CHECK_FALSE(key.starts_with("metrics.scalar.[77_agent_value/"));
    write(true);
    const auto selected = anet::ConfigManager(path).GetConfigData();
    CHECK(anet::rl::dqn::DefaultDQNAgentConfig(selected).learner.intrinsic.decomposed);
    int rows = 0;
    std::set<std::string> keys;
    for (const auto& [key, value] : selected.Map()) {
        if (!key.starts_with("metrics.scalar.[77_agent_value/")) continue;
        ++rows;
        const auto start = key.find('[') + 1;
        CHECK(key.size() - start - 1 <= 40);
        keys.insert(value.substr(0, value.find(' ')));
    }
    const std::set<std::string> expected{
        "extrinsic_loss", "intrinsic_loss", "extrinsic_q_sa_mean", "intrinsic_q_sa_mean", "intrinsic_q_share",
        "intrinsic.reward_scaler.scale", "extrinsic_head_grad_norm", "intrinsic_head_grad_norm", "extrinsic_grad_norm", "intrinsic_grad_norm",
        "extrinsic_q_argmax_disagreement_rate", "extrinsic_q_regret_mean", "extrinsic_q_gap_mean", "intrinsic_q_gap_mean",
        "extrinsic_q_session_argmax_disagreement_rate", "extrinsic_q_session_regret_mean",
        "extrinsic_q_session_gap_mean", "intrinsic_q_session_gap_mean"};
    CHECK(keys == expected);
    CHECK(rows == 22);
    for (const auto& [key, value] : baseline.Map()) {
        if (key.starts_with("metrics.scalar.[")) CHECK(selected.Get(key) == value);
    }
    CHECK_NOTHROW(anet::rl::ObserverFactory(selected));
}

TEST_CASE("DQN shipped profiles provide default values, actors and metrics", "[prd121][shipped_config][config][prd104][prd061][actor_inheritance][shipped_actors][attention][prd093][profiles]")
{
    SECTION("PRD104 defaults and auxiliary arms preserve metrics and propagate Actor limits") {
        const auto repo = std::filesystem::current_path();
        const auto config_root = (repo / "apps/runner/config").generic_string();
        const auto output = repo / "out/test-tmp/prd104-config";
        std::filesystem::create_directories(output);
        // 既定は env の設定を含めない共通の設定（agent.txt）で確かめる。env の設定が Actor ごとに選んだ値はここでは見ない。
        {
            const auto input = output / "main_only.txt";
            { std::ofstream file(input); file << "$include <" << config_root << "/_main.txt>\n"; }
            const anet::rl::dqn::DefaultDQNAgentConfig baseline(anet::ConfigManager(input).GetConfigData());
            CHECK(baseline.learner.skip.target_window == "decision");
            REQUIRE_FALSE(baseline.actor.empty());
            for (const auto& [key, actor] : baseline.actor) {
                CAPTURE(key);
                CHECK(actor.skip.max_length_mode == "full");
                CHECK_FALSE(actor.skip.max_length.has_value());
            }
        }
        for (const std::string env : {"GridMaze", "LunarLander", "Atari", "DropMerge"}) {
            CAPTURE(env);
            const auto input = output / (env + ".txt");
            const auto includes = "$include <" + config_root + "/_main.txt>\n$include <" + config_root + "/" + env + ".txt>\n"
                + "$include <" + (repo / "src/core/base/testdata/prd104/arms.txt").generic_string() + ">\n";
            { std::ofstream file(input); file << includes; }
            const auto defaults = anet::ConfigManager(input).GetConfigData();
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
    SECTION("PRD061 target Actor inherits final eval policy with explicit override priority") {
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
            // 右側のtarget差分と、Actor自身に指定した設定キーの優先順位も維持する。
            CHECK(config.Get("DefaultDQNAgent.actor.[eval_target].network") == "target");
            CHECK(config.Get("DefaultDQNAgent.actor.[eval].network") == "online");
        }
    }
    SECTION("PRD061 shipped configurations resolve every Actor reference without panel overrides") {
        const auto repo = std::filesystem::current_path();
        const auto root = repo / "out/test-tmp/prd061-shipped-actors-dqn";
        std::filesystem::create_directories(root);
        for (const auto env : {"Atari", "LunarLander", "DropMerge", "GridMaze", "CartPole"}) {
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
                else if (agent_class == "RainbowAgent") typed = anet::rl::dqn::RainbowAgentConfig(config).ToJson();
                else FAIL("Unexpected DQN Agent: " << agent_class);

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
    SECTION("Attention Run profiles preserve existing metric definitions") {
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
}

TEST_CASE("DQN Actor catalogs validate fields and network choices", "[config][prd117][prd061]")
{
    SECTION("The public initializer registers its Agent factories") {
        anet::rl::algo::InitDqn();
        const auto defaultDQNAgentFactory = anet::rl::AgentRepository::Instance().GetAgentFactory("DefaultDQNAgent");
        REQUIRE(defaultDQNAgentFactory);
        CHECK(defaultDQNAgentFactory->GetTargetAgentClassId() == "DefaultDQNAgent");
        const auto rainbowAgentFactory = anet::rl::AgentRepository::Instance().GetAgentFactory("RainbowAgent");
        REQUIRE(rainbowAgentFactory);
        CHECK(rainbowAgentFactory->GetTargetAgentClassId() == "RainbowAgent");
    }
    SECTION("Agent Actor catalogs reject unread keys independently of prior reads") {
        const std::string owner = GENERATE("DefaultDQNAgent", "RainbowAgent");
        CAPTURE(owner);
        anet::ConfigData data;
        const auto prefix = owner + ".actor.[custom]";
        data.Set(prefix + ".clone_model", false);
        if (owner == "DefaultDQNAgent") {
            data.Set(owner + ".nn_forward.[test].noisy.mode", "mu_only");
            data.Set(prefix + ".nn_forward_key", "test");
            data.Set(owner + ".learner.current.nn_forward_key", "test");
            data.Set(owner + ".learner.target.nn_forward_key", "test");
        }
        const auto read = [&] {
            if (owner == "DefaultDQNAgent") { const anet::rl::dqn::DefaultDQNAgentConfig config(data); }
            else { const anet::rl::dqn::RainbowAgentConfig config(data); }
        };
        CHECK_NOTHROW(read());
        data.Set(prefix + ".typo", 9);
        CHECK(data.Get<int>(prefix + ".typo") == 9);
        CHECK_THROWS_WITH(read(), Catch::Matchers::ContainsSubstring(prefix)
            && Catch::Matchers::ContainsSubstring("typo = 9"));
    }
    SECTION("Actor catalog validates networks") {
        anet::ConfigData invalid;
        invalid.Set("DefaultDQNAgent.actor.[probe].network", "missing");
        CHECK_THROWS(anet::rl::dqn::DefaultDQNAgentConfig(invalid));
        invalid.Set("RainbowAgent.actor.[probe].network", "missing");
        CHECK_THROWS(anet::rl::dqn::RainbowAgentConfig(invalid));
    }
}

TEST_CASE("DQN feature diagnostics validate their learner settings", "[config][prd093][rename][attention]")
{
    SECTION("Final feature branch is a shared learner configuration") {
        // 特定診断に依存しない公開設定名で境界を指定する。
        anet::ConfigData data;
        data.Set("DefaultDQNAgent.nn_forward.[mean].noisy.mode", "mu_only");
        data.Set("DefaultDQNAgent.actor.[train].nn_forward_key", "mean");
        data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "mean");
        data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "mean");
        // baseline未適用のDTOは未指定を保持し、購読時の必須検証へ渡す。
        CHECK(anet::rl::dqn::DefaultDQNAgentConfig(data).learner.final_feature_branch_key.empty());
        data.Set("DefaultDQNAgent.learner.final_feature_branch_key", "boundary");
        const anet::rl::dqn::DefaultDQNAgentConfig config(data);
        CHECK(config.learner.final_feature_branch_key == "boundary");
    }
    SECTION("Attention probe batch size validates even without a subscription") {
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
    }
}
