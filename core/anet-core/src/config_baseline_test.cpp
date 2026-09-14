#include "anet/catch_test.hpp"
#include "anet/config.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <wx/cmdline.h>
#include <wx/utils.h>

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
