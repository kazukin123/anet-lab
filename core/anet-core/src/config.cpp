#include "anet/config.hpp"
#include <sstream>
#include <fstream>
#include <wx/string.h>
#include "anet/str_util.hpp"

namespace anet {

    std::string Properties::Trim(const std::string& s) {
        const char* ws = " \t\r\n";
        size_t b = s.find_first_not_of(ws);
        if (b == std::string::npos) return "";
        size_t e = s.find_last_not_of(ws);
        return s.substr(b, e - b + 1);
    }

    void Properties::Load(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs) throw std::runtime_error("Properties: Cannot open: " + filename);

        // BOM スキップ
        char bom[3] = { 0 };
        ifs.read(bom, 3);
        if (!(bom[0] == '\xEF' && bom[1] == '\xBB' && bom[2] == '\xBF')) {
            // BOM がなければ読み取り位置を戻す
            ifs.seekg(0);
        }

        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue; // コメント行スキップ

            size_t posHash = line.find('#');   // '#' 以降はコメント扱い
            if (posHash != std::string::npos)
                line = line.substr(0, posHash);

            size_t posSlash = line.find("//");  // '//' 以降はコメント扱い
            if (posSlash != std::string::npos)
                line = line.substr(0, posSlash);

            line = Trim(line);  // 前後の空白除去
            if (line.empty()) continue;

            size_t pos = line.find('=');    // '=' または ':' で区切る
            if (pos == std::string::npos)
                pos = line.find(':');
            if (pos == std::string::npos)
                continue;

            std::string key = Trim(line.substr(0, pos));
            std::string value = Trim(line.substr(pos + 1));

            // 末尾 ';' を除去
            while (!value.empty() && value.back() == ';')
                value.pop_back();
            value = Trim(value);

            if (!key.empty()) {
                configData.Set(key, value);
            }
        }
    }

    // ---- Config ----

    Config::Config(const std::string& config_prefix) : config_prefix_(config_prefix)
    {
    }

    Config::Config(const ConfigData& config_data, const std::string& config_prefix) :
        config_prefix_(config_prefix)
    {
    }

    std::string Config::ToString() const {
        return ToJson().dump(2);
    }


    // ---- ConfigManager ----

    ConfigManager::ConfigManager(const std::string& filePath, const wxCmdLineParser* cmdLine)
    {
        LoadFromFile(filePath);
        if (cmdLine) {
            ApplyCmdLineOverrides(*cmdLine);
        }
        AutoMerge();
    }

    void ConfigManager::LoadFromFile(const std::string& filePath)
    {
        Properties props(filePath);
        map_ = props.ToConfigData().Map();
    }

    void ConfigManager::ApplyCmdLineOverrides(const wxCmdLineParser& cmdLine)
    {
        // 例: agent.lr=0.001 をパラメータとして渡す
        // executable agent.lr=0.001 train.max_steps=20000
        const int count = cmdLine.GetParamCount();
        for (int i = 0; i < count; ++i) {
            wxString s = cmdLine.GetParam(i);
            std::string p = std::string(s.mb_str());

            const auto pos = p.find('=');
            if (pos == std::string::npos) continue;

            const std::string key = p.substr(0, pos);
            const std::string val = p.substr(pos + 1);

            if (!key.empty()) {
                map_.Set(key, val);
            }
        }
    }

    static constexpr const char* MERGE_KEYWORD = ".$";

    void ConfigManager::AutoMerge()
    {
        // env.$ = env.common > env.trunk
        // env.xxx = 1
        // env.yyy = 2
        // env.common.yyy = 10
        // env.common.zzz = 20
        // env.trunk.zzz = 200
        // 
        //   ↓
        // 
        // env.xxx = 1
        // env.yyy = 10
        // env.zzz = 200
        //

        ConfigData::MapType new_map;
        ConfigData::MapType map = map_;

		std::vector<std::string> merge_keys;

        // マージキー以外をそのままコピー
        for (const auto& kv : map) {
            const std::string key = kv.first;
            const std::string val = kv.second;
            if (!anet::EndsWith(key, MERGE_KEYWORD)) {
                new_map.Set(key, val);
            } else {
				merge_keys.push_back(key);
            }
        }

        // マージキーの上書き処理
        for (const auto& merge_key : merge_keys) {
			auto base_key = anet::RemoveSuffix(merge_key, MERGE_KEYWORD);   // env.$ -> env

			auto merge_val = map.Get(merge_key);                                             // "env.common > env.trunk"
			std::vector<std::string> merge_target_keys = Split(merge_val, { ">" }, true);    // { env.common, env.trunk }
            if (merge_target_keys.empty()) continue;

            for (auto merge_target_key : merge_target_keys) {   // env.common, env.trunk
                if (merge_target_key.empty()) continue;

                for (const auto& kv2 : map) {
                    std::string key2 = kv2.first;
                    std::string val2 = kv2.second;
                    if (anet::StartsWith(key2, merge_target_key)) {                    // env.common, env.common.yyy, env.common.zzz
                        // ERASE: env.common, env.common.yyy, env.common.zzz
                        new_map.Erase(key2);

                        // SKIP env.common
                        if (merge_target_key == key2) continue;

                        auto key_suffix = anet::RemovePrefix(key2, merge_target_key);  // .yyy, .zzz
                        auto target_key = base_key + key_suffix;                       // env.yyy, env.zzz

                        // マージ対象のValueをマージ元のKeyでSet
                        new_map.Set(target_key, val2);
                    }
                }
            }
        }

        map_ = new_map;
    }

} // namespace anet
