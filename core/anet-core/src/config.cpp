#include "anet/config.hpp"
#include <sstream>
#include <fstream>
#include <wx/string.h>

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

    ConfigManager::ConfigManager(const std::string& filePath, const wxCmdLineParser* cmdLine) {
        LoadFromFile(filePath);
        if (cmdLine) {
            ApplyCmdLineOverrides(*cmdLine);
        }
    }

    void ConfigManager::LoadFromFile(const std::string& filePath) {
        Properties props(filePath);
        data_ = props.ToConfigData();
    }

    void ConfigManager::ApplyCmdLineOverrides(const wxCmdLineParser& cmdLine) {
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
                data_.Set(key, val);
            }
        }
    }

    std::string ConfigManager::ResolveModule(const std::string& module, const std::string& defaultPreset) const {
        const std::string presetKey = module + ".preset";
        const std::string preset = data_.Get(presetKey, std::string());

        if (!preset.empty()) return preset;
        if (!defaultPreset.empty()) return defaultPreset;

        return module;
    }

    ConfigData ConfigManager::Make(const std::string& module, const std::string& defaultPreset) const {
        ConfigData out;

        // (1) resolvedModule の決定
        const std::string resolved = ResolveModule(module, defaultPreset);
        const std::string prefix = resolved + ".";

        // (2) 元の data_ から resolved.* のみ抽出
        for (const auto& kv : data_.Map()) {   // Raw() は map<string,string> 取得
            const std::string& key = kv.first;
            const std::string& val = kv.second;

            // prefix に一致する場合だけ抽出
            if (key.compare(0, prefix.size(), prefix) == 0) {
                // 先頭の "resolved." を除去してローカルキーへ
                const std::string localKey = key.substr(prefix.size());
                out.Set(localKey, val);
            }
        }

        return out;
    }

    Config::Config() {
    }

    Config::Config(const ConfigData& configData, const std::string& className, const std::string& moduleName) :
            configData_(configData), className_(className), moduleName_(moduleName)
    {
            //apply(configData);
    }

    std::string Config::ToString() const {
        std::ostringstream oss;
        oss << "[class=" << className_ << ", module=" << moduleName_ << "]\n";

        for (const auto& kv : configData_.Map()) {
            oss << kv.first << " = " << kv.second << "\n";
        }

        return oss.str();
    }

    nlohmann::json Config::ToJson() const {
        nlohmann::json j;
        j["class"] = className_;
        j["module"] = moduleName_;

        nlohmann::json params = nlohmann::json::object();
        for (const auto& kv : configData_.Map()) {
            const auto& key = kv.first;
            const auto& val = kv.second;

            // 数値かどうか簡易判定（小数点と符号は許容）
            bool numeric = true;
            for (char c : val) {
                if (!(std::isdigit(c) || c == '.' || c == '-')) {
                    numeric = false;
                    break;
                }
            }

            if (numeric) {
                params[key] = std::stod(val);
            }
            else {
                params[key] = val;
            }
        }

        j["params"] = params;
        return j;
    }

} // namespace anet
