#pragma once
#include <string>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include "anet/util.hpp"

// ---- 読み込みマクロ（group と field を指定する）----
#ifndef ANET_READ_PROPS
#define ANET_READ_PROPS(configData, group, field) \
        (configData).Read(anet::config_key(std::string(group), #field), (field), (field))
#endif

namespace anet {

    // ---- key 結合 ("group" + "." + field) ----
    static inline std::string config_key(const std::string& group, const char* field) {
        if (!group.empty() && group.back() == '.')
            return group + field;
        return group + "." + field;
    }

    class ConfigData {
    public:
        ConfigData() {}
        ConfigData(const ConfigData& from) {
            this->kv_ = from.kv_;
        }

        void Set(const std::string& key, const std::string& value) {
            kv_[key] = value;
        }

        bool Has(const std::string& key) const {
            return kv_.find(key) != kv_.end();
        }
    public:
        std::string Get(const std::string& key, const char* defaultValue) const {
            std::string v(defaultValue);
            Read(key, v, v);
            return v;
        }

        template<typename T>
        T Get(const std::string& key, T defaultValue = T()) const {
            T v = defaultValue;
            Read(key, v, defaultValue);
            return v;
        }
    public:
        bool Read(const std::string& key, std::string& value, const std::string& defaultValue) const {
            auto it = kv_.find(key);
            if (it == kv_.end()) { value = defaultValue; return false; }
            value = (*it).second;
            return true;
        }

        bool Read(const std::string& key, int& value, int defaultValue) const {
            auto it = kv_.find(key);
            if (it == kv_.end()) { value = defaultValue; return false; }
            try { value = std::stoi((*it).second); }
            catch (...) { value = defaultValue; return false; }
            return true;
        }

        bool Read(const std::string& key, float& value, float defaultValue) const {
            auto it = kv_.find(key);
            if (it == kv_.end()) { value = defaultValue; return false; }
            try { value = std::stof((*it).second); }
            catch (...) { value = defaultValue; return false; }
            return true;
        }

        bool Read(const std::string& key, double& value, double defaultValue) const {
            auto it = kv_.find(key);
            if (it == kv_.end()) { value = defaultValue; return false; }
            try { value = std::stod((*it).second); }
            catch (...) { value = defaultValue; return false; }
            return true;
        }

        bool Read(const std::string& key, bool& value, bool defaultValue) const {
            auto it = kv_.find(key);
            if (it == kv_.end()) { value = defaultValue; return false; }
            const auto& v = (*it).second;
            if (v == "true" || v == "TRUE" || v == "1" || v == "yes" || v == "on") { value = true; return true; }
            if (v == "false" || v == "FALSE" || v == "0" || v == "no" || v == "off") { value = false; return true; }
            value = defaultValue;
            return false;
        }

    private:
        anet::LinkedHashMap<std::string, std::string> kv_;
    };

    class Properties {
    public:
        explicit Properties(const std::string& filename) {
            Load(filename);
        }
        ConfigData ToConfigData() const {
            return configData;
        }
    private:
        ConfigData configData;

        static std::string Trim(const std::string& s) {
            const char* ws = " \t\r\n";
            size_t b = s.find_first_not_of(ws);
            if (b == std::string::npos) return "";
            size_t e = s.find_last_not_of(ws);
            return s.substr(b, e - b + 1);
        }

        void Load(const std::string& filename) {
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
    };

    // サンプル
    //struct RLAgent::Param {
    //    static constexpr const char* GROUP = "agent";
    //
    //    float alpha = 1e-3f;
    //    float gamma = 0.99f;
    //
    //    void Load(const Properties& props) {
    //        ANET_READ_PROPS(props, GROUP, alpha);
    //        ANET_READ_PROPS(props, GROUP, gamma);
    //    }
    //};

}
