#pragma once
#include <string>
#include <unordered_map>
#include <fstream>
#include <stdexcept>

// ---- key 結合 ("group" + "." + field) ----
static inline std::string PROPS_JOIN(const std::string& group, const char* field) {
    if (!group.empty() && group.back() == '.')
        return group + field;
    return group + "." + field;
}

// ---- 読み込みマクロ（group と field を指定する）----
#ifndef ANET_READ_PROPS
#define ANET_READ_PROPS(props, group, field) \
    (props)->Read(PROPS_JOIN(std::string(group), #field), (field), (field))
#endif

class Properties {
public:
    explicit Properties(const std::string& filename) {
        Load(filename);
    }

    void Set(const std::string& key, const std::string& value) {
        kv_[key] = value;
    }

    bool Has(const std::string& key) const {
        return kv_.find(key) != kv_.end();
    }

public:
    // --- std::string 用オーバーロード ---
    std::string Get(const std::string& key, const std::string& defaultValue = "") const
    {
        auto it = kv_.find(key);
        return (it != kv_.end()) ? it->second : defaultValue;
    }

    // --- 汎用テンプレート ---
    template<typename T>
    T Get(const std::string& key, const T& defaultValue = T()) const
    {
        auto it = kv_.find(key);
        if (it == kv_.end()) return defaultValue;

        std::istringstream iss(it->second);
        T value;
        if (!(iss >> value)) return defaultValue;
        return value;
    }

public:

    void Read(const std::string& key, std::string& value, const std::string& defaultValue) const {
        auto it = kv_.find(key);
        value = (it != kv_.end()) ? it->second : defaultValue;
    }

    void Read(const std::string& key, int& value, int defaultValue) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) { value = defaultValue; return; }
        try { value = std::stoi(it->second); }
        catch (...) { value = defaultValue; }
    }

    void Read(const std::string& key, float& value, float defaultValue) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) { value = defaultValue; return; }
        try { value = std::stof(it->second); }
        catch (...) { value = defaultValue; }
    }

    void Read(const std::string& key, double& value, double defaultValue) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) { value = defaultValue; return; }
        try { value = std::stod(it->second); }
        catch (...) { value = defaultValue; }
    }

    void Read(const std::string& key, bool& value, bool defaultValue) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) { value = defaultValue; return; }
        const auto& v = it->second;
        if (v == "true" || v == "TRUE" || v == "1" || v == "yes" || v == "on") { value = true; return; }
        if (v == "false" || v == "FALSE" || v == "0" || v == "no" || v == "off") { value = false; return; }
        value = defaultValue;
    }

private:
    std::unordered_map<std::string, std::string> kv_;

    static std::string Trim(const std::string& s) {
        const char* ws = " \t\rn";
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
            if (line.empty() || line[0] == '#' || line[0] == '!') continue;

            size_t posComment = line.find('#');
            if (posComment != std::string::npos)
                line = line.substr(0, posComment);

            line = Trim(line);
            if (line.empty()) continue;

            size_t pos = line.find('=');
            if (pos == std::string::npos)
                pos = line.find(':');
            if (pos == std::string::npos)
                continue;

            std::string key = Trim(line.substr(0, pos));
            std::string value = Trim(line.substr(pos + 1));
            if (!key.empty())
                kv_[key] = value;
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

