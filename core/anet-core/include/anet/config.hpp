#pragma once
#include <string>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <wx/cmdline.h>
#include <nlohmann/json.hpp>
#include "anet/util.hpp"


// ---- 読み込みマクロ ----
#ifndef ANET_APPLY_CONFIG
#define ANET_APPLY_CONFIG(configData,field) \
        (configData).Read((#field), (field), (field))
#endif

namespace anet {

    ///  Key-Valueベースの設定データ
    class ConfigData {
    public:
        ConfigData() {}
        ConfigData(const ConfigData& from) {
            this->kv_ = from.kv_;
        }

        void Set(const std::string& key, const std::string& value) {
            kv_.Set(key, value);
        }

        bool Has(const std::string& key) const {
            return kv_.find(key) != kv_.end();
        }

        const anet::OrderedMap<std::string, std::string> Map() const {
            return kv_;
        }
    public:
        std::string Get(const std::string& key, const char* defaultValue = "") const {
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

        bool Read(const std::string& key, uint64_t& value, uint64_t defaultValue) const {
            auto it = kv_.find(key);
            if (it == kv_.end()) { value = defaultValue; return false; }
            try { value = std::stoull((*it).second); }
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
        anet::OrderedMap<std::string, std::string> kv_;
    };

    /// Properties類似形式の設定ファイル操作クラス
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

        static std::string Trim(const std::string& s);
        void Load(const std::string& filename);
    };

    /// 設定マネージャー。コマンドラインオプションとPropertiesファイルを元にConfigDataを生成。
    class ConfigManager {
    public:
        ConfigManager(const std::string& filePath,const wxCmdLineParser* cmdLine = nullptr);

        ConfigData Make(const std::string& module, const std::string& defaultPreset = "") const;    ///< @todo ConfigDataに移す
        const ConfigData& GetConfigData() const { return data_; }
    private:
        void LoadFromFile(const std::string& filePath);
        void ApplyCmdLineOverrides(const wxCmdLineParser& cmdLine);

        // (module, defaultPreset) → resolvedModule を返す
        std::string ResolveModule(const std::string& module,
            const std::string& defaultPreset) const;
    private:
        ConfigData data_;
    };

    /// モジュール別Configクラス実装用の基底クラス
    class Config {
    public:
        Config();
        Config(const ConfigData& configData, const std::string& className = "", const std::string& moduleName = "");

        std::string ToString() const;
        nlohmann::json ToJson() const;
    protected:
        ConfigData configData_;
        std::string className_;
        std::string moduleName_;
    };

} // namespace anet
