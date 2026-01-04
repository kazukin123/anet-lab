#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <wx/cmdline.h>
#include <nlohmann/json.hpp>
#include "anet/util.hpp"
#include "anet/str_util.hpp"

namespace anet {

    ///  Key-Valueベースの設定データ
    class ConfigData {
    public:
		using MapType = anet::OrderedMap<std::string, std::string>;
    public:
        ConfigData() {}
        ConfigData(const MapType& map) : map_(map) { }
        ConfigData(const ConfigData& from) : map_(from.map_) { }

        //ConfigData Make(const std::string& sub_class_id) const;
    public:
        void Set(const std::string& key, const std::string& value)
        {
            map_.Set(key, value);
        }

        template<typename T>
        void Set(const std::string& key, const T& value)
        {
            std::stringstream ss;
            ss << value;
            map_.Set(key, ss.str());
        }

        bool Has(const std::string& key) const
        {
            return map_.find(key) != map_.end();
        }

        const anet::OrderedMap<std::string, std::string>& Map() const
        {
            return map_;
        }
    public:
        std::string Get(const std::string& key, const char* defaultValue = "") const
        {
            std::string v(defaultValue);
            Read(key, v, v);
            return v;
        }

        template<typename T>
        T Get(const std::string& key, T defaultValue = T()) const
        {
            T v = defaultValue;
            Read(key, v, defaultValue);
            return v;
        }
    public:
        bool Read(const std::string& key, std::string& value, const std::string& defaultValue) const
        {
            auto it = map_.find(key);
            if (it == map_.end()) { value = defaultValue; return false; }
            value = (*it).second;
            return true;
        }

        bool Read(const std::string& key, int& value, int defaultValue) const
        {
            auto it = map_.find(key);
            if (it == map_.end()) { value = defaultValue; return false; }
            auto str = anet::ReplaceAll((*it).second, ",", ""); // カンマ除去
            try { value = std::stoi(str.c_str()); }
            catch (...) { value = defaultValue; return false; }
            return true;
        }

        bool Read(const std::string& key, float& value, float defaultValue) const
        {
            auto it = map_.find(key);
            if (it == map_.end()) { value = defaultValue; return false; }
            auto str = anet::ReplaceAll((*it).second, ",", ""); // カンマ除去
            try { value = std::stof(str.c_str()); }
            catch (...) { value = defaultValue; return false; }
            return true;
        }

        bool Read(const std::string& key, double& value, double defaultValue) const
        {
            auto it = map_.find(key);
            if (it == map_.end()) { value = defaultValue; return false; }
            auto str = anet::ReplaceAll((*it).second, ",", ""); // カンマ除去
            try { value = std::stod(str.c_str()); }
            catch (...) { value = defaultValue; return false; }
            return true;
        }

        bool Read(const std::string& key, uint64_t& value, uint64_t defaultValue) const
        {
            auto it = map_.find(key);
            if (it == map_.end()) { value = defaultValue; return false; }
            auto str = anet::ReplaceAll((*it).second, ",", ""); // カンマ除去
            try { value = std::stoull(str); }
            catch (...) { value = defaultValue; return false; }
            return true;
        }

        bool Read(const std::string& key, bool& value, bool defaultValue) const
        {
            auto it = map_.find(key);
            if (it == map_.end()) { value = defaultValue; return false; }
            const auto& v = (*it).second;
            if (v == "true" || v == "TRUE" || v == "1" || v == "yes" || v == "on") { value = true; return true; }
            if (v == "false" || v == "FALSE" || v == "0" || v == "no" || v == "off") { value = false; return true; }
            value = defaultValue;
            return false;
        }

        bool Read(const std::string& key, std::vector<float>& value, std::vector<float> defaultValue) const
        {
            auto it = map_.find(key);
            if (it == map_.end()) { value = defaultValue; return false; }
            try {
                auto str_vec = anet::Split((*it).second, { " ", "　" }, true);
                value.resize(str_vec.size());
                for (int i = 0; i < value.size(); i++) {
                    value[i] = std::stof(str_vec[i]);
                }
            } catch (...) {
                value = defaultValue;
                return false;
            }
            return true;
        }

        bool Read(const std::string& key, std::vector<std::string>& value, std::vector<std::string> defaultValue) const
        {
            auto it = map_.find(key);
            if (it == map_.end()) { value = defaultValue; return false; }
            try {
                auto str_vec = anet::Split((*it).second, { " ", "　" }, true);
                value.resize(str_vec.size());
                for (int i = 0; i < value.size(); i++) {
                    value[i] = str_vec[i];
                }
            } catch (...) {
                value = defaultValue;
                return false;
            }
            return true;
        }

    private:
        //std::vector<std::string> ResolveModule(const std::string& module) const;
    private:
        MapType map_;
    };

	const ConfigData EmptyConfigData;

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
        void Load(const std::string& filename, int depth = 0);
    };

    /// モジュール別Configクラス実装用の基底クラス
    class Config {
    public:
        Config(const std::string& config_prefix_);
        Config(const ConfigData& config_data, const std::string& config_prefix_);   ///< @todo インスタンスグループ対応（例：同じENV設定でもTrainとEvalで設定を分ける）

        std::string ToString() const;
        std::string ToConfigString() const;
        anet::json ToJson() const { return my_config_json_; }
        std::string GetConfigPrefix() const { return config_prefix_; }
        anet::ConfigData GetConfigData() const { return my_config_data_; }
    protected:
        template<typename T>
        void ReadConfig(const ConfigData& config_data, const std::string& key, T& value)
        {
			std::string config_data_key = (config_prefix_.empty() ? "" : config_prefix_ + ".") + key;
            config_data.Read(config_data_key, value, value);
            my_config_data_.Set(key, value);
            my_config_json_[key] = value;
		}
    protected:
        /// Configの値としてConfigは含まめず、あくまでもフラットな設定データ構造とする

        std::string config_prefix_;
        ConfigData my_config_data_; ///< Key=String Value=String
        anet::json my_config_json_; ///< JSONデータとして元の型情報を覚えておく
    };

    /// 設定マネージャー。コマンドラインオプションとPropertiesファイルを元にConfigDataを生成。
    class ConfigManager {
    public:
        ConfigManager(const std::string& filePath, const wxCmdLineParser* cmdLine = nullptr);

        ConfigData GetConfigData() const { return { map_ }; }
    private:
        void LoadFromFile(const std::string& filePath);
        void ApplyCmdLineOverrides(const wxCmdLineParser& cmdLine);
        void AutoMerge();
    private:
        ConfigData::MapType map_;
    };

}

// namespace anet
// ---- 読み込みマクロ ----
#ifndef ANET_READ_CONFIG
#define ANET_READ_CONFIG(config_data, field) \
        ReadConfig(config_data, (#field), (field))
#endif