#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include "anet/util.hpp"
#include "anet/str_util.hpp"

class wxCmdLineParser;

namespace anet {

    template<typename T>
    struct ConfigReader {
        static constexpr bool kEnabled = false;
    };

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
        std::unordered_map<std::string, ConfigData> MakeSubConfigData(const std::string& prefix) const;
		std::unordered_set<std::string> GetSubConfigTags(const std::string& prefix) const;
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

        template<typename T>
        void Set(const std::string& key, const std::vector<T>& value)
        {
            std::stringstream ss;
            for (size_t i = 0; i < value.size(); ++i) {
                if (i > 0) ss << " ";
                ss << value[i];
            }
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
        bool Read(const std::string& key, std::string& value, const std::string& defaultValue) const;
        bool Read(const std::string& key, int& value, int defaultValue) const;
        bool Read(const std::string& key, float& value, float defaultValue) const;
        bool Read(const std::string& key, double& value, double defaultValue) const;
        bool Read(const std::string& key, uint64_t& value, uint64_t defaultValue) const;
        bool Read(const std::string& key, int64_t& value, int64_t defaultValue) const;
        bool Read(const std::string& key, bool& value, bool defaultValue) const;
        bool Read(const std::string& key, std::vector<float>& value, std::vector<float> defaultValue) const;
        bool Read(const std::string& key, std::vector<int64_t>& value, std::vector<int64_t> defaultValue) const;
        bool Read(const std::string& key, std::vector<std::string>& value, std::vector<std::string> defaultValue) const;

        anet::json ToJson() const
        {
            anet::json j;
            for (const auto& kv : map_) {
				j[kv.first] = kv.second;
            }
            return j;
        }

        std::string ToString() const
        {
            std::stringstream ss;
            ss << "{";
            bool first = true;
            for (const auto& kv : map_) {
                if (!first) ss << ", ";
                ss << kv.first << "=" << kv.second;
                first = false;
            }
            ss << "}";
            return ss.str();
		}
    private:
        //std::vector<std::string> ResolveModule(const std::string& module) const;
    private:
        MapType map_;
    };

	const ConfigData EmptyConfigData;

    struct ConfigManagerOptions {
        std::optional<std::vector<std::filesystem::path>> config_search_dirs;
    };

    /// Properties類似形式の設定ファイル操作クラス
    class Properties {
    public:
        explicit Properties(const std::string& filename, ConfigManagerOptions options = {})
            : options_(options)
        {
            Load(filename);
        }
        ConfigData ToConfigData() const {
            return configData;
        }
    private:
        ConfigData configData;
        ConfigManagerOptions options_;

        static std::string Trim(const std::string& s);
        void Load(const std::filesystem::path& filename, int depth = 0);
    };

    /// モジュール別Configクラス実装用の基底クラス
    class Config {
    public:
        Config(const std::string& config_prefix_);
        Config(const ConfigData& config_data, const std::string& default_prefix);
        Config(const ConfigData& config_data, const std::string& default_prefix, const std::string& prefix);

        std::string ToString() const;
        std::string ToConfigString() const;
        anet::json ToJson() const { return my_config_json_; }
        
        std::string GetConfigPrefix() const { return default_prefix_; }
        std::string GetOverridePrefix() const { return override_prefix_; }
        anet::ConfigData GetConfigData() const { return my_config_data_; }
    protected:
        template<typename T>
        void ReadConfig(const ConfigData& config_data, const std::string& key, T& value)
        {
            if constexpr (ConfigReader<T>::kEnabled) {
                ConfigReader<T>::Read(*this, config_data, key, value);
            } else {
                // default_prefixで設定取得
			    std::string default_config_key = (default_prefix_.empty() ? "" : default_prefix_ + ".") + key;
                config_data.Read(default_config_key, value, value);

			    // override_prefixで上書き設定があれば取得
			    std::string override_config_key = (override_prefix_.empty() ? "" : override_prefix_ + ".") + key;
                config_data.Read(override_config_key, value, value);

			    // 取り込んだKeyとValueを保存
                my_config_data_.Set(key, value);
                my_config_json_[key] = value;
            }
		}

        template<typename T>
        void ReadSubConfig(
            const ConfigData& config_data,
            const std::string& root_key,
            const std::string& sub_key,
            T& value)
        {
            ReadConfig(config_data, root_key.empty() ? sub_key : root_key + "." + sub_key, value);
        }

        std::string MakeTaggedSubConfigKey(
            const std::string& root_key,
            const std::string& sub_key,
            const std::string& tag) const
        {
            const std::string key = sub_key + ".[" + tag + "]";
            return root_key.empty() ? key : root_key + "." + key;
        }

        template<typename T>
        friend struct ConfigReader;
    protected:
        /// Configの値としてConfigは含まめず、あくまでもフラットな設定データ構造とする

        std::string default_prefix_;
        std::string override_prefix_;
        ConfigData my_config_data_; ///< Key=String Value=String
        anet::json my_config_json_; ///< JSONデータとして元の型情報を覚えておく
    };

    /// 設定マネージャー。コマンドラインオプションとPropertiesファイルを元にConfigDataを生成。
    class ConfigManager {
    public:
        ConfigManager(
            const std::string& filePath,
            const wxCmdLineParser* cmdLine = nullptr,
            ConfigManagerOptions options = {});

        ConfigData GetConfigData() const { return { map_ }; }
    private:
        void LoadFromFile(const std::string& filePath);
        void ApplyCmdLineOverrides(const wxCmdLineParser& cmdLine);
        void AutoMerge();
    private:
        ConfigManagerOptions options_;
        ConfigData::MapType map_;
    };

}

// namespace anet
// ---- 読み込みマクロ ----
#ifndef ANET_READ_CONFIG
#define ANET_READ_CONFIG(config_data, field) \
        ReadConfig(config_data, (#field), (field))
#endif
