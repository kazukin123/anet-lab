#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <set>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include "anet/json_util.hpp"
#include "anet/util.hpp"
#include "anet/str_util.hpp"

class wxCmdLineParser;

namespace anet {

    /// 読んだ欄の相対キーと、値を与えた完全キーとの対応。
    class ConfigFullKeyTable {
    public:
        std::string GetFullKey(const std::string& key) const;
        bool Contains(const std::string& key) const;
        void Set(const std::string& key, std::string full_key);
        ConfigFullKeyTable Extract(const std::string& prefix) const;
        void CopyFieldFrom(const std::string& key, const ConfigFullKeyTable& source, const std::string& source_key);
    private:
        std::unordered_map<std::string, std::string> keys_;
    };

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
        ConfigData(const MapType& map) { for (const auto& [key, value] : map) Set(key, value); }
        ConfigData(const ConfigData& from) = default;

        //ConfigData Make(const std::string& sub_class_id) const;
    public:
        std::unordered_map<std::string, ConfigData> MakeSubConfigData(const std::string& prefix) const;
        /// 値を与えた完全キーを返す。照会は読み取りとして記録しない。
        std::string GetFullKey(const std::string& key) const;
		std::unordered_set<std::string> GetSubConfigTags(const std::string& prefix) const;
        /// 元の設定の読み取り記録を引き継いで、相対キーへ値を写す。
        void CopyValueFrom(const std::string& key, const ConfigData& source, const std::string& source_key);
        /// 読み口が消費しなかった設定キーをキー順に返す。
        std::vector<std::pair<std::string, std::string>> GetUnreadKeys() const;
        /// 同じ出典を保ち、今回の読み口だけを検証する範囲を作る。
        ConfigData CreateReadScope(const std::string& prefix = "") const;
        /// 合成したカタログ項目に、その設定を所有する項目の置き場を継承する。
        void UseLocationOf(const ConfigData& source);
        /// 読まれなくても誤りでない prefix を、その設定の持ち主が宣言する。
        void MarkInventory(const std::string& prefix, const std::vector<std::string>& excluded_prefixes = {}) const;
        /// この範囲に読まれないキーが残っていれば、読み口の受け付けるキーとともに止める。
        void RequireAllKeysRead(std::string_view detail = {}) const;
        void MergeFromChecked(const ConfigData& other);
        void OverwriteFrom(const ConfigData& other);
        std::string ToPropertiesString() const;
        void SaveProperties(const std::filesystem::path& path) const;
    public:
        void Set(const std::string& key, const std::string& value)
        {
            SetString(key, value);
        }

        void Set(const std::string& key, bool value)
        {
            SetString(key, value ? "true" : "false");
        }

        template<typename T>
        void Set(const std::string& key, const T& value)
        {
            std::stringstream ss;
            ss << value;
            SetString(key, ss.str());
        }

        template<typename T>
        void Set(const std::string& key, const std::vector<T>& value)
        {
            std::stringstream ss;
            for (size_t i = 0; i < value.size(); ++i) {
                if (i > 0) ss << " ";
                ss << value[i];
            }
            SetString(key, ss.str());
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
        /// キー欠落時だけvalue_if_missingを使い、存在する不正値は例外にする。
        bool Read(const std::string& key, std::string& value, const std::string& value_if_missing) const;
        bool Read(const std::string& key, int& value, int value_if_missing) const;
        bool Read(const std::string& key, float& value, float value_if_missing) const;
        bool Read(const std::string& key, double& value, double value_if_missing) const;
        bool Read(const std::string& key, uint64_t& value, uint64_t value_if_missing) const;
        bool Read(const std::string& key, int64_t& value, int64_t value_if_missing) const;
        bool Read(const std::string& key, bool& value, bool value_if_missing) const;
        bool Read(const std::string& key, std::vector<float>& value, std::vector<float> value_if_missing) const;
        bool Read(const std::string& key, std::vector<int64_t>& value, std::vector<int64_t> value_if_missing) const;
        bool Read(const std::string& key, std::vector<std::string>& value, std::vector<std::string> value_if_missing) const;

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
        struct ReadEntry {
            std::atomic_bool read = false;
            std::atomic_bool inventory = false;
            std::vector<std::shared_ptr<ReadEntry>> sources;
            void MarkRead();
            void MarkInventory();
        };
        struct ReadScope {
            std::mutex mutex;
            std::set<std::string> attempted;
        };
        void SetString(const std::string& key, const std::string& value);
        void MarkRead(const std::string& key) const;
        MapType map_;
        // 別の置き場から写した項目は、現在のキーとは別に出典を保持する。
        std::unordered_map<std::string, std::string> full_keys_;
        std::string location_;
        std::string scope_prefix_;
        std::unordered_map<std::string, std::shared_ptr<ReadEntry>> read_entries_;
        std::shared_ptr<ReadScope> read_scope_;
    };

    /// Run の構築を終えた呼び出し側が、未読キーを 1 件の WARN で一覧する。
    void WarnUnreadConfigKeys(const ConfigData& config_data);

	const ConfigData EmptyConfigData;

    struct ConfigManagerOptions {
        std::optional<std::vector<std::filesystem::path>> config_search_dirs;
        ConfigData injected_config;
        std::vector<std::filesystem::path> overwrite_config_paths;
    };

    /// Properties類似形式の設定ファイル操作クラス
    class Properties {
    public:
        explicit Properties(const std::filesystem::path& filename, ConfigManagerOptions options = {})
            : options_(options)
        {
            Load(filename);
        }
        ConfigData ToConfigData() const {
            return configData;
        }
        /// 解決入力の既定キーを読み取り専用で参照する。返却参照の寿命はPropertiesと同じ。
        const std::unordered_set<std::string>& DefaultKeys() const {
            return default_keys_;
        }
        const std::unordered_set<std::string>& SelectionSourcePrefixes() const {
            return selection_source_prefixes_;
        }
    private:
        // 読み込み途中だけ保持し、公開ConfigDataへは渡さない。
        std::unordered_set<std::string> default_keys_;
        std::unordered_set<std::string> selection_source_prefixes_;
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
        anet::ConfigData GetScopedConfigData() const;
        const ConfigFullKeyTable& GetFullKeys() const { return full_keys_; }
        std::string GetFullKey(const std::string& key) const { return full_keys_.GetFullKey(key); }
    protected:
        /// default/overrideの完全キーを保ち、指定した項目だけを新しく検証する。
        ConfigData MakeReadScope(const ConfigData& config_data, const std::string& key) const;

        void RecordFullKey(const std::string& key, std::string full_key)
        {
            full_keys_.Set(key, std::move(full_key));
        }

        template<typename T>
        void ReadConfig(const ConfigData& config_data, const std::string& key, T& value, bool record = true)
        {
            if constexpr (ConfigReader<T>::kEnabled) {
                ConfigReader<T>::Read(*this, config_data, key, value);
            } else {
                // default_prefixで設定取得
                const std::string default_config_key = MakeDefaultConfigKey(key);
                config_data.Read(default_config_key, value, value);

			    // override_prefixで上書き設定があれば取得
                std::string source_key = default_config_key;
                if (!override_prefix_.empty()) {
                    const std::string override_key = MakeOverrideConfigKey(key);
                    if (config_data.Read(override_key, value, value)) source_key = override_key;
                }
                RecordFullKey(key, config_data.GetFullKey(source_key));

			    // 取り込んだKeyとValueを保存
                if (record) RecordConfigValue(key, value);
            }
		}

        // 読み取った値のうち、実効契約が使う欄だけを記録する。
        template<typename T>
        void RecordConfigValue(const std::string& key, const T& value)
        {
            my_config_data_.Set(key, value);
            my_config_json_[key] = value;
        }

        template<typename T>
        void ReadSubConfig(
            const ConfigData& config_data,
            const std::string& root_key,
            const std::string& sub_key,
            T& value, bool record = true)
        {
            ReadConfig(config_data, root_key.empty() ? sub_key : root_key + "." + sub_key, value, record);
        }

        template<typename T>
        void ReadInheritedSubConfig(const ConfigData& data, const std::string& prefix, const std::string& field,
            T& value, const ConfigFullKeyTable& inherited, bool record = true)
        {
            const auto key = prefix.empty() ? field : prefix + "." + field;
            ReadConfig(data, key, value, record);
            // 型付きの読み口は内側の各欄を扱う。直接読む欄は、明示指定が無ければ継承元を保つ。
            if constexpr (!ConfigReader<T>::kEnabled) {
                if (inherited.Contains(field) && !HasConfigValue(data, key)) {
                    RecordFullKey(key, inherited.GetFullKey(field));
                }
            }
        }

        std::string MakeTaggedSubConfigKey(
            const std::string& root_key,
            const std::string& sub_key,
            const std::string& tag) const
        {
            const std::string key = sub_key + ".[" + tag + "]";
            return root_key.empty() ? key : root_key + "." + key;
        }

        bool HasConfigValue(const ConfigData& config_data, const std::string& key) const
        {
            const std::string default_config_key = MakeDefaultConfigKey(key);
            if (config_data.Has(default_config_key)) {
                return true;
            }

            return !override_prefix_.empty() && config_data.Has(MakeOverrideConfigKey(key));
        }

        std::string MakeDefaultConfigKey(const std::string& key) const
        {
            return (default_prefix_.empty() ? "" : default_prefix_ + ".") + key;
        }

        std::string MakeOverrideConfigKey(const std::string& key) const
        {
            return (override_prefix_.empty() ? "" : override_prefix_ + ".") + key;
        }

        template<typename T>
        friend struct ConfigReader;

        /// Configの値としてConfigは含まめず、あくまでもフラットな設定データ構造とする

        std::string default_prefix_;
        std::string override_prefix_;
        ConfigData my_config_data_; ///< Key=String Value=String
        anet::json my_config_json_; ///< JSONデータとして元の型情報を覚えておく
        ConfigFullKeyTable full_keys_;
    };

    /// ConfigDataで扱う設定キーの値のoptional。未指定ならDTOの初期値を保持する。
    template<typename T>
    struct ConfigReader<std::optional<T>> {
        static constexpr bool kEnabled = true;

        static void Read(Config& owner, const ConfigData& config_data,
            const std::string& key, std::optional<T>& value)
        {
            static_assert(!ConfigReader<T>::kEnabled,
                "Optional configuration supports native ConfigData leaf values.");
            // 不在を記録値に変えず、既定値も無い場合は読み取り自体を省く。
            if (!value.has_value() && !owner.HasConfigValue(config_data, key)) {
                return;
            }
            // prefix順序・変換・記録は通常値の共通経路へ委譲する。
            T parsed = value.value_or(T{});
            owner.ReadConfig(config_data, key, parsed);
            value = std::move(parsed);
        }
    };

    /// 設定マネージャー。コマンドラインオプションとPropertiesファイルを元にConfigDataを生成。
    class ConfigManager {
    public:
        ConfigManager(
            const std::filesystem::path& filePath,
            const wxCmdLineParser* cmdLine = nullptr,
            ConfigManagerOptions options = {});

        ConfigData GetConfigData() const { return effective_config_; }
        anet::json GetResolutionJson() const { return resolution_json_; }
    private:
        void LoadFromFile(const std::filesystem::path& filePath);
        void OverwriteFromFile(const std::filesystem::path& filePath);
        ConfigData::MapType ReadCmdLineOverrides(const wxCmdLineParser& cmdLine);
    private:
        ConfigManagerOptions options_;
        ConfigData::MapType map_;
        ConfigData effective_config_;
        std::unordered_set<std::string> default_keys_;
        std::unordered_set<std::string> selection_source_prefixes_;
        anet::json resolution_json_;
    };

}

// namespace anet
// ---- 読み込みマクロ ----
#ifndef ANET_READ_CONFIG
#define ANET_READ_CONFIG(config_data, field) \
        ReadConfig(config_data, (#field), (field))
#endif
