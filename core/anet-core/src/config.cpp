#include "anet/config.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <type_traits>
#include <utility>
#include <wx/string.h>
#include <wx/cmdline.h>
#include "anet/app_util.hpp"
#include "anet/common.hpp"
#include "anet/json_util.hpp"
#include "anet/str_util.hpp"
#include "anet/log.hpp"

namespace LOG = anet::log;

namespace anet {

    namespace {

    [[noreturn]] void ThrowReadFailure(
        const std::string& key, const std::string& value, const char* expected_type)
    {
        ANET_SYSTEM_ERROR(
            "ConfigData::Read failed. key=" << key
            << " value=\"" << value << "\" expected=" << expected_type);
    }

    std::string TrimConfigValue(const std::string& value)
    {
        const auto first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return {};
        }
        const auto last = value.find_last_not_of(" \t\r\n");
        return value.substr(first, last - first + 1);
    }

    template<typename T, typename Parser>
    T ParseNumericConfigValue(
        const std::string& key,
        const std::string& raw_value,
        const std::string& parse_value,
        const char* expected_type,
        Parser&& parser)
    {
        // 数値表現を正規化し、既存互換としてカンマ位置を問わず除去する。
        const auto value = anet::ReplaceAll(TrimConfigValue(parse_value), ",", "");
        if (value.empty() || (std::is_unsigned_v<T> && value.front() == '-')) {
            ThrowReadFailure(key, raw_value, expected_type);
        }

        // 変換結果は一旦ローカルへ保持し、全体を正しく解釈できた場合だけ呼出側へ返す。
        size_t parsed_length = 0;
        T parsed{};
        try {
            parsed = parser(value, &parsed_length);
        } catch (const std::exception&) {
            ThrowReadFailure(key, raw_value, expected_type);
        }
        if (parsed_length != value.size()) {
            ThrowReadFailure(key, raw_value, expected_type);
        }
        if constexpr (std::is_floating_point_v<T>) {
            if (!std::isfinite(parsed)) {
                ThrowReadFailure(key, raw_value, expected_type);
            }
        }
        return parsed;
    }

    template<typename T, typename Parser>
    T ParseNumericConfigValue(
        const std::string& key,
        const std::string& raw_value,
        const char* expected_type,
        Parser&& parser)
    {
        return ParseNumericConfigValue<T>(
            key, raw_value, raw_value, expected_type, std::forward<Parser>(parser));
    }

    bool PathExists(const std::filesystem::path& path)
    {
        std::error_code ec;
        return std::filesystem::exists(path, ec);
    }

    std::filesystem::path NormalizeAbsolutePath(const std::filesystem::path& path)
    {
        std::error_code ec;
        auto absolute_path = path.is_absolute()
            ? path
            : std::filesystem::absolute(path, ec);
        if (ec) {
            absolute_path = path;
        }

        const auto canonical_path = std::filesystem::weakly_canonical(absolute_path, ec);
        if (!ec) {
            return canonical_path.lexically_normal();
        }
        return absolute_path.lexically_normal();
    }

    bool IsPathUnderDirectory(const std::filesystem::path& path, const std::filesystem::path& directory)
    {
        const auto normalized_path = NormalizeAbsolutePath(path);
        const auto normalized_directory = NormalizeAbsolutePath(directory);

        auto path_it = normalized_path.begin();
        for (auto dir_it = normalized_directory.begin(); dir_it != normalized_directory.end(); ++dir_it, ++path_it) {
            if (path_it == normalized_path.end() || *path_it != *dir_it) {
                return false;
            }
        }
        return true;
    }

    std::vector<std::filesystem::path> GetConfigSearchDirs(const ConfigManagerOptions& options)
    {
        if (options.config_search_dirs.has_value()) {
            return *options.config_search_dirs;
        }
        return { GetExecutableConfigDir() };
    }

    std::string FormatPathList(const std::vector<std::filesystem::path>& paths)
    {
        std::ostringstream oss;
        oss << "[";
        bool first = true;
        for (const auto& path : paths) {
            if (!first) {
                oss << ", ";
            }
            oss << path.string();
            first = false;
        }
        oss << "]";
        return oss.str();
    }

    std::optional<std::filesystem::path> ResolveFromConfigSearchDirs(
        const std::filesystem::path& relative_path,
        const ConfigManagerOptions& options)
    {
        if (relative_path.empty() || !relative_path.is_relative()) {
            return std::nullopt;
        }

        for (const auto& config_dir : GetConfigSearchDirs(options)) {
            if (config_dir.empty()) {
                continue;
            }

            const auto candidate = (config_dir / relative_path).lexically_normal();
            if (!IsPathUnderDirectory(candidate, config_dir)) {
                continue;
            }
            if (PathExists(candidate)) {
                return candidate;
            }
        }

        return std::nullopt;
    }

    std::optional<std::filesystem::path> ResolveIncludePath(
        const std::filesystem::path& include_path,
        const std::filesystem::path& parent_dir,
        const ConfigManagerOptions& options)
    {
        if (include_path.empty()) {
            return std::nullopt;
        }
        if (include_path.is_absolute()) {
            return PathExists(include_path) ? std::optional<std::filesystem::path>(include_path) : std::nullopt;
        }

        const auto parent_candidate = (parent_dir / include_path).lexically_normal();
        if (PathExists(parent_candidate)) {
            return parent_candidate;
        }

        return ResolveFromConfigSearchDirs(include_path, options);
    }

    std::filesystem::path ResolveMainConfigPath(
        const std::filesystem::path& file_path,
        const ConfigManagerOptions& options)
    {
        if (file_path.empty()) {
            ANET_SYSTEM_ERROR("ConfigManager: config file path must not be empty.");
            return {};
        }

        if (file_path.is_absolute()) {
            if (PathExists(file_path)) {
                return file_path;
            }
            ANET_SYSTEM_ERROR("ConfigManager: config file not found. path=" << file_path.string());
            return {};
        }

        const auto cwd_candidate = NormalizeAbsolutePath(file_path);
        if (PathExists(cwd_candidate)) {
            return cwd_candidate;
        }

        if (const auto resolved_path = ResolveFromConfigSearchDirs(file_path, options)) {
            return *resolved_path;
        }

        ANET_SYSTEM_ERROR(
            "ConfigManager: config file not found. path=" << file_path.string()
            << " cwd_candidate=" << cwd_candidate.string()
            << " config_search_dirs=" << FormatPathList(GetConfigSearchDirs(options)));
        return {};
    }

    } // namespace

    // train.eval.[greedy].interval = 100
    // train.eval.[greedy].run_mode = 1
    // train.eval.[greedy].env.init.x_range = 0.0
    //   prefix = "train.eval"
    //   key_prefix = "train.eval.["
    //   key_suffix = "]"
    //   tag = greedy
	//   sub_key = interval, run_mode, env.init.x_range
	//   value = 100, 1, 0.0

    bool ConfigData::Read(const std::string& key, std::string& value, const std::string& value_if_missing) const
    {
        auto it = map_.find(key);
        if (it == map_.end()) { value = value_if_missing; return false; }
        value = (*it).second;
        return true;
    }

    bool ConfigData::Read(const std::string& key, int& value, int value_if_missing) const
    {
        auto it = map_.find(key);
        if (it == map_.end()) { value = value_if_missing; return false; }
        value = ParseNumericConfigValue<int>(
            key, (*it).second, "int",
            [](const std::string& text, size_t* pos) { return std::stoi(text, pos); });
        return true;
    }

    bool ConfigData::Read(const std::string& key, float& value, float value_if_missing) const
    {
        auto it = map_.find(key);
        if (it == map_.end()) { value = value_if_missing; return false; }
        value = ParseNumericConfigValue<float>(
            key, (*it).second, "float",
            [](const std::string& text, size_t* pos) { return std::stof(text, pos); });
        return true;
    }

    bool ConfigData::Read(const std::string& key, double& value, double value_if_missing) const
    {
        auto it = map_.find(key);
        if (it == map_.end()) { value = value_if_missing; return false; }
        value = ParseNumericConfigValue<double>(
            key, (*it).second, "double",
            [](const std::string& text, size_t* pos) { return std::stod(text, pos); });
        return true;
    }

    bool ConfigData::Read(const std::string& key, uint64_t& value, uint64_t value_if_missing) const
    {
        auto it = map_.find(key);
        if (it == map_.end()) { value = value_if_missing; return false; }
        value = ParseNumericConfigValue<uint64_t>(
            key, (*it).second, "uint64_t",
            [](const std::string& text, size_t* pos) { return std::stoull(text, pos); });
        return true;
    }

    bool ConfigData::Read(const std::string& key, int64_t& value, int64_t value_if_missing) const
    {
        auto it = map_.find(key);
        if (it == map_.end()) { value = value_if_missing; return false; }
        value = ParseNumericConfigValue<int64_t>(
            key, (*it).second, "int64_t",
            [](const std::string& text, size_t* pos) { return std::stoll(text, pos); });
        return true;
    }

    bool ConfigData::Read(const std::string& key, bool& value, bool value_if_missing) const
    {
        auto it = map_.find(key);
        if (it == map_.end()) { value = value_if_missing; return false; }
        const auto v = TrimConfigValue((*it).second);
        if (v == "true" || v == "TRUE" || v == "1" || v == "yes" || v == "on") { value = true; return true; }
        if (v == "false" || v == "FALSE" || v == "0" || v == "no" || v == "off") { value = false; return true; }
        ThrowReadFailure(key, (*it).second, "bool");
    }

    bool ConfigData::Read(const std::string& key, std::vector<float>& value, std::vector<float> value_if_missing) const
    {
        auto it = map_.find(key);
        if (it == map_.end()) { value = value_if_missing; return false; }
        auto str_vec = anet::Split((*it).second, { " ", "　" }, true);
        std::vector<float> parsed_value;
        parsed_value.reserve(str_vec.size());
        for (const auto& token : str_vec) {
            parsed_value.push_back(ParseNumericConfigValue<float>(
                key, (*it).second, token, "float vector",
                [](const std::string& text, size_t* pos) { return std::stof(text, pos); }));
        }
        value = std::move(parsed_value);
        return true;
    }

    bool ConfigData::Read(const std::string& key, std::vector<int64_t>& value, std::vector<int64_t> value_if_missing) const
    {
        auto it = map_.find(key);
        if (it == map_.end()) { value = value_if_missing; return false; }
        auto str_vec = anet::Split((*it).second, { " ", "　" }, true);
        std::vector<int64_t> parsed_value;
        parsed_value.reserve(str_vec.size());
        for (const auto& token : str_vec) {
            parsed_value.push_back(ParseNumericConfigValue<int64_t>(
                key, (*it).second, token, "int64_t vector",
                [](const std::string& text, size_t* pos) { return std::stoll(text, pos); }));
        }
        value = std::move(parsed_value);
        return true;
    }

    bool ConfigData::Read(const std::string& key, std::vector<std::string>& value, std::vector<std::string> value_if_missing) const
    {
        auto it = map_.find(key);
        if (it == map_.end()) { value = value_if_missing; return false; }
        value = anet::Split((*it).second, { " ", "　" }, true);
        return true;
    }

    std::unordered_map<std::string, ConfigData> ConfigData::MakeSubConfigData(const std::string& prefix) const
    {
        //std::unordered_map<std::string, std::unordered_map<std::string, std::string>> tag_block_map;
        std::unordered_map<std::string, ConfigData> tag_sub_config;

        auto config_map = Map();
        for (const auto& kv : config_map) {
            const std::string& config_key = kv.first;
            const std::string& config_value = kv.second;

            // 設定Keyからtagを抽出
			const std::string key_prefix = prefix + ".[";
            const std::string key_suffix = "]";
            auto config_tag = anet::ExtractBetween(config_key, key_prefix.c_str(), key_suffix.c_str());
            if (config_tag.empty()) {
                continue;
            }

            // サブキーを抽出
            auto pos = config_key.find(key_suffix);
            auto size = config_key.size();
            if ((pos + 2) >= config_key.size()) continue;
            auto config_sub_key = config_key.substr(pos + 2);
            if (config_sub_key.empty()) continue;

            // 保存
            tag_sub_config[config_tag].Set(config_sub_key, config_value);
        }

		return tag_sub_config;
    }

    std::unordered_set<std::string> ConfigData::GetSubConfigTags(const std::string& prefix) const
    {
        std::unordered_set<std::string> tags;

        auto config_map = Map();
        for (const auto& kv : config_map) {
            const std::string& config_key = kv.first;
            const std::string& config_value = kv.second;

            // 設定Keyからtagを抽出
            const std::string key_prefix = prefix + ".[";
            const std::string key_suffix = "]";
            auto config_tag = anet::ExtractBetween(config_key, key_prefix.c_str(), key_suffix.c_str());
            if (!config_tag.empty()) {
                tags.insert(config_tag);
            }
        }
        return tags;
    }

    std::string Properties::Trim(const std::string& s)
    {
        const char* ws = " \t\r\n";
        size_t b = s.find_first_not_of(ws);
        if (b == std::string::npos) return "";
        size_t e = s.find_last_not_of(ws);
        return s.substr(b, e - b + 1);
    }

    void Properties::Load(const std::filesystem::path& filename, int depth)
    {
        if (depth >= 10) {
            ANET_SYSTEM_ERROR("Properties: include depth limit exceeded. max=10");
        }

        std::ifstream ifs(filename);
        if (!ifs) {
            // ファイルが見つからない場合は警告のみで続行
            LOG::warn() << "Properties: Failed to open file: " << filename.string();
            return;
        }

        // BOM スキップ
        char bom[3] = { 0 };
        ifs.read(bom, 3);
        if (!(bom[0] == '\xEF' && bom[1] == '\xBB' && bom[2] == '\xBF')) {
            // BOM がなければ読み取り位置を戻す
            ifs.seekg(0);
        }

        // カレントファイルのディレクトリを取得（相対パス解決用）
        std::filesystem::path parent_dir = filename.parent_path();

        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue; // コメント行スキップ

            size_t pos_hash = line.find('#');   // '#' 以降はコメント扱い
            if (pos_hash != std::string::npos)
                line = line.substr(0, pos_hash);

            size_t pos_slash = line.find("//");  // '//' 以降はコメント扱い
            if (pos_slash != std::string::npos)
                line = line.substr(0, pos_slash);

            line = Trim(line);  // 前後の空白除去
            if (line.empty()) continue;

            // ---- $include 処理 ----
            // $とincludeの間にはスペースが入る可能性があるため、先頭から解析
            if (line[0] == '$') {
                std::string s = line.substr(1); // '$'除去

                // 先頭の空白スキップ
                size_t b = s.find_first_not_of(" \t");
                if (b != std::string::npos) {
                    s = s.substr(b);
                    // "include" (case insensitive) チェック
                    if (s.size() >= 7) {
                        std::string prefix = s.substr(0, 7);
                        std::transform(prefix.begin(), prefix.end(), prefix.begin(), ::tolower);

                        if (prefix == "include") {
                            // include以降のパス部分を抽出
                            std::string path_part = s.substr(7);
                            path_part = Trim(path_part); // 前後の空白除去

                            if (path_part.empty()) {
                                LOG::error() << "Properties: Empty include path in " << filename.string();
                                continue;
                            }

                            // 囲み文字処理
                            char start_char = path_part.front();
                            char end_char = path_part.back();

                            if (start_char == '<' || start_char == '[' || start_char == '(' || start_char == '"') {
                                char expected_end = 0;
                                if (start_char == '<') expected_end = '>';
                                else if (start_char == '[') expected_end = ']';
                                else if (start_char == '(') expected_end = ')';
                                else if (start_char == '"') expected_end = '"';

                                if (end_char == expected_end) {
                                    // 囲み文字を除去
                                    path_part = path_part.substr(1, path_part.size() - 2);
                                    path_part = Trim(path_part); // 中身もトリム
                                } else {
                                    // 囲み文字が閉じられていない場合はエラー
                                    LOG::error() << "Properties: Mismatched include brackets in " << filename.string() << ": " << line;
                                    continue;
                                }
                            }

                            if (path_part.empty()) {
                                LOG::error() << "Properties: Empty filename after parsing include in " << filename.string();
                                continue;
                            }

                            // include元ファイル基準を優先し、見つからない場合だけconfig search dirを試す。
                            const std::filesystem::path include_path = path_part;
                            const auto resolved_include_path = ResolveIncludePath(include_path, parent_dir, options_);
                            if (!resolved_include_path.has_value()) {
                                LOG::warn() << "Properties: Failed to open include file. path=" << include_path.string()
                                    << " from=" << filename.string()
                                    << " config_search_dirs=" << FormatPathList(GetConfigSearchDirs(options_));
                                continue;
                            }

                            // 再帰読み込み
                            Load(*resolved_include_path, depth + 1);
                            continue; // 次の行へ
                        }
                    }
                }
            }
            // ---------------------

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

    Config::Config(const std::string& default_prefix)
        : default_prefix_(default_prefix)
    {
    }

    Config::Config(const ConfigData& config_data, const std::string& default_prefix)
        : default_prefix_(default_prefix)
    {
    }

    Config::Config(const ConfigData& config_data, const std::string& default_prefix, const std::string& override_prefix)
        : default_prefix_(default_prefix)
        , override_prefix_(override_prefix)
    {
        ;
    }

    std::string Config::ToString() const
    {
        return ToJson().dump(2);
    }

    std::string Config::ToConfigString() const
    {
        std::ostringstream oss;

        auto json = round_numbers(my_config_json_);
        for (auto kv : my_config_data_.Map()) {
            auto key = kv.first;
            auto value = json[key];
            if (value.is_array()) {
                oss << default_prefix_ << "." << key << " =";
                for (auto& v : value) {
                    if (v.is_string()) {
                        oss << " " << v.get<std::string>();
                    } else {
                        oss << " " << v;
                    }
                }
                oss << std::endl;

                //    json arr = json::array();
                //for (auto& v : j) arr.push_back(round_numbers(v, precision));
            } else if (value.is_string()) {
                oss << default_prefix_ << "." << key << " = " << value.get<std::string>() << std::endl;
            } else {
                oss << default_prefix_ << "." << key << " = " << value << std::endl;
            }
        }

        return oss.str();
    }

    // ---- ConfigManager ----

    ConfigManager::ConfigManager(
        const std::string& filePath,
        const wxCmdLineParser* cmdLine,
        ConfigManagerOptions options)
        : options_(options)
    {
        // ベース読み込み
        LoadFromFile(filePath);

        // コマンドライン引数を上書き
        if (cmdLine) {
            ApplyCmdLineOverrides(*cmdLine);
        }

        // マージ実行（.$ = を展開）
        AutoMerge();

		// 再度コマンドライン引数を上書き（マージされた値よりコマンドライン指定を優先させるため）
        if (cmdLine) {
            ApplyCmdLineOverrides(*cmdLine);
        }
    }

    void ConfigManager::LoadFromFile(const std::string& filePath)
    {
        const auto resolved_path = ResolveMainConfigPath(filePath, options_);
        Properties props(resolved_path.string(), options_);
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

                // 設定階層の子孫だけを対象にし、env.common_value のような単なる前方一致を除外する。
                const std::string merge_target_prefix = merge_target_key + ".";

                for (const auto& kv2 : map) {
                    std::string key2 = kv2.first;
                    std::string val2 = kv2.second;
                    if (anet::StartsWith(key2, merge_target_prefix)) {                 // env.common.yyy, env.common.zzz
                        // ERASE: env.common.yyy, env.common.zzz
                        //new_map.Erase(key2);	// 2回目のマージで困るので消さない

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
