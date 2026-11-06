#include "config_impl.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "anet/diag.hpp"
#include "anet/json_util.hpp"
#include "anet/profile.hpp"
#include "anet/str_util.hpp"

using namespace anet;
using namespace anet::detail;

class anet::detail::ResolutionEngine {
public:
    ResolutionEngine(const ConfigData::MapType& source_map, const ConfigData::MapType& cli_overrides,
        const std::unordered_set<std::string>& default_keys)
        : working_map_(source_map), cli_overrides_(cli_overrides), default_keys_(default_keys)
    {
        // Runの選択と、Run素材自体へのCLI指定を先に確定する。
        for (const auto& [key, value] : cli_overrides_) {
            working_map_.Set(key, value);
            default_keys_.erase(key);
        }
    }

    ConfigResolverResult Resolve()
    {
        ANET_PROFILE_SCOPE(resolve);
        const auto before_run = working_map_;
        const auto before_defaults = default_keys_;
        ExpandNamedTrunk();
        for (const auto& [key, value] : cli_overrides_) {
            working_map_.Set(key, value);
            default_keys_.erase(key);
        }
        input_map_ = working_map_;
        BuildDependencies();
        ValidateAndRecord();
        BuildValues();

        // Runが直接指定した葉だけを外し、同じ依存グラフで適用前の最終値を求める。
        // CLIが同じ葉を指定していてもtoはRun値、実効値はCLI値として区別する。
        for (const auto& [key, value] : trunk_leaves_) {
            auto previous_input = input_map_;
            auto previous_defaults = default_keys_;
            if (before_defaults.contains(key)) previous_defaults.insert(key);
            else previous_defaults.erase(key);
            if (before_run.Has(key)) {
                previous_input.Set(key, before_run.Get(key));
            } else {
                previous_input.Erase(key);
            }
            std::unordered_map<std::string, std::string> cache;
            std::vector<std::string> path;
            const auto previous = Evaluate(key, previous_input, previous_defaults, cache, path);
            if (previous != value) {
                overrides_.push_back({ { "key", key }, { "by", trunk_leaf_sources_.Get(key) },
                    { "from", previous }, { "to", value } });
            }
        }
        ExpandReferences();
        // 葉の配置移行と独立した、参照元キー順の診断にする。
        std::stable_sort(references_.begin(), references_.end(), [](const auto& left, const auto& right) {
            return left.at("source").template get<std::string>() < right.at("source").template get<std::string>();
        });
        return {
            .effective_map = std::move(effective_map_),
            .resolution_json = {
                { "schema_version", 1 }, { "selections", std::move(selections_) },
                { "references", std::move(references_) }, { "overrides", std::move(overrides_) },
            },
        };
    }

private:
    static constexpr const char* kSelectionSuffix = ".$";
    static constexpr const char* kNamedTrunkSelectionKey = "run.$";
    static constexpr const char* kNamedTrunkOwner = "run";
    static constexpr int kMaxSelectionDepth = 10;

    struct Declaration {
        std::string key;
        std::string owner;
        std::vector<std::string> terms;
        bool active = false;
    };

    struct Edge {
        size_t declaration;
        std::string source;
        std::string term;
    };

    bool IsSourceRegion(const Declaration& declaration, const std::string& prefix) const
    {
        return std::any_of(declaration.terms.begin(), declaration.terms.end(), [&](const auto& term) {
            const auto source = ResolveSelectionTerm(declaration.owner, term);
            return prefix == source || StartsWith(prefix, source + ".");
        });
    }

    void Activate(size_t index)
    {
        auto& declaration = declarations_[index];
        if (declaration.active) {
            return;
        }
        declaration.active = true;
        for (const auto& term : declaration.terms) {
            if (term.empty()) {
                continue;
            }
            const auto source = ResolveSelectionTerm(declaration.owner, term);
            if (declaration.owner == source || StartsWith(declaration.owner, source + ".")) {
                ANET_SYSTEM_ERROR("ConfigResolver: selection self-supply detected. selection=" << declaration.key
                    << " term=" << term << " resolved=" << source << " path=" << declaration.key << " -> " << source);
            }
            edges_.push_back({ .declaration = index, .source = source, .term = term });
        }
    }

    std::string DefinitionOrigin(const std::string& key) const
    {
        std::unordered_set<std::string> visited;
        auto current = key;
        while (!input_map_.Has(current) && value_sources_.Has(current)) {
            if (!visited.insert(current).second) {
                ANET_SYSTEM_ERROR("ConfigResolver: selection cycle in definition dependency. path=" << key << " -> " << current);
            }
            current = value_sources_.Get(current);
        }
        return current;
    }

    void BuildDependencies()
    {
        ANET_PROFILE_SCOPE(dependencies);
        // 宣言のidentityは常に入力の定義位置。コピー先に選択命令を新設しない。
        for (const auto& [key, value] : input_map_) {
            if (key == kNamedTrunkSelectionKey) {
                continue;
            }
            keys_.Set(key, value);
            if (EndsWith(key, kSelectionSuffix)) {
                declaration_indices_[key] = declarations_.size();
                declarations_.push_back({ .key = key, .owner = RemoveSuffix(key, kSelectionSuffix),
                    .terms = Split(value, { ">" }, true) });
            }
        }
        for (size_t i = 0; i < declarations_.size(); ++i) {
            if (!HasMaterialSegment(declarations_[i].key)) {
                Activate(i);
            }
        }

        // 最終キー集合と、参照された内側定義の有効化を収束まで求める。
        // 定義参照も元キーへの依存として運び、短い@の解釈は変更しない。
        std::unordered_map<std::string, std::vector<std::pair<size_t, std::string>>> growth_paths;
        bool changed;
        do {
            changed = false;
            const auto keys = keys_.Order();
            for (const auto& edge : edges_) {
                const auto& declaration = declarations_[edge.declaration];
                for (const auto& key : keys) {
                    if (!StartsWith(key, edge.source + ".")) {
                        continue;
                    }
                    const auto target = declaration.owner + RemovePrefix(key, edge.source);
                    if (EndsWith(key, kSelectionSuffix) && !HasMaterialSegment(target)) {
                        continue;
                    }
                    if (IsSourceRegion(declaration, target)) {
                        ANET_SYSTEM_ERROR("ConfigResolver: selection self-supply detected. selection=" << declaration.key
                            << " term=" << edge.term << " resolved=" << edge.source << " path=" << key << " -> " << target);
                    }
                    if (!keys_.Has(target)) {
                        auto path = growth_paths[key];
                        if (std::any_of(path.begin(), path.end(), [&](const auto& step) {
                            return step.first == edge.declaration
                                && EndsWith(key, "." + RemovePrefix(step.second, edge.source + "."));
                        })) {
                            ANET_SYSTEM_ERROR("ConfigResolver: selection cycle in key dependencies. selection=" << declaration.key
                                << " term=" << edge.term << " resolved=" << edge.source << " path=" << key << " -> " << target);
                        }
                        path.emplace_back(edge.declaration, key);
                        growth_paths[target] = std::move(path);
                        keys_.Set(target, "");
                        changed = true;
                    }
                }
            }
            // 同一場所では具体的なownerを優先し、同じownerではtermの右側を優先する。
            auto ordered = edges_;
            std::stable_sort(ordered.begin(), ordered.end(), [&](const auto& left, const auto& right) {
                const auto& a = declarations_[left.declaration].owner;
                const auto& b = declarations_[right.declaration].owner;
                return std::count(a.begin(), a.end(), '.') < std::count(b.begin(), b.end(), '.');
            });
            value_sources_ = {};
            for (const auto& edge : ordered) {
                for (const auto& [key, value] : keys_) {
                    if (!StartsWith(key, edge.source + ".")) {
                        continue;
                    }
                    const auto target = declarations_[edge.declaration].owner + RemovePrefix(key, edge.source);
                    if (keys_.Has(target) && (!EndsWith(key, kSelectionSuffix) || HasMaterialSegment(target))) {
                        value_sources_.Set(target, key);
                    }
                }
            }
            const auto requests = edges_;
            for (const auto& edge : requests) {
                for (const auto& [key, value] : keys_) {
                    if (StartsWith(key, edge.source + ".") && EndsWith(key, kSelectionSuffix)
                        && !HasMaterialSegment(RemovePrefix(key, edge.source + "."))) {
                        const auto origin = DefinitionOrigin(key);
                        const auto found = declaration_indices_.find(origin);
                        if (found != declaration_indices_.end() && !declarations_[found->second].active) {
                            Activate(found->second);
                            changed = true;
                        }
                    }
                }
            }
        } while (changed);
    }

    void VisitSource(const std::string& prefix, std::vector<std::string>& path, std::vector<size_t>& dependencies)
    {
        if (std::find(path.begin(), path.end(), prefix) != path.end()) {
            std::vector<std::string> selection_path;
            for (const auto index : dependencies) {
                selection_path.push_back(declarations_[index].key);
            }
            selection_path.push_back(prefix + kSelectionSuffix);
            ANET_SYSTEM_ERROR("ConfigResolver: selection cycle detected. path=" << FormatPath(selection_path, "")
                << " scopes=" << FormatPath(path, prefix));
        }
        path.push_back(prefix);
        // 要求された部分へ届く選択だけを辿り、兄弟の参照を循環へ混同しない。
        for (size_t i = 0; i < declarations_.size(); ++i) {
            const auto& declaration = declarations_[i];
            if (!declaration.active) {
                continue;
            }
            if (declaration.owner == prefix || (StartsWith(declaration.owner, prefix + ".")
                && !HasMaterialSegment(RemovePrefix(declaration.owner, prefix + ".")))) {
                VisitDeclaration(i, "", path, dependencies);
            } else if (StartsWith(prefix, declaration.owner + ".") && !IsSourceRegion(declaration, prefix)) {
                const auto suffix = RemovePrefix(prefix, declaration.owner);
                const bool contributes = std::any_of(declaration.terms.begin(), declaration.terms.end(), [&](const auto& term) {
                    const auto source = ResolveSelectionTerm(declaration.owner, term) + suffix + ".";
                    return std::any_of(keys_.begin(), keys_.end(), [&](const auto& entry) {
                        return StartsWith(entry.first, source);
                    });
                });
                if (contributes) {
                    VisitDeclaration(i, suffix, path, dependencies);
                }
            }
        }
        path.pop_back();
    }

    void VisitDeclaration(size_t index, const std::string& suffix,
        std::vector<std::string>& path, std::vector<size_t>& dependencies)
    {
        const auto& declaration = declarations_[index];
        if (recorded_.insert(index).second) {
            selections_.push_back({ { "key", declaration.key },
                { "chain", MakeSelectionChain(declaration.owner, declaration.terms) } });
        }
        const bool new_dependency = std::find(dependencies.begin(), dependencies.end(), index) == dependencies.end();
        if (new_dependency) {
            dependencies.push_back(index);
        }
        if (dependencies.size() > kMaxSelectionDepth) {
            std::vector<std::string> selection_path;
            for (const auto dependency : dependencies) {
                selection_path.push_back(declarations_[dependency].key);
            }
            ANET_SYSTEM_ERROR("ConfigResolver: selection depth limit exceeded. max=" << kMaxSelectionDepth
                << " selection=" << declaration.key << " term=" << input_map_.Get(declaration.key)
                << " resolved=" << MakeSelectionChain(declaration.owner, declaration.terms).dump()
                << " path=" << FormatPath(selection_path, ""));
        }
        for (const auto& term : declaration.terms) {
            if (!term.empty()) {
                const auto source = ResolveSelectionTerm(declaration.owner, term) + suffix;
                // 存在しない内側在庫を外側の選択へ戻して架空の依存を増殖させない。
                if (!suffix.empty() && !std::any_of(keys_.begin(), keys_.end(), [&](const auto& entry) {
                    return StartsWith(entry.first, source + ".");
                })) {
                    continue;
                }
                VisitSource(source, path, dependencies);
            }
        }
        if (new_dependency) {
            dependencies.pop_back();
        }
    }

    void ValidateAndRecord()
    {
        ANET_PROFILE_SCOPE(validate);
        std::vector<std::string> path;
        std::vector<size_t> dependencies;
        for (size_t i = 0; i < declarations_.size(); ++i) {
            if (!HasMaterialSegment(declarations_[i].key)) {
                VisitDeclaration(i, "", path, dependencies);
            }
        }
        for (const auto& edge : edges_) {
            if (!HasMaterialSegment(edge.source) && !HasCatalogSegment(edge.source)) {
                continue;
            }
            const bool defined = std::any_of(keys_.begin(), keys_.end(), [&](const auto& entry) {
                return StartsWith(entry.first, edge.source + ".");
            });
            if (!defined) {
                ANET_SYSTEM_ERROR("ConfigResolver: " << (HasMaterialSegment(edge.source) ? "material" : "catalog")
                    << " selection target not found. selection=" << declarations_[edge.declaration].key
                    << " term=" << edge.term << " resolved=" << edge.source << " scope=" << declarations_[edge.declaration].owner);
            }
        }
    }

    std::string Evaluate(const std::string& key, const ConfigData::MapType& input,
        const std::unordered_set<std::string>& defaults,
        std::unordered_map<std::string, std::string>& cache, std::vector<std::string>& path) const
    {
        if (const auto found = cache.find(key); found != cache.end()) {
            return found->second;
        }
        if (std::find(path.begin(), path.end(), key) != path.end()) {
            ANET_SYSTEM_ERROR("ConfigResolver: selection cycle detected. path=" << FormatPath(path, key));
        }
        path.push_back(key);
        // 既定葉だけはベースの後へ回す。選択元は自身の最終値を返す。
        const auto value = input.Has(key) && !defaults.contains(key) ? input.Get(key)
            : value_sources_.Has(key) ? Evaluate(value_sources_.Get(key), input, defaults, cache, path)
            : input.Has(key) ? input.Get(key) : std::string{};
        path.pop_back();
        cache.emplace(key, value);
        return value;
    }

    void BuildValues()
    {
        ANET_PROFILE_SCOPE(values);
        std::unordered_map<std::string, std::string> cache;
        std::vector<std::string> path;
        for (const auto& [key, value] : keys_) {
            if (EndsWith(key, kSelectionSuffix)) {
                continue;
            }
            const auto final_value = Evaluate(key, input_map_, default_keys_, cache, path);
            working_map_.Set(key, final_value);
            if (!HasMaterialSegment(key)) {
                effective_map_.Set(key, final_value);
            }
        }
    }

    static bool HasMaterialSegment(const std::string& key)
    {
        for (const auto& segment : Split(key, { "." }, false)) {
            if (!segment.empty() && segment.front() == '@') {
                return true;
            }
        }
        return false;
    }

    static bool HasCatalogSegment(const std::string& key)
    {
        for (const auto& segment : Split(key, { "." }, false)) {
            if (segment.size() >= 2 && segment.front() == '[' && segment.back() == ']') {
                return true;
            }
        }
        return false;
    }

    static bool IsRelativeMaterialTerm(const std::string& term)
    {
        return !term.empty() && term.front() == '@' && term.find('.') == std::string::npos;
    }

    static bool IsResolverInputKey(const std::string& key)
    {
        return EndsWith(key, kSelectionSuffix) || HasMaterialSegment(key);
    }

    static std::string ResolveSelectionTerm(const std::string& owner, const std::string& term)
    {
        if (!IsRelativeMaterialTerm(term)) {
            return term;
        }
        // プロファイル自身のベースは、兄弟プロファイルを定義元で参照する。
        const auto dot = owner.rfind('.');
        const auto segment = dot == std::string::npos ? owner : owner.substr(dot + 1);
        const auto scope = !segment.empty() && segment.front() == '@'
            ? (dot == std::string::npos ? std::string{} : owner.substr(0, dot)) : owner;
        return scope.empty() ? term : scope + "." + term;
    }

    static std::string FormatPath(const std::vector<std::string>& path, const std::string& tail)
    {
        std::ostringstream oss;
        for (const auto& item : path) {
            if (oss.tellp() > 0) {
                oss << " -> ";
            }
            oss << item;
        }
        if (!tail.empty()) {
            if (oss.tellp() > 0) {
                oss << " -> ";
            }
            oss << tail;
        }
        return oss.str();
    }

    static json MakeSelectionChain(
        const std::string& owner,
        const std::vector<std::string>& terms)
    {
        json chain = json::array();
        for (const auto& term : terms) {
            if (!term.empty()) {
                chain.push_back({
                    { "term", term },
                    { "resolved", ResolveSelectionTerm(owner, term) },
                });
            }
        }
        return chain;
    }

    void ExpandNamedTrunk()
    {
        if (!working_map_.Has(kNamedTrunkSelectionKey)) {
            return;
        }

        // 幹の選択経路を通常 selection と同じ形式で先頭に記録する。
        const auto terms = Split(working_map_.Get(kNamedTrunkSelectionKey), { ">" }, true);
        selections_.push_back({
            { "key", kNamedTrunkSelectionKey },
            { "chain", MakeSelectionChain(kNamedTrunkOwner, terms) },
        });

        // 幹素材の子をrootへ後書きし、通常 selection のスナップショットへ渡す。
        for (const auto& term : terms) {
            if (term.empty()) {
                continue;
            }
            const auto resolved = ResolveSelectionTerm(kNamedTrunkOwner, term);
            const auto source_prefix = resolved + ".";
            std::vector<std::pair<std::string, std::string>> source_entries;
            for (const auto& [source_key, source_value] : working_map_) {
                if (StartsWith(source_key, source_prefix)) {
                    source_entries.emplace_back(source_key, source_value);
                }
            }

            if (source_entries.empty()
                && (IsRelativeMaterialTerm(term) || HasMaterialSegment(resolved))) {
                ANET_SYSTEM_ERROR(
                    "ConfigResolver: material selection target not found. selection="
                    << kNamedTrunkSelectionKey << " term=" << term << " resolved=" << resolved
                    << " scope=" << kNamedTrunkOwner);
            }

            for (const auto& [source_key, source_value] : source_entries) {
                const auto target_key = RemovePrefix(source_key, source_prefix);
                if (target_key == kNamedTrunkSelectionKey) {
                    ANET_SYSTEM_ERROR(
                        "ConfigResolver: named trunk must not select another trunk. material="
                        << resolved << " key=" << source_key);
                }
                working_map_.Set(target_key, source_value);
                default_keys_.erase(target_key);
                if (!IsResolverInputKey(target_key)) {
                    trunk_leaves_.Set(target_key, source_value);
                    trunk_leaf_sources_.Set(target_key, resolved);
                }
            }
        }
    }

    void ExpandReferences()
    {
        // 展開前 snapshot を固定し、走査順によって2段参照が通らないようにする。
        auto reference_values = working_map_;
        std::vector<std::pair<std::string, std::string>> entries;
        for (const auto& [key, value] : effective_map_) {
            entries.emplace_back(key, value);
            reference_values.Set(key, value);
        }

        for (const auto& [source_key, source_value] : entries) {
            std::string expanded = source_value;
            size_t search_pos = 0;
            while (true) {
                const auto token_begin = expanded.find("${", search_pos);
                if (token_begin == std::string::npos) {
                    break;
                }
                const auto token_end = expanded.find('}', token_begin + 2);
                if (token_end == std::string::npos) {
                    ANET_SYSTEM_ERROR(
                        "ConfigResolver: unresolved value reference token. source="
                        << source_key << " value=" << expanded);
                }

                const auto target_key = expanded.substr(
                    token_begin + 2,
                    token_end - token_begin - 2);
                if (!reference_values.Has(target_key)) {
                    ANET_SYSTEM_ERROR(
                        "ConfigResolver: value reference target not found. source="
                        << source_key << " target=" << target_key);
                }
                const auto& target_value = reference_values.Get(target_key);
                if (target_value.find("${") != std::string::npos) {
                    ANET_SYSTEM_ERROR(
                        "ConfigResolver: chained value reference is not supported. source="
                        << source_key << " target=" << target_key
                        << " target_value=" << target_value);
                }

                expanded.replace(
                    token_begin,
                    token_end - token_begin + 1,
                    target_value);
                references_.push_back({
                    { "source", source_key },
                    { "target", target_key },
                    { "value", target_value },
                });
                search_pos = token_begin + target_value.size();
            }

            if (expanded.find("${") != std::string::npos) {
                ANET_SYSTEM_ERROR(
                    "ConfigResolver: unresolved value reference token. source="
                    << source_key << " value=" << expanded);
            }
            effective_map_.Set(source_key, expanded);
        }
    }

    ConfigData::MapType working_map_;
    ConfigData::MapType input_map_;
    ConfigData::MapType cli_overrides_;
    ConfigData::MapType effective_map_;
    ConfigData::MapType trunk_leaves_;
    ConfigData::MapType trunk_leaf_sources_;
    ConfigData::MapType keys_;
    ConfigData::MapType value_sources_;
    std::unordered_set<std::string> default_keys_;
    std::vector<Declaration> declarations_;
    std::vector<Edge> edges_;
    std::unordered_map<std::string, size_t> declaration_indices_;
    std::unordered_set<size_t> recorded_;
    json selections_ = json::array();
    json references_ = json::array();
    json overrides_ = json::array();
};

ConfigResolverResult ConfigResolver::Resolve(
    const ConfigData::MapType& source_map,
    const ConfigData::MapType& cli_overrides,
    const std::unordered_set<std::string>& default_keys)
{
    return ResolutionEngine(source_map, cli_overrides, default_keys).Resolve();
}
