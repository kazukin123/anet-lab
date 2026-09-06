#include "config_impl.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "anet/diag.hpp"
#include "anet/json_util.hpp"
#include "anet/str_util.hpp"

using namespace anet;
using namespace anet::detail;

class anet::detail::ResolutionEngine {
public:
    ResolutionEngine(
        const ConfigData::MapType& source_map,
        const ConfigData::MapType& cli_overrides)
        : working_map_(source_map), cli_overrides_(cli_overrides)
    {
        // 全 CLI override を解決入力へ先出しし、源プレフィクス形も selection に渡す。
        for (const auto& [key, value] : cli_overrides_) {
            working_map_.Set(key, value);
        }
    }

    ConfigResolverResult Resolve()
    {
        // CLI 第1相を反映済みの幹を、通常 selection の入力へ先に展開する。
        ExpandNamedTrunk();

        std::vector<std::string> root_selection_keys;

        // 未選択素材を dormant のまま保持し、実効 map にはデフォルト直書きだけを置く。
        for (const auto& [key, value] : working_map_) {
            if (EndsWith(key, kSelectionSuffix)) {
                if (key != kNamedTrunkSelectionKey && !HasMaterialSegment(key)) {
                    root_selection_keys.push_back(key);
                }
            } else if (!HasMaterialSegment(key)) {
                effective_map_.Set(key, value);
            }
        }

        // source 上で有効な selection を宣言順に解決する。
        for (const auto& selection_key : root_selection_keys) {
            std::vector<std::string> path;
            ResolveSelection(
                selection_key,
                working_map_.Get(selection_key),
                selection_key,
                path,
                1);
        }

        // 実効 leaf は第1相に重ねて再適用し、selection と幹の産物より後に勝たせる。
        for (const auto& [key, value] : cli_overrides_) {
            if (!IsResolverInputKey(key)) {
                effective_map_.Set(key, value);
            }
        }

        // leaf override 後の最終値を参照し、値同期 token を 1 段だけ展開する。
        ExpandReferences();

        return {
            .effective_map = std::move(effective_map_),
            .resolution_json = {
                { "schema_version", 1 },
                { "selections", std::move(selections_) },
                { "references", std::move(references_) },
            },
        };
    }

private:
    static constexpr const char* kSelectionSuffix = ".$";
    static constexpr const char* kNamedTrunkSelectionKey = "run.$";
    static constexpr const char* kNamedTrunkOwner = "run";
    static constexpr int kMaxSelectionDepth = 10;

    static bool HasMaterialSegment(const std::string& key)
    {
        for (const auto& segment : Split(key, { "." }, false)) {
            if (!segment.empty() && segment.front() == '@') {
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
        return IsRelativeMaterialTerm(term) ? owner + "." + term : term;
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

    void ResolveSelection(
        const std::string& selection_key,
        const std::string& selection_value,
        const std::string& declaration_key,
        std::vector<std::string>& path,
        int depth)
    {
        if (depth > kMaxSelectionDepth) {
            ANET_SYSTEM_ERROR(
                "ConfigResolver: selection depth limit exceeded. max=" << kMaxSelectionDepth
                << " path=" << FormatPath(path, declaration_key));
        }
        if (std::find(path.begin(), path.end(), declaration_key) != path.end()) {
            ANET_SYSTEM_ERROR(
                "ConfigResolver: selection cycle detected. path="
                << FormatPath(path, declaration_key));
        }

        path.push_back(declaration_key);
        const auto owner = RemoveSuffix(selection_key, kSelectionSuffix);
        const auto terms = Split(selection_value, { ">" }, true);
        selections_.push_back({
            { "key", selection_key },
            { "chain", MakeSelectionChain(owner, terms) },
        });

        // 各項の direct leaf を反映してから、その項が生成した nested selection を解決する。
        for (const auto& term : terms) {
            if (!term.empty()) {
                ApplyTerm(selection_key, owner, term, path, depth);
            }
        }
        path.pop_back();
    }

    void ApplyTerm(
        const std::string& selection_key,
        const std::string& owner,
        const std::string& term,
        std::vector<std::string>& path,
        int depth)
    {
        const auto resolved = ResolveSelectionTerm(owner, term);
        const auto source_prefix = resolved + ".";
        std::vector<std::pair<std::string, std::string>> source_entries;
        for (const auto& [source_key, source_value] : working_map_) {
            if (StartsWith(source_key, source_prefix)) {
                source_entries.emplace_back(source_key, source_value);
            }
        }

        if (source_entries.empty() && (IsRelativeMaterialTerm(term) || HasMaterialSegment(resolved))) {
            ANET_SYSTEM_ERROR(
                "ConfigResolver: material selection target not found. selection="
                << selection_key << " term=" << term << " resolved=" << resolved
                << " scope=" << owner);
        }

        std::vector<std::tuple<std::string, std::string, std::string>> nested_selections;
        for (const auto& [source_key, source_value] : source_entries) {
            const auto suffix = RemovePrefix(source_key, resolved);
            const auto target_key = owner + suffix;
            working_map_.Set(target_key, source_value);

            if (EndsWith(target_key, kSelectionSuffix)) {
                if (!HasMaterialSegment(target_key)) {
                    nested_selections.emplace_back(target_key, source_value, source_key);
                }
            } else if (!HasMaterialSegment(target_key)) {
                effective_map_.Set(target_key, source_value);
            }
        }

        for (const auto& [nested_key, nested_value, declaration_key] : nested_selections) {
            ResolveSelection(
                nested_key,
                nested_value,
                declaration_key,
                path,
                depth + 1);
        }
    }

    ConfigData::MapType working_map_;
    ConfigData::MapType cli_overrides_;
    ConfigData::MapType effective_map_;
    json selections_ = json::array();
    json references_ = json::array();
};

ConfigResolverResult ConfigResolver::Resolve(
    const ConfigData::MapType& source_map,
    const ConfigData::MapType& cli_overrides)
{
    return ResolutionEngine(source_map, cli_overrides).Resolve();
}
