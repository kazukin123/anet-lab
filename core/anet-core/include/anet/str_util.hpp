// anet/str_util.hpp
#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

namespace anet {

    // ----- trim -----
    inline std::string TrimCopy(const std::string & s) {
        auto begin = std::find_if_not(s.begin(), s.end(), ::isspace);
        auto end = std::find_if_not(s.rbegin(), s.rend(), ::isspace).base();
        if (begin >= end) return "";
        return std::string(begin, end);
    }

    // ----- find: 複数 delimiter のうち最小位置のものを探す -----
    inline std::pair<size_t, size_t> FindDelim(const std::string & s,
        size_t start,
        const std::vector<std::string>&delims)
    {
        size_t pos = std::string::npos;
        size_t len = 0;

        for (const auto& d : delims) {
            size_t p = s.find(d, start);
            if (p != std::string::npos && (pos == std::string::npos || p < pos)) {
                pos = p;
                len = d.length();
            }
        }
        return { pos, len };    // 見つかった位置と、その長さ
    }

    // ----- split -----
    inline std::vector<std::string> Split(const std::string & s,
        const std::vector<std::string>&delimiters,
        bool trim)
    {
        std::vector<std::string> result;
        size_t start = 0;

        while (true) {
            auto [pos, len] = FindDelim(s, start, delimiters);
            if (pos == std::string::npos) break;  // もう無い

            std::string token = s.substr(start, pos - start);
            std::string trimmed_token = trim ? TrimCopy(token) : token;
            if (!trimmed_token.empty()) result.push_back(trimmed_token);
            start = pos + len;
        }

        // 最後のトークン
        std::string token = s.substr(start);
        std::string trimmed_token = trim ? TrimCopy(token) : token;
        if (!trimmed_token.empty()) result.push_back(trimmed_token);

        return result;
    }

    inline bool StartsWith(const std::string& str, const std::string& prefix)
    {
        return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
    }

    inline bool StartsWith(const std::string& str, const char* prefix)
    {
        return StartsWith(str, std::string(prefix));
    }

    inline bool EndsWith(const std::string& str, const std::string& suffix)
    {
        if (str.size() < suffix.size()) return false;
        return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    inline bool EndsWith(const std::string& str, const char* suffix)
    {
        return EndsWith(str, std::string(suffix));
    }

    inline std::string RemoveSuffix(const std::string& str, const std::string& keyword)
    {
        if (EndsWith(str, keyword)) {
            return str.substr(0, str.size() - keyword.size());
        }
        return str;
    }

    inline std::string RemovePrefix(const std::string& a, const std::string& b)
    {
        if (a.rfind(b, 0) == 0) { // a の先頭に b がある?
            return a.substr(b.size());
        }
        return a; // 先頭にない場合はそのまま
    }

    inline std::string ExtractBetween(const std::string& src, const char* prefix, const char* suffix)
    {
        if (!prefix || !suffix) return "";

        std::string pre(prefix);
        std::string suf(suffix);

        size_t start = src.find(pre);
        if (start == std::string::npos) return "";

        start += pre.length();
        size_t end = src.find(suf, start);
        if (end == std::string::npos) return "";

        return src.substr(start, end - start);
    }
}
