#include "anet/str_util.hpp"

using namespace anet;

std::vector<std::string> anet::Split(
    const std::string& s, const std::vector<std::string>& delimiters, bool trim)
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

std::string anet::ExtractBetween(const std::string& src, const char* prefix, const char* suffix)
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

std::string anet::ReplaceAll(std::string str, const std::string& from, const std::string& to)
{
    if (from.empty()) return str; // 検索対象が空ならそのまま返す

    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

// 3桁区切りのためのカスタムファセット
struct comma_numpunct : std::numpunct<char> {
    virtual char do_thousands_sep() const { return ','; } // 区切り文字
    virtual std::string do_grouping() const { return "\3"; } // 3桁ごと
};

std::string anet::FormatWithCommas(uint64_t value)
{
    std::stringstream ss;
    // カンマ区切りのロケールをストリームに注入
    ss.imbue(std::locale(std::locale(), new comma_numpunct));
    ss << value;

    return ss.str();
}

