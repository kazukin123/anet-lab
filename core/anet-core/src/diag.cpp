#include "anet/diag.hpp"

#include <cmath>
#include <sstream>
#include "anet/exception.hpp"
#include "anet/log.hpp"

using namespace anet;
namespace LOG = anet::log;

void anet::ThrowError(const char* file, int line, const char* prefix, const char* cond, const std::string& msg)
{
    // 例外情報に含めるmessageを作る
    std::ostringstream ss;
    ss << prefix;
    if (cond != nullptr)
        ss << "  " << cond;
    ss << "\n  " << msg;
    ss << "\n\n[" << file << ":" << line << "]";

    // ログ出力
    auto ex_msg = ss.str();
    LOG::fatal() << "Exception: " << ex_msg;

    // 例外を作って投げる
    throw anet::AnetException(ex_msg);
}

anet::json anet::round_numbers(const anet::json& j, int precision)
{
    if (j.is_number_float()) {
        double val = j.get<double>();
        double scale = std::pow(10.0, precision);
        return std::round(val * scale) / scale;
    } else if (j.is_object()) {
        json res;
        for (auto& [k, v] : j.items()) res[k] = round_numbers(v, precision);
        return res;
    } else if (j.is_array()) {
        json arr = json::array();
        for (auto& v : j) arr.push_back(round_numbers(v, precision));
        return arr;
    }
    return j;
}
