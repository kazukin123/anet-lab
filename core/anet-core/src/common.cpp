#include "anet/common.hpp"
#include <stdexcept>
#include <sstream>
#include "anet/exception.hpp"

void anet::ThrowError(const char* file, int line, const char* prefix, const char* cond, const std::string& msg)
{
    std::ostringstream ss;
    ss << prefix;
    if (cond != nullptr)
        ss << "  " << cond;
    ss << "\n  " << msg;
    ss << "\n\n[" << __FILE__ << ":" << __LINE__ << "]";
    throw anet::AnetException(ss.str());
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