#include "anet/common.hpp"
#include <stdexcept>
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


// =============================================================
// TensorSpec
// =============================================================

static std::string SpaceTypeToString(SpaceType type)
{
    switch (type) {
    case SpaceType::Vector: return "Vector";
    case SpaceType::Grid: return "Grid";
    case SpaceType::Sequence: return "Sequence";
    default: return "Unknown";
    }
}

anet::json TensorSpec::ToJson() const
{
    anet::json j;
    j["type"] = SpaceTypeToString(type);
    j["shape"] = shape;
    j["dtype"] = std::string(c10::toString(dtype)); /// @todo LibTorchの型名変換を利用
    j["num_classes"] = num_classes;
    j["labels"] = labels;
    j["min_values"] = min_values;
    j["max_values"] = max_values;
    return j;
}

std::string TensorSpec::ToString() const
{
    return ToJson().dump(2); // 2-space indent for pretty print
}
