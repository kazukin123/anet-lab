#include "anet/diag.hpp"

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
