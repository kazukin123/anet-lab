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

