// anet/diag.hpp

#pragma once

#include <sstream>
#include <string>

#if ANET_ENABLE_DEBUGINFO

#ifndef ANET_ENABLE_ASSERT
#define ANET_ENABLE_ASSERT 1
#endif

#else

#ifndef ANET_ENABLE_ASSERT
#define ANET_ENABLE_ASSERT 0
#endif

#endif


#if ANET_ENABLE_ASSERT

#define ANET_ASSERT(cond)                                                           \
    do {                                                                            \
        if (!(cond)) {                                                              \
            anet::ThrowError(__FILE__, __LINE__, "ANET_ASSERT failed: ", #cond);    \
        }                                                                           \
    } while (0)

#define ANET_ASSERT_MSG(cond, stream_args)                                          \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::ostringstream ss;                                                  \
            ss << stream_args;                                                      \
            anet::ThrowError(__FILE__, __LINE__, "ANET_ASSERT failed: ", #cond, ss.str());  \
        }                                                                           \
    } while (0)

#else
#define ANET_ASSERT(cond) do {} while (0)
#define ANET_ASSERT_MSG(cond, msg) do {} while (0)
#endif


#define ANET_CHECK(cond)                                                            \
    do {                                                                            \
        if (!(cond)) {                                                              \
            anet::ThrowError(__FILE__, __LINE__, "ANET_CHECK failed: ", #cond);     \
        }                                                                           \
    } while (0)

#define ANET_CHECK_MSG(cond, stream_args)                                           \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::ostringstream ss;                                                  \
            ss << stream_args;                                                      \
            anet::ThrowError(__FILE__, __LINE__, "ANET_CHECK failed: ", #cond, ss.str());  \
        }                                                                           \
    } while (0)

#define ANET_SYSTEM_ERROR(stream_args)                                              \
    do {                                                                            \
        std::ostringstream ss;                                                      \
        ss << stream_args;                                                          \
        anet::ThrowError(__FILE__, __LINE__, "System error: ", nullptr, ss.str());  \
    } while (0)

namespace anet {

    void ThrowError(const char* file, int line, const char* prefix, const char* cond, const std::string& msg = "");

} // namespace anet
