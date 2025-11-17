

#include <sstream>
#include <stdexcept>

#ifndef ANET_ENABLE_ASSERT
#define ANET_ENABLE_ASSERT 1
#endif


#if ANET_ENABLE_ASSERT
#define ANET_ASSERT(cond)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::stringstream ss;                                              \
            ss << "ANET_ASSERT failed: "                                       \
               << " | Condition: " << #cond                                    \
               << " | File: " << __FILE__ << ":" << __LINE__;                  \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (0)
#define ANET_ASSERT_MSG(cond, msg)                                             \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::stringstream ss;                                              \
            ss << "ANET_ASSERT failed: " << (msg)                              \
               << " | Condition: " << #cond                                    \
               << " | File: " << __FILE__ << ":" << __LINE__;                  \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while (0)
#else
#define ANET_ASSERT(cond, msg) do {} while (0)
#define ANET_ASSERT_MSG(cond, msg) do {} while (0)
#endif

