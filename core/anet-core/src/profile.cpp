#include "anet/profile.hpp"
#include <cstdint>
#include <nvtx3/nvtx3.hpp>


using namespace anet;

ProfileRange::ProfileRange(const char* name)
    : active_(true)
{
    nvtxRangePushA(name);
}

ProfileRange::ProfileRange(const char* name, int idx)
    : active_(true)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "%s[%d]", name, idx);
    nvtxRangePushA(buf);
}

ProfileRange::ProfileRange(const char* name, ProfileRange& previous)
    : active_(true)
{
    previous.End();
    nvtxRangePushA(name);
}

ProfileRange::ProfileRange(const char* name, int idx, ProfileRange& previous)
    : active_(true)
{
    previous.End();

    char buf[128];
    snprintf(buf, sizeof(buf), "%s[%d]", name, idx);
    nvtxRangePushA(buf);
}


void ProfileRange::End() {
    if (active_) {
        nvtxRangePop();
        active_ = false;
    }
}

ProfileRange::~ProfileRange() {
    if (active_) {
        nvtxRangePop();
    }
}

ProfileRange::ProfileRange(ProfileRange&& other) noexcept
    : active_(other.active_)
{
    other.active_ = false; // 移動元のPOPは無効化
}

ProfileRange& ProfileRange::operator=(ProfileRange&& other) noexcept
{
    if (this != &other) {
        if (active_) {
            nvtxRangePop();
        }
        active_ = other.active_;
        other.active_ = false;
    }
    return *this;
}

// --------------

#ifdef _WIN32
#include <windows.h>
static inline uint32_t GetOsThreadId() {
    return ::GetCurrentThreadId();
}
#else
#include <sys/syscall.h>
#include <unistd.h>
static inline uint32_t GetOsThreadId() {
    return static_cast<uint32_t>(::syscall(SYS_gettid));
}
#endif

// --------------

ProfileThreadName::ProfileThreadName(const char* name)
{
    uint32_t tid = GetOsThreadId();
    snprintf(buf_, sizeof(buf_), "%s", name);
    nvtxNameOsThreadA(tid, buf_);
}

ProfileThreadName::ProfileThreadName(const char* base, int idx)
{
    uint32_t tid = GetOsThreadId();

    // base + "_" + idx を stack で構築
    snprintf(buf_, sizeof(buf_), "%s_%d", base, idx);

    nvtxNameOsThreadA(tid, buf_);
}

