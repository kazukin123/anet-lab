#include "anet/profile.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef ANET_ENABLE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

using namespace anet;

namespace {

constexpr bool HasProfileBackend()
{
#if defined(ANET_ENABLE_TRACY) || defined(ANET_ENABLE_NVTX)
    return true;
#else
    return false;
#endif
}

void PushNvtxRange(const char* name)
{
#ifdef ANET_ENABLE_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
}

void PopNvtxRange()
{
#ifdef ANET_ENABLE_NVTX
    nvtxRangePop();
#endif
}

void NameNvtxThread(uint32_t tid, const char* name)
{
#ifdef ANET_ENABLE_NVTX
    nvtxNameOsThreadA(tid, name);
#else
    (void)tid;
    (void)name;
#endif
}

#ifdef _WIN32
#include <windows.h>
uint32_t GetOsThreadId()
{
    return ::GetCurrentThreadId();
}
#else
#include <sys/syscall.h>
#include <unistd.h>
uint32_t GetOsThreadId()
{
    return static_cast<uint32_t>(::syscall(SYS_gettid));
}
#endif

} // namespace

// ============================================================
// ProfileRange
// ============================================================

ProfileRange::ProfileRange(const char* name, const std::source_location& loc)
    : active_(HasProfileBackend())
{
    PushNvtxRange(name);
#ifdef ANET_ENABLE_TRACY
    // 直接コンストラクタは互換用。新規コードは静的 source-location マクロを使う。
    const uint64_t srcloc = ___tracy_alloc_srcloc_name(
        loc.line(),
        loc.file_name(), strlen(loc.file_name()),
        loc.function_name(), strlen(loc.function_name()),
        name, strlen(name),
        0
    );
    tracy_ctx_ = ___tracy_emit_zone_begin_alloc(srcloc, 1);
#else
    (void)loc;
#endif
}

ProfileRange::ProfileRange(const char* name, int idx, const std::source_location& loc)
    : active_(HasProfileBackend())
{
    char buf[128];
    snprintf(buf, sizeof(buf), "%s[%d]", name, idx);
    PushNvtxRange(buf);

#ifdef ANET_ENABLE_TRACY
    const uint64_t srcloc = ___tracy_alloc_srcloc_name(
        loc.line(),
        loc.file_name(), strlen(loc.file_name()),
        loc.function_name(), strlen(loc.function_name()),
        buf, strlen(buf),
        0
    );
    tracy_ctx_ = ___tracy_emit_zone_begin_alloc(srcloc, 1);
#else
    (void)loc;
#endif
}

ProfileRange::ProfileRange(const char* name, ProfileRange& previous, const std::source_location& loc)
    : active_(HasProfileBackend())
{
    previous.End();
    PushNvtxRange(name);

#ifdef ANET_ENABLE_TRACY
    const uint64_t srcloc = ___tracy_alloc_srcloc_name(
        loc.line(),
        loc.file_name(), strlen(loc.file_name()),
        loc.function_name(), strlen(loc.function_name()),
        name, strlen(name),
        0
    );
    tracy_ctx_ = ___tracy_emit_zone_begin_alloc(srcloc, 1);
#else
    (void)loc;
#endif
}

ProfileRange::ProfileRange(const char* name, int idx, ProfileRange& previous, const std::source_location& loc)
    : active_(HasProfileBackend())
{
    previous.End();

    char buf[128];
    snprintf(buf, sizeof(buf), "%s[%d]", name, idx);
    PushNvtxRange(buf);

#ifdef ANET_ENABLE_TRACY
    const uint64_t srcloc = ___tracy_alloc_srcloc_name(
        loc.line(),
        loc.file_name(), strlen(loc.file_name()),
        loc.function_name(), strlen(loc.function_name()),
        buf, strlen(buf),
        0
    );
    tracy_ctx_ = ___tracy_emit_zone_begin_alloc(srcloc, 1);
#else
    (void)loc;
#endif
}

ProfileRange::ProfileRange(
    profile_detail::ProfileRangeTag,
    const char* nvtx_name,
    const profile_detail::SourceLocationData* srcloc)
    : active_(HasProfileBackend())
{
    PushNvtxRange(nvtx_name);
#ifdef ANET_ENABLE_TRACY
    tracy_ctx_ = ___tracy_emit_zone_begin(srcloc, 1);
#else
    (void)srcloc;
#endif
}

ProfileRange::ProfileRange(
    profile_detail::ProfileRangeTag,
    const char* nvtx_name,
    int idx,
    const profile_detail::SourceLocationData* srcloc)
    : active_(HasProfileBackend())
{
    char buf[128];
    snprintf(buf, sizeof(buf), "%s[%d]", nvtx_name, idx);
    PushNvtxRange(buf);

#ifdef ANET_ENABLE_TRACY
    tracy_ctx_ = ___tracy_emit_zone_begin(srcloc, 1);
    ___tracy_emit_zone_name(tracy_ctx_, buf, strlen(buf));
#else
    (void)srcloc;
#endif
}

ProfileRange::ProfileRange(
    profile_detail::ProfileRangeTag,
    const char* nvtx_name,
    ProfileRange& previous,
    const profile_detail::SourceLocationData* srcloc)
    : active_(HasProfileBackend())
{
    previous.End();
    PushNvtxRange(nvtx_name);
#ifdef ANET_ENABLE_TRACY
    tracy_ctx_ = ___tracy_emit_zone_begin(srcloc, 1);
#else
    (void)srcloc;
#endif
}

ProfileRange::ProfileRange(ProfileRange&& other) noexcept
    : active_(other.active_)
{
#ifdef ANET_ENABLE_TRACY
    tracy_ctx_ = other.tracy_ctx_;
#endif
    other.active_ = false;
}

ProfileRange& ProfileRange::operator=(ProfileRange&& other) noexcept
{
    if (this != &other) {
        End();
        active_ = other.active_;
#ifdef ANET_ENABLE_TRACY
        tracy_ctx_ = other.tracy_ctx_;
#endif
        other.active_ = false;
    }
    return *this;
}

void ProfileRange::End()
{
    if (active_) {
        PopNvtxRange();
#ifdef ANET_ENABLE_TRACY
        ___tracy_emit_zone_end(tracy_ctx_);
#endif
        active_ = false;
    }
}

ProfileRange::~ProfileRange()
{
    End();
}

// ============================================================
// ProfileThreadName
// ============================================================

ProfileThreadName::ProfileThreadName(const char* name)
{
    const uint32_t tid = GetOsThreadId();
    snprintf(buf_, sizeof(buf_), "%s", name);
    NameNvtxThread(tid, buf_);
#ifdef ANET_ENABLE_TRACY
    ___tracy_set_thread_name(name);
#endif
}

ProfileThreadName::ProfileThreadName(const char* base, int idx)
{
    const uint32_t tid = GetOsThreadId();
    snprintf(buf_, sizeof(buf_), "%s_%d", base, idx);
    NameNvtxThread(tid, buf_);
#ifdef ANET_ENABLE_TRACY
    ___tracy_set_thread_name(buf_);
#endif
}
