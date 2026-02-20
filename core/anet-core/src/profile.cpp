#include "anet/profile.hpp"
#include <cstdint>
#include <nvtx3/nvtx3.hpp>


using namespace anet;


// ============================================================
// ProfileRange
// ============================================================

ProfileRange::ProfileRange(const char* name, const std::source_location& loc)
    : active_(true)
{
    nvtxRangePushA(name);
#ifdef ANET_ENABLE_TRACY
    // Tracyの動的ソースロケーション登録APIを使用
    uint64_t srcloc = ___tracy_alloc_srcloc_name(
        loc.line(),
        loc.file_name(), strlen(loc.file_name()),
        loc.function_name(), strlen(loc.function_name()),
        name, strlen(name),
        0
    );
    // 割り当てたIDを使ってゾーン開始
    tracy_ctx_ = ___tracy_emit_zone_begin_alloc(srcloc, 1);
#endif
}

ProfileRange::ProfileRange(const char* name, int idx, const std::source_location& loc)
    : active_(true)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "%s[%d]", name, idx);
    nvtxRangePushA(buf);

#ifdef ANET_ENABLE_TRACY
    uint64_t srcloc = ___tracy_alloc_srcloc_name(
        loc.line(),
        loc.file_name(), strlen(loc.file_name()),
        loc.function_name(), strlen(loc.function_name()),
        buf, strlen(buf),
        0
    );
    tracy_ctx_ = ___tracy_emit_zone_begin_alloc(srcloc, 1);
#endif
}

ProfileRange::ProfileRange(const char* name, ProfileRange& previous, const std::source_location& loc)
    : active_(true)
{
    previous.End();
    nvtxRangePushA(name);

#ifdef ANET_ENABLE_TRACY
    uint64_t srcloc = ___tracy_alloc_srcloc_name(
        loc.line(),
        loc.file_name(), strlen(loc.file_name()),
        loc.function_name(), strlen(loc.function_name()),
        name, strlen(name),
        0
    );
    tracy_ctx_ = ___tracy_emit_zone_begin_alloc(srcloc, 1);
#endif
}

ProfileRange::ProfileRange(const char* name, int idx, ProfileRange& previous, const std::source_location& loc)
    : active_(true)
{
    previous.End();

    char buf[128];
    snprintf(buf, sizeof(buf), "%s[%d]", name, idx);
    nvtxRangePushA(buf);

#ifdef ANET_ENABLE_TRACY
    uint64_t srcloc = ___tracy_alloc_srcloc_name(
        loc.line(),
        loc.file_name(), strlen(loc.file_name()),
        loc.function_name(), strlen(loc.function_name()),
        buf, strlen(buf),
        0
    );
    tracy_ctx_ = ___tracy_emit_zone_begin_alloc(srcloc, 1);
#endif
}

#ifdef ANET_ENABLE_TRACY

ProfileRange::ProfileRange(const char* name, const ___tracy_source_location_data* srcloc)
    : active_(true)
{
    nvtxRangePushA(name);
    tracy_ctx_ = ___tracy_emit_zone_begin(srcloc, 1);
}

ProfileRange::ProfileRange(const char* name, int idx, const ___tracy_source_location_data* srcloc)
    : active_(true)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "%s[%d]", name, idx);
    nvtxRangePushA(buf);

    // ベースとなる静的情報は最速で登録し、名前だけ後から動的に上書きする
    tracy_ctx_ = ___tracy_emit_zone_begin(srcloc, 1);
    ___tracy_emit_zone_name(tracy_ctx_, buf, strlen(buf));
}

ProfileRange::ProfileRange(const char* name, ProfileRange& previous, const ___tracy_source_location_data* srcloc)
    : active_(true)
{
    previous.End();
    nvtxRangePushA(name);
    tracy_ctx_ = ___tracy_emit_zone_begin(srcloc, 1);
}

#endif // ANET_ENABLE_TRACY

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
        nvtxRangePop();
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

// ============================================================
// ProfileThreadName
// ============================================================

ProfileThreadName::ProfileThreadName(const char* name)
{
    uint32_t tid = GetOsThreadId();
    snprintf(buf_, sizeof(buf_), "%s", name);
    nvtxNameOsThreadA(tid, buf_);
#ifdef ANET_ENABLE_TRACY
    ___tracy_set_thread_name(name);
#endif
}

ProfileThreadName::ProfileThreadName(const char* base, int idx)
{
    uint32_t tid = GetOsThreadId();

    // base + "_" + idx を stack で構築
    snprintf(buf_, sizeof(buf_), "%s_%d", base, idx);

    nvtxNameOsThreadA(tid, buf_);

#ifdef ANET_ENABLE_TRACY
    ___tracy_set_thread_name(buf_);
#endif
}

