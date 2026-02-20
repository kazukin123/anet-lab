// anet/profile.hpp
#pragma once

#include <source_location>

#ifdef ANET_ENABLE_TRACY
#include <TracyC.h>
#endif

namespace anet {

    ///  @todo リリースビルドやNVTX無効時はマクロで実態無し版に切換え

    class ProfileRange {
    public:
        explicit ProfileRange(const char* name, const std::source_location& loc = std::source_location::current());
        explicit ProfileRange(const char* name, int idx, const std::source_location& loc = std::source_location::current());
        explicit ProfileRange(const char* name, ProfileRange& previous, const std::source_location& loc = std::source_location::current());
        explicit ProfileRange(const char* name, int idx, ProfileRange& previous, const std::source_location& loc = std::source_location::current());
#ifdef ANET_ENABLE_TRACY
        explicit ProfileRange(const char* name, const ___tracy_source_location_data* srcloc);
        explicit ProfileRange(const char* name, int idx, const ___tracy_source_location_data* srcloc);
        explicit ProfileRange(const char* name, ProfileRange& previous, const ___tracy_source_location_data* srcloc);
#endif
        ProfileRange(const ProfileRange&) = delete;
        ProfileRange& operator=(const ProfileRange&) = delete;
        ProfileRange(ProfileRange&& other) noexcept;
        ~ProfileRange();

        ProfileRange& operator=(ProfileRange&& other) noexcept;
        void End();
    private:
        bool active_;
#ifdef ANET_ENABLE_TRACY
        TracyCZoneCtx tracy_ctx_; // C APIの軽量コンテキスト
#endif
    };

    class ProfileThreadName {
    public:
        explicit ProfileThreadName(const char* name);
        ProfileThreadName(const char* base, int idx);
    private:
        char buf_[64];
    };
}

// 使い方例
//   anet::ProfileRange p("ResBlockModule::Forward");
//   ANET_PROFILE_RANGE(p, "ResBlockModule::Forward");


#define ANET_PROFILE_CONCAT_IMPL(x, y) x##y
#define ANET_PROFILE_CONCAT(x, y) ANET_PROFILE_CONCAT_IMPL(x, y)

#ifdef ANET_ENABLE_TRACY
#define ANET_PROFILE_RANGE(var_name, name_literal) \
        static constexpr ___tracy_source_location_data ANET_PROFILE_CONCAT(__tracy_sl_, __LINE__) { \
            name_literal, __FUNCTION__, __FILE__, (uint32_t)__LINE__, 0 \
        }; \
        anet::ProfileRange var_name(name_literal, &ANET_PROFILE_CONCAT(__tracy_sl_, __LINE__))

#define ANET_PROFILE_RANGE_IDX(var_name, base_name_literal, idx) \
        static constexpr ___tracy_source_location_data ANET_PROFILE_CONCAT(__tracy_sl_, __LINE__) { \
            base_name_literal, __FUNCTION__, __FILE__, (uint32_t)__LINE__, 0 \
        }; \
        anet::ProfileRange var_name(base_name_literal, idx, &ANET_PROFILE_CONCAT(__tracy_sl_, __LINE__))

#define ANET_PROFILE_RANGE_PREV(var_name, name_literal, prev_var) \
        static constexpr ___tracy_source_location_data ANET_PROFILE_CONCAT(__tracy_sl_, __LINE__) { \
            name_literal, __FUNCTION__, __FILE__, (uint32_t)__LINE__, 0 \
        }; \
        anet::ProfileRange var_name(name_literal, prev_var, &ANET_PROFILE_CONCAT(__tracy_sl_, __LINE__))

#else
// Tracy無効時は通常のコンストラクタ（std::source_location版）へフォールバック
#define ANET_PROFILE_RANGE(var_name, name) \
        anet::ProfileRange var_name(name)

#define ANET_PROFILE_RANGE_IDX(var_name, name, idx) \
        anet::ProfileRange var_name(name, idx)

#define ANET_PROFILE_RANGE_PREV(var_name, name, prev_var) \
        anet::ProfileRange var_name(name, prev_var)
#endif