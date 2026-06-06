// anet/profile.hpp

#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <source_location>
#include <string_view>

#ifdef ANET_ENABLE_TRACY
#include <TracyC.h>
#endif

namespace anet::profile_detail {

#ifdef ANET_ENABLE_TRACY
    using SourceLocationData = ___tracy_source_location_data;
#else
    struct SourceLocationData {
        const char* name;
        const char* function;
        const char* file;
        uint32_t line;
        uint32_t color;
    };
#endif

    struct ProfileRangeTag {
    };

    template <size_t N>
    struct FixedString {
        char value[N];

        consteval FixedString(const char (&text)[N])
        {
            for (size_t i = 0; i < N; ++i) {
                value[i] = text[i];
            }
        }
    };

    constexpr size_t npos = static_cast<size_t>(-1);

    consteval size_t FindChar(std::string_view s, char c)
    {
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == c) {
                return i;
            }
        }
        return npos;
    }

    consteval size_t FindNameStart(std::string_view sig, size_t end)
    {
        size_t start = 0;
        for (size_t i = 0; i < end; ++i) {
            if (sig[i] == ' ') {
                start = i + 1;
            }
        }
        return start;
    }

    consteval size_t FindLastScope(std::string_view s, size_t begin, size_t end)
    {
        size_t last = npos;
        for (size_t i = begin; i + 1 < end; ++i) {
            if (s[i] == ':' && s[i + 1] == ':') {
                last = i;
                ++i;
            }
        }
        return last;
    }

    consteval size_t FindPreviousScope(std::string_view s, size_t begin, size_t end, size_t last)
    {
        size_t prev = npos;
        for (size_t i = begin; i + 1 < end && i < last; ++i) {
            if (s[i] == ':' && s[i + 1] == ':') {
                prev = i;
                ++i;
            }
        }
        return prev;
    }

    consteval size_t ShortNameStart(std::string_view sig, size_t name_start, size_t name_end)
    {
        const size_t last_scope = FindLastScope(sig, name_start, name_end);
        if (last_scope == npos) {
            return name_start;
        }

        const size_t prev_scope = FindPreviousScope(sig, name_start, name_end, last_scope);
        if (prev_scope == npos) {
            return name_start;
        }
        return prev_scope + 2;
    }

    template <FixedString Signature>
    consteval auto MakeProfileFunctionName()
    {
        constexpr std::string_view sig(Signature.value, sizeof(Signature.value) - 1);
        constexpr size_t open_paren = FindChar(sig, '(');
        constexpr size_t name_end = open_paren == npos ? sig.size() : open_paren;
        constexpr size_t name_start = FindNameStart(sig, name_end);
        constexpr size_t short_start = ShortNameStart(sig, name_start, name_end);
        constexpr size_t len = name_end - short_start;

        std::array<char, len + 1> result{};
        for (size_t i = 0; i < len; ++i) {
            result[i] = sig[short_start + i];
        }
        result[len] = '\0';
        return result;
    }

    template <size_t FunctionNameSize, size_t PhaseNameSize>
    consteval auto MakeProfilePhaseName(
        const std::array<char, FunctionNameSize>& function_name,
        const char (&phase_name)[PhaseNameSize])
    {
        std::array<char, FunctionNameSize + PhaseNameSize> result{};
        size_t out = 0;
        for (size_t i = 0; i + 1 < FunctionNameSize; ++i) {
            result[out++] = function_name[i];
        }
        result[out++] = '.';
        for (size_t i = 0; i < PhaseNameSize; ++i) {
            result[out++] = phase_name[i];
        }
        return result;
    }

} // namespace anet::profile_detail

namespace anet {

    class ProfileRange {
    public:
        [[deprecated("Use ANET_PROFILE_FUNC, ANET_PROFILE_SCOPE, ANET_PROFILE_SCOPE_NAMED, or ANET_PROFILE_SCOPE_FULL instead.")]]
        explicit ProfileRange(const char* name, const std::source_location& loc = std::source_location::current());
        [[deprecated("Use ANET_PROFILE_SCOPE_NAMED or ANET_PROFILE_SCOPE_FULL instead.")]]
        explicit ProfileRange(const char* name, int idx, const std::source_location& loc = std::source_location::current());
        [[deprecated("Use ANET_PROFILE_SCOPE_NEXT or ANET_PROFILE_SCOPE_NEXT_NAMED instead.")]]
        explicit ProfileRange(const char* name, ProfileRange& previous, const std::source_location& loc = std::source_location::current());
        [[deprecated("Use ANET_PROFILE_SCOPE_NEXT or ANET_PROFILE_SCOPE_NEXT_NAMED instead.")]]
        explicit ProfileRange(const char* name, int idx, ProfileRange& previous, const std::source_location& loc = std::source_location::current());

        explicit ProfileRange(
            profile_detail::ProfileRangeTag,
            const char* nvtx_name,
            const profile_detail::SourceLocationData* srcloc);
        explicit ProfileRange(
            profile_detail::ProfileRangeTag,
            const char* nvtx_name,
            int idx,
            const profile_detail::SourceLocationData* srcloc);
        explicit ProfileRange(
            profile_detail::ProfileRangeTag,
            const char* nvtx_name,
            ProfileRange& previous,
            const profile_detail::SourceLocationData* srcloc);

        ProfileRange(const ProfileRange&) = delete;
        ProfileRange& operator=(const ProfileRange&) = delete;
        ProfileRange(ProfileRange&& other) noexcept;
        ~ProfileRange();

        ProfileRange& operator=(ProfileRange&& other) noexcept;
        void End();
    private:
        bool active_;
    #ifdef ANET_ENABLE_TRACY
        TracyCZoneCtx tracy_ctx_;
    #endif
    };

    namespace profile_detail {

        class ProfileScopeCurrentGuard {
        public:
            ProfileScopeCurrentGuard(ProfileRange& scope, ProfileScopeCurrentGuard*& current)
                : scope_(scope), current_(current), previous_(current)
            {
                current_ = this;
            }

            ProfileScopeCurrentGuard(const ProfileScopeCurrentGuard&) = delete;
            ProfileScopeCurrentGuard& operator=(const ProfileScopeCurrentGuard&) = delete;

            ~ProfileScopeCurrentGuard()
            {
                if (current_ == this) {
                    current_ = previous_;
                }
            }

            ProfileRange& scope()
            {
                return scope_;
            }

            void End()
            {
                scope_.End();
                if (current_ == this) {
                    current_ = previous_;
                }
            }

        private:
            ProfileRange& scope_;
            ProfileScopeCurrentGuard*& current_;
            ProfileScopeCurrentGuard* previous_;
        };

        template <FixedString Signature>
        inline ProfileScopeCurrentGuard*& CurrentProfileScopeGuard()
        {
            thread_local ProfileScopeCurrentGuard* current = nullptr;
            return current;
        }

        inline ProfileRange& RequireCurrentProfileScope(ProfileScopeCurrentGuard* current)
        {
            assert(current != nullptr);
            return current->scope();
        }

    } // namespace profile_detail

    class ProfileThreadName {
    public:
        explicit ProfileThreadName(const char* name);
        ProfileThreadName(const char* base, int idx);
    private:
        char buf_[64];
    };

} // namespace anet

#define ANET_PROFILE_CONCAT_IMPL(x, y) x##y
#define ANET_PROFILE_CONCAT(x, y) ANET_PROFILE_CONCAT_IMPL(x, y)
#define ANET_PROFILE_SCOPE_VAR(phase) ANET_PROFILE_CONCAT(anet_profile_scope_, phase)
#define ANET_PROFILE_SCOPE_GUARD_VAR(var_name) ANET_PROFILE_CONCAT(var_name, _guard)

#ifdef _MSC_VER
#define ANET_PROFILE_FUNCTION_SIGNATURE __FUNCSIG__
#else
#define ANET_PROFILE_FUNCTION_SIGNATURE __PRETTY_FUNCTION__
#endif

#define ANET_PROFILE_CURRENT_SCOPE_GUARD() \
    anet::profile_detail::CurrentProfileScopeGuard<ANET_PROFILE_FUNCTION_SIGNATURE>()

#ifdef ANET_ENABLE_PROFILE

#define ANET_PROFILE_MAKE_SOURCE_LOCATION(var_name, zone_name) \
    static constexpr anet::profile_detail::SourceLocationData var_name { \
        zone_name, ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__).data(), __FILE__, (uint32_t)__LINE__, 0 \
    }

#define ANET_PROFILE_FUNC() \
    static constexpr auto ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__) = \
        anet::profile_detail::MakeProfileFunctionName<ANET_PROFILE_FUNCTION_SIGNATURE>(); \
    ANET_PROFILE_MAKE_SOURCE_LOCATION(ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__), \
        ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__).data()); \
    anet::ProfileRange ANET_PROFILE_CONCAT(__anet_profile_scope_, __LINE__) { \
        anet::profile_detail::ProfileRangeTag{}, \
        ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__).data(), \
        &ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__) \
    }

#define ANET_PROFILE_SCOPE(phase) \
    ANET_PROFILE_SCOPE_NAMED(ANET_PROFILE_SCOPE_VAR(phase), #phase)

#define ANET_PROFILE_SCOPE_NAMED(var_name, phase_literal) \
    static constexpr auto ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__) = \
        anet::profile_detail::MakeProfileFunctionName<ANET_PROFILE_FUNCTION_SIGNATURE>(); \
    static constexpr auto ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__) = \
        anet::profile_detail::MakeProfilePhaseName(ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__), phase_literal); \
    ANET_PROFILE_MAKE_SOURCE_LOCATION(ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__), \
        ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__).data()); \
    anet::ProfileRange var_name { \
        anet::profile_detail::ProfileRangeTag{}, \
        ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__).data(), \
        &ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__) \
    }; \
    anet::profile_detail::ProfileScopeCurrentGuard ANET_PROFILE_SCOPE_GUARD_VAR(var_name) { \
        var_name, ANET_PROFILE_CURRENT_SCOPE_GUARD() \
    }

#define ANET_PROFILE_SCOPE_NEXT(phase) \
    ANET_PROFILE_SCOPE_NEXT_CURRENT(phase, #phase)

#define ANET_PROFILE_SCOPE_NEXT_FROM(phase, prev_phase) \
    ANET_PROFILE_SCOPE_NEXT_EXPLICIT(phase, prev_phase, #phase)

#define ANET_PROFILE_SCOPE_NEXT_CURRENT(phase, phase_literal) \
    static constexpr auto ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__) = \
        anet::profile_detail::MakeProfileFunctionName<ANET_PROFILE_FUNCTION_SIGNATURE>(); \
    static constexpr auto ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__) = \
        anet::profile_detail::MakeProfilePhaseName(ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__), phase_literal); \
    ANET_PROFILE_MAKE_SOURCE_LOCATION(ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__), \
        ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__).data()); \
    anet::ProfileRange ANET_PROFILE_SCOPE_VAR(phase) { \
        anet::profile_detail::ProfileRangeTag{}, \
        ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__).data(), \
        anet::profile_detail::RequireCurrentProfileScope(ANET_PROFILE_CURRENT_SCOPE_GUARD()), \
        &ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__) \
    }; \
    anet::profile_detail::ProfileScopeCurrentGuard ANET_PROFILE_SCOPE_GUARD_VAR(ANET_PROFILE_SCOPE_VAR(phase)) { \
        ANET_PROFILE_SCOPE_VAR(phase), ANET_PROFILE_CURRENT_SCOPE_GUARD() \
    }

#define ANET_PROFILE_SCOPE_NEXT_EXPLICIT(phase, prev_phase, phase_literal) \
    static constexpr auto ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__) = \
        anet::profile_detail::MakeProfileFunctionName<ANET_PROFILE_FUNCTION_SIGNATURE>(); \
    static constexpr auto ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__) = \
        anet::profile_detail::MakeProfilePhaseName(ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__), phase_literal); \
    ANET_PROFILE_MAKE_SOURCE_LOCATION(ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__), \
        ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__).data()); \
    anet::ProfileRange ANET_PROFILE_SCOPE_VAR(phase) { \
        anet::profile_detail::ProfileRangeTag{}, \
        ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__).data(), \
        ANET_PROFILE_SCOPE_VAR(prev_phase), \
        &ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__) \
    }; \
    anet::profile_detail::ProfileScopeCurrentGuard ANET_PROFILE_SCOPE_GUARD_VAR(ANET_PROFILE_SCOPE_VAR(phase)) { \
        ANET_PROFILE_SCOPE_VAR(phase), ANET_PROFILE_CURRENT_SCOPE_GUARD() \
    }

#define ANET_PROFILE_SCOPE_NEXT_NAMED(var_name, prev_var, phase_literal) \
    static constexpr auto ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__) = \
        anet::profile_detail::MakeProfileFunctionName<ANET_PROFILE_FUNCTION_SIGNATURE>(); \
    static constexpr auto ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__) = \
        anet::profile_detail::MakeProfilePhaseName(ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__), phase_literal); \
    ANET_PROFILE_MAKE_SOURCE_LOCATION(ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__), \
        ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__).data()); \
    anet::ProfileRange var_name { \
        anet::profile_detail::ProfileRangeTag{}, \
        ANET_PROFILE_CONCAT(__anet_profile_phase_name_, __LINE__).data(), \
        prev_var, \
        &ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__) \
    }; \
    anet::profile_detail::ProfileScopeCurrentGuard ANET_PROFILE_SCOPE_GUARD_VAR(var_name) { \
        var_name, ANET_PROFILE_CURRENT_SCOPE_GUARD() \
    }

#define ANET_PROFILE_SCOPE_END(phase) \
    ANET_PROFILE_SCOPE_GUARD_VAR(ANET_PROFILE_SCOPE_VAR(phase)).End()

#define ANET_PROFILE_SCOPE_END_NAMED(var_name) \
    ANET_PROFILE_SCOPE_GUARD_VAR(var_name).End()

#define ANET_PROFILE_SCOPE_FULL(var_name, full_name_literal) \
    static constexpr auto ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__) = \
        anet::profile_detail::MakeProfileFunctionName<ANET_PROFILE_FUNCTION_SIGNATURE>(); \
    ANET_PROFILE_MAKE_SOURCE_LOCATION(ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__), full_name_literal); \
    anet::ProfileRange var_name { \
        anet::profile_detail::ProfileRangeTag{}, \
        full_name_literal, \
        &ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__) \
    }; \
    anet::profile_detail::ProfileScopeCurrentGuard ANET_PROFILE_SCOPE_GUARD_VAR(var_name) { \
        var_name, ANET_PROFILE_CURRENT_SCOPE_GUARD() \
    }

#define ANET_PROFILE_RANGE(var_name, name_literal) \
    ANET_PROFILE_SCOPE_FULL(var_name, name_literal)

#define ANET_PROFILE_RANGE_IDX(var_name, base_name_literal, idx) \
    static constexpr auto ANET_PROFILE_CONCAT(__anet_profile_func_name_, __LINE__) = \
        anet::profile_detail::MakeProfileFunctionName<ANET_PROFILE_FUNCTION_SIGNATURE>(); \
    ANET_PROFILE_MAKE_SOURCE_LOCATION(ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__), base_name_literal); \
    anet::ProfileRange var_name { \
        anet::profile_detail::ProfileRangeTag{}, \
        base_name_literal, \
        idx, \
        &ANET_PROFILE_CONCAT(__anet_profile_sl_, __LINE__) \
    }; \
    anet::profile_detail::ProfileScopeCurrentGuard ANET_PROFILE_SCOPE_GUARD_VAR(var_name) { \
        var_name, ANET_PROFILE_CURRENT_SCOPE_GUARD() \
    }

#define ANET_PROFILE_RANGE_PREV(var_name, name_literal, prev_var) \
    ANET_PROFILE_SCOPE_NEXT_NAMED(var_name, prev_var, name_literal)

#else

#define ANET_PROFILE_FUNC() \
    anet::ProfileRange ANET_PROFILE_CONCAT(__anet_profile_scope_, __LINE__) { \
        anet::profile_detail::ProfileRangeTag{}, \
        nullptr, \
        static_cast<const anet::profile_detail::SourceLocationData*>(nullptr) \
    }

#define ANET_PROFILE_SCOPE(phase) \
    ANET_PROFILE_SCOPE_NAMED(ANET_PROFILE_SCOPE_VAR(phase), #phase)

#define ANET_PROFILE_SCOPE_NAMED(var_name, phase_literal) \
    anet::ProfileRange var_name { \
        anet::profile_detail::ProfileRangeTag{}, \
        nullptr, \
        static_cast<const anet::profile_detail::SourceLocationData*>(nullptr) \
    }; \
    anet::profile_detail::ProfileScopeCurrentGuard ANET_PROFILE_SCOPE_GUARD_VAR(var_name) { \
        var_name, ANET_PROFILE_CURRENT_SCOPE_GUARD() \
    }

#define ANET_PROFILE_SCOPE_NEXT(phase) \
    ANET_PROFILE_SCOPE_NEXT_CURRENT(phase, #phase)

#define ANET_PROFILE_SCOPE_NEXT_FROM(phase, prev_phase) \
    ANET_PROFILE_SCOPE_NEXT_EXPLICIT(phase, prev_phase, #phase)

#define ANET_PROFILE_SCOPE_NEXT_CURRENT(phase, phase_literal) \
    anet::ProfileRange ANET_PROFILE_SCOPE_VAR(phase) { \
        anet::profile_detail::ProfileRangeTag{}, \
        nullptr, \
        anet::profile_detail::RequireCurrentProfileScope(ANET_PROFILE_CURRENT_SCOPE_GUARD()), \
        static_cast<const anet::profile_detail::SourceLocationData*>(nullptr) \
    }; \
    anet::profile_detail::ProfileScopeCurrentGuard ANET_PROFILE_SCOPE_GUARD_VAR(ANET_PROFILE_SCOPE_VAR(phase)) { \
        ANET_PROFILE_SCOPE_VAR(phase), ANET_PROFILE_CURRENT_SCOPE_GUARD() \
    }

#define ANET_PROFILE_SCOPE_NEXT_EXPLICIT(phase, prev_phase, phase_literal) \
    anet::ProfileRange ANET_PROFILE_SCOPE_VAR(phase) { \
        anet::profile_detail::ProfileRangeTag{}, \
        nullptr, \
        ANET_PROFILE_SCOPE_VAR(prev_phase), \
        static_cast<const anet::profile_detail::SourceLocationData*>(nullptr) \
    }; \
    anet::profile_detail::ProfileScopeCurrentGuard ANET_PROFILE_SCOPE_GUARD_VAR(ANET_PROFILE_SCOPE_VAR(phase)) { \
        ANET_PROFILE_SCOPE_VAR(phase), ANET_PROFILE_CURRENT_SCOPE_GUARD() \
    }

#define ANET_PROFILE_SCOPE_NEXT_NAMED(var_name, prev_var, phase_literal) \
    anet::ProfileRange var_name { \
        anet::profile_detail::ProfileRangeTag{}, \
        nullptr, \
        prev_var, \
        static_cast<const anet::profile_detail::SourceLocationData*>(nullptr) \
    }; \
    anet::profile_detail::ProfileScopeCurrentGuard ANET_PROFILE_SCOPE_GUARD_VAR(var_name) { \
        var_name, ANET_PROFILE_CURRENT_SCOPE_GUARD() \
    }

#define ANET_PROFILE_SCOPE_END(phase) \
    ANET_PROFILE_SCOPE_GUARD_VAR(ANET_PROFILE_SCOPE_VAR(phase)).End()

#define ANET_PROFILE_SCOPE_END_NAMED(var_name) \
    ANET_PROFILE_SCOPE_GUARD_VAR(var_name).End()

#define ANET_PROFILE_SCOPE_FULL(var_name, full_name_literal) \
    ANET_PROFILE_SCOPE_NAMED(var_name, full_name_literal)

#define ANET_PROFILE_RANGE(var_name, name_literal) \
    ANET_PROFILE_SCOPE_FULL(var_name, name_literal)

#define ANET_PROFILE_RANGE_IDX(var_name, name_literal, idx) \
    ANET_PROFILE_SCOPE_NAMED(var_name, name_literal)

#define ANET_PROFILE_RANGE_PREV(var_name, name_literal, prev_var) \
    ANET_PROFILE_SCOPE_NEXT_NAMED(var_name, prev_var, name_literal)

#endif
