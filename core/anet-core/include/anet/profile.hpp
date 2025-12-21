// anet/profile.hpp
#pragma once

namespace anet {

    ///  @todo リリースビルドやNVTX無効時はマクロで実態無し版に切換え

    class ProfileRange {
    public:
        explicit ProfileRange(const char* name);
        explicit ProfileRange(const char* name, int idx);
        explicit ProfileRange(const char* name, ProfileRange& previous);
        explicit ProfileRange(const char* name, int idx, ProfileRange& previous);

        ProfileRange(const ProfileRange&) = delete;
        ProfileRange& operator=(const ProfileRange&) = delete;
        ProfileRange(ProfileRange&& other) noexcept;
        ~ProfileRange();

        ProfileRange& operator=(ProfileRange&& other) noexcept;
        void End();
    private:
        bool active_;
    };

    class ProfileThreadName {
    public:
        explicit ProfileThreadName(const char* name);
        ProfileThreadName(const char* base, int idx);
    private:
        char buf_[64];
    };
}