// app_util.cpp
#include "anet/app_util.hpp"

#include <atomic>
#include <array>
#include <cstdio>
#include <iostream>
#include <thread>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

#include "anet/log.hpp"

namespace LOG = anet::log;

namespace {

#ifdef _WIN32

void CloseHandleIfValid(HANDLE& handle)
{
    if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
        handle = nullptr;
    }
}

#endif

bool RedirectStandardStreamFallback(FILE* stream, const std::filesystem::path& path)
{
    return std::freopen(path.string().c_str(), "a", stream) != nullptr;
}

} // namespace

namespace anet {

struct StandardStreamLogger::Impl {
    ~Impl()
    {
        Stop();
    }

    void Start(const std::filesystem::path& run_dir)
    {
        std::filesystem::create_directories(run_dir);
        stdout_log_path_ = run_dir / "stdout.log";
        stderr_log_path_ = run_dir / "stderr.log";

        // GUIアプリで失われるプロセス標準出力をrun directoryへ退避する。
        stdout_redirected_ = StartStdout(stdout_log_path_);
        stderr_redirected_ = StartStderr(stderr_log_path_);

        if (stderr_redirected_) {
            std::setvbuf(stderr, nullptr, _IONBF, 0);
        }
    }

    void LogStatus() const
    {
        if (stdout_redirected_) {
            LOG::info() << "stdout redirected to " << stdout_log_path_.string();
        } else {
            LOG::warn() << "Failed to redirect stdout to " << stdout_log_path_.string();
        }

        if (stderr_redirected_) {
            LOG::info() << "stderr redirected to " << stderr_log_path_.string();
        } else {
            LOG::warn() << "Failed to redirect stderr to " << stderr_log_path_.string();
        }
    }

    void Flush()
    {
        // wxLog以外へ直接書かれた標準ストリームも、pause等の可視境界で同期する。
        std::cout.flush();
        std::cerr.flush();
        std::clog.flush();
        std::fflush(stdout);
        std::fflush(stderr);
        stdout_capture.Flush();
        stderr_capture.Flush();
    }

    void Stop()
    {
        stdout_capture.Stop();
        stderr_capture.Stop();
    }

private:
    bool StartStdout(const std::filesystem::path& path)
    {
#ifdef _WIN32
        return stdout_capture.Start(stdout, 1, STD_OUTPUT_HANDLE, path);
#else
        return stdout_capture.Start(stdout, 1, 0, path);
#endif
    }

    bool StartStderr(const std::filesystem::path& path)
    {
#ifdef _WIN32
        return stderr_capture.Start(stderr, 2, STD_ERROR_HANDLE, path);
#else
        return stderr_capture.Start(stderr, 2, 0, path);
#endif
    }

    struct StreamCapture {
#ifdef _WIN32
        ~StreamCapture()
        {
            Stop();
        }

        bool Start(FILE* stream, int target_fd, DWORD std_handle_id, const std::filesystem::path& path)
        {
            stream_ = stream;
            target_fd_ = target_fd;
            std_handle_id_ = std_handle_id;
            original_std_handle_ = GetStdHandle(std_handle_id_);

            log_file_ = CreateFileW(
                path.wstring().c_str(),
                FILE_APPEND_DATA,
                FILE_SHARE_READ,
                nullptr,
                OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                nullptr);
            if (log_file_ == INVALID_HANDLE_VALUE) {
                return false;
            }

            SECURITY_ATTRIBUTES security_attributes{};
            security_attributes.nLength = sizeof(security_attributes);
            if (!CreatePipe(&read_pipe_, &write_pipe_, &security_attributes, 0)) {
                CloseForStartFailure();
                return false;
            }

            if (!SetStdHandle(std_handle_id_, write_pipe_)) {
                CloseForStartFailure();
                return false;
            }
            std_handle_redirected_ = true;

            HANDLE crt_write_handle = nullptr;
            if (!DuplicateHandle(
                    GetCurrentProcess(),
                    write_pipe_,
                    GetCurrentProcess(),
                    &crt_write_handle,
                    0,
                    FALSE,
                    DUPLICATE_SAME_ACCESS)) {
                CloseForStartFailure();
                return false;
            }

            const int pipe_fd = _open_osfhandle(reinterpret_cast<intptr_t>(crt_write_handle), _O_TEXT);
            if (pipe_fd < 0) {
                CloseHandle(crt_write_handle);
                CloseForStartFailure();
                return false;
            }

            if (!EnsureStreamFileDescriptor()) {
                _close(pipe_fd);
                CloseForStartFailure();
                return false;
            }

            if (_dup2(pipe_fd, target_fd_) != 0) {
                _close(pipe_fd);
                CloseForStartFailure();
                return false;
            }
            _close(pipe_fd);

            if (stream_ == stderr) {
                std::setvbuf(stderr, nullptr, _IONBF, 0);
            }
            std::clearerr(stream_);
            std::cout.clear();
            std::cerr.clear();
            std::clog.clear();

            started_.store(true);
            try {
                worker_ = std::thread([this] { ReadLoop(); });
            } catch (...) {
                Stop();
                return false;
            }
            return true;
        }

        void Flush()
        {
            if (write_pipe_ != nullptr && write_pipe_ != INVALID_HANDLE_VALUE) {
                FlushFileBuffers(write_pipe_);
            }
            if (log_file_ != INVALID_HANDLE_VALUE) {
                FlushFileBuffers(log_file_);
            }
        }

        void Stop()
        {
            if (!started_.exchange(false)) {
                CloseForStartFailure();
                return;
            }

            if (stream_ != nullptr) {
                std::fflush(stream_);
            }

            if (std_handle_redirected_) {
                SetStdHandle(std_handle_id_, original_std_handle_);
                std_handle_redirected_ = false;
            }

            RedirectTargetToNull();
            CloseHandleIfValid(write_pipe_);

            if (worker_.joinable()) {
                worker_.join();
            }

            CloseHandleIfValid(read_pipe_);
            if (log_file_ != INVALID_HANDLE_VALUE) {
                FlushFileBuffers(log_file_);
                CloseHandle(log_file_);
                log_file_ = INVALID_HANDLE_VALUE;
            }
        }

    private:
        bool EnsureStreamFileDescriptor()
        {
            int stream_fd = _fileno(stream_);
            if (stream_fd < 0) {
                // GUIアプリでは標準FILEが未接続の場合があるため、NULで有効化してからpipeへ差し替える。
                FILE* reopened = nullptr;
                if (_wfreopen_s(&reopened, L"NUL", L"w", stream_) != 0 || reopened == nullptr) {
                    return false;
                }
                stream_fd = _fileno(stream_);
            }

            if (stream_fd < 0) {
                return false;
            }
            target_fd_ = stream_fd;
            return true;
        }

        void ReadLoop()
        {
            std::array<char, 4096> buffer{};
            while (true) {
                DWORD bytes_read = 0;
                const BOOL read_ok = ReadFile(
                    read_pipe_,
                    buffer.data(),
                    static_cast<DWORD>(buffer.size()),
                    &bytes_read,
                    nullptr);
                if (!read_ok || bytes_read == 0) {
                    break;
                }
                if (!WriteAll(buffer.data(), bytes_read)) {
                    break;
                }
            }
            Flush();
        }

        bool WriteAll(const char* data, DWORD size)
        {
            DWORD total_written = 0;
            while (total_written < size) {
                DWORD bytes_written = 0;
                const BOOL write_ok = WriteFile(
                    log_file_,
                    data + total_written,
                    size - total_written,
                    &bytes_written,
                    nullptr);
                if (!write_ok || bytes_written == 0) {
                    return false;
                }
                total_written += bytes_written;
            }
            return true;
        }

        void RedirectTargetToNull()
        {
            const int null_fd = _open("NUL", _O_WRONLY | _O_TEXT);
            if (null_fd >= 0) {
                _dup2(null_fd, target_fd_);
                _close(null_fd);
            } else {
                _close(target_fd_);
            }
        }

        void CloseForStartFailure()
        {
            if (std_handle_redirected_) {
                SetStdHandle(std_handle_id_, original_std_handle_);
                std_handle_redirected_ = false;
            }
            CloseHandleIfValid(write_pipe_);
            CloseHandleIfValid(read_pipe_);
            if (log_file_ != INVALID_HANDLE_VALUE) {
                CloseHandle(log_file_);
                log_file_ = INVALID_HANDLE_VALUE;
            }
        }

        FILE* stream_ = nullptr;
        int target_fd_ = -1;
        DWORD std_handle_id_ = 0;
        HANDLE original_std_handle_ = nullptr;
        HANDLE read_pipe_ = nullptr;
        HANDLE write_pipe_ = nullptr;
        HANDLE log_file_ = INVALID_HANDLE_VALUE;
        bool std_handle_redirected_ = false;
        std::atomic<bool> started_{ false };
        std::thread worker_;
#else
        bool Start(FILE* stream, int, int, const std::filesystem::path& path)
        {
            stream_ = stream;
            redirected_ = RedirectStandardStreamFallback(stream_, path);
            return redirected_;
        }

        void Flush()
        {
            if (redirected_ && stream_ != nullptr) {
                std::fflush(stream_);
            }
        }

        void Stop()
        {
            Flush();
        }

        FILE* stream_ = nullptr;
        bool redirected_ = false;
#endif
    };

    StreamCapture stdout_capture;
    StreamCapture stderr_capture;
    std::filesystem::path stdout_log_path_;
    std::filesystem::path stderr_log_path_;
    bool stdout_redirected_ = false;
    bool stderr_redirected_ = false;
};

StandardStreamLogger::StandardStreamLogger()
    : impl_(std::make_unique<Impl>())
{
}

StandardStreamLogger::~StandardStreamLogger() = default;

void StandardStreamLogger::Start(const std::filesystem::path& run_dir)
{
    impl_->Start(run_dir);
}

void StandardStreamLogger::LogStatus() const
{
    impl_->LogStatus();
}

void StandardStreamLogger::Flush() const
{
    impl_->Flush();
}

void StandardStreamLogger::Stop()
{
    impl_->Stop();
}

} // namespace anet
