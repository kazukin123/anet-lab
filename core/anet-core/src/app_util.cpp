// app_util.cpp

#include "anet/app_util.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <cstring>
#else
#include <unistd.h>
#endif
#include "app_util_impl.hpp"
#include "anet/diag.hpp"
#include "anet/log.hpp"

namespace LOG = anet::log;

namespace {

std::optional<std::filesystem::path> GetEnvironmentPath(const char* name)
{
#ifdef _WIN32
    char* value = nullptr;
    size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || value == nullptr || value[0] == '\0') {
        std::free(value);
        return std::nullopt;
    }
    const std::filesystem::path path(value);
    std::free(value);
    return path;
#else
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    return std::filesystem::path(value);
#endif
}

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

constexpr size_t kMaxHistoryEntries = 10;

static std::string Trim(const std::string& value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static std::filesystem::path PathFromUtf8(const std::string& value)
{
    const std::u8string text(
        reinterpret_cast<const char8_t*>(value.data()), value.size());
    return std::filesystem::path(text);
}

static std::string PathToUtf8(const std::filesystem::path& path)
{
    const auto text = path.u8string();
    return { reinterpret_cast<const char*>(text.data()), text.size() };
}

static bool HasUncRoot(const std::string& input)
{
    // UNC 分類専用に区切りを揃え、混在区切りも同じ root として扱う。
    auto normalized_separators = input;
    std::replace(normalized_separators.begin(), normalized_separators.end(), '/', '\\');
    const auto normalized_path = PathFromUtf8(normalized_separators).lexically_normal();
#ifdef _WIN32
    const auto root_name = normalized_path.root_name().native();
    return root_name.starts_with(LR"(\\)");
#else
    const auto normalized_text = PathToUtf8(normalized_path);
    return normalized_text.starts_with(R"(\\)");
#endif
}

static std::optional<std::string> GetWorkspacePathError(const std::string& raw_input)
{
    const auto input = Trim(raw_input);
    if (input.empty()) {
        return "value must not be empty";
    }
    if (input.find('#') != std::string::npos) {
        return "value must not contain '#'";
    }
    if (input.find("//") != std::string::npos) {
        return "value must not contain '//'";
    }
    if (input.ends_with(';')) {
        return "value must not end with ';'";
    }
    if (HasUncRoot(input)) {
        return "UNC roots are not supported";
    }
    return std::nullopt;
}

static bool IsSingleRelativeName(const std::filesystem::path& path)
{
    if (path.empty() || path.is_absolute() || path.has_root_name()) {
        return false;
    }
    auto it = path.begin();
    if (it == path.end()) {
        return false;
    }
    const auto component = *it;
    ++it;
    return it == path.end() && component != "." && component != "..";
}

static bool HasWindowsDrivePrefix(const std::string& input)
{
    return input.size() >= 2
        && std::isalpha(static_cast<unsigned char>(input.front())) != 0
        && input[1] == ':';
}

namespace anet {

std::filesystem::path GetExecutablePath()
{
#ifdef _WIN32
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0) {
        throw std::runtime_error("GetExecutablePath: GetModuleFileNameW failed. error=" + std::to_string(GetLastError()));
    }
    if (length >= buffer.size()) {
        throw std::runtime_error("GetExecutablePath: executable path is too long.");
    }
    buffer.resize(length);
    return std::filesystem::path(buffer);
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        throw std::runtime_error("GetExecutablePath: _NSGetExecutablePath failed.");
    }
    buffer.resize(std::strlen(buffer.c_str()));
    return std::filesystem::path(buffer);
#else
    std::array<char, 4096> buffer{};
    const ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size());
    if (length < 0) {
        throw std::runtime_error("GetExecutablePath: readlink(/proc/self/exe) failed.");
    }
    if (static_cast<size_t>(length) >= buffer.size()) {
        throw std::runtime_error("GetExecutablePath: executable path is too long.");
    }
    return std::filesystem::path(std::string(buffer.data(), static_cast<size_t>(length)));
#endif
}

std::filesystem::path GetExecutableDir()
{
    return GetExecutablePath().parent_path();
}

std::filesystem::path GetExecutableRootDir()
{
    return GetExecutableDir().parent_path().parent_path();
}

std::filesystem::path GetExecutableConfigDir()
{
    return GetExecutableRootDir() / "config";
}

namespace internal {

std::filesystem::path ResolveAppDataDir(
    const std::filesystem::path& executable_root,
    const std::optional<std::filesystem::path>& user_config_root)
{
    // 配布物の隣に appdata が用意されている場合だけ portable mode とする。
    const auto portable_dir = executable_root / "appdata";
    if (std::filesystem::is_directory(portable_dir)) {
        return portable_dir;
    }

    if (!user_config_root.has_value() || user_config_root->empty()) {
        ANET_SYSTEM_ERROR("GetAppDataDir: user configuration directory is unavailable.");
    }

    // user mode の保存先は利用前に確実に作成する。
    const auto app_data_dir = *user_config_root / "anet-lab" / "runner";
    std::filesystem::create_directories(app_data_dir);
    return app_data_dir;
}

} // namespace internal

std::filesystem::path GetAppDataDir()
{
#ifdef _WIN32
    const auto user_config_root = GetEnvironmentPath("APPDATA");
#else
    auto user_config_root = GetEnvironmentPath("XDG_CONFIG_HOME");
    if (!user_config_root.has_value()) {
        const auto home = GetEnvironmentPath("HOME");
        if (home.has_value()) {
            user_config_root = *home / ".config";
        }
    }
#endif
    return internal::ResolveAppDataDir(GetExecutableRootDir(), user_config_root);
}

AppConfigMode DetermineAppConfigMode(
    bool has_config,
    bool has_workspace,
    bool force_workspace_selection)
{
    // 指定された入力源を列挙し、競合時に利用者が直せる診断を組み立てる。
    std::vector<std::string> specified_options;
    if (has_config) {
        specified_options.push_back("--config");
    }
    if (has_workspace) {
        specified_options.push_back("--workspace");
    }
    if (force_workspace_selection) {
        specified_options.push_back("--select-workspace");
    }
    if (specified_options.size() > 1) {
        std::ostringstream oss;
        for (size_t i = 0; i < specified_options.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << specified_options[i];
        }
        ANET_SYSTEM_ERROR(
            "Conflicting application config sources. specified=" << oss.str()
            << " expected=exactly one of --config, --workspace, or --select-workspace");
    }

    // direct config を先に返し、呼出側が workspace 状態を構築しない分岐を固定する。
    if (has_config) {
        return AppConfigMode::DirectConfig;
    }
    if (has_workspace) {
        return AppConfigMode::ExplicitWorkspace;
    }
    return AppConfigMode::WorkspaceFlow;
}

WorkspaceService::WorkspaceService(
    std::filesystem::path runner_root,
    std::filesystem::path app_data_dir)
    : runner_root_(std::move(runner_root))
    , app_data_dir_(std::move(app_data_dir))
{
}

std::optional<std::string> WorkspaceService::GetNewWorkspaceNameError(
    const std::string& raw_input) const
{
    // workspace 共通の禁止形式を先に判定し、UI と Resolve で同じ理由を返す。
    const auto input = Trim(raw_input);
    if (const auto error = GetWorkspacePathError(input)) {
        return error;
    }

    // 新規作成欄は OS によらず単一名に限定し、path 指定は履歴または参照へ分離する。
    if (input.find('/') != std::string::npos || input.find('\\') != std::string::npos) {
        return "value must not contain path separators ('/' or '\\')";
    }
    const auto input_path = PathFromUtf8(input);
    if (HasWindowsDrivePrefix(input) || !IsSingleRelativeName(input_path)) {
        return "value must be a single relative workspace name other than '.' or '..'";
    }
    return std::nullopt;
}

std::string WorkspaceService::ValidateAndTrim(const std::string& raw_input) const
{
    // 永続化・解決に使う値を先に正規化し、禁止形式は filesystem 操作前に拒否する。
    const auto input = Trim(raw_input);
    if (const auto error = GetWorkspacePathError(input)) {
        ANET_SYSTEM_ERROR(
            "Invalid workspace path. value=\"" << raw_input << "\" reason=" << *error
            << " expected=non-empty local path without '#', '//', trailing ';', or UNC root");
    }
    return input;
}

WorkspacePaths WorkspaceService::Resolve(const std::string& raw_input, bool allow_create) const
{
    // UTF-8 入力を native path へ変換し、相対指定だけを application root の workspaces 基準へ展開する。
    const auto input = ValidateAndTrim(raw_input);
    const auto input_path = PathFromUtf8(input);
    const bool relative = input_path.is_relative() && !input_path.has_root_name();
    auto root = relative ? runner_root_ / "workspaces" / input_path : input_path;
    root = root.lexically_normal();

    // 新規作成は相対1語だけに限定し、絶対・多階層の誤指定を自動生成しない。
    if (!std::filesystem::exists(root)) {
        if (!allow_create) {
            ANET_SYSTEM_ERROR(
                "Workspace directory does not exist. input=\"" << input
                << "\" resolved_path=" << PathToUtf8(root));
        }
        if (const auto error = GetNewWorkspaceNameError(input)) {
            ANET_SYSTEM_ERROR(
                "Invalid new workspace name. value=\"" << input << "\" reason=" << *error
                << " expected=single relative workspace name");
        }
        CreateWorkspace(root);
    }
    if (!std::filesystem::is_directory(root)) {
        ANET_SYSTEM_ERROR("Workspace path is not a directory. path=" << PathToUtf8(root));
    }

    // 既存ディレクトリも、明示選択された場合だけ不足する workspace config を補完する。
    const auto config_file = root / "config" / "_main.txt";
    if (std::filesystem::exists(config_file) && !std::filesystem::is_regular_file(config_file)) {
        ANET_SYSTEM_ERROR(
            "Workspace config path is not a regular file. path=" << PathToUtf8(config_file));
    }
    if (!std::filesystem::exists(config_file)) {
        if (allow_create) {
            CreateWorkspace(root);
        } else {
            ANET_SYSTEM_ERROR(
                "Workspace config file does not exist. workspace=" << PathToUtf8(root)
                << " expected=" << PathToUtf8(config_file));
        }
    }
    if (!std::filesystem::is_regular_file(config_file)) {
        ANET_SYSTEM_ERROR(
            "Workspace config file does not exist. workspace=" << PathToUtf8(root)
            << " expected=" << PathToUtf8(config_file));
    }
    const auto runs_dir = root / "runs";
    std::filesystem::create_directories(runs_dir);

    // config 内へ注入する path は UTF-8 とし、相対指定の可搬性を維持する。
    const auto runs_config_value = relative
        ? PathToUtf8(std::filesystem::relative(runs_dir, runner_root_))
        : PathToUtf8(std::filesystem::absolute(runs_dir).lexically_normal());
    return WorkspacePaths{
        .input = input,
        .root = std::filesystem::absolute(root).lexically_normal(),
        .config_file = std::filesystem::absolute(config_file).lexically_normal(),
        .runs_dir = std::filesystem::absolute(runs_dir).lexically_normal(),
        .runs_config_value = runs_config_value,
    };
}

bool WorkspaceService::IsResolvable(const std::string& raw_input) const
{
    // UI や CLI の可用性確認では生成せず、完成済みまたはテンプレートで補完可能かだけを確認する。
    const auto input = Trim(raw_input);
    if (GetWorkspacePathError(input)) {
        return false;
    }
    const auto input_path = PathFromUtf8(input);
    const bool relative = input_path.is_relative() && !input_path.has_root_name();
    const auto root = (relative ? runner_root_ / "workspaces" / input_path : input_path).lexically_normal();
    if (!std::filesystem::is_directory(root)) {
        return false;
    }
    const auto config_dir = root / "config";
    const auto config_file = config_dir / "_main.txt";
    if (std::filesystem::is_regular_file(config_file)) {
        return true;
    }
    if (std::filesystem::exists(config_file)
        || (std::filesystem::exists(config_dir) && !std::filesystem::is_directory(config_dir))) {
        return false;
    }
    const auto runs_dir = root / "runs";
    if (std::filesystem::exists(runs_dir) && !std::filesystem::is_directory(runs_dir)) {
        return false;
    }
    return std::filesystem::is_regular_file(runner_root_ / "config" / "_workspace_template.txt");
}

std::vector<std::string> WorkspaceService::ScanLocalWorkspaces() const
{
    // 過去 Run だけを移した未初期化フォルダも含め、直下の全ディレクトリを列挙する。
    std::vector<std::string> result;
    const auto workspaces_dir = runner_root_ / "workspaces";
    if (!std::filesystem::is_directory(workspaces_dir)) {
        return result;
    }
    for (const auto& entry : std::filesystem::directory_iterator(workspaces_dir)) {
        if (entry.is_directory()) {
            result.push_back(PathToUtf8(entry.path().filename()));
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::string> WorkspaceService::LoadHistory() const
{
    // application data が無い初回起動は空履歴として扱う。
    std::vector<std::string> history;
    const auto path = app_data_dir_ / "history.txt";
    if (!std::filesystem::is_regular_file(path)) {
        return history;
    }

    ConfigData data;
    try {
        data = Properties(path).ToConfigData();
    } catch (const std::exception& e) {
        LOG::warn() << "Workspace history could not be read. path=" << PathToUtf8(path)
            << " error=" << e.what() << ". Falling back to an empty history.";
        return history;
    }

    // 手編集で壊れた entry だけを WARN して除外し、他の有効な履歴は利用する。
    for (size_t i = 0; i < kMaxHistoryEntries; ++i) {
        const auto key = "workspace.history." + std::to_string(i);
        if (!data.Has(key)) {
            continue;
        }
        const auto raw_input = data.Get(key);
        const auto input = Trim(raw_input);
        if (const auto error = GetWorkspacePathError(input)) {
            LOG::warn() << "Workspace history entry ignored. path=" << PathToUtf8(path)
                << " key=" << key << " value=\"" << raw_input << "\" reason=" << *error;
            continue;
        }
        history.push_back(input);
    }
    return history;
}

void WorkspaceService::RecordHistory(const std::string& adopted_input) const
{
    // 採用値をMRU先頭へ移動し、完全一致の重複と上限超過を除去する。
    const auto input = ValidateAndTrim(adopted_input);
    auto history = LoadHistory();
    history.erase(std::remove(history.begin(), history.end(), input), history.end());
    history.insert(history.begin(), input);
    if (history.size() > kMaxHistoryEntries) {
        history.resize(kMaxHistoryEntries);
    }

    // Properties の原子的置換で履歴責務だけを保存する。
    ConfigData data;
    for (size_t i = 0; i < history.size(); ++i) {
        data.Set("workspace.history." + std::to_string(i), history[i]);
    }
    data.SaveProperties(app_data_dir_ / "history.txt");
}

void WorkspaceService::SaveLastWorkspace(const WorkspacePaths& workspace) const
{
    // 補助 launcher との受け渡しは、解決済み絶対 path の UTF-8 bytes に固定する。
    const auto path = app_data_dir_ / "last_workspace.txt";
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    ANET_CHECK_MSG(ofs, "Failed to open last workspace file. path=" << PathToUtf8(path));
    ofs << PathToUtf8(workspace.root);
    ofs.flush();
    ANET_CHECK_MSG(ofs, "Failed to write last workspace file. path=" << PathToUtf8(path));
}

void WorkspaceService::CreateWorkspace(const std::filesystem::path& root) const
{
    // 追跡対象テンプレートの存在を確認してから箱を作る。
    const auto template_path = runner_root_ / "config" / "_workspace_template.txt";
    if (!std::filesystem::is_regular_file(template_path)) {
        ANET_SYSTEM_ERROR("Workspace template does not exist. path=" << PathToUtf8(template_path));
    }

    // 既存内容を保持したまま不足するテンプレートを配置し、最後に runs を用意する。
    std::filesystem::create_directories(root / "config");
    std::filesystem::copy_file(
        template_path,
        root / "config" / "_main.txt",
        std::filesystem::copy_options::none);
    std::filesystem::create_directories(root / "runs");
}

std::unique_ptr<ConfigManager> CreateWorkspaceConfigManager(
    const WorkspacePaths& workspace,
    const std::filesystem::path& common_config_dir,
    const wxCmdLineParser* cmd_line)
{
    // 共通設定へ runs 導出値と workspace config を順番に重ねる。
    ConfigManagerOptions options;
    options.config_search_dirs = { common_config_dir };
    options.injected_config.Set("app.runs_dir", workspace.runs_config_value);
    options.overwrite_config_paths = { workspace.config_file };
    auto manager = std::make_unique<ConfigManager>(
        common_config_dir / "_main.txt", cmd_line, options);

    // AutoMerge と CLI 完了後に、自己完結 workspace の不変条件を検証する。
    ValidateWorkspaceRunsDir(manager->GetConfigData(), workspace);
    return manager;
}

void ValidateWorkspaceRunsDir(
    const ConfigData& config_data,
    const WorkspacePaths& workspace)
{
    // 表記を正規化せず、注入した UTF-8 文字列との完全一致を要求する。
    const auto actual = config_data.Get("app.runs_dir");
    if (actual != workspace.runs_config_value) {
        ANET_SYSTEM_ERROR(
            "Workspace config changed app.runs_dir. actual=\"" << actual
            << "\" expected=\"" << workspace.runs_config_value << "\"");
    }
}

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

            // 子プロセス(ffmpeg 等。wxExecute は bInheritHandles=TRUE で起動)にこのwriteハンドルが継承されると、
            // Stop() 時に read 側が EOF に達せず ReadLoop/join がハングするので継承を切る
            HANDLE dup_h = reinterpret_cast<HANDLE>(_get_osfhandle(target_fd_));
            if (dup_h != INVALID_HANDLE_VALUE) SetHandleInformation(dup_h, HANDLE_FLAG_INHERIT, 0);
            SetHandleInformation(write_pipe_, HANDLE_FLAG_INHERIT, 0);  // 念のため両方

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
