#include "anet/exception.hpp"
#include <atomic>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string_view>
#include <wx/stackwalk.h>
#include "anet/log.hpp"

#ifdef _MSC_VER
#include <new.h>
#include <windows.h>
#endif

using namespace anet;
namespace LOG = anet::log;

// --- 内部クラス: スタックウォーカーの実装 ---
// (ヘッダに公開する必要がないのでここに隠蔽します)
class StackDumpWalker : public wxStackWalker {
public:
    StackDumpWalker(std::stringstream& ss)
        : ss_(ss)
    {
    }
protected:
    void OnStackFrame(const wxStackFrame& frame) override
    {
        // レベル、関数名
        ss_ << "[" << frame.GetLevel() << "] " << frame.GetName();

        // ファイル名と行番号（デバッグ情報がある場合のみ）
        if (frame.HasSourceLocation()) {
            ss_ << " (" << frame.GetFileName() << ":" << frame.GetLine() << ")";
        }
        ss_ << "\n";
    }

private:
    std::stringstream& ss_;
};

// --- AnetException 実装 ---

AnetException::AnetException(const std::string& message)
    : std::runtime_error(message)
{
    std::stringstream ss;

    // インスタンス化された瞬間にウォーカーを走らせる
    StackDumpWalker walker(ss);

    // Walk(2) の意味:
    // 0: OnStackFrame
    // 1: Walk() 自身
    // 2: AnetException のコンストラクタ (これ)
    // 呼び出し元(ANET_CHECKなど)から表示したいので 2 フレームスキップします
    walker.Walk(2);

    stack_trace_ = ss.str();
}

const char* AnetException::stack_trace() const noexcept
{
    return stack_trace_.c_str();
}

std::string AnetException::full_info() const
{
    std::stringstream ss;
    ss << what() << "\n\n--- Stack Trace ---\n" << stack_trace_;
    return ss.str();
}


// --- 確保失敗ハンドラ実装 ---

namespace anet::alloc_failure {

    // 失敗の記録。bad_alloc を捕まえた側(showFatalError)から読むためプロセス全体で保持する。
    std::atomic<size_t> last_request_bytes{ 0 };
    std::atomic<uint64_t> failure_count{ 0 };

#ifdef _MSC_VER

    // スタックは最初の失敗 1 回だけ出す。2 回目以降は要求サイズの行だけ残す。
    std::atomic<bool> stack_reported{ false };

    // 記録処理自体が確保に失敗しても handler を再帰させないための再入ガード。
    thread_local bool in_handler = false;

    // 1 回の採取で辿るフレーム数と、その整形結果を置くスタックバッファの容量。
    constexpr size_t kMaxStackFrames = 32;
    constexpr size_t kStackTextCapacity = 4096;

    /// in_handler を例外経路でも確実に戻すための RAII ガード。
    class ReentryGuard {
    public:
        ReentryGuard() { in_handler = true; }
        ~ReentryGuard() { in_handler = false; }
        ReentryGuard(const ReentryGuard&) = delete;
        ReentryGuard& operator=(const ReentryGuard&) = delete;
    };

    /// stderr へ確保せずに書く。
    /// stderr は StandardStreamLogger が run dir の stderr.log へ捕捉し無バッファ化しているので、
    /// メモリ枯渇中でもこの層だけは残る。
    void WriteStderr(const char* data, size_t size)
    {
        std::fwrite(data, 1, size, stderr);
    }

    /// フルパスからファイル名部分の先頭を返す。
    const char* BaseName(const char* path, size_t length)
    {
        const char* name = path;
        for (size_t i = 0; i < length; ++i) {
            if (path[i] == '/' || path[i] == '\\') {
                name = path + i + 1;
            }
        }
        return name;
    }

    /// 現在のスタックを "module+offset" 形式で text へ整形し、書いた長さを返す。
    /// DbgHelp を使わないので確保もシンボル解決も伴わず、メモリ枯渇中でも動く。
    /// 関数名と行番号は、出力の module 名と offset を PDB へ突き合わせて事後に解決する。
    size_t FormatCurrentStack(char* text, size_t capacity)
    {
        // skip=0。handler 自身のフレームも残しておくと、どの層から来たか追いやすい。
        void* frames[kMaxStackFrames] = {};
        const USHORT count = ::RtlCaptureStackBackTrace(
            0, static_cast<DWORD>(kMaxStackFrames), frames, nullptr);

        size_t length = 0;
        for (USHORT i = 0; i < count; ++i) {
            // アドレスの所属モジュールを引く。参照カウントは増やさない。
            HMODULE module = nullptr;
            char path[MAX_PATH] = {};
            const char* name = "<unknown>";
            size_t offset = reinterpret_cast<size_t>(frames[i]);
            if (::GetModuleHandleExA(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                static_cast<LPCSTR>(frames[i]), &module)) {
                const DWORD path_length =
                    ::GetModuleFileNameA(module, path, static_cast<DWORD>(sizeof(path)));
                name = BaseName(path, path_length);
                offset -= reinterpret_cast<size_t>(module);
            }

            // バッファに収まらなくなった時点で打ち切る。切り詰めても前半の frame は残る。
            const int written = std::snprintf(text + length, capacity - length,
                "  [%u] %s+0x%zx\n", static_cast<unsigned>(i), name, offset);
            if (written <= 0 || static_cast<size_t>(written) >= capacity - length) {
                break;
            }
            length += static_cast<size_t>(written);
        }
        return length;
    }

    /// 最初の失敗だけスタックを stderr と通常ログの両方へ残す。
    void ReportStackOnce(size_t size)
    {
        bool expected = false;
        if (!stack_reported.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return;
        }

        // 1) 確保ゼロの層。整形もスタックバッファ上で行うので枯渇中でも残る。
        char text[kStackTextCapacity];
        const size_t length = FormatCurrentStack(text, sizeof(text));
        WriteStderr(text, length);

        // 2) 通常ログへも同じ内容を出す。wxLog は確保するので best-effort。
        //    stderr を先に書いてあるので、ここで失敗しても記録自体は残る。
        try {
            LOG::error() << "Allocation failed. requested_bytes=" << size
                << "\n--- Stack (module+offset) ---\n" << std::string_view(text, length);
        } catch (...) {
            const char kLogFailed[] = "[E] Allocation failure log write failed.\n";
            WriteStderr(kLogFailed, sizeof(kLogFailed) - 1);
        }
    }

    /// operator new の確保失敗時に CRT の _callnewh() から呼ばれる。
    /// 0 を返すと CRT が従来どおり std::bad_alloc を投げるため、例外の送出経路は変わらない。
    int OnAllocationFailure(size_t size)
    {
        // 1) 再入していたら何もしない。記録側の確保失敗で handler が再帰するのを止める。
        if (in_handler) {
            return 0;
        }
        ReentryGuard guard;

        // 2) showFatalError() が読む記録を更新する。
        last_request_bytes.store(size, std::memory_order_relaxed);
        const uint64_t count = failure_count.fetch_add(1, std::memory_order_relaxed) + 1;

        // 3) 確保ゼロの層。枯渇していてもここだけは必ず残す。
        char line[128];
        const int length = std::snprintf(line, sizeof(line),
            "[E] Allocation failed. requested_bytes=%zu failure_count=%llu\n",
            size, static_cast<unsigned long long>(count));
        if (length > 0) {
            WriteStderr(line, static_cast<size_t>(length));
        }

        // 4) 初回だけスタックを採る。
        ReportStackOnce(size);

        // 5) 0 を返して CRT に std::bad_alloc を投げさせる。
        return 0;
    }

#endif // _MSC_VER

} // namespace anet::alloc_failure

void anet::InstallAllocationFailureLogger()
{
#ifdef _MSC_VER
    // handler はプロセス全体で 1 つなので、登録も 1 回だけにする。
    static std::once_flag installed;
    std::call_once(installed, [] {
        _set_new_handler(&alloc_failure::OnAllocationFailure);
        });
#endif
}

anet::AllocationFailureInfo anet::GetAllocationFailureInfo()
{
    return AllocationFailureInfo{
        .last_request_bytes = alloc_failure::last_request_bytes.load(std::memory_order_relaxed),
        .failure_count = alloc_failure::failure_count.load(std::memory_order_relaxed),
    };
}
