#include "anet/exception.hpp"
#include <sstream>
#include <wx/stackwalk.h>

using namespace anet;

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
