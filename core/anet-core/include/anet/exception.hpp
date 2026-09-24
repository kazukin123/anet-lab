// anet/exception.hpp

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace anet {

    // スタックトレースを自動保持する例外クラス
    class AnetException : public std::runtime_error {
    public:
        explicit AnetException(const std::string& message);
        const char* stack_trace() const noexcept;
        std::string full_info() const;
    private:
        std::string stack_trace_;
    };


    // ============================================================
    // 確保失敗の診断
    // ============================================================

    /// operator new の確保失敗として記録した内容。
    struct AllocationFailureInfo {
        size_t last_request_bytes = 0;  ///< 直近に失敗した確保の要求バイト数
        uint64_t failure_count = 0;     ///< プロセス開始からの確保失敗回数
    };

    /// operator new の確保失敗を stderr とログへ記録するハンドラを登録する。
    /// bad_alloc は catch した時点でスタックが巻き戻っており確保地点を取れないため、
    /// 例外が投げられる前の失敗スレッド上で要求サイズとスタックを残す。
    /// 多重呼び出しは無視する。
    void InstallAllocationFailureLogger();

    /// これまでに記録した確保失敗の要約を返す。
    AllocationFailureInfo GetAllocationFailureInfo();

}   // namespace anet

