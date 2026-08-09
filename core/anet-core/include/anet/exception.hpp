// anet/exception.hpp

#pragma once

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

}   // namespace anet

