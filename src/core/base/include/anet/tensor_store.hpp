#pragma once

#include <cstdint>
#include <span>
#include "anet/tensor_util.hpp"

namespace anet {

    struct LaneTimeIndex {
        int64_t lane;
        int64_t time;
    };

    /// 同じキー・型・形・device の lane 一式を時系列にため、指定した位置をまとめて読む。
    class LaneTimeTensorStore {
    public:
        explicit LaneTimeTensorStore(int64_t capacity);
        LaneTimeTensorStore(const LaneTimeTensorStore&) = delete;
        LaneTimeTensorStore& operator=(const LaneTimeTensorStore&) = delete;
        LaneTimeTensorStore(LaneTimeTensorStore&&) = default;
        LaneTimeTensorStore& operator=(LaneTimeTensorStore&&) = default;

        void Append(const TensorDict& lanes);
        TensorDict Gather(std::span<const LaneTimeIndex> indices) const;
        /// たまっている範囲の全体を [時刻, lane, ...] の形で返す。保存領域を共有し、複製しない。
        /// 返した値は、次の Append や Clear のあとに中身が変わりうる。
        TensorDict GetAll() const;
        void Clear() { time_count_ = 0; }
        int64_t GetTimeCount() const { return time_count_; }
    private:
        int64_t capacity_;
        int64_t lane_count_ = 0;
        int64_t time_count_ = 0;
        TensorDict storage_;
    };

} // namespace anet
