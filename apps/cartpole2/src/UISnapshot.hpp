// UISnapshot.hpp
#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include "anet/rl.hpp"

struct UISnapshot {
    anet::rl::StepCounts counts;                // 現在のステップ状況
    anet::rl::BatchExperience train_exp;        // CPU化済みの Experience（描画対象）
    //float train_reward_ema = 0.0f;    // 学習報酬EMA（UI表示用）
    //std::shared_ptr<const anet::rl::Agent> agent; // 観測専用(sharedでOK)
};

class UISnapshotStore {
public:
    UISnapshotStore() = default;

    // Trainer側から更新（コピーを保持）
    void Update(const UISnapshot& s) {
        std::lock_guard<std::mutex> lock(m_);
        snap_ = s;
        data_exists_.store(true);
    }

    // UI側から取得（コピーを返す）
    std::optional<UISnapshot> Get() const {
        if (data_exists_.load() == false) return std::nullopt;
        std::lock_guard<std::mutex> lock(m_);
        return snap_;
    }

private:
    mutable std::mutex m_;
    UISnapshot snap_; // 最新状態の保持
    std::atomic<bool> data_exists_{ false };
};
