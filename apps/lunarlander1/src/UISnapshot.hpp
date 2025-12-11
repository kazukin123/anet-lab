// UISnapshot.hpp
#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include "anet/rl.hpp"
#include "LunarLanderEnv.hpp"

struct TerrainPoint {
    float x;
    float y;
};

struct TerrainPolyline {
    std::vector<TerrainPoint> points;
};

struct UISnapshot
{
    // Step
    anet::rl::step_t step;

    // RL由来情報
    anet::rl::Experience exp;

    // ENV非可観測情報
    std::optional<TerrainPolyline> terrain;  // episode_start時のみセット
    LunarLanderEnv::PadInfo pad;             // episode_startでセット
    float wind_x;                            // 毎step更新

    // world transform (Canvas描画座標系)
    float world_min_x;
    float world_max_x;
    float world_min_y;
    float world_max_y;
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
