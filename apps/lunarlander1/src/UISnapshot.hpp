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

struct UISegment {
    float x0, y0;
    float x1, y1;
};

struct Legs {
    UISegment left_leg;       // 左脚の座標情報
    UISegment right_leg;      // 右脚の座標情報
};

struct TerrainPolyline {
    std::vector<TerrainPoint> points;
};

struct UISnapshot
{
    // Step
    anet::rl::step_t step;

    // RL由来情報
    anet::rl::SingleState state;
    std::int64_t action;
    float reward;

    // ENV由来情報
    std::unordered_map<std::string, torch::Tensor> aux;

    // 表示用情報
    std::optional<float> total_reward;

    /// @todo 毎回取る情報と取らない情報を分離整理
};

class UISnapshotStore {
public:
    UISnapshotStore() = default;

    bool IsDataRequest() {
        return data_request_.load();
    }

    // Trainer側から更新（コピーを保持）
    void Update(const UISnapshot& s) {
        std::lock_guard<std::mutex> lock(m_);
        snap_ = s;
        data_exists_.store(true);
        data_request_.store(false);
    }

    // UI側から取得（コピーを返す）
    std::optional<UISnapshot> Get() {
        if (data_exists_.load() == false) return std::nullopt;
        std::lock_guard<std::mutex> lock(m_);
        data_request_.store(true);
        return snap_;
    }

private:
    mutable std::mutex m_;
    UISnapshot snap_; // 最新状態の保持
    std::atomic<bool> data_exists_{ false };
    std::atomic<bool> data_request_{ true };
};
