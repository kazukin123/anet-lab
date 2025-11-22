#include <stdexcept>
#include <wx/log.h>
#include "anet/rl.hpp"
#include "anet/common.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/tensor_check.hpp"

namespace anet::rl {

    nlohmann::json StateDimInfo::ToJson() const {
        nlohmann::json j;
        j["coords"] = coords;
        j["min_value"] = min_value;
        j["max_value"] = max_value;
        j["name"] = name;
        j["description"] = description;
        return j;
    }

    std::string StateDimInfo::ToString() const {
        return ToJson().dump(2); // 2-space indent for pretty print
    }

    int64_t StateSpec::CalcStateDim() const
    {
        ANET_ASSERT_MSG(!shape.empty(),
            "StateSpec::CalcStateDim: shape must not be empty.");

        int64_t dim = 1;
        for (auto v : shape) {
            ANET_ASSERT_MSG(v > 0,
                "StateSpec::CalcStateDim: shape elements must be > 0.");
            dim *= v;
        }
        return dim;
    }

    const StateDimInfo* StateSpec::FindDim(const std::vector<int64_t>& coords) const {
        for (auto& d : dims)
            if (d.coords == coords)
                return &d;
        return nullptr;
    }

    bool StateSpec::MatchesShape(const torch::Tensor& obs) const
    {
        ANET_ASSERT_MSG(
            obs.dim() == static_cast<int64_t>(shape.size()),
            "StateSpec::MatchesShape: dimension mismatch.");

        for (size_t i = 0; i < shape.size(); i++) {
            int64_t e = shape[i];
            int64_t a = obs.size(i);
            if (e == ANET_SHAPE_ANY) continue;
            ANET_ASSERT_MSG(
                e == a,
                "StateSpec::MatchesShape: shape mismatch.");
        }
        return true;
    }

    bool StateSpec::MatchesRange(const torch::Tensor& obs) const
    {
        ANET_CHECK_DTYPE(obs, torch::kFloat32);

        if (dims.empty()) return true;

        for (const auto& d : dims) {
            float mn = d.min_value;
            float mx = d.max_value;

            if (d.coords.empty()) {
                // @todo 最適化の余地あり（全要素スキャン）
                auto flat = obs.flatten();
                const int64_t n = flat.size(0);
                for (int64_t i = 0; i < n; i++) {
                    float v = flat[i].item<float>();
                    ANET_ASSERT_MSG(
                        v >= mn && v <= mx,
                        "StateSpec::MatchesRange: value out of range.");
                }
                continue;
            }

            // coords が多次元 index の場合
            // 例: coords = {c, h, w}, obs[c][h][w]
            ANET_ASSERT_MSG(
                static_cast<int64_t>(d.coords.size()) <= obs.dim(),
                "StateSpec::MatchesRange: coords dim mismatch.");

            const torch::Tensor* cur = &obs;
            torch::Tensor tmp;  // スコープ内で生存する一時保持用

            for (size_t k = 0; k < d.coords.size(); k++) {
                int64_t idx = d.coords[k];

                ANET_ASSERT_MSG(
                    idx >= 0 && idx < cur->size(0),
                    "StateSpec::MatchesRange: coords index OOB.");

                // select の結果を tmp に保持
                tmp = cur->select(0, idx);

                // cur を tmp へ向ける
                cur = &tmp;
            }

            float v = cur->item<float>();
            ANET_ASSERT_MSG(
                v >= mn && v <= mx,
                "StateSpec::MatchesRange: coord value out of range.");
        }
        return true;
    }

    bool StateSpec::MatchesRangeFlat(const torch::Tensor& flat_obs) const
    {
        ANET_CHECK_DTYPE(flat_obs, torch::kFloat32);
        ANET_ASSERT_MSG(
            flat_obs.dim() == 1,
            "StateSpec::MatchesRangeFlat: expected 1D tensor.");

        auto data = flat_obs;
        const int64_t total = data.size(0);

        if (dims.empty()) return true;

        for (const auto& d : dims) {
            float mn = d.min_value;
            float mx = d.max_value;

            // coords 指定なし → 全要素検査
            if (d.coords.empty()) {
                for (int64_t i = 0; i < total; i++) {
                    float v = data[i].item<float>();
                    ANET_ASSERT_MSG(
                        v >= mn && v <= mx,
                        "StateSpec::MatchesRangeFlat: value out of range.");
                }
                continue;
            }

            // coords 指定あり → 1D として扱う
            for (auto idx : d.coords) {
                ANET_ASSERT_MSG(
                    idx >= 0 && idx < total,
                    "StateSpec::MatchesRangeFlat: coords index OOB.");
                float v = data[idx].item<float>();
                ANET_ASSERT_MSG(
                    v >= mn && v <= mx,
                    "StateSpec::MatchesRangeFlat: coord value out of range.");
            }
        }
        return true;
    }

    nlohmann::json StateSpec::ToJson() const {
        nlohmann::json j;
        j["shape"] = shape;

        j["dims"] = nlohmann::json::array();
        for (const auto& d : dims) {
            j["dims"].push_back(d.ToJson());
        }

        j["options"] = nlohmann::json::object();
        for (const auto& kv : options) {
            j["options"][kv.first] = kv.second;
        }

        return j;
    }

    std::string StateSpec::ToString() const {
        return ToJson().dump(2); // pretty JSON
    }

    nlohmann::json ActionDimInfo::ToJson() const {
        nlohmann::json j;
        j["min_value"] = min_value;
        j["max_value"] = max_value;
        j["name"] = name;
        j["description"] = description;
        return j;
    }

    std::string ActionDimInfo::ToString() const {
        return ToJson().dump(2);
    }

    nlohmann::json ActionSpec::ToJson() const {
        nlohmann::json j;
        j["is_discrete"] = is_discrete;

        // 離散アクションラベル
        j["value_labels"] = value_labels;

        // 連続アクション次元
        j["dims"] = nlohmann::json::array();
        for (const auto& d : dims) {
            j["dims"].push_back(d.ToJson());
        }

        // オプション
        j["options"] = nlohmann::json::object();
        for (const auto& kv : options) {
            j["options"][kv.first] = kv.second;
        }

        return j;
    }

    std::string ActionSpec::ToString() const {
        return ToJson().dump(2);
    }

    nlohmann::json EnvSpec::ToJson() const {
        nlohmann::json j;

        j["state_spec"] = state.ToJson();
        j["action_spec"] = action.ToJson();

        j["reward_range"] = {
            reward_range.first,
            reward_range.second
        };

        j["options"] = nlohmann::json::object();
        for (const auto& kv : options) {
            j["options"][kv.first] = kv.second;
        }

        return j;
    }

    std::string EnvSpec::ToString() const {
        return ToJson().dump(2);
    }
    // -----------------------------------------

    std::string BatchState::ToString() const
    {
        std::ostringstream oss;
        oss << "BatchState{";
        oss << "obs=" << anet::ToString(obs);
        oss << ", done=" << anet::ToString(done);
        oss << ", truncated=" << anet::ToString(truncated);
        oss << ", episode_start=" << anet::ToString(episode_start);
        oss << "}";
        return oss.str();
    }

    std::string BatchActionInfo::ToString() const
    {
        std::ostringstream oss;
        oss << "BatchActionInfo{";
        oss << "action=" << anet::ToString(action);
        oss << ", is_random=" << anet::ToString(is_random);
        oss << "}";
        return oss.str();
    }

    std::string BatchStepResult::ToString() const
    {
        std::ostringstream oss;
        oss << "BatchStepResult{";
        oss << "next_state=" << next_state.ToString();
        oss << ", reward=" << anet::ToString(reward);
        oss << "}";
        return oss.str();
    }
    std::string SingleState::ToString() const
    {
        std::ostringstream oss;
        oss << "SingleState{";
        oss << "obs=" << anet::ToString(obs);
        oss << ", done=" << done;
        oss << ", truncated=" << truncated;
        oss << ", episode_start=" << episode_start;
        oss << "}";
        return oss.str();
    }

    std::string Experience::ToString() const
    {
        std::ostringstream oss;
        oss << "Experience{";
        oss << "state=" << state.ToString();
        oss << ", action_shape=" << action.sizes();
        oss << ", reward=" << reward;
        oss << ", next_state=" << next_state.ToString();
        oss << "}";
        return oss.str();
    }

    // -----------------------------------------

    std::vector<Experience> BatchExperience::ToExperienceList() const
    {
        // ---- N (batch 次元) の取得 ----
        ANET_CHECK_DTYPE(state.obs, torch::kFloat32);
        ANET_CHECK_DTYPE(next_state.obs, torch::kFloat32);
        ANET_CHECK_SHAPE(state.done, { ANET_SHAPE_ANY });
        ANET_CHECK_SHAPE(state.truncated, { ANET_SHAPE_ANY });
        ANET_CHECK_SHAPE(state.episode_start, { ANET_SHAPE_ANY });

        const int64_t N = state.obs.size(0);

        // ---- batch 次元の整合検査 ----
        ANET_ASSERT_MSG(next_state.obs.size(0) == N,
            "MakeFromBatch: state.obs and next_states.obs batch size mismatch.");
        ANET_ASSERT_MSG(state.done.size(0) == N,
            "MakeFromBatch: state.done batch size mismatch.");
        ANET_ASSERT_MSG(state.truncated.size(0) == N,
            "MakeFromBatch: state.truncated batch size mismatch.");
        ANET_ASSERT_MSG(state.episode_start.size(0) == N,
            "MakeFromBatch: state.episode_start batch size mismatch.");
        ANET_ASSERT_MSG(next_state.done.size(0) == N,
            "MakeFromBatch: next_state.done batch size mismatch.");
        ANET_ASSERT_MSG(next_state.truncated.size(0) == N,
            "MakeFromBatch: next_state.truncated batch size mismatch.");
        ANET_ASSERT_MSG(next_state.episode_start.size(0) == N,
            "MakeFromBatch: next_state.episode_start batch size mismatch.");

        // ---- actions の整合検査 ----
        ANET_CHECK_DTYPE(action.action, torch::kInt64);
        ANET_ASSERT_MSG(action.action.size(0) == N,
            "MakeFromBatch: action.action batch size mismatch.");

        // ---- rewards の shape チェック ----
        ANET_CHECK_DTYPE(reward, torch::kFloat32);
        ANET_ASSERT_MSG(reward.size(0) == N,
            "MakeFromBatch: reward batch size mismatch.");

        // ---- obs の最低限の次元検査 ----
        ANET_ASSERT_MSG(state.obs.dim() >= 2,
            "MakeFromBatch: state.obs must have at least 2 dims (N, ...).");
        ANET_ASSERT_MSG(next_state.obs.dim() >= 2,
            "MakeFromBatch: next_state.obs must have at least 2 dims (N, ...).");

        // ---- flatten 前の要素数チェック（破損検出）----
        ANET_ASSERT_MSG(state.obs.numel() % N == 0,
            "MakeFromBatch: state.obs total elements not divisible by batch size.");
        ANET_ASSERT_MSG(next_state.obs.numel() % N == 0,
            "MakeFromBatch: next_state.obs total elements not divisible by batch size.");

        // ---- main loop ----
        std::vector<Experience> out;
        out.reserve(N);

        for (int64_t i = 0; i < N; ++i) {
            SingleState s = {
                state.obs[i],
                state.done[i].item<bool>(),
                state.truncated[i].item<bool>(),
                state.episode_start[i].item<bool>()
            };

            SingleState ns = {
                next_state.obs[i],
                next_state.done[i].item<bool>(),
                next_state.truncated[i].item<bool>(),
                next_state.episode_start[i].item<bool>()
            };
            out.push_back({
                s,
                action.action.index({i}),
                reward[i].item<float>(),
                ns
                });
        }

        return out;
    }

    std::string BatchExperience::ToString() const
    {
        ANET_CHECK_DTYPE(reward, torch::kFloat32);
        ANET_CHECK_SHAPE(reward, { ANET_SHAPE_ANY });
        std::ostringstream oss;
        oss << "BatchExperience{\n";
        oss << "  state      = " << state.ToString() << "\n";
        oss << "  action     = " << action.ToString() << "\n";
        oss << "  reward     = " << anet::ToString(reward) << "\n";
        oss << "  next_state = " << next_state.ToString() << "\n";
        oss << "}";
        return oss.str();
    }

    //std::unique_ptr<HeatMap> MakeStateHeatMapPtr(
    //    const anet::rl::StateSpaceInfo& info,
    //    int idx_x,
    //    int idx_y,
    //    int width,
    //    int height,
    //    size_t max_points,
    //    uint32_t flags)
    //{
    //    if (!info.low.defined() || !info.high.defined())
    //        throw std::runtime_error("StateSpaceInfo.low/high are undefined.");

    //    auto dim = info.low.size(0);
    //    if (idx_x >= dim || idx_y >= dim)
    //        throw std::runtime_error("MakeStateHeatMapPtr: axis index out of range.");

    //    float x_min = info.low[idx_x].item<float>();
    //    float x_max = info.high[idx_x].item<float>();
    //    float y_min = info.low[idx_y].item<float>();
    //    float y_max = info.high[idx_y].item<float>();

    //    return std::make_unique<HeatMap>(
    //        width,
    //        height,
    //        x_min, x_max,
    //        y_min, y_max,
    //        max_points,
    //        flags);
    //}

    //std::unique_ptr<TimeHeatMap> MakeStateTimeHeatMapPtr(
    //    const anet::rl::StateSpaceInfo& info,
    //    int idx_x,
    //    int width, int height,
    //    size_t max_points,
    //    uint32_t flags,
    //    TimeFrameMode mode)
    //{
    //    if (!info.low.defined() || !info.high.defined())
    //        throw std::runtime_error("StateSpaceInfo.low/high are undefined.");

    //    auto dim = info.low.size(0);
    //    if (idx_x >= dim)
    //        throw std::runtime_error("MakeStateHeatMapPtr: axis index out of range.");

    //    float x_min = info.low[idx_x].item<float>();
    //    float x_max = info.high[idx_x].item<float>();

    //    return std::make_unique<TimeHeatMap>(
    //        width,
    //        height,
    //        x_min, x_max,
    //        flags,
    //        max_points,
    //        mode);
    //}

} // namespace anet::rl
