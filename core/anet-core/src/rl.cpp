#include <stdexcept>
#include <wx/log.h>
#include "anet/rl.hpp"
#include "anet/common.hpp"
#include "anet/util.hpp"
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

    int64_t StateSpec::CalcFlattenSize() const
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

    const StateDimInfo* StateSpec::FindDim(int64_t flatten_index) const
    {
        // shape が空なら対応不可
        if (shape.empty()) return nullptr;

        // flatten size 超えは無効
        const int64_t flat_size = CalcFlattenSize();
        if (flatten_index < 0 || flatten_index >= flat_size)
            return nullptr;

        // flatten_index → coords（多次元インデクス）へ逆変換
        std::vector<int64_t> coords(shape.size(), 0);

        int64_t idx = flatten_index;
        for (int i = (int)shape.size() - 1; i >= 0; i--) {
            int64_t dim = shape[i];
            coords[i] = idx % dim;
            idx /= dim;
        }

        // coords に一致する StateDimInfo を検索
        return FindDim(coords);
    }

    bool StateSpec::MatchesShape(const torch::Tensor& obs) const
    {
        //wxLogDebug("dim=%d size=%d", static_cast<int>(obs.dim()), static_cast<int>(shape.size()));
        ANET_ASSERT_MSG(
            obs.dim() == static_cast<int64_t>(shape.size() + 1),
            "StateSpec::MatchesShape: dimension mismatch.");

        for (size_t i = 1; i < shape.size(); i++) {
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

        const int64_t N = obs.size(0);  // 0次元目 = 環境数

        for (const auto& d : dims) {
            float mn = d.min_value;
            float mx = d.max_value;

            for (int64_t n = 0; n < N; n++) {
                torch::Tensor env = obs[n];  // [D1, D2, ...]

                if (d.coords.empty()) {
                    auto flat = env.flatten();
                    const int64_t M = flat.size(0);
                    for (int64_t i = 0; i < M; i++) {
                        float v = flat[i].item<float>();
                        ANET_ASSERT_MSG(
                            v >= mn && v <= mx,
                            "StateSpec::MatchesRange: value out of range.");
                    }
                    continue;
                }

                // coords を env に適用
                torch::Tensor cur = env;

                for (size_t k = 0; k < d.coords.size(); k++) {
                    int64_t idx = d.coords[k];

                    ANET_ASSERT_MSG(
                        idx >= 0 && idx < cur.size(0),
                        "StateSpec::MatchesRange: coords index OOB.");

                    cur = cur.select(0, idx);
                }

                ANET_ASSERT_MSG(
                    cur.dim() == 0,
                    "StateSpec::MatchesRange: coords did not resolve to scalar.");

                float v = cur.item<float>();
                ANET_ASSERT_MSG(
                    v >= mn && v <= mx,
                    "StateSpec::MatchesRange: coord value out of range.");
            }
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

        j["info"] = nlohmann::json::object();
        for (const auto& kv : info) {
            j["info"][kv.first] = kv.second;
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
        j["info"] = nlohmann::json::object();
        for (const auto& kv : info) {
            j["info"][kv.first] = kv.second;
        }

        return j;
    }

    std::string ActionSpec::ToString() const {
        return ToJson().dump(2);
    }

    nlohmann::json EnvSpec::ToJson() const {
        nlohmann::json j;

        j["state_spec"] = state_spec.ToJson();
        j["action_spec"] = action_spec.ToJson();

        j["reward_range"] = {
            reward_range.first,
            reward_range.second
        };

        j["info"] = nlohmann::json::object();
        for (const auto& kv : info) {
            j["info"][kv.first] = kv.second;
        }

        return j;
    }

    std::string EnvSpec::ToString() const {
        return ToJson().dump(2);
    }

    nlohmann::json BatchEnvSpec::ToJson() const {
        nlohmann::json j;
        j["batch_size"] = batch_size;
        j["num_threads"] = num_threads;
        return j;
    }

    std::string BatchEnvSpec::ToString() const {
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
        oss << "reward=" << anet::ToString(reward);
        oss << ", next_state=" << next_state.ToString();
        oss << ", continue_state=" << continue_state.ToString();
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
    std::string SingleDiscreteActionInfo::ToString() const
    {
        std::ostringstream oss;
        oss << "SingleDiscreteActionInfo{";
        oss << "action=" << action;
        oss << ", is_random=" << is_random;
        oss << "}";
        return oss.str();
    }
    std::string SingleStepResult::ToString() const
    {
        std::ostringstream oss;
        oss << "SingleStepResult{";
        oss << "reward=" << reward;
        oss << ", next_state=" << next_state.ToString();
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

    std::optional<torch::Tensor> BatchExperience::GetTensor(
        const std::string& key) const
    {
        if (key == NEXT_STATE_OBS)
            return next_state.obs;
        if (key == REWARD)
            return reward;
        if (key == ACTION_ACTION)
            return action.action;
        if (key == STATE_OBS)
            return state.obs;

        if (key == STATE_DONE)
            return state.done;
        if (key == STATE_TRUNCATED)
            return state.truncated;
        if (key == NEXT_STATE_EPISODE_START)
            return state.episode_start;
        
        if (key == NEXT_STATE_DONE)
            return next_state.done;
        if (key == NEXT_STATE_TRUNCATED)
            return next_state.truncated;
        if (key == NEXT_STATE_EPISODE_START)
            return next_state.episode_start;

        if (key == ACTION_IS_RANDOM)
            return action.is_random;

        return std::nullopt;
    }

    std::optional<std::vector<torch::Tensor>>
        BatchExperience::GetTensorVector(const std::string& key) const
    {
        auto t = GetTensor(key);
        if (!t.has_value()) return std::nullopt;
        return std::vector<torch::Tensor>{ *t };
    }

    BatchExperience BatchExperience::to(torch::Device d) const {
        BatchExperience out;
        out.state = state.to(d);
        out.action = action.to(d);
        out.reward = reward.to(d);
        out.next_state = next_state.to(d);
        return out;
    }

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

    VectorizedDiscreteBatchEnv::VectorizedDiscreteBatchEnv(
        std::shared_ptr<SingleDiscreteEnvFactory> factory, int batch_size)
        : batch_spec_({ batch_size, 1 })
    {
        ANET_ASSERT(batch_size > 0);
        envs_.reserve(batch_size);

        // 最初の1個を生成してEnvSpec取得
        auto env = factory->Create();
        spec_ = std::make_unique<EnvSpec>(env->GetSpec());
        envs_.push_back(std::move(env));

        // 残り batch_size-1 個を生成
        for (int i = 1; i < batch_size; ++i) {
            envs_.push_back(factory->Create());
        }
    }

    EnvSpec VectorizedDiscreteBatchEnv::GetSpec() const
    {
        return *spec_;
    }

    BatchEnvSpec VectorizedDiscreteBatchEnv::GetBatchSpec() const
    {
        return batch_spec_;
    }

    static std::vector<int64_t> prepend_batch(int N, at::IntArrayRef shape) {
        std::vector<int64_t> out;
        out.reserve(shape.size() + 1);
        out.push_back(N);
        out.insert(out.end(), shape.begin(), shape.end());
        return out;
    }

    BatchState VectorizedDiscreteBatchEnv::Reset(RunMode mode)
    {
        const int N = batch_spec_.batch_size;

        // 最初のenvで obs の shape を取得
        SingleState s0 = envs_[0]->Reset(mode);
        ANET_CHECK_SHAPE(s0.obs, { ANET_SHAPE_ENDANY });
        ANET_CHECK_DTYPE(s0.obs, torch::kFloat32);

        const auto obs_sizes = s0.obs.sizes();  // state_dim...

        // BatchState 用 Tensor を作成 (N,state_dim...)
        std::vector<int64_t> batch_obs_sizes;
        batch_obs_sizes.reserve(obs_sizes.size() + 1);
        batch_obs_sizes.push_back(N);
        batch_obs_sizes.insert(batch_obs_sizes.end(), obs_sizes.begin(), obs_sizes.end());

        // 戻り用のBatchSteteを準備
        auto obs_options_float = torch::TensorOptions().dtype(torch::kFloat32).device(s0.obs.device());
        auto obs_options_bool = torch::TensorOptions().dtype(torch::kBool).device(s0.obs.device());
        BatchState batch_state {
            torch::empty(batch_obs_sizes, obs_options_float), // obs           (N,state_dim...) kFloat
            torch::empty({ N }, obs_options_bool),   // done          (N) kBool      
            torch::empty({ N }, obs_options_bool),   // truncated     (N) kBool
            torch::empty({ N }, obs_options_bool)    // episode_start (N) kBool
        };

        // 最初の環境の結果を詰める
        batch_state.obs.index_put_({ 0 }, s0.obs);
        batch_state.done.index_put_({ 0 }, s0.done);
        batch_state.truncated.index_put_({ 0 }, s0.truncated);
        batch_state.episode_start.index_put_({ 0 }, s0.episode_start);

        // 残りの環境を順次リセットして直接詰める
        for (int i = 1; i < N; ++i) {
            SingleState s = envs_[i]->Reset(mode);
            batch_state.obs.index_put_({ i }, s.obs);
            batch_state.done.index_put_({ i }, s.done);
            batch_state.truncated.index_put_({ i }, s.truncated);
            batch_state.episode_start.index_put_({ i }, s.episode_start);
        }

        state_ = batch_state;

        return state_;
    }

    BatchStepResult VectorizedDiscreteBatchEnv::Step(const torch::Tensor& actions, RunMode mode)
    {
        const int64_t N = batch_spec_.batch_size;
        ANET_CHECK_DTYPE_MSG(actions, torch::kInt64,
            "VectorizedDiscreteBatchEnv supports discrete action only. actions should be kInt64.");
        ANET_CHECK_SHAPE(actions, { N });

        // 1つ目の環境で Step を実行
        auto a0 = actions[0].item<int64_t>();
        SingleStepResult r0 = envs_[0]->Step(a0, mode);
        const auto obs_sizes = r0.next_state.obs.sizes();

        // obs shape を確定
        std::vector<int64_t> batch_obs_sizes;
        batch_obs_sizes.reserve(obs_sizes.size() + 1);
        batch_obs_sizes.push_back(N);
        batch_obs_sizes.insert(batch_obs_sizes.end(), obs_sizes.begin(), obs_sizes.end());

        // BatchStepResult のテンソル群を作成
        auto float_opt = torch::TensorOptions().dtype(torch::kFloat32).device(r0.next_state.obs.device());
        auto bool_opt = torch::TensorOptions().dtype(torch::kBool).device(r0.next_state.obs.device());
        BatchStepResult result{
            torch::empty({ N }, float_opt),                 // reward        (N) kFloat32
            BatchState {    // next_state
                torch::empty(batch_obs_sizes, float_opt),   // obs           (N, state_dim..) kFloat32
                torch::empty({ N }, bool_opt),              // done          (N) kBool
                torch::empty({ N }, bool_opt),              // truncated     (N) kBool
                torch::empty({ N }, bool_opt)               // episode_start (N) kBool
            },
            BatchState {    // continue_state
                torch::empty(batch_obs_sizes, float_opt),   // obs           (N, state_dim..) kFloat32
                torch::empty({ N }, bool_opt),              // done          (N) kBool
                torch::empty({ N }, bool_opt),              // truncated     (N) kBool
                torch::empty({ N }, bool_opt)               // episode_start (N) kBool
            },
        };

        // ----- 最初の結果を詰める -----
        result.next_state.obs.index_put_({ 0 }, r0.next_state.obs);
        result.next_state.done.index_put_({ 0 }, r0.next_state.done);
        result.next_state.truncated.index_put_({ 0 }, r0.next_state.truncated);
        result.next_state.episode_start.index_put_({ 0 }, r0.next_state.episode_start);
        result.reward.index_put_({ 0 }, r0.reward);

        // ---- 最初分のcontinue_stateを詰める
        if (r0.next_state.done || r0.next_state.truncated) {
            SingleState rs = envs_[0]->Reset(mode);
            result.continue_state.obs.index_put_({ 0 }, rs.obs);
            result.continue_state.done.index_put_({ 0 }, rs.done);
            result.continue_state.truncated.index_put_({ 0 }, rs.truncated);
            result.continue_state.episode_start.index_put_({ 0 }, rs.episode_start);
        } else {
            result.continue_state.obs.index_put_({ 0 }, r0.next_state.obs);
            result.continue_state.done.index_put_({ 0 }, false);
            result.continue_state.truncated.index_put_({ 0 }, false);
            result.continue_state.episode_start.index_put_({ 0 }, r0.next_state.episode_start);
        }

        // ----- 残りの環境を順次実行 -----
        for (int i = 1; i < N; ++i) {
            auto a = actions[i].item<int64_t>();
            SingleStepResult ri = envs_[i]->Step(a, mode);

            result.next_state.obs.index_put_({ i }, ri.next_state.obs);
            result.next_state.done.index_put_({ i }, ri.next_state.done);
            result.next_state.truncated.index_put_({ i }, ri.next_state.truncated);
            result.next_state.episode_start.index_put_({ i }, ri.next_state.episode_start);
            result.reward.index_put_({ i }, ri.reward);

            // continue_state
            if (ri.next_state.done || ri.next_state.truncated) {
                SingleState rs = envs_[i]->Reset(mode);
                result.continue_state.obs.index_put_({ i }, rs.obs);
                result.continue_state.done.index_put_({ i }, rs.done);
                result.continue_state.truncated.index_put_({ i }, rs.truncated);
                result.continue_state.episode_start.index_put_({ i }, rs.episode_start);
            } else {
                result.continue_state.obs.index_put_({ i }, ri.next_state.obs);
                result.continue_state.done.index_put_({ i }, false);
                result.continue_state.truncated.index_put_({ i }, false);
                result.continue_state.episode_start.index_put_({ i }, ri.next_state.episode_start);
            }
        }

        // 継続用Stateを保存
        state_ = result.continue_state;

        return result;
    }

    BatchState VectorizedDiscreteBatchEnv::GetState() const
    {
        return state_;
    }

} // namespace anet::rl
