#include "anet/rl.hpp"
#include <stdexcept>
#include <wx/log.h>
#include "anet/common.hpp"
#include "anet/profile.hpp"
#include "anet/util.hpp"
#include "anet/tensor_util.hpp"
#include "anet/tensor_check.hpp"
#include "anet/log.hpp"

using namespace anet::rl;
namespace LOG = anet::log;

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

BatchStepResult::BatchStepResult(torch::Tensor reward_in, BatchState next_state_in, BatchState continue_state_in, uint32_t n_transitions_in,uint32_t n_done_in)
    : reward(std::move(reward_in))
    , next_state(std::move(next_state_in))
    , continue_state(std::move(continue_state_in))
    , n_transitions(n_transitions_in)
    , n_done(n_done_in)
{
}

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
    oss << "BatchStepResult{\n";
    oss << "  reward=" << anet::ToString(reward) << "\n";
    oss << "  , next_state=" << next_state.ToString() << "\n";
    oss << "  , continue_state=" << continue_state.ToString() << "\n";
    oss << "  , n_transitions=" << n_transitions;
    oss << "  , n_done=" << n_done;
    oss << "  , auxs={";
    auto auxs = GetAuxDataList();
    for (auto aux : auxs) {
        oss << "  [\n";
        for (auto kv : aux) {
            oss << "    " << kv.first << "=" << anet::ToString(kv.second) << "\n";
        }
        oss << "  ]\n";
    }
    oss << "  }}";
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
    oss << ", aux=[";
    auto aux = GetAuxData();
    for (auto kv : aux) {
        oss << " " << kv.first << "=" << anet::ToString(kv.second);
    }
    oss << "]}";
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
    const std::string& key, int index) const
{
    /// @todo index指定対応

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
    BatchExperience::GetTensorVector(const std::string& key, int index) const
{
    /// @todo index指定対応

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
        SingleState cs = {
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

ExperienceSample ExperienceSample::Flatten() const {
    return ExperienceSample{
        obs.flatten(1), // obs
        actions,        // action
        rewards,        // reward
        {
            next_states.obs.flatten(1), // next_states.obs
            next_states.dones,          // next_states.dones
            next_states.truncateds,     // next_states.truncateds
            next_states.episode_start   // next_states.episode_start
        }
    };
}

std::string ExperienceSample::ToString() const
{
    std::ostringstream oss;
    oss << "ExperienceSample{\n";
    oss << "  obs     = " << anet::ToString(obs) << "\n";
    oss << "  action  = " << anet::ToString(actions) << "\n";
    oss << "  reward  = " << anet::ToString(rewards) << "\n";
    oss << "  next_state.obs           = " << anet::ToString(next_states.obs) << "\n";
    oss << "  next_state.dones         = " << anet::ToString(next_states.dones) << "\n";
    oss << "  next_state.truncateds    = " << anet::ToString(next_states.truncateds) << "\n";
    oss << "  next_state.episode_start = " << anet::ToString(next_states.episode_start) << "\n";
    oss << "}";
    return oss.str();
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

Notifier::Notifier()
{
    ;
}

void Notifier::Attach(std::shared_ptr<BeforeStepObserver> obs)
{
    before_step_observers_.push_back(obs);
}
void Notifier::Attach(std::shared_ptr<TrainObserver> obs)
{
    train_observers_.push_back(obs);
}
void Notifier::Attach(std::shared_ptr<LearnObserver> obs)
{
    learn_observers_.push_back(obs);
}


void Notifier::Detach(std::shared_ptr<BeforeStepObserver> obs)
{
    before_step_observers_.erase(
        std::remove_if(
            before_step_observers_.begin(), before_step_observers_.end(),
            [&](const std::shared_ptr<BeforeStepObserver>& o) {
                return o == obs;
            }
        ),
        before_step_observers_.end()
    );
}
void Notifier::Detach(std::shared_ptr<TrainObserver> obs)
{
    train_observers_.erase(
        std::remove_if(
            train_observers_.begin(), train_observers_.end(),
            [&](const std::shared_ptr<TrainObserver>& o) {
                return o == obs;
            }
        ),
        train_observers_.end()
    );
}
void Notifier::Detach(std::shared_ptr<LearnObserver> obs)
{
    learn_observers_.erase(
        std::remove_if(
            learn_observers_.begin(), learn_observers_.end(),
            [&](const std::shared_ptr<LearnObserver>& o) {
                return o == obs;
            }
        ),
        learn_observers_.end()
    );
}

void Notifier::Notify(const BeforeStepEvent& event)
{
    anet::ProfileRange r("Notifier::Notify");

    for (auto obs : before_step_observers_) {
        obs->OnBeforeStep(event);
    }
}
void Notifier::Notify(const TrainEvent& event)
{
    anet::ProfileRange r("Notifier::Notify");

    for (auto obs : train_observers_) {
        obs->OnTrain(event);
    }
}
void Notifier::Notify(const LearnEvent& event)
{
    anet::ProfileRange r("Notifier::Notify");

    for (auto obs : learn_observers_) {
        obs->OnLearn(event);
    }
}

void Notifier::LogObservers() const
{
    int idx = 0;
    for (auto obs : train_observers_) {
        LOG::info() << "Notifier: TRAIN [" << idx << "] " << obs->ToString();
        idx++;
    }
    idx = 0;
    for (auto obs : learn_observers_) {
        LOG::info() << "Notifier: LEARN [" << idx << "] " << obs->ToString();
       idx++;
    }
}
