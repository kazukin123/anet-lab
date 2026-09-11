#include "anet/rl.hpp"
#include <stdexcept>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

#include <wx/log.h>
#include "anet/common.hpp"
#include "anet/profile.hpp"
#include "anet/str_util.hpp"
#include "anet/util.hpp"
#include "anet/tensor_util.hpp"
#include "anet/tensor_check.hpp"
#include "anet/log.hpp"

using namespace anet::rl;
namespace LOG = anet::log;

namespace anet::rl {

anet::TraceCallback MakeActionTraceCallback(anet::TensorDict& trace)
{
    return [&trace](std::string_view key, const torch::Tensor& activation) {
        trace.Set(std::string(key), activation.slice(0, 0, 1).detach().to(torch::kFloat32).clone());
    };
}

void AppendTraceAux(AuxData& aux, const anet::TensorDict& trace)
{
    for (const auto& [key, tensor] : trace) {
        aux[std::string(kNnTracePrefix) + key] = tensor;
    }
}

anet::TensorDict ExtractNnTrace(const AuxData& aux)
{
    anet::TensorDict dict;
    for (const auto& [key, tensor] : aux) {
        if (key.starts_with(kNnTracePrefix)) {
            dict.Set(key.substr(kNnTracePrefix.size()), tensor);
        }
    }
    return dict;
}

} // namespace anet::rl

static std::optional<torch::Tensor> GetObservationTensor(
    const anet::TensorDict& obs, const std::string& key, const char* base_key)
{
    if (key == base_key) {
        return anet::rl::ToUnifiedObservation(obs);
    }

    const std::string prefix = std::string(base_key) + ".";
    if (!anet::StartsWith(key, prefix)) {
        return std::nullopt;
    }

    const std::string sub_key = anet::RemovePrefix(key, prefix);
    if (sub_key.empty()) {
        ANET_SYSTEM_ERROR("BatchExperience::GetTensor: empty observation subkey. key=" << key);
    }

    auto tensor = obs.Get(sub_key);
    if (!tensor.has_value()) {
        ANET_SYSTEM_ERROR(
            "BatchExperience::GetTensor: unknown observation subkey. key=" << key
            << " subkey=" << sub_key);
    }
    return *tensor;
}


// =============================================================
// StateSpec
// =============================================================

void StateSpec::AssertSanity() const
{
    for (const auto& kv : obs_spec) {
        const auto& key = kv.first;
        const auto& spec = kv.second;

        if (spec.IsDiscrete()) {
            // 離散空間: float系は不可
            ANET_CHECK_MSG(!torch::isFloatingType(spec.dtype),
                "TensorSpec Sanity Error: [" << key << "] is discrete (num_classes > 0) but dtype is float.");
        } else {
            // 連続空間: float系が必須
            //ANET_CHECK_MSG(torch::isFloatingType(spec.dtype),
            //    "TensorSpec Sanity Error: [" << key << "] is continuous (num_classes == 0) but dtype is not float.");
        }

        // トポロジーとShapeの整合性 (Vectorは1次元配列であること)
        if (spec.type == SpaceType::Vector) {
            ANET_CHECK_MSG(spec.shape.size() == 1,
                "TensorSpec Sanity Error: [" << key << "] is Vector but shape is not 1D (size is " << spec.shape.size() << ").");
        }
    }
}

bool StateSpec::ValidateObservation(const anet::TensorDict & obs, bool is_batched) const
{
    if (obs.empty() && !obs_spec.empty()) {
        ANET_SYSTEM_ERROR(
            "ValidateObservation failed: TensorDict is empty but obs_spec is not.");
        return false;
    }

    for (const auto& kv : obs_spec) {
        const auto& key = kv.first;
        const auto& spec = kv.second;

        // キーの存在チェック
        if (!obs.Contains(key)) {
            ANET_SYSTEM_ERROR(
                "ValidateObservation failed: Missing key in TensorDict: " << key);
            return false;
        }

        const auto& tensor = obs.At(key);

        // Dtypeのチェック
        if (tensor.dtype() != spec.dtype) {
            ANET_SYSTEM_ERROR(
                "ValidateObservation failed: Dtype mismatch for key: " << key
                << " | Expected: " << spec.dtype << ", Actual: " << tensor.dtype());
            return false;
        }

        // 次元数とShapeのチェック
        int64_t expected_dims = spec.shape.size() + (is_batched ? 1 : 0);
        if (tensor.dim() != expected_dims) {
            ANET_SYSTEM_ERROR(
                "ValidateObservation failed: Dimension mismatch for key: " << key
                << " | Expected dims: " << expected_dims << ", Actual: " << tensor.dim());
            return false;
        }

        for (size_t i = 0; i < spec.shape.size(); ++i) {
            int64_t tensor_dim_idx = is_batched ? (static_cast<int64_t>(i) + 1) : static_cast<int64_t>(i);
            if (tensor.size(tensor_dim_idx) != spec.shape[i]) {
                ANET_SYSTEM_ERROR(
                    "ValidateObservation failed: Shape mismatch at dim " << i
                    << " for key: " << key << " | Expected: " << spec.shape[i] << ", Actual: " << tensor.size(tensor_dim_idx));
                return false;
            }
        }

        // Rangeのチェック (float型でmin/maxが定義されている場合のみ)
        if (tensor.is_floating_point() && !spec.min_values.empty() && !spec.max_values.empty()) {
            float spec_min = *std::min_element(spec.min_values.begin(), spec.min_values.end());
            float spec_max = *std::max_element(spec.max_values.begin(), spec.max_values.end());

            float t_min = tensor.min().item<float>();
            float t_max = tensor.max().item<float>();

            const float eps = 1e-4f;
            if (t_min < spec_min - eps || t_max > spec_max + eps) {
                ANET_SYSTEM_ERROR(
                    "ValidateObservation failed: Values out of range for key: " << key
                    << " | Spec Range: [" << spec_min << ", " << spec_max
                    << "], Actual Range: [" << t_min << ", " << t_max << "]");
                return false;
            }
        }
    }
    return true;
}

anet::json StateSpec::ToJson() const
{
    anet::json j;
    anet::json dict_j;
    for (const auto& [key, spec] : obs_spec) {
        dict_j[key] = spec.ToJson();
    }
    j["obs_spec"] = dict_j;
    j["info"] = info;
    return j;
}

std::string StateSpec::ToString() const
{
    return ToJson().dump(2);
}


// =============================================================
// ActionDimInfo
// =============================================================

anet::json ActionDimInfo::ToJson() const
{
    anet::json j;
    j["min_value"] = min_value;
    j["max_value"] = max_value;
    j["name"] = name;
    j["description"] = description;
    return j;
}

std::string ActionDimInfo::ToString() const
{
    return ToJson().dump(2);
}

anet::json ActionSpec::ToJson() const
{
    anet::json j;
    j["is_discrete"] = is_discrete;

    // 離散アクションラベル
    j["value_labels"] = value_labels;

    // 連続アクション次元
    j["dims"] = anet::json::array();
    for (const auto& d : dims) {
        j["dims"].push_back(d.ToJson());
    }

    // オプション
    j["info"] = anet::json::object();
    for (const auto& kv : info) {
        j["info"][kv.first] = kv.second;
    }

    return j;
}

std::string ActionSpec::ToString() const
{
    return ToJson().dump(2);
}


// =============================================================
// EnvSpec
// =============================================================

void EnvSpec::CheckSameStateActionSpec(const EnvSpec& actual) const
{
    // infoは参考情報なので除外し、NN入出力へ実際に影響する通常fieldだけを比較する。
    auto expected_state = state_spec;
    auto actual_state = actual.state_spec;
    expected_state.info.clear();
    actual_state.info.clear();
    ANET_CHECK_MSG(expected_state.ToJson() == actual_state.ToJson(),
        "Env state spec mismatch. expected=" << expected_state.ToJson().dump()
        << " actual=" << actual_state.ToJson().dump());

    auto expected_action = action_spec;
    auto actual_action = actual.action_spec;
    expected_action.info.clear();
    actual_action.info.clear();
    ANET_CHECK_MSG(expected_action.ToJson() == actual_action.ToJson(),
        "Env action spec mismatch. expected=" << expected_action.ToJson().dump()
        << " actual=" << actual_action.ToJson().dump());
}

anet::json EnvSpec::ToJson() const
{
    anet::json j;

    j["state_spec"] = state_spec.ToJson();
    j["action_spec"] = action_spec.ToJson();

    j["reward_range"] = {
        reward_range.first,
        reward_range.second
    };

    j["info"] = anet::json::object();
    for (const auto& kv : info) {
        j["info"][kv.first] = kv.second;
    }

    return j;
}

std::string EnvSpec::ToString() const
{
    return ToJson().dump(2);
}


// =============================================================
// BatchEnvSpec
// =============================================================

anet::json BatchEnvSpec::ToJson() const
{
    anet::json j;
    j["num_envs"] = num_envs;
    j["num_threads"] = num_threads;
    switch (episode_scope) {
    case EpisodeScope::PER_LANE:
        j["episode_scope"] = "per_lane";
        break;
    case EpisodeScope::SHARED:
        j["episode_scope"] = "shared";
        break;
    default:
        ANET_SYSTEM_ERROR("Unknown BatchEnvSpec episode_scope="
            << static_cast<int>(episode_scope) << ".");
    }
    return j;
}

std::string BatchEnvSpec::ToString() const
{
    return ToJson().dump(2);
}


// =============================================================

torch::Tensor anet::rl::ToUnifiedObservation(const anet::TensorDict& obs, bool is_batched)
{
    if (obs.empty()) return torch::Tensor();

    // 要素が1つだけの場合
    if (obs.size() == 1) {
        const auto& key = obs.begin()->first;
        // 唯一の要素がActionMaskなら、空を返す（学習データがない状態）
        if (key == ObsKeys::kActionMask) return torch::Tensor();
        return obs.begin()->second;
    }

    // 複数要素の場合
    int flatten_dim = is_batched ? 1 : 0;

    // 辞書順ソートを維持しつつ、性質で分けるバケット
    std::map<std::string, torch::Tensor> vector_bucket;
    std::map<std::string, torch::Tensor> grid_bucket;

    for (auto it = obs.begin(); it != obs.end(); ++it) {
        const auto& key = it->first;
        const auto& tensor = it->second;

        // ActionMask は 入力（Legacy Tensor）には含めない
        if (key == ObsKeys::kActionMask) continue;

        // Keyが"vector" か次元数が 1 (バッチ含め 2) なら Vector 扱い
        bool is_vector_key = (key == ObsKeys::kVector);
        int effective_dims = is_batched ? (tensor.dim() - 1) : tensor.dim();
        if (is_vector_key || effective_dims <= 1) {
            vector_bucket[key] = tensor.to(torch::kFloat32).flatten(flatten_dim);
        } else {
            grid_bucket[key] = tensor.to(torch::kFloat32).flatten(flatten_dim);
        }
    }

    // Vector系(辞書順) -> Grid系(辞書順) の順で結合して、旧仕様のメモリレイアウトを模倣
    std::vector<torch::Tensor> concat_list;
    for (auto& kv : vector_bucket) concat_list.push_back(kv.second);
    for (auto& kv : grid_bucket)   concat_list.push_back(kv.second);
    if (concat_list.empty()) return torch::Tensor();

    return torch::cat(concat_list, flatten_dim);
}

// -----------------------------------------

BatchStepResult::BatchStepResult(torch::Tensor reward_in, BatchState next_state_in, BatchState continue_state_in, uint32_t n_transitions_in,uint32_t n_episode_end_in)
    : reward(std::move(reward_in))
    , next_state(std::move(next_state_in))
    , continue_state(std::move(continue_state_in))
    , n_transitions(n_transitions_in)
    , n_episode_end(n_episode_end_in)
{
}

std::string BatchState::ToString() const
{
    std::ostringstream oss;
    oss << "BatchState{";
    oss << "obs=" << obs.ToString();
    oss << ", done=" << anet::ToString(done);
    oss << ", truncated=" << anet::ToString(truncated);
    oss << ", episode_start=" << anet::ToString(episode_start);
    oss << "}";
    return oss.str();
}

ReplayInitialPriorityHint::ReplayInitialPriorityHint(torch::Tensor payload)
    : payload_(std::move(payload))
{
    // 共通carrierはpayloadの意味を解釈せず、運搬に必要な形式だけを全buildで検証する。
    if (!payload_.defined()) {
        ANET_SYSTEM_ERROR("ReplayInitialPriorityHint payload must be defined.");
    }
    if (payload_.dim() != 2) {
        ANET_SYSTEM_ERROR("ReplayInitialPriorityHint payload must have rank 2. actual=" << payload_.sizes());
    }
    if (payload_.scalar_type() != torch::kFloat32) {
        ANET_SYSTEM_ERROR("ReplayInitialPriorityHint payload must use float32. actual=" << payload_.scalar_type());
    }
    if (payload_.size(0) <= 0 || payload_.size(1) <= 0) {
        ANET_SYSTEM_ERROR("ReplayInitialPriorityHint payload dimensions must be positive. actual=" << payload_.sizes());
    }
    payload_ = payload_.detach().contiguous();
}

const torch::Tensor& ReplayInitialPriorityHint::GetPayloadCpu() const
{
    ANET_PROFILE_FUNC();
    // packed tensor全体を初回だけ同期転送し、env行ごとのD2Hへ分解しない。
    if (!payload_cpu_.defined()) {
        payload_cpu_ = payload_.device().is_cpu()
            ? payload_
            : payload_.to(torch::kCPU, torch::kFloat32, /*non_blocking=*/false);
        payload_cpu_ = payload_cpu_.contiguous();
    }
    return payload_cpu_;
}

torch::Tensor BatchActionInfo::GetAction(torch::Device device) const
{
    /// @todo GPU→CPUのno_blockingは同期が必要で面倒なので使わない

    // --- CPU requested ---
    if (device.is_cpu()) {
        if (!action_cpu_.defined()) {
            if (!gpu_)
                ANET_SYSTEM_ERROR("BatchActionInfo: no tensor available to create CPU action");
            action_cpu_ = gpu_->second.to(torch::kCPU);
        }
        return action_cpu_;
    }


    // --- GPU requested ---
    if (!device.is_cuda()) {
        ANET_SYSTEM_ERROR("BatchActionInfo: unsupported device type");
    }

    // インデックスが指定されていない("cuda")場合、デフォルトで 0 を割り当てる("cuda:0" にする)
    torch::Device req_dev = device;
    if (!req_dev.has_index()) {
        req_dev = torch::Device(torch::kCUDA, 0);
    }

    if (!gpu_) {
        if (!action_cpu_.defined())
            ANET_SYSTEM_ERROR("BatchActionInfo: no tensor available to create GPU action");
        gpu_ = std::make_pair(device, action_cpu_.to(req_dev));
        return gpu_->second;
    }
    if (gpu_->first != req_dev) {
        ANET_SYSTEM_ERROR(
            "BatchActionInfo: GPU device mismatch (cached="<< gpu_->first.str() << ", requested=" << device.str() << ")"
        );
    }

    return gpu_->second;
}

std::string BatchActionInfo::ToString() const
{
    std::ostringstream oss;
    oss << "BatchActionInfo{";
    oss << "action=" << anet::ToString(GetAction(torch::kCPU));
    oss << ", aux={";
    for (auto kv : aux_) {
        oss << "\n    " << kv.first << "=" << anet::ToString(kv.second);
    }
    oss << "  }}";
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
    oss << "  , n_episode_end=" << n_episode_end;
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
    oss << "obs=" << obs.ToString();
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

std::string SingleExperience::ToString() const
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


// =============================================================
// BatchExperience
// =============================================================

std::optional<torch::Tensor> BatchExperience::GetTensor(const std::string& key, int64_t index) const
{
    /// @todo index指定対応

    if (auto obs = GetObservationTensor(next_state.obs, key, NEXT_STATE_OBS); obs.has_value())
        return obs;
    if (auto obs = GetObservationTensor(state.obs, key, STATE_OBS); obs.has_value())
        return obs;

    if (key == REWARD)
        return reward;
    if (key == ACTION_ACTION)
        return action->GetAction();

    if (key == STATE_DONE)
        return state.done;
    if (key == STATE_TRUNCATED)
        return state.truncated;
    if (key == STATE_EPISODE_START)
        return state.episode_start;

    if (key == NEXT_STATE_DONE)
        return next_state.done;
    if (key == NEXT_STATE_TRUNCATED)
        return next_state.truncated;
    if (key == NEXT_STATE_EPISODE_START)
        return next_state.episode_start;

    if (key.find("action.") == 0) {
        auto sub_key = anet::RemovePrefix(key, "action.");
        auto aux = action->GetAuxData();
        auto aux_itr = aux.find(sub_key);
        if (aux_itr == aux.end()) return std::nullopt;
        const auto& aux_tensor = aux_itr->second;
        return aux_tensor;

    }

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>>
    BatchExperience::GetTensorVector(const std::string& key, int64_t index) const
{
    /// @todo index指定対応

    auto t = GetTensor(key);
    if (!t.has_value()) return std::nullopt;
    return std::vector<torch::Tensor>{ *t };
}

BatchExperience BatchExperience::To(torch::Device d, bool non_blocking) const {
    BatchExperience out {
        state.To(d, non_blocking),
        action->To(d),// GPU→CPUは明示的な同期が必要でバグの温床になるのでnon_blocking無しにしておく
        reward.to(d, non_blocking),
        next_state.To(d, non_blocking)
    };
    return out;
}

std::vector<SingleExperience> BatchExperience::ToExperienceList() const
{
    // TensorDictの不整合チェック ----
    ANET_ASSERT_MSG(state.obs.IsValid(), "state.obs tensors have is not valid.");
    ANET_ASSERT_MSG(next_state.obs.IsValid(), "next_state.obs tensors is not valid.");

    // ---- N (batch 次元) の取得 ----
    //ANET_ASSERT_DTYPE(state.obs, torch::kFloat32);
    //ANET_ASSERT_DTYPE(next_state.obs, torch::kFloat32);
    ANET_ASSERT_SHAPE(state.done, { ANET_SHAPE_ANY });
    ANET_ASSERT_SHAPE(state.truncated, { ANET_SHAPE_ANY });
    ANET_ASSERT_SHAPE(state.episode_start, { ANET_SHAPE_ANY });

    const int64_t N = state.obs.Size(0);

    // ---- batch 次元の整合検査 ----
    ANET_ASSERT_MSG(next_state.obs.Size(0) == N,
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
    ANET_ASSERT_DTYPE(action->GetAction(), torch::kInt64);
    ANET_ASSERT_MSG(action->GetAction().size(0) == N, "MakeFromBatch: action.action batch size mismatch.");

    // ---- rewards の shape チェック ----
    ANET_ASSERT_DTYPE(reward, torch::kFloat32);
    ANET_ASSERT_MSG(reward.size(0) == N, "MakeFromBatch: reward batch size mismatch.");

    // ---- obs の最低限の次元検査 ----
    //ANET_ASSERT_MSG(state.obs.dim() >= 2,
    //    "MakeFromBatch: state.obs must have at least 2 dims (N, ...).");
    //ANET_ASSERT_MSG(next_state.obs.dim() >= 2,
    //    "MakeFromBatch: next_state.obs must have at least 2 dims (N, ...).");

    // ---- flatten 前の要素数チェック（破損検出）----
    //ANET_ASSERT_MSG(state.obs.numel() % N == 0,
    //    "MakeFromBatch: state.obs total elements not divisible by batch size.");
    //ANET_ASSERT_MSG(next_state.obs.numel() % N == 0,
    //    "MakeFromBatch: next_state.obs total elements not divisible by batch size.");

    // ReplayBufferがCPU前提なので、ここで一括してCPUに移す (GPU->CPU転送コストを1回に集約)
    // unbind(0) で N個の Tensor (view または copy) のリストに分解しておく

    // Observation / NextObservation
    auto state_obs_cpu = state.obs.Cpu();
    auto state_obs_list = state_obs_cpu.Unbind(0);
    auto next_obs_cpu = next_state.obs.Cpu();
    auto next_obs_list = next_obs_cpu.Unbind(0);

    // Action
    auto action_cpu = action->GetAction().cpu();
    auto action_list = action_cpu.unbind(0);

    // Scalars (Done, Truncated, EpisodeStart, Reward)
    auto s_done_cpu = state.done.to(torch::kCPU).contiguous();
    auto s_trunc_cpu = state.truncated.to(torch::kCPU).contiguous();
    auto s_start_cpu = state.episode_start.to(torch::kCPU).contiguous();
    auto ns_done_cpu = next_state.done.to(torch::kCPU).contiguous();
    auto ns_trunc_cpu = next_state.truncated.to(torch::kCPU).contiguous();
    auto ns_start_cpu = next_state.episode_start.to(torch::kCPU).contiguous();
    auto reward_cpu = reward.to(torch::kCPU).contiguous();

    // 生ポインタ取得 (オーバーヘッド回避)
    const bool* s_done_ptr = s_done_cpu.data_ptr<bool>();
    const bool* s_trunc_ptr = s_trunc_cpu.data_ptr<bool>();
    const bool* s_start_ptr = s_start_cpu.data_ptr<bool>();
    const bool* ns_done_ptr = ns_done_cpu.data_ptr<bool>();
    const bool* ns_trunc_ptr = ns_trunc_cpu.data_ptr<bool>();
    const bool* ns_start_ptr = ns_start_cpu.data_ptr<bool>();
    const float* reward_ptr = reward_cpu.data_ptr<float>();

    // =========================================================
    // ループ構築
    // =========================================================
    std::vector<SingleExperience> out;
    out.reserve(N);

    for (int64_t i = 0; i < N; ++i) {
        SingleState s = {
            state_obs_list[i],  // CPU Tensor
            s_done_ptr[i],
            s_trunc_ptr[i],
            s_start_ptr[i]
        };

        SingleState ns = {
            next_obs_list[i],   // CPU Tensor
            ns_done_ptr[i],
            ns_trunc_ptr[i],
            ns_start_ptr[i]
        };

        out.push_back({
            std::move(s),
            action_list[i],     // CPU Tensor
            reward_ptr[i],      // float
            std::move(ns)
            });
    }

    return out;
}


// =============================================================
// ExperienceSamples
// =============================================================

ExperienceSamples ExperienceSamples::To(torch::Device device, bool non_blocking) const
{
    ANET_PROFILE_FUNC();

    // GPUデバイスへの転送時のみストリームガードを有効にする
    if (device.is_cuda()) {
        auto stream = at::cuda::getCurrentCUDAStream();
        at::cuda::CUDAStreamGuard guard(stream);

        return {
            .obs = obs.To(device, non_blocking),
            .actions = anet::To(actions, device, non_blocking),
            .target_returns = anet::To(target_returns, device, non_blocking),
            .next_state = {
                .next_obs = next_state.next_obs.To(device, non_blocking),
                .terminals = anet::To(next_state.terminals, device, non_blocking),
            },
            .n_steps = anet::To(n_steps, device, non_blocking),
            // replay item key/source は ReplayBuffer の CPU metadata なので learner device へは送らない。
            .replay_item_keys = replay_item_keys.defined()
                ? (replay_item_keys.device().is_cpu() ? replay_item_keys : replay_item_keys.to(torch::kCPU))
                : replay_item_keys,
            .is_weights = anet::To(is_weights, device, non_blocking),
            .per_priority_sources = per_priority_sources.defined()
                ? (per_priority_sources.device().is_cpu() ? per_priority_sources : per_priority_sources.to(torch::kCPU))
                : per_priority_sources,
			.info = info.To(device, non_blocking)
        };
    }

    // CPUの場合はそのまま転送
    return {
        .obs = obs.To(device, non_blocking),
        .actions = actions.to(device, non_blocking),
        .target_returns = target_returns.to(device, non_blocking),
        .next_state = {
            .next_obs = next_state.next_obs.To(device, non_blocking),
            .terminals = next_state.terminals.to(device, non_blocking),
        },
        .n_steps = n_steps.defined() ? n_steps.to(device, non_blocking) : n_steps,
        .replay_item_keys = replay_item_keys.defined() ? replay_item_keys.to(torch::kCPU) : replay_item_keys,
        .is_weights = is_weights.defined() ? is_weights.to(device, non_blocking) : is_weights,
        .per_priority_sources = per_priority_sources.defined()
            ? per_priority_sources.to(torch::kCPU)
            : per_priority_sources,
        .info = info.To(device, non_blocking)
    };
}


std::string ExperienceSamples::ToString() const
{
    std::ostringstream oss;
    oss << "ExperienceSamples{\n";
    oss << "  obs = " << obs.ToString() << "\n";
    oss << "  action  = " << anet::ToString(actions) << "\n";
    oss << "  target_returns  = " << anet::ToString(target_returns) << "\n";
    oss << "  next_state.next_obs = " << next_state.next_obs.ToString() << "\n";
    oss << "  next_state.terminals     = " << anet::ToString(next_state.terminals) << "\n";
    oss << "  n_steps                  = " << anet::ToString(n_steps) << "\n";
    oss << "  replay_item_keys         = " << anet::ToString(replay_item_keys) << "\n";
    oss << "  is_weights               = " << anet::ToString(is_weights) << "\n";
    oss << "  per_priority_sources     = " << anet::ToString(per_priority_sources) << "\n";
    oss << "  info = " << info.ToString() << "\n";
    oss << "}";
    return oss.str();
}


// -----------------------------------------------------------------
// BatchExperience
// -----------------------------------------------------------------

std::string BatchExperience::ToString() const
{
    ANET_ASSERT_DTYPE(reward, torch::kFloat32);
    ANET_ASSERT_SHAPE(reward, { ANET_SHAPE_ANY });
    std::ostringstream oss;
    oss << "BatchExperience{\n";
    oss << "  state      = " << state.ToString() << "\n";
    oss << "  action     = " << action->ToString() << "\n";
    oss << "  reward     = " << anet::ToString(reward) << "\n";
    oss << "  next_state = " << next_state.ToString() << "\n";
    oss << "}";
    return oss.str();
}


// -----------------------------------------------------------------
// RunnerScopedTrainObserver
// -----------------------------------------------------------------

RunnerScopedTrainObserver::RunnerScopedTrainObserver(std::shared_ptr<TrainObserver> real_observer, std::shared_ptr<const Runner> target_runner)
    : real_observer_(real_observer), target_runner_(target_runner)
{
    ANET_CHECK(target_runner_ != nullptr);
}

void RunnerScopedTrainObserver::OnTrain(const TrainEvent& event)
{
    if (event.runner == target_runner_) {
        real_observer_->OnTrain(event);
    }
}

std::string RunnerScopedTrainObserver::ToString() const
{
    return "RunnerScopedTrainObserver(" + real_observer_->ToString() + ")";
}


// -----------------------------------------------------------------
// RunnerScopedLearnObserver
// -----------------------------------------------------------------
RunnerScopedLearnObserver::RunnerScopedLearnObserver(std::shared_ptr<LearnObserver> real_observer, std::shared_ptr<const Runner> target_runner)
    : real_observer_(real_observer), target_runner_(target_runner)
{
}

void RunnerScopedLearnObserver::OnLearn(const LearnEvent& event)
{
    if (event.runner == target_runner_) {
        real_observer_->OnLearn(event);
    }
}

std::string RunnerScopedLearnObserver::ToString() const
{
    return "RunnerScopedLearnObserver(" + real_observer_->ToString() + ")";
}


// -----------------------------------------------------------------
// RunnerScopedEpisodeEndObserver
// -----------------------------------------------------------------
RunnerScopedEpisodeEndObserver::RunnerScopedEpisodeEndObserver(std::shared_ptr<EpisodeEndObserver> real_observer, std::shared_ptr<const Runner> target_runner)
    : real_observer_(real_observer), target_runner_(target_runner)
{
    ANET_CHECK(target_runner_ != nullptr);
}

void RunnerScopedEpisodeEndObserver::OnEpisodeEnd(const EpisodeEndEvent& event)
{
    if (event.runner == target_runner_) {
        real_observer_->OnEpisodeEnd(event);
    }
}

std::string RunnerScopedEpisodeEndObserver::ToString() const
{
    return "RunnerScopedEpisodeEndObserver(" + real_observer_->ToString() + ")";
}


RunnerScopedSessionEndObserver::RunnerScopedSessionEndObserver(std::shared_ptr<SessionEndObserver> real_observer, std::shared_ptr<const Runner> target_runner)
    : real_observer_(real_observer), target_runner_(target_runner)
{
    ANET_CHECK(target_runner_ != nullptr);
}

void RunnerScopedSessionEndObserver::OnSessionEnd(const SessionEndEvent& event)
{
    if (event.runner == target_runner_) {
        real_observer_->OnSessionEnd(event);
    }
}

std::string RunnerScopedSessionEndObserver::ToString() const
{
    return "RunnerScopedSessionEndObserver(" + real_observer_->ToString() + ")";
}


// =============================================================
// Notifier
// =============================================================

Notifier::Notifier()
{
    ;
}

std::shared_ptr<TrainObserver> Notifier::Attach(std::shared_ptr<TrainObserver> obs)
{
    train_observers_.push_back(obs);
    return obs;
}

std::shared_ptr<LearnObserver> Notifier::Attach(std::shared_ptr<LearnObserver> obs)
{
    learn_observers_.push_back(obs);
    return obs;
}

std::shared_ptr<EpisodeEndObserver> Notifier::Attach(std::shared_ptr<EpisodeEndObserver> obs)
{
    episode_end_observers_.push_back(obs);
    return obs;
}

std::shared_ptr<SessionEndObserver> Notifier::Attach(std::shared_ptr<SessionEndObserver> obs)
{
    session_end_observers_.push_back(obs);
    return obs;
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

void Notifier::Detach(std::shared_ptr<EpisodeEndObserver> obs)
{
    episode_end_observers_.erase(
        std::remove_if(
            episode_end_observers_.begin(), episode_end_observers_.end(),
            [&](const std::shared_ptr<EpisodeEndObserver>& o) {
                return o == obs;
            }
        ),
        episode_end_observers_.end()
    );
}

void Notifier::Detach(std::shared_ptr<SessionEndObserver> obs)
{
    session_end_observers_.erase(
        std::remove_if(
            session_end_observers_.begin(), session_end_observers_.end(),
            [&](const std::shared_ptr<SessionEndObserver>& o) {
                return o == obs;
            }
        ),
        session_end_observers_.end()
    );
}

void Notifier::Detach(const TrainObserver* observer)
{
    train_observers_.erase(
        std::remove_if(
            train_observers_.begin(), train_observers_.end(),
            [&](const std::shared_ptr<TrainObserver>& o) {
                return o.get() == observer;
            }
        ),
        train_observers_.end()
    );
}

void Notifier::Detach(const LearnObserver* observer)
{
    learn_observers_.erase(
        std::remove_if(
            learn_observers_.begin(), learn_observers_.end(),
            [&](const std::shared_ptr<LearnObserver>& o) {
                return o.get() == observer;
            }
        ),
        learn_observers_.end()
    );
}

void Notifier::Detach(const EpisodeEndObserver* observer)
{
    episode_end_observers_.erase(
        std::remove_if(
            episode_end_observers_.begin(), episode_end_observers_.end(),
            [&](const std::shared_ptr<EpisodeEndObserver>& o) {
                return o.get() == observer;
            }
        ),
        episode_end_observers_.end()
    );
}

void Notifier::Detach(const SessionEndObserver* observer)
{
    session_end_observers_.erase(
        std::remove_if(
            session_end_observers_.begin(), session_end_observers_.end(),
            [&](const std::shared_ptr<SessionEndObserver>& o) {
                return o.get() == observer;
            }
        ),
        session_end_observers_.end()
    );
}

void Notifier::Clear()
{
    train_observers_.clear();
    learn_observers_.clear();
    episode_end_observers_.clear();
    session_end_observers_.clear();
}

void Notifier::Notify(const TrainEvent& event)
{
    ANET_PROFILE_FUNC();

    for (auto obs : train_observers_) {
        obs->OnTrain(event);
    }
}

void Notifier::Notify(const LearnEvent& event)
{
    ANET_PROFILE_FUNC();

    for (auto obs : learn_observers_) {
        obs->OnLearn(event);
    }
}

void Notifier::Notify(const EpisodeEndEvent& event)
{
    ANET_PROFILE_FUNC();

    for (auto obs : episode_end_observers_) {
        obs->OnEpisodeEnd(event);
    }
}

void Notifier::Notify(const SessionEndEvent& event)
{
    ANET_PROFILE_FUNC();

    for (auto obs : session_end_observers_) {
        obs->OnSessionEnd(event);
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
    idx = 0;
    for (auto obs : episode_end_observers_) {
        LOG::info() << "Notifier: EPISODE_END [" << idx << "] " << obs->ToString();
        idx++;
    }
    idx = 0;
    for (auto obs : session_end_observers_) {
        LOG::info() << "Notifier: SESSION_END [" << idx << "] " << obs->ToString();
        idx++;
    }
}
