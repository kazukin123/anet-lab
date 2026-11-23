#include "anet/rl.hpp"
#include <algorithm>
#include <stdexcept>
#include <charconv>
#include <utility>
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
#include "anet/nn.hpp"

using namespace anet::rl;
namespace LOG = anet::log;

ForcedActions ForcedActions::ByLane(int64_t lane, int64_t action)
{
    return ForcedActions(Lane{.index = lane, .action = action});
}

ForcedActions ForcedActions::ByMask(torch::Tensor mask, torch::Tensor actions)
{
    return ForcedActions(Mask{.lanes = std::move(mask), .actions = std::move(actions)});
}

static torch::Tensor SelectForcedActions(const torch::Tensor& mask, const torch::Tensor& actions,
    int64_t lane_count, int64_t action_count)
{
    // マスクと行動が lane ごとに 1 要素ずつ対応することを確認する。
    const torch::IntArrayRef expected_shape(&lane_count, 1);
    if (!mask.defined() || mask.scalar_type() != torch::kBool || mask.sizes() != expected_shape
        || !actions.defined() || actions.scalar_type() != torch::kInt64 || actions.sizes() != expected_shape) {
        const auto mask_description = mask.defined() ? anet::ToDefString(mask) : "undefined";
        const auto actions_description = actions.defined() ? anet::ToDefString(actions) : "undefined";
        ANET_SYSTEM_ERROR("Invalid forced action mask or values. mask=" << mask_description
            << " actions=" << actions_description << " expected bool mask and int64 actions of shape "
            << expected_shape << " action_count=" << action_count << ".");
    }
    // 強制対象だけを取り出して行動の範囲を検査し、エラーには元の lane を示す。
    const auto lanes = mask.to(actions.device());
    const auto selected = actions.masked_select(lanes);
    const auto invalid = selected.lt(0).logical_or(selected.ge(action_count)).nonzero();
    if (invalid.numel() != 0) {
        const auto index = invalid[0].item<int64_t>();
        ANET_SYSTEM_ERROR("Invalid forced action. lane=" << lanes.nonzero()[index].item<int64_t>()
            << " action=" << selected[index].item<int64_t>() << " expected action=[0, " << action_count - 1
            << "] action_count=" << action_count << ".");
    }
    return selected;
}

void ForcedActions::Validate(const ActionSpec& spec, int64_t lane_count) const
{
    if (!spec.is_discrete) {
        ANET_SYSTEM_ERROR("ForcedActions requires a discrete action space. lane_count=" << lane_count);
    }
    ValidateDiscrete(lane_count, spec.GetNumActions());
}

torch::Tensor ForcedActions::Apply(const torch::Tensor& policy_actions, int64_t action_count) const
{
    ANET_PROFILE_FUNC();
    const auto lane_count = policy_actions.size(0);
    // マスク指定では選択した lane の行動だけをまとめて置き換える。
    if (const auto* mask = std::get_if<Mask>(&selection_)) {
        const auto selected = SelectForcedActions(mask->lanes, mask->actions, lane_count, action_count);
        return policy_actions.masked_scatter(MakeMask(lane_count, policy_actions.device()), selected.to(policy_actions.device()));
    }
    // 単一 lane の指定では方策の出力を複製し、その lane だけを書き換える。
    ValidateDiscrete(lane_count, action_count);
    const auto& lane = std::get<Lane>(selection_);
    auto actions = policy_actions.clone();
    actions[lane.index].fill_(lane.action);
    return actions;
}

torch::Tensor ForcedActions::MakeMask(int64_t lane_count, torch::Device device) const
{
    if (const auto* mask = std::get_if<Mask>(&selection_)) return mask->lanes.to(device);
    const auto& lane = std::get<Lane>(selection_);
    auto mask = torch::zeros({lane_count}, torch::TensorOptions().device(device).dtype(torch::kBool));
    mask[lane.index].fill_(true);
    return mask;
}

void ForcedActions::ValidateDiscrete(int64_t lane_count, int64_t action_count) const
{
    if (const auto* mask = std::get_if<Mask>(&selection_)) {
        SelectForcedActions(mask->lanes, mask->actions, lane_count, action_count);
        return;
    }
    const auto& lane = std::get<Lane>(selection_);
    if (lane.index < 0 || lane.index >= lane_count || lane.action < 0 || lane.action >= action_count) {
        ANET_SYSTEM_ERROR("Invalid forced action. lane=" << lane.index << " action=" << lane.action
            << " expected lane=[0, " << lane_count - 1 << "] action=[0, " << action_count - 1
            << "] action_count=" << action_count << ".");
    }
}

void anet::rl::AppendNnTraceAux(AuxData& aux, anet::ForwardCapture& capture)
{
    ANET_PROFILE_FUNC();
    // lane 0 だけを独立した FP32 の記憶域へ移し、batch 全体の保持を終える。
    for (const auto& [key, activation] : capture.GetAll(anet::nn::kNnTracePrefix)) {
        aux[key] = activation.slice(0, 0, 1).detach().to(torch::kFloat32).clone();
    }
    capture.Reset();
}

anet::TensorDict anet::rl::ExtractNnTrace(const AuxData& aux)
{
    anet::TensorDict dict;
    for (const auto& [key, tensor] : aux) {
        if (key.starts_with(anet::nn::kNnTracePrefix)) {
            dict.Set(key.substr(anet::nn::kNnTracePrefix.size()), tensor);
        }
    }
    return dict;
}

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

        if (spec.scale == ValueScale::Unscaled && (spec.IsDiscrete() || c10::isIntegralType(spec.dtype, /*includeBool=*/false))) {
            ANET_SYSTEM_ERROR("Invalid TensorSpec scale. key='" << key << "' scale=Unscaled dtype="
                << spec.dtype << " num_classes=" << spec.num_classes
                << "; expected a non-integral dtype with no discrete classes.");
        }

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

std::optional<float> BatchActionInfo::GetScalar(const std::string& key, int64_t lane) const
{
    // lane指定の実行行動だけを共通に読み、scalar集計は派生型へ任せる。
    if (lane < 0 || (key != "action" && !key.starts_with("action.["))) return std::nullopt;
    const auto action = GetAction(torch::kCPU);
    ANET_CHECK_MSG(action.defined() && lane < action.size(0),
        "Invalid action scalar lane. key='" << key << "' lane=" << lane << " expected=an existing lane.");
    if (key == "action") {
        ANET_CHECK_MSG(action.dim() == 1,
            "Continuous action requires action.[dimension]. key='" << key << "' expected=action.[dimension].");
        return action[lane].item<float>();
    }
    // 次元指定は文字列全体を解析し、suffixや負値・範囲外を受理しない。
    int64_t dimension = -1;
    bool valid = key.size() > 9 && key.ends_with("]");
    if (valid) {
        const auto end = key.data() + key.size() - 1;
        const auto parsed = std::from_chars(key.data() + 8, end, dimension);
        valid = parsed.ec == std::errc{} && parsed.ptr == end;
    }
    if (!valid || action.dim() != 2 || dimension < 0 || dimension >= action.size(1)) {
        ANET_SYSTEM_ERROR("Invalid continuous action dimension. key='" << key
            << "' expected=action.[integer dimension] within the action shape.");
    }
    return action[lane][dimension].item<float>();
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

void ExperienceSamples::FinalizeReturns(std::span<const float> gammas)
{
    if (!sampled_rewards.defined() && !sampled_window_rewards.defined()) return;
    ANET_PROFILE_FUNC();
    // ExperienceBuilderと同じ逆順のfloat演算で、消費時の収益を確定する。
    const auto evaluate = [&](const torch::Tensor& rewards, torch::Tensor& result) {
        if (!rewards.defined()) return;
        if (gammas.empty()) ANET_SYSTEM_ERROR("Return gammas count=0; expected at least one component.");
        if (rewards.dim() == 3) {
            if (gammas.size() != static_cast<size_t>(rewards.size(2)))
                ANET_SYSTEM_ERROR("Return gammas count=" << gammas.size() << "; expected " << rewards.size(2) << ".");
            auto values = torch::zeros({rewards.size(0), rewards.size(2)}, rewards.options());
            // 各成分の演算順序は単一成分と同じ逆順に保つ。
            for (int64_t component = 0; component < rewards.size(2); ++component) {
                auto value = values.select(1, component);
                for (int64_t offset = rewards.size(1); offset-- > 0;)
                    value.copy_(rewards.select(1, offset).select(1, component) + gammas[component] * value);
            }
            result = needs_return_recompute.defined() ? torch::where(needs_return_recompute, values, result) : values;
            return;
        }
        result = torch::zeros({rewards.size(0)}, rewards.options());
        for (int64_t offset = rewards.size(1); offset-- > 0;) {
            result = rewards.select(1, offset) + gammas.front() * result;
        }
    };
    evaluate(sampled_rewards, target_returns);
    evaluate(sampled_window_rewards, window.returns);
}

ExperienceSamples ExperienceSamples::To(torch::Device device, bool non_blocking) const
{
    ANET_PROFILE_FUNC();
    ExperienceSamples result;
    // device に送る値と CPU metadata の違いだけを扱い、必須性は利用側へ残す。
    VisitFields(*this, result, [&](const auto& source, auto& destination, const char*, bool cpu_only) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(source)>, anet::TensorDict>)
            destination = source.To(device, non_blocking);
        else
            destination = anet::To(source, cpu_only ? torch::Device(torch::kCPU) : device,
                cpu_only ? false : non_blocking);
    });
    return result;
}


std::string ExperienceSamples::ToString() const
{
    std::ostringstream oss;
    oss << "ExperienceSamples{\n";
    VisitFields(*this, *this, [&](const auto& value, const auto&, const char* name, bool) {
        oss << "  " << name << " = ";
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(value)>, anet::TensorDict>)
            oss << value.ToString();
        else
            oss << anet::ToString(value);
        oss << "\n";
    });
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

void RunnerScopedTrainObserver::Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode)
{
    real_observer_->Shutdown(deadline, mode);
}

bool RunnerScopedTrainObserver::WillBlockOnShutdown() const
{
    return real_observer_->WillBlockOnShutdown();
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

void RunnerScopedLearnObserver::Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode)
{
    real_observer_->Shutdown(deadline, mode);
}

bool RunnerScopedLearnObserver::WillBlockOnShutdown() const
{
    return real_observer_->WillBlockOnShutdown();
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

void RunnerScopedEpisodeEndObserver::Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode)
{
    real_observer_->Shutdown(deadline, mode);
}

bool RunnerScopedEpisodeEndObserver::WillBlockOnShutdown() const
{
    return real_observer_->WillBlockOnShutdown();
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

void RunnerScopedSessionEndObserver::Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode)
{
    real_observer_->Shutdown(deadline, mode);
}

bool RunnerScopedSessionEndObserver::WillBlockOnShutdown() const
{
    return real_observer_->WillBlockOnShutdown();
}

std::string RunnerScopedSessionEndObserver::ToString() const
{
    return "RunnerScopedSessionEndObserver(" + real_observer_->ToString() + ")";
}


// =============================================================
// Notifier
// =============================================================

Notifier::Notifier() = default;

template <class Observer>
Notifier::ObserverList<Observer> Notifier::GetObserverSnapshot(const ObserverList<Observer>& observers) const
{
    const std::lock_guard lock(observers_mutex_);
    return observers;
}

Notifier::ObserverLists Notifier::GetObserverSnapshot() const
{
    const std::lock_guard lock(observers_mutex_);
    return observers_;
}

template <class Observer>
std::shared_ptr<Observer> Notifier::AttachObserver(
    ObserverList<Observer>& observers, std::shared_ptr<Observer> observer)
{
    const std::lock_guard lock(observers_mutex_);
    // 進行中の通知は取得済みの一覧を使う。追加した一覧は次の通知から見える。
    auto updated = std::make_shared<std::vector<std::shared_ptr<Observer>>>(*observers);
    updated->push_back(observer);
    observers = std::move(updated);
    return observer;
}

template <class Observer>
void Notifier::DetachObserver(ObserverList<Observer>& observers, const Observer* observer)
{
    ObserverList<Observer> previous;
    {
        const std::lock_guard lock(observers_mutex_);
        const auto matches = [observer](const auto& entry) { return entry.get() == observer; };
        if (std::none_of(observers->begin(), observers->end(), matches)) return;

        auto updated = std::make_shared<std::vector<std::shared_ptr<Observer>>>(*observers);
        std::erase_if(*updated, matches);
        previous = std::exchange(observers, std::move(updated));
    }
    // 最後の参照を手放すと Observer が破棄されるので、ロックの外で行う。
}

std::shared_ptr<TrainObserver> Notifier::Attach(std::shared_ptr<TrainObserver> observer)
{
    return AttachObserver(observers_.train, std::move(observer));
}

void Notifier::Detach(std::shared_ptr<TrainObserver> observer)
{
    Detach(observer.get());
}

void Notifier::Detach(const TrainObserver* observer)
{
    DetachObserver(observers_.train, observer);
}

std::shared_ptr<LearnObserver> Notifier::Attach(std::shared_ptr<LearnObserver> observer)
{
    return AttachObserver(observers_.learn, std::move(observer));
}

void Notifier::Detach(std::shared_ptr<LearnObserver> observer)
{
    Detach(observer.get());
}

void Notifier::Detach(const LearnObserver* observer)
{
    DetachObserver(observers_.learn, observer);
}

std::shared_ptr<EpisodeEndObserver> Notifier::Attach(std::shared_ptr<EpisodeEndObserver> observer)
{
    return AttachObserver(observers_.episode_end, std::move(observer));
}

void Notifier::Detach(std::shared_ptr<EpisodeEndObserver> observer)
{
    Detach(observer.get());
}

void Notifier::Detach(const EpisodeEndObserver* observer)
{
    DetachObserver(observers_.episode_end, observer);
}

std::shared_ptr<SessionEndObserver> Notifier::Attach(std::shared_ptr<SessionEndObserver> observer)
{
    return AttachObserver(observers_.session_end, std::move(observer));
}

void Notifier::Detach(std::shared_ptr<SessionEndObserver> observer)
{
    Detach(observer.get());
}

void Notifier::Detach(const SessionEndObserver* observer)
{
    DetachObserver(observers_.session_end, observer);
}

void Notifier::Clear()
{
    ObserverLists previous;
    {
        const std::lock_guard lock(observers_mutex_);
        std::swap(previous, observers_);
    }
    // 古い一覧の参照をロックの外で手放す。取得済みの通知は古い一覧を使い続ける。
}

void Notifier::Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode)
{
    const auto observers = GetObserverSnapshot();
    // observer の登録列を順に排水する。例外は呼び出し元へそのまま伝える。
    for (const auto& observer : *observers.train) observer->Shutdown(deadline, mode);
    for (const auto& observer : *observers.learn) observer->Shutdown(deadline, mode);
    for (const auto& observer : *observers.episode_end) observer->Shutdown(deadline, mode);
    for (const auto& observer : *observers.session_end) observer->Shutdown(deadline, mode);
}

bool Notifier::WillBlockOnShutdown() const
{
    const auto observers = GetObserverSnapshot();
    // どれか一つでも WAIT を伴うなら、終了UIへ通知する。
    for (const auto& observer : *observers.train)
        if (observer->WillBlockOnShutdown()) return true;
    for (const auto& observer : *observers.learn)
        if (observer->WillBlockOnShutdown()) return true;
    for (const auto& observer : *observers.episode_end)
        if (observer->WillBlockOnShutdown()) return true;
    for (const auto& observer : *observers.session_end)
        if (observer->WillBlockOnShutdown()) return true;
    return false;
}

void Notifier::Notify(const TrainEvent& event)
{
    ANET_PROFILE_FUNC();

    // 一覧の取得にだけロックを使い、Observer はロックの外で呼ぶ。
    const auto observers = GetObserverSnapshot(observers_.train);
    for (const auto& observer : *observers) {
        observer->OnTrain(event);
    }
}

void Notifier::Notify(const LearnEvent& event)
{
    ANET_PROFILE_FUNC();

    // 一覧の取得にだけロックを使い、Observer はロックの外で呼ぶ。
    const auto observers = GetObserverSnapshot(observers_.learn);
    for (const auto& observer : *observers) {
        observer->OnLearn(event);
    }
}

void Notifier::Notify(const EpisodeEndEvent& event)
{
    ANET_PROFILE_FUNC();

    // 一覧の取得にだけロックを使い、Observer はロックの外で呼ぶ。
    const auto observers = GetObserverSnapshot(observers_.episode_end);
    for (const auto& observer : *observers) {
        observer->OnEpisodeEnd(event);
    }
}

void Notifier::Notify(const SessionEndEvent& event)
{
    ANET_PROFILE_FUNC();

    // 一覧の取得にだけロックを使い、Observer はロックの外で呼ぶ。
    const auto observers = GetObserverSnapshot(observers_.session_end);
    for (const auto& observer : *observers) {
        observer->OnSessionEnd(event);
    }
}

void Notifier::LogObservers() const
{
    const auto observers = GetObserverSnapshot();
    const auto log = [](const auto& list, const char* name) {
        int index = 0;
        for (const auto& observer : list) {
            LOG::info() << "Notifier: " << name << " [" << index << "] " << observer->ToString();
            ++index;
        }
    };
    log(*observers.train, "TRAIN");
    log(*observers.learn, "LEARN");
    log(*observers.episode_end, "EPISODE_END");
    log(*observers.session_end, "SESSION_END");
}

// 結果が所有するauxのスカラーだけを読み、非スカラーは未知の読出しとして扱う。
std::optional<float> SingleEnvResult::GetScalar(const std::string& key) const
{
    const auto found = aux_.find(key);
    if (found == aux_.end() || !found->second.defined() || found->second.numel() != 1) return std::nullopt;
    return found->second.item<float>();
}

const torch::Tensor& BatchStepResult::GetEpisodeEndMask() const
{
    // 初回の並行読出しも同期し、公開値を変えずにCPU転送を結果ごとに一度へまとめる。
    std::call_once(episode_end_once_, [this]() {
        episode_end_cpu_ = next_state.done.logical_or(next_state.truncated).to(torch::kCPU).contiguous();
    });
    return episode_end_cpu_;
}

std::optional<float> BatchStepResult::GetScalar(const std::string& key, int64_t lane) const
{
    if (lane < 0 || lane >= reward.size(0)) return std::nullopt;
    if (key == "reward") return reward[lane].item<float>();
    if (key == "done") return next_state.done[lane].item<float>();
    if (key == "truncated") return next_state.truncated[lane].item<float>();
    return std::nullopt;
}
