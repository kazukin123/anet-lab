// image_log.cpp （抜粋）

#include "anet/probe.hpp"
#include <sstream>
#include "anet/log.hpp"
#include "anet/profile.hpp"

using namespace anet::rl;
namespace LOG = anet::log;


//std::optional<float> MetricsScalarProbe::GetFloat(const TrainEvent& event) const
//{
//    return event.update_result->GetScalar(key_);
//}

// ============================================================
// StaticScalarProbe
// ============================================================

std::optional<float> StaticScalarProbe::GetFloat(const TrainEvent& event) const
{
    return value_;
}

// ============================================================
// FunctionScalarProbe
// ============================================================

std::optional<float> FunctionScalarProbe::GetFloat(const TrainEvent& event) const
{
    return fn_(event);
}

// ============================================================
// BatchExperience*Probe
// ============================================================

BatchExperienceStateProbe::BatchExperienceStateProbe(
    int64_t state_index, const anet::rl::StateSpec* spec, bool for_next_state, std::optional<std::string> name)
    : state_index_(state_index), for_next_state_(for_next_state)
{
    ANET_ASSERT(state_index >= 0);

    std::optional<std::string> name_spec;

    /// @todo v2暫定: デフォルトのVector観測から情報を取得
    if (spec != nullptr && spec->obs_spec.count(anet::rl::ObsKeys::kVector) > 0) {
        const auto& tspec = spec->obs_spec.at(anet::rl::ObsKeys::kVector);
        ANET_ASSERT(state_index < tspec.CalcFlattenDim());

        min_ = tspec.GetMin(state_index);
        max_ = tspec.GetMax(state_index);
        if (state_index < tspec.labels.size()) {
            name_spec = tspec.labels[state_index];
        }
    }

    // nameの個別指定があれば優先
    if (name.has_value()) {
        name_ = *name;
    } else {
        // 個別指定が無ければ自動生成
        std::stringstream ss;
        ss << BatchExperience::STATE_OBS << "[" << state_index << "]";
        if (name_spec.has_value()) {
            ss << "(" + *name_spec + ")";
        }
        name_ = ss.str();
    }
}

std::optional<std::vector<float>> BatchExperienceStateProbe::GetVector(const UpdateEvent& event) const
{
    ANET_PROFILE_FUNC();

    // 対象obs（現状態 or next_state）
    auto obs = (for_next_state_) ?
        event.experience.GetTensor(anet::rl::BatchExperience::NEXT_STATE_OBS) :
        event.experience.GetTensor(anet::rl::BatchExperience::STATE_OBS);
    ANET_ASSERT(obs.has_value());

    if (!obs->defined())
        return std::nullopt;

    // CPU 転送
    ANET_ASSERT_DTYPE(*obs, torch::kFloat32);
    auto obs_cpu = obs->to(torch::kCPU);

    // state: (N, dim1, dim2, ..., dimK)
    ANET_ASSERT(obs_cpu.dim() >= 2);
    const int64_t N = obs_cpu.size(0);

    // flatten: dim1～(K) を flatten -> (N, flat_dim)
    torch::Tensor flat = obs_cpu.flatten(1);
    ANET_ASSERT_SHAPE(flat, { N, ANET_SHAPE_ANY });
    const int64_t flat_size = flat.size(1);
    ANET_ASSERT(state_index_ >= 0 && state_index_ < flat_size);

    // 結果用
    std::vector<float> out;
    out.resize(N);

    auto accessor = flat.accessor<float, 2>();
    for (int64_t i = 0; i < N; i++) {
            out[i] = accessor[i][state_index_];
        }

    return out;
}

BatchExperienceRewardProbe::BatchExperienceRewardProbe(const anet::rl::EnvSpec* spec, std::optional<std::string> name)
{
    if (spec != nullptr) {
        min_ = spec->reward_range.first;
        max_ = spec->reward_range.second;
    }
    
    if (name.has_value()) {
        name_ = *name;
    } else {
        name_ = BatchExperience::REWARD;
    }
}

std::optional<std::vector<float>> BatchExperienceRewardProbe::GetVector(const UpdateEvent& event) const
{
    auto tensor = event.experience.GetTensor(anet::rl::BatchExperience::REWARD);
    ANET_ASSERT(tensor.has_value());
    ANET_ASSERT_SHAPE(*tensor, { ANET_SHAPE_ANY });   // (N)
    ANET_ASSERT_DTYPE(*tensor, torch::kFloat32);

    torch::Tensor flat = tensor->flatten().to(torch::kCPU);
    ANET_ASSERT_SHAPE(flat, { ANET_SHAPE_ANY });

    const int64_t n = flat.size(0);
    std::vector<float> out;
    out.resize(n);
    std::memcpy(out.data(), flat.data_ptr<float>(), sizeof(float) * static_cast<size_t>(n));

    return out;
}

// ============================================================
// BatchExperienceVectorProbe
// ============================================================

BatchExperienceVectorProbe::BatchExperienceVectorProbe(
    const std::string& key, int index,
    const anet::rl::StateSpec* state_spec, const anet::rl::ActionSpec* action_spec,
    std::optional<float> min_override, std::optional<float> max_override,
    std::optional<std::string> name)
    : key_(key), index_(index)
{
    std::optional<std::string> name_spec;

    /// @todo v2暫定:EnvSpec の state_spec から min/max を取得 (暫定Vector指定)
    if (state_spec != nullptr && index_ >= 0 && state_spec->obs_spec.count(anet::rl::ObsKeys::kVector) > 0) {
        const auto& tspec = state_spec->obs_spec.at(anet::rl::ObsKeys::kVector);
        ANET_ASSERT(index_ < (int)tspec.CalcFlattenDim());

        min_ = tspec.GetMin(index_);
        max_ = tspec.GetMax(index_);
        if (index_ < tspec.labels.size()) {
            name_spec = tspec.labels[index_];
        }
    } else if (action_spec != nullptr) {
        //  ActionSpec では index_ は使わない（離散を想定）
        // ActionSpec から範囲を取る（離散を想定）
        ANET_ASSERT(action_spec->is_discrete);

        //  離散 [0, n_actions-1]
        const int n_actions = action_spec->GetNumActions();
        ANET_ASSERT(n_actions > 0);
        //if (auto_scale_mode == AutoScaleMode::DISABLE) {
            min_ = 0.0f;
            max_ = static_cast<float>(n_actions - 1);
        //}
        ANET_CHECK_MSG(index_ < 0 || index_ < action_spec->value_labels.size(), "Invalid index:" << index_);
        if (index_ >= 0) name_spec = action_spec->value_labels[index_];
    }

    // min/max の明示指定があればそれを最優先
    if (min_override.has_value()) {
        min_ = min_override;
    }
    if (max_override.has_value()) {
        max_ = max_override;
    }

    if (name.has_value()) {
        name_ = *name;
    } else {
        std::stringstream ss;
        ss << key_ << "[" << index_ << "]";
        if (name_spec.has_value()) {
            ss << "(" << *name_spec + ")";
        }
        name_ = ss.str();
    }

    // minだけ指定、maxだけ指定でもOK
}

std::optional<std::vector<float>> BatchExperienceVectorProbe::GetVector(const UpdateEvent& event) const
{
    ANET_PROFILE_FUNC();

    ANET_PROFILE_SCOPE(get_tensor_vector);

    auto opt_vec = event.experience.GetTensorVector(key_);

    //ANET_ASSERT(opt_vec.has_value());
    if (!opt_vec.has_value()) {
        LOG::warn() << "BatchExperienceVectorProbe::GetVector() failed to get value." << this->GetName();
        return std::nullopt;
    }

    ANET_PROFILE_SCOPE_NEXT(dump, get_tensor_vector);
    const auto& tvec = opt_vec.value();
    std::vector<float> out;

    // ---- index 指定なし：flatten して全て連結 ----
    if (index_ < 0) {
        for (const auto& t : tvec) {
            torch::Tensor flat = t.flatten().to(torch::kCPU);
            auto dtype = flat.dtype();

            ANET_ASSERT(dtype == torch::kFloat32 ||
                dtype == torch::kInt64 ||
                dtype == torch::kBool);

            const int64_t n = flat.size(0);
            const size_t old = out.size();
            out.resize(old + n);

            if (dtype == torch::kFloat32) {
                std::memcpy(out.data() + old, flat.data_ptr<float>(), n * sizeof(float));
            } else if (dtype == torch::kInt64) {
                const int64_t* src = flat.data_ptr<int64_t>();
                float* dst = out.data() + old;
                for (int64_t i = 0; i < n; i++)
                    dst[i] = static_cast<float>(src[i]);
            } else if (dtype == torch::kBool) {
                const bool* src = flat.data_ptr<bool>();
                float* dst = out.data() + old;
                for (int64_t i = 0; i < n; i++)
                    dst[i] = src[i] ? 1.0f : 0.0f;
            }
        }

        return out;
    }

    ANET_PROFILE_SCOPE_NEXT(extract, dump);
    out.reserve(1024); // optional
    for (const auto& t : tvec) {
        ANET_LOG_DEBUG("t=" << anet::ToDefString(t));

        // t must be 2-D (ReplayBuffer)
        ANET_ASSERT_SHAPE(t, { ANET_SHAPE_ANY, ANET_SHAPE_ANY });
        ANET_ASSERT(t.dim() == 2);   // [rows, D]

        const int64_t rows = t.size(0);
        const int64_t dim = t.size(1);

        ANET_ASSERT(index_ < dim);

        // 1列だけ抽出
        torch::Tensor col = t.index({
            torch::indexing::Slice(),
            index_
            }).to(torch::kCPU);  // [rows]
        if (!col.is_contiguous()) {
            col = col.contiguous();
        }

        auto dtype = col.dtype();
        ANET_ASSERT(dtype == torch::kFloat32 || dtype == torch::kInt64 || dtype == torch::kBool);

        const int64_t n = col.size(0);
        const size_t old = out.size();
        out.resize(old + n);

        float* dst = out.data() + old;

        if (dtype == torch::kFloat32) {
            std::memcpy(dst, col.data_ptr<float>(), n * sizeof(float));
            //if (auto_scale_mode_ == AutoScaleMode::PER_STEP) {
            //    min_ = col.min().item<float>();
            //    max_ = col.max().item<float>();
            //} else if (auto_scale_mode_ == AutoScaleMode::GLOBAL) {
            //    const float min = col.min().item<float>();
            //    const float max = col.max().item<float>();
            //    min_ = std::min(min_.value_or(min), min);
            //    max_ = std::max(max_.value_or(max), max);
            //}
        } else if (dtype == torch::kInt64) {
            const int64_t* src = col.data_ptr<int64_t>();
            for (int64_t i = 0; i < n; i++)
                dst[i] = static_cast<float>(src[i]);
        } else { // bool
            const bool* src = col.data_ptr<bool>();
            for (int64_t i = 0; i < n; i++)
                dst[i] = src[i] ? 1.0f : 0.0f;
        }
    }

    return out;
}

// ============================================================
// BatchUpdateResultTensorToVectorProbe
// ============================================================

BatchUpdateResultTensorToVectorProbe::BatchUpdateResultTensorToVectorProbe(const std::string& key, std::optional<float> min, std::optional<float> max)
    : key_(key), min_(min), max_(max)
{
    name_ = "BatchUpdateResultTensorToVectorProbe[" + key + "]";
}

std::optional<std::vector<float>> BatchUpdateResultTensorToVectorProbe::GetVector(const UpdateEvent& event) const
{
    const auto& result_list = event.update_result_list;
    std::vector<float> out;
    bool has_value = false;

    // BatchUpdateResultListで回す
    for (auto result : result_list) {
        // 値取得
        auto tensor = result->GetTensor(key_);

        // 値取得判定
        //ANET_ASSERT(tensor.has_value());   // key誤りによるバグ防止のため
        if (!tensor.has_value()) continue;
        if (!tensor->defined()) continue;
        has_value = true;

        // データ取り出し準備
        torch::Tensor flat = tensor->flatten().to(torch::kCPU).contiguous();
        ANET_ASSERT_SHAPE(flat, { ANET_SHAPE_ANY });
        ANET_ASSERT_DTYPE(flat, torch::kFloat32);

        // outに詰める（追加）
        const float* ptr = flat.data_ptr<float>();
        out.insert(out.end(), ptr, ptr + flat.numel());
    }

    if (has_value) return out;
    return std::nullopt;
}

//BatchActionInfoToVectorProbe::BatchActionInfoToVectorProbe(const std::string& key, std::optional<float> min, std::optional<float> max)
//    : key_(key), min_(min), max_(max)
//{
//    name_ = "BatchActionInfoToVectorProbe[" + key + "]";
//}
//
//std::optional<std::vector<float>> BatchActionInfoToVectorProbe::GetVector(const TrainEvent& event) const
//{
//    ProfileRange r("BatchActionInfoToVectorProbe::GetVector");
//
//    auto itr = event.action_info.GetAuxData().find(key_);
//    if (itr == event.action_info.GetAuxData().end()) return std::nullopt;
//    auto tensor = itr->second;
//
//    //ANET_ASSERT(tensor.has_value());   // key誤りによるバグ防止のため
//    if (!tensor.defined()) return std::nullopt;
//
//    torch::Tensor flat = tensor.flatten().to(torch::kCPU);
//    ANET_ASSERT_SHAPE(flat, { ANET_SHAPE_ANY });
//    ANET_ASSERT_DTYPE(flat, torch::kFloat32);
//
//    std::vector<float> out;
//    out.resize(flat.size(0));
//    std::memcpy(out.data(), flat.data_ptr<float>(), flat.size(0) * sizeof(float));
//    return out;
//}

// ============================================================
// AgentTensorVectorProbe
// ============================================================

AgentTensorVectorProbe::AgentTensorVectorProbe(
    const std::string& key, int index,
    const anet::rl::StateSpec* state_spec, const anet::rl::ActionSpec* action_spec,
    AutoScaleMode auto_scale_mode,
    std::optional<float> min_override, std::optional<float> max_override,
    std::optional<std::string> name)
    : key_(key), index_(index), auto_scale_mode_(auto_scale_mode)
{
    std::optional<std::string> name_spec;

    /// @todo v2暫定: EnvSpec の state_spec から min/max を取得 (暫定Vector指定)
    if (state_spec != nullptr && index_ >= 0 && state_spec->obs_spec.count(anet::rl::ObsKeys::kVector) > 0) {
        const auto& tspec = state_spec->obs_spec.at(anet::rl::ObsKeys::kVector);
        ANET_ASSERT(index_ < (int)tspec.CalcFlattenDim());

        if (auto_scale_mode == AutoScaleMode::DISABLE) {
            min_ = tspec.GetMin(index_);
            max_ = tspec.GetMax(index_);
        }
        if (index_ < tspec.labels.size()) {
            name_spec = tspec.labels[index_];
        }
    } else if (action_spec != nullptr) {
        //  ActionSpec では index_ は使わない（離散を想定）
        // ActionSpec から範囲を取る（離散を想定）
        ANET_ASSERT(action_spec->is_discrete);

        //  離散 [0, n_actions-1]
        const int n_actions = action_spec->GetNumActions();
        ANET_ASSERT(n_actions > 0);
        if (auto_scale_mode == AutoScaleMode::DISABLE) {
            min_ = 0.0f;
            max_ = static_cast<float>(n_actions - 1);
        }
        ANET_CHECK(index < action_spec->value_labels.size());
        name_spec = action_spec->value_labels[index];
    }

    // min/max の明示指定があればそれを最優先
    if (min_override.has_value()) {
        min_ = min_override;
    }
    if (max_override.has_value()) {
        max_ = max_override;
    }

    if (name.has_value()) {
        name_ = *name;
    } else {
        std::stringstream ss;
        ss << key_ << "[" << index_ << "]";
        if (name_spec.has_value()) {
            ss << "(" << *name_spec + ")";
        }
        name_ = ss.str();
    }

    // minだけ指定、maxだけ指定でもOK
}

std::optional<std::vector<float>> AgentTensorVectorProbe::GetVector(const UpdateEvent& event) const
{
    ANET_PROFILE_FUNC();

    ANET_PROFILE_SCOPE(get_tensor_vector);
    auto opt_vec = event.agent->GetTensorVector(key_);
    //ANET_ASSERT(opt_vec.has_value());
    if (!opt_vec.has_value()) {
        //LOG::warn() << "AgentTensorVectorProbe::GetVector() failed to get value." << this->GetName();
        return std::nullopt;
    }

    ANET_PROFILE_SCOPE_NEXT(dump, get_tensor_vector);
    const auto& tvec = opt_vec.value();
    std::vector<float> out;
    // ---- index 指定なし：flatten して全て連結 ----
    if (index_ < 0) {
        for (const auto& t : tvec) {
            torch::Tensor flat = t.flatten().to(torch::kCPU);
            auto dtype = flat.dtype();

            ANET_ASSERT(dtype == torch::kFloat32 ||
                dtype == torch::kInt64 ||
                dtype == torch::kBool);

            const int64_t n = flat.size(0);
            const size_t old = out.size();
            out.resize(old + n);

            if (dtype == torch::kFloat32) {
                std::memcpy(out.data() + old, flat.data_ptr<float>(), n * sizeof(float));
            } else if (dtype == torch::kInt64) {
                const int64_t* src = flat.data_ptr<int64_t>();
                float* dst = out.data() + old;
                for (int64_t i = 0; i < n; i++)
                    dst[i] = static_cast<float>(src[i]);
            } else if (dtype == torch::kBool) {
                const bool* src = flat.data_ptr<bool>();
                float* dst = out.data() + old;
                for (int64_t i = 0; i < n; i++)
                    dst[i] = src[i] ? 1.0f : 0.0f;
            }
        }

        return out;
    }

    ANET_PROFILE_SCOPE_NEXT(extract, dump);
    out.reserve(1024); // optional
    for (const auto& t : tvec) {
        ANET_LOG_DEBUG("t=" << anet::ToDefString(t));

        // t must be 2-D (ReplayBuffer)
        ANET_ASSERT_SHAPE(t, { ANET_SHAPE_ANY, ANET_SHAPE_ANY });
        ANET_ASSERT(t.dim() == 2);   // [rows, D]

        const int64_t rows = t.size(0);
        const int64_t dim = t.size(1);

        ANET_ASSERT(index_ < dim);

        // 1列だけ抽出
        torch::Tensor col = t.index({
            torch::indexing::Slice(),
            index_
            }).to(torch::kCPU);  // [rows]
        if (!col.is_contiguous()) {
            col = col.contiguous();
        }

        auto dtype = col.dtype();
        ANET_ASSERT(dtype == torch::kFloat32 || dtype == torch::kInt64 || dtype == torch::kBool);
        
        const int64_t n = col.size(0);
        const size_t old = out.size();
        out.resize(old + n);

        float* dst = out.data() + old;

        if (dtype == torch::kFloat32) {
            std::memcpy(dst, col.data_ptr<float>(), n * sizeof(float));
            if (auto_scale_mode_ == AutoScaleMode::PER_STEP) {
                min_ = col.min().item<float>(); 
                max_ = col.max().item<float>();
            } else if (auto_scale_mode_ == AutoScaleMode::GLOBAL) {
                const float min = col.min().item<float>();
                const float max = col.max().item<float>();
                min_ = std::min(min_.value_or(min), min);
                max_ = std::max(max_.value_or(max), max);
            }
        } else if (dtype == torch::kInt64) {
            const int64_t* src = col.data_ptr<int64_t>();
            for (int64_t i = 0; i < n; i++)
                dst[i] = static_cast<float>(src[i]);
        } else { // bool
            const bool* src = col.data_ptr<bool>();
            for (int64_t i = 0; i < n; i++)
                dst[i] = src[i] ? 1.0f : 0.0f;
        }
    }

    return out;
}

// ============================================================
// StateSweepProcessor
// ============================================================

StateSweepProcessor::StateSweepProcessor(
    const anet::rl::StateSpec& state_spec,
    int x_index,
    int y_index,
    ValueExtractFunction value_extract_fn,
    const torch::Device& device,
    std::optional<torch::Tensor> base_state,
    std::optional<float> x_min_override, std::optional<float> x_max_override,
    std::optional<float> y_min_override, std::optional<float> y_max_override)
    : state_spec_(state_spec)
    , device_(device)
    , x_index_(x_index)
    , y_index_(y_index)
    , value_extract_fn_(value_extract_fn)
{
    /// @todo v2暫定: Vector次元をベースとする
    int64_t flat_size = 1;
    if (state_spec_.obs_spec.count(anet::rl::ObsKeys::kVector) > 0) {
        flat_size = state_spec_.obs_spec.at(anet::rl::ObsKeys::kVector).CalcFlattenDim();
    }

    if (base_state.has_value()) {
        base_flatten_ = base_state.value().clone();
    } else {
        base_flatten_ = torch::zeros({ flat_size }, torch::kFloat32);
    }

    /// @todo v2暫定:min/max：StateSpec から
    float xs_min = -1.0f;
    float xs_max = 1.0f;
    float ys_min = -1.0f;
    float ys_max = 1.0f;

    if (state_spec_.obs_spec.count(anet::rl::ObsKeys::kVector) > 0) {
        const auto& tspec = state_spec_.obs_spec.at(anet::rl::ObsKeys::kVector);
        xs_min = tspec.GetMin(x_index_).value_or(-1.0f);
        xs_max = tspec.GetMax(x_index_).value_or(1.0f);
        ys_min = tspec.GetMin(y_index_).value_or(-1.0f);
        ys_max = tspec.GetMax(y_index_).value_or(1.0f);
    }

    x_min_overridden_ = x_min_override.has_value();
    x_max_overridden_ = x_max_override.has_value();
    y_min_overridden_ = y_min_override.has_value();
    y_max_overridden_ = y_max_override.has_value();

    x_min_ = x_min_override.value_or(xs_min);
    x_max_ = x_max_override.value_or(xs_max);
    y_min_ = y_min_override.value_or(ys_min);
    y_max_ = y_max_override.value_or(ys_max);
}

// ===========================================================
// ISweepInputGenerator
// ===========================================================

void StateSweepProcessor::ApplyGridSize(int width, int height)
{
    if (width > 0) grid_w_ = width;
    if (height> 0) grid_h_ = height;
}

std::pair<int, int> StateSweepProcessor::GetGridSize() const
{
    return { grid_w_, grid_h_ };
}

torch::Tensor StateSweepProcessor::BuildInputTensor()
{
    ANET_ASSERT(grid_w_ > 0);
    ANET_ASSERT(grid_h_ > 0);

    const int64_t flat_size = base_flatten_.size(0);
    const int64_t grid_num = static_cast<int64_t>(grid_w_) * static_cast<int64_t>(grid_h_);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    // base_flatten_ を device に移動し、全セル分に複製
    torch::Tensor base = base_flatten_.to(opts);
    torch::Tensor batch = base.unsqueeze(0).repeat({ grid_num, 1 });

    // X軸補間値作成
    torch::Tensor xs;
    if (grid_w_ > 1) {
        xs = torch::linspace(0.0f, 1.0f, grid_w_, opts);
    } else {
        xs = torch::zeros({ 1 }, opts);
    }
    torch::Tensor xv = x_min_ + xs * (x_max_ - x_min_);
    xv = xv.repeat({ grid_h_ });

    // Y軸補間値作成
    torch::Tensor ys;
    if (grid_h_ > 1) {
        //ys = torch::linspace(1.0f, 0.0f, grid_h_, opts);    // Y軸反転
        ys = torch::linspace(0.0f, 1.0f, grid_h_, opts);    // Y軸反転
    } else {
        ys = torch::zeros({ 1 }, opts);
    }
    torch::Tensor yv = y_min_ + ys * (y_max_ - y_min_);
    yv = yv.repeat_interleave(grid_w_);

    // 行インデックス [0 .. grid_num-1]
    torch::Tensor idx = torch::arange(
        grid_num, torch::TensorOptions().dtype(torch::kLong).device(device_));

    // 指定された state index へ X/Y 値を上書き
    batch.index_put_({ idx, static_cast<int64_t>(x_index_) }, xv);
    batch.index_put_({ idx, static_cast<int64_t>(y_index_) }, yv);

    // Shape 検証
    ANET_ASSERT_SHAPE(batch, { grid_num, flat_size });

    return batch;  // [W*H, flat_size] on device_
}

int64_t StateSweepProcessor::GetFlattenSize() const
{
    return base_flatten_.size(0);
}

ExtractResult StateSweepProcessor::Extract(const torch::Tensor& batched_out,
    const std::unordered_set<std::string>& required_labels)
{
    const int64_t grid_num = static_cast<int64_t>(grid_w_) * static_cast<int64_t>(grid_h_);
    ANET_ASSERT_SHAPE(batched_out, { grid_num, ANET_SHAPE_ENDANY });

    auto extract_result = value_extract_fn_(batched_out, required_labels);
    ANET_ASSERT_SHAPE(extract_result.grid, { grid_num });

    return extract_result;
}

//struct ExtractResult {
//    torch::Tensor grid;                          // HeatMap 用
//    std::vector<std::string> labels;             // それぞれの scalar 名
//    std::vector<torch::Tensor> scalars;          // 個別 scalar 値（GPU Tensor）
//};

// ==== Extractors

namespace anet::rl::extractor {

    // -------

    template <class Map, class Key>
    bool map_contains(const Map& m, const Key& k) {
        return m.find(k) != m.end();
    }

    void push_scalar(std::vector<std::string>& labels, std::vector<torch::Tensor>& scalars,
        const std::string& label, const torch::Tensor& scalar_tensor)
    {
        labels.push_back(label);
        scalars.push_back(scalar_tensor);
    }

    // -------

    ExtractResult MaxExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req)
    {
        return MaxIdxExtractor(t, req, -1);
    }

    ExtractResult MinExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req)
    {
        return MinIdxExtractor(t, req, -1);
    }

    ExtractResult MeanExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req)
    {
        return MeanIdxExtractor(t, req, -1);
    }

    ExtractResult ArgmaxExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req)
    {
        return ArgmaxIdxExtractor(t, req, -1);
    }

    ExtractResult StdExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req)
    {
        return StdIdxExtractor(t, req, -1);
    }

    static torch::Tensor ApplyIndex(torch::Tensor t,int idx)
    {
        // t: [W*H, 4, 51] (例)
        torch::Tensor target_t;

        if (idx < 0) {
            // 全次元を集計対象にする場合
            // [W*H, 4, 51] -> [W*H, 4*51] (dim=1以降を平坦化)
            target_t = t.flatten(1);
        } else {
            // 特定の次元(4の次元)を指定してスライスする場合
            // [W*H, 4, 51] -> [W*H, 51] (dim=1 の idx を選択)
            target_t = t.select(1, idx);
        }

        return target_t;
    }

    ExtractResult MaxIdxExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req, int idx)
    {
        // idxを適用して展開・選択
        auto target_t = ApplyIndex(t, idx);

        // 描画用Tensor
        torch::Tensor grid;

        // target_t がすでに [W*H] の1次元になっている場合、
        // これ以上の集計は不要なので、そのままグリッドとして使う
        if (target_t.dim() == 1) {
            grid = target_t;
        } else {
            // 特徴量の次元(dim=1)で集計して [W*H] にする
            grid = std::get<0>(target_t.max(1));
        }

        // grid全体の集計
        auto grid_val = grid.max();

        return { grid,
            { "max" },
            { grid_val }
        };
    }

    ExtractResult MinIdxExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req, int idx)
    {
        // idxを適用して展開・選択
        auto target_t = ApplyIndex(t, idx);

        // 描画用Tensor
        torch::Tensor grid;

        // target_t がすでに [W*H] の1次元になっている場合、
        // これ以上の集計は不要なので、そのままグリッドとして使う
        if (target_t.dim() == 1) {
            grid = target_t;
        } else {
            // 特徴量の次元(dim=1)で集計して [W*H] にする
            grid = std::get<0>(target_t.min(1));
        }

        // grid全体の集計
        auto grid_val = grid.min();

        return { grid,
            { "min" },
            { grid_val }
        };
    }

    ExtractResult MeanIdxExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req, int idx)
    {
        // idxを適用して展開・選択
        auto target_t = ApplyIndex(t, idx);

        // 描画用Tensor
        torch::Tensor grid;

        // target_t がすでに [W*H] の1次元になっている場合、
        // これ以上の集計は不要なので、そのままグリッドとして使う
        if (target_t.dim() == 1) {
            grid = target_t;
        } else {
            // 特徴量の次元(dim=1)で集計して [W*H] にする
            grid = target_t.mean(1);
        }

        // grid全体の集計
        auto grid_val = grid.mean();

        return { grid,
            { "mean" },
            { grid_val }
        };
    }

    ExtractResult StdIdxExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req, int idx)
    {
        // idxを適用して展開・選択
        auto target_t = ApplyIndex(t, idx);

        // 描画用Tensor
        torch::Tensor grid;

        // target_t がすでに [W*H] の1次元になっている場合、
        // これ以上の集計は不要なので、そのままグリッドとして使う
        if (target_t.dim() == 1) {
            // 要素が1つ(スカラー)の場合、標準偏差は定義上 0
            grid = torch::zeros_like(target_t);
        } else {
            // 特徴量の次元(dim=1)で集計して [W*H] にする
            grid = target_t.std(1);
        }

        // grid全体の集計
        auto grid_val = grid.std();

        return { grid,
            { "std" },
            { grid_val }
        };
    }

    ExtractResult ArgmaxIdxExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req, int idx)
    {
        // idxを適用して展開・選択
        auto target_t = ApplyIndex(t, idx);

        // 描画用Tensor
        torch::Tensor grid;

        // --- 修正箇所 ---
        if (target_t.dim() == 1) {
            // 1次元（スカラー列）の場合、"最大値のインデックス" は常に 0
            // 値そのものではなく、0 で埋め尽くされたテンソルを作成して返す
            grid = torch::zeros_like(target_t);
        } else {
            // 2次元以上なら通常通り argmax を計算
            grid = target_t.argmax(1).to(torch::kFloat);
        }

        auto grid_val = grid.max(); // インデックスの最大値 (例: 3.0)

        return { grid,
            { "argmax" },
            { grid_val }
        };
    }

    ExtractResult DiffIndexExtractor(
        const torch::Tensor& t, const std::unordered_set<std::string>& req, int plus_idx, int minus_idx) {
        using namespace torch::indexing;
        auto plus_val = t.index({ Slice(), plus_idx });
        auto minus_va = t.index({ Slice(), minus_idx });
        return { plus_val - minus_va };
    }
    ExtractResult PairDiffExtractor(
        const torch::Tensor& t, const std::unordered_set<std::string>& req, int n_actions) {
        using namespace torch::indexing;
        ANET_ASSERT_SHAPE(t, { ANET_SHAPE_ANY, n_actions * 2 });
        auto q_online = t.index({ Slice(), Slice(0, n_actions) });               // [N, n_actions]
        auto q_target = t.index({ Slice(), Slice(n_actions, n_actions * 2) });   // [N, n_actions]
        auto diff = (q_online - q_target).abs().mean(1);                         // [N]
        return { diff };
    }
    ExtractResult QdeltaQmaxCombined(
        const torch::Tensor& t, const std::unordered_set<std::string>& req,
        int n_actions,
        float qdelta_scale,      // ex: 0.5f
        float qmax_threshold)    // ex: 20.0f
    {
        using namespace torch::indexing;

        auto q_online = t.index({ Slice(), Slice(0, n_actions) });
        auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

        // Qdelta = |online - target| の平均
        auto qdelta = (q_online - q_target).abs().mean(1);       // [N]

        // Qmax = |online| の最大値
        auto qmax = std::get<0>(q_online.abs().max(1));       // [N]

        // 正規化（0〜1）
        auto qdelta_norm = (qdelta / qdelta_scale).clamp(0.0f, 1.0f);
        auto qmax_norm = (qmax / qmax_threshold).clamp(0.0f, 1.0f);

        // 合成
        auto combined = qdelta_norm * qmax_norm;
        return { combined };
    }
    // =============================
    // QdeltaQmaxCombinedAuto
    // -----------------
    //    Qdelta 高 × Qmax 高
    //    → 発散で地形が壊れて target 追従不能の領域
    //    Qdelta 高 × Qmax 低
    //    → target追従不足（でも発散ではない）
    //    Qdelta 低 × Qmax 高
    //    → Qの発散だけが起きている領域（target が遅れて青くなる前兆）
    //    両方低
    //    → 安定
    // -----------------
    //    発散領域（本当に最悪の赤）
    //    → 真っ赤に浮かび上がる
    //    （Qdelta_norm ≈ 1＆Qmax_norm ≈ 1）
    //    Qmax が高いのに Qdelta はまだ小さい（発散初期段階）
    //    → 暗オレンジ色に現れる
    //    → 崩壊の前兆が見える
    //    赤いけれど発散ではない Qdelta の赤
    //    → 黄色〜緑程度で止まる
    //    Qdelta が高いが Qmax はまだ青い（target追従だけ遅れ）
    //    → 緑〜黄に現れる
    // =============================
    ExtractResult QdeltaQmaxCombinedAuto(
        const torch::Tensor& t, const std::unordered_set<std::string>& req,
        int n_actions)
    {
        using namespace torch::indexing;

        auto q_online = t.index({ Slice(), Slice(0, n_actions) });
        auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

        auto qdelta = (q_online - q_target).abs().mean(1);
        auto qmax = std::get<0>(q_online.abs().max(1));

        // GPU 上の max（scalar-tensor）
        auto qdelta_max = qdelta.max();  // shape [], device same as qdelta
        auto qmax_max = qmax.max();    // shape [], device same as qmax

        // EPS を GPU Tensor で作る
        auto eps = torch::full({}, 1e-6, qdelta.options());   // shape=[]

        // GPU同士の除算 → 完全GPU処理
        auto qdelta_norm = qdelta / (qdelta_max + eps);
        auto qmax_norm = qmax / (qmax_max + eps);

        return { qdelta_norm * qmax_norm };
    }
    /// QDELTA × |Qdiff|
    ExtractResult QdeltaQdiffCombinedAuto(
        const torch::Tensor& t, const std::unordered_set<std::string>& req,
        int n_actions,
        int action_index_a,
        int action_index_b)
    {
        using namespace torch::indexing;

        ANET_ASSERT(t.dim() == 2);
        ANET_ASSERT(n_actions > 0);
        ANET_ASSERT(t.size(1) == n_actions * 2);
        ANET_ASSERT(action_index_a >= 0 && action_index_a < n_actions);
        ANET_ASSERT(action_index_b >= 0 && action_index_b < n_actions);

        // t: [N, 2 * n_actions] = [q_online, q_target]
        auto q_online = t.index({ Slice(), Slice(0, n_actions) });
        auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

        // Qdelta(s) = mean_a |Q_online(s,a) - Q_target(s,a)|
        auto qdelta = (q_online - q_target).abs().mean(1);  // [N]

        // Qdiff(s) = Q_online(s, b) - Q_online(s, a)
        auto qdiff = q_online.index({ Slice(), action_index_b })
            - q_online.index({ Slice(), action_index_a }); // [N]
        auto qdiff_abs = qdiff.abs();                      // [N]

        // GPU 上での max（0 次元 Tensor）
        auto qdelta_max = qdelta.max();    // []
        auto qdiff_max = qdiff_abs.max();  // []

        // EPS を GPU Tensor で生成
        auto eps = torch::full({}, 1e-6f, t.options());

        // 自動正規化（0〜1）
        auto qdelta_norm = qdelta / (qdelta_max + eps);
        auto qdiff_norm = qdiff_abs / (qdiff_max + eps);

        // 合成: target 乖離 × 境界ゆらぎ
        auto combined = qdelta_norm * qdiff_norm;          // [N]

        return { combined };
    }
    ExtractResult BoundaryMaskFromQdiffAuto(
        const torch::Tensor& t, const std::unordered_set<std::string>& req,
        int n_actions,
        int action_index_a,
        int action_index_b)
    {
        using namespace torch::indexing;

        ANET_ASSERT(t.dim() == 2);
        ANET_ASSERT(n_actions > 0);
        ANET_ASSERT(t.size(1) == n_actions);
        ANET_ASSERT(action_index_a >= 0 && action_index_a < n_actions);
        ANET_ASSERT(action_index_b >= 0 && action_index_b < n_actions);

        // t: [N, 2 * n_actions] = [q_online, q_target]
        auto q_online = t.index({ Slice(), Slice(0, n_actions) });

        // Qdiff(s) = Q_online(s, b) - Q_online(s, a)
        auto qdiff = q_online.index({ Slice(), action_index_b }) -
            q_online.index({ Slice(), action_index_a });  // [N]
        auto qdiff_abs = qdiff.abs();                             // [N]

        // GPU 上で max を取得（0 次元 Tensor）
        auto qdiff_max = qdiff_abs.max();                         // []

        // EPS を GPU Tensor として生成
        auto eps = torch::full({}, 1e-6f, t.options());

        // 正規化: 0〜1 （境界からの距離）
        auto qdiff_norm = qdiff_abs / (qdiff_max + eps);          // [N], 0〜1

        // 境界強度: 境界付近ほど 1.0、遠いほど 0.0
        auto boundary_strength = 1.0f - qdiff_norm;               // [N]

        return { boundary_strength };
    }

    ExtractResult BoundaryMaskedQdeltaAuto(
        const torch::Tensor& t, const std::unordered_set<std::string>& req,
        int n_actions,
        int action_index_a,
        int action_index_b)
    {
        using namespace torch::indexing;

        ANET_ASSERT(t.dim() == 2);
        ANET_ASSERT(n_actions > 0);
        ANET_ASSERT(t.size(1) == n_actions * 2);
        ANET_ASSERT(action_index_a >= 0 && action_index_a < n_actions);
        ANET_ASSERT(action_index_b >= 0 && action_index_b < n_actions);

        // t: [N, 2 * n_actions] = [q_online, q_target]
        auto q_online = t.index({ Slice(), Slice(0, n_actions) });
        auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

        // Qdelta(s) = mean_a |Q_online - Q_target|
        auto qdelta = (q_online - q_target).abs().mean(1);   // [N]

        // Qdiff(s) = Q_online(s,b) - Q_online(s,a)
        auto qdiff = q_online.index({ Slice(), action_index_b })
            - q_online.index({ Slice(), action_index_a });    // [N]
        auto qdiff_abs = qdiff.abs();

        // 正規化用 max
        auto qdelta_max = qdelta.max();
        auto qdiff_max = qdiff_abs.max();

        auto eps = torch::full({}, 1e-6f, t.options());

        // 0〜1 に normalize
        auto qdelta_norm = qdelta / (qdelta_max + eps);
        auto qdiff_norm = qdiff_abs / (qdiff_max + eps);

        // 境界強度（1が境界付近）
        auto boundary_strength = 1.0f - qdiff_norm;          // [N]

        // HeatMap 用（正規化された combined）
        auto combined = boundary_strength * qdelta_norm;     // [N]

        std::vector<std::string> labels;
        std::vector<torch::Tensor> scalars;

        if (map_contains(req, "raw_qdelta_mean")) {
            push_scalar(labels, scalars, "raw_qdelta_mean", qdelta.mean());
        }
        if (map_contains(req, "raw_qdelta_max")) {
            push_scalar(labels, scalars, "raw_qdelta_max", qdelta.max());
        }
        if (map_contains(req, "raw_boundary_mean")) {
            push_scalar(labels, scalars, "raw_boundary_mean", boundary_strength.mean());
        }
        if (map_contains(req, "boundary_area")) {
            push_scalar(labels, scalars, "boundary_area", (boundary_strength > 0.5f).sum());
        }
        if (map_contains(req, "combined_mean")) {
            push_scalar(labels, scalars, "combined_mean", combined.mean());
        }
        if (map_contains(req, "combined_max")) {
            push_scalar(labels, scalars, "combined_max", combined.max());
        }

        return { combined, labels, scalars };
    }

}
