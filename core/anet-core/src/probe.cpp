// image_log.cpp （抜粋）

#include "anet/probe.hpp"

namespace anet {

    std::optional<float> MetricsScalarProbe::GetFloat(
        int step,
        const anet::rl::Experience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    {
        return result->GetScalar(key_);  ///< @todo GetScalar 実装と合わせる
    }

    std::optional<float> StaticScalarProbe::GetFloat(
        int,
        const anet::rl::Experience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    {
        return value_;
    }

    //void TensorInputProbe::SetTensor(const torch::Tensor& t)
    //{
    //    latest_tensor_ = t;
    //    cached_value_.reset();

    //    if (!latest_tensor_.defined()) return;

    //    auto flat = latest_tensor_.flatten();
    //    const auto numel = flat.numel();

    //    if (index_ < 0 || index_ >= numel) return;

    //    cached_value_ = flat[index_].item<float>();
    //}

    //std::optional<float> TensorInputProbe::GetFloat(
    //    int step,
    //    const anet::rl::Experience& experience,
    //    std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    //{
    //    // 現状は SetTensor 経由でのみ更新する設計
    //    /// @todo 必要に応じて BatchExperience/BUR から Tensor を取得する経路を追加
    //}

    std::optional<float> FunctionFloatProbe::GetFloat(
        int step,
        const anet::rl::Experience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    {
        return fn_(step, experience, std::move(result));
    }

     StateAxisProbe::StateAxisProbe(int state_index, const anet::rl::StateSpec* spec, bool for_next_state)
        : state_index_(state_index), for_next_state_(for_next_state)
    {
        // EnvSpec の state_spec から min/max を取得
        if (spec != nullptr && state_index >= 0 && state_index < (int)spec->CalcFlattenSize()) {
            // 指定されたindexの定義情報を取得
            const anet::rl::StateDimInfo* s = spec->FindDim(state_index);

            // 定義情報を取得出来たらmin/maxをセット
            if (s != nullptr) {
                min_ = s->min_value;
                max_ = s->max_value;
            }
        }
    }

    std::optional<float> StateAxisProbe::GetFloat(
        int step,
        const anet::rl::Experience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    {
        const auto& flat = experience.next_state.Flatten();
        ANET_CHECK_SHAPE(flat, { ANET_SHAPE_ANY });     // (N)
        int64_t numel = flat.numel();
        if (state_index_ < 0 || state_index_ >= numel) return std::nullopt;

        return flat[state_index_].item<float>();
    }

    RewardProbe::RewardProbe(const anet::rl::EnvSpec* spec)
    {
        if (spec != nullptr) {
            min_ = spec->reward_range.first;
            max_ = spec->reward_range.second;
        }
    }

    std::optional<float> RewardProbe::GetFloat(
        int step,
        const anet::rl::Experience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    {
        return experience.reward;
    }


    RLStateSweepProcessor::RLStateSweepProcessor(
        const anet::rl::EnvSpec& env_spec,
        int x_index,
        int y_index,
        std::function<float(const torch::Tensor&)> value_extractor,
        std::optional<torch::Tensor> base_state,
        std::optional<float> x_min_override,
        std::optional<float> x_max_override,
        std::optional<float> y_min_override,
        std::optional<float> y_max_override
        )
        : env_spec_(env_spec)
        , state_spec_(env_spec.state_spec)
        , x_index_(x_index)
        , y_index_(y_index)
        , value_extractor_(value_extractor)
    {
        int64_t flat_size = state_spec_.CalcFlattenSize();

        if (base_state.has_value()) {
            base_flatten_ = base_state.value().clone();
        }
        else {
            base_flatten_ = torch::zeros({ flat_size }, torch::kFloat32);
        }

        // min/max：StateSpec から
        const auto* dx = state_spec_.FindDim(x_index_);
        const auto* dy = state_spec_.FindDim(y_index_);

        float xs_min = dx ? dx->min_value : -1.0f;
        float xs_max = dx ? dx->max_value : 1.0f;
        float ys_min = dy ? dy->min_value : -1.0f;
        float ys_max = dy ? dy->max_value : 1.0f;

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

    void RLStateSweepProcessor::ApplyGridSize(int width, int height)
    {
        if (width > 0) grid_w_ = width;
        if (height> 0) grid_h_ = height;
    }

    std::pair<int, int> RLStateSweepProcessor::GetGridSize() const
    {
        return { grid_w_, grid_h_ };
    }

    torch::Tensor RLStateSweepProcessor::BuildInputTensor(int gx, int gy)
    {
        torch::Tensor input = base_flatten_.clone();

        float xf = (grid_w_ > 1) ? float(gx) / float(grid_w_ - 1) : 0.f;
        float yf = (grid_h_ > 1) ? float(gy) / float(grid_h_ - 1) : 0.f;

        float xv = x_min_ + xf * (x_max_ - x_min_);
        float yv = y_min_ + yf * (y_max_ - y_min_);

        input[x_index_] = xv;
        input[y_index_] = yv;

        return input;  // (state_dim)
    }

    int64_t RLStateSweepProcessor::GetFlattenSize() const
    {
        return base_flatten_.size(0);
    }

    // ===========================================================
    // ISweepOutputExtractor
    // ===========================================================

    float RLStateSweepProcessor::ExtractValue(const torch::Tensor& batched_out, int gx, int gy)
    {
        int idx = gy * grid_w_ + gx;
        auto sample = batched_out[idx];
        return value_extractor_(sample);
    }

}
