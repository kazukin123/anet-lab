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

    using ValueExtractFunction = std::function<float(const torch::Tensor&)>;


    RLStateSweepProcessor::RLStateSweepProcessor(
        const anet::rl::StateSpec& state_spec,
        int x_index,
        int y_index,
        ValueExtractFn value_extract_fn,
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
        int64_t flat_size = state_spec_.CalcFlattenSize();

        if (base_state.has_value()) {
            base_flatten_ = base_state.value().clone();
        } else {
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

    torch::Tensor RLStateSweepProcessor::BuildInputTensor()
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
            ys = torch::linspace(0.0f, 1.0f, grid_h_, opts);
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
        ANET_CHECK_SHAPE(batch, { grid_num, flat_size });

        return batch;  // [W*H, flat_size] on device_
    }

    int64_t RLStateSweepProcessor::GetFlattenSize() const
    {
        return base_flatten_.size(0);
    }

    torch::Tensor RLStateSweepProcessor::ExtractValue(const torch::Tensor& batched_out)
    {
        const int64_t grid_num = static_cast<int64_t>(grid_w_) * static_cast<int64_t>(grid_h_);
        ANET_CHECK_SHAPE(batched_out, { grid_num, ANET_SHAPE_ENDANY });

        torch::Tensor grid_values = value_extract_fn_(batched_out);
        ANET_CHECK_SHAPE(grid_values, { grid_num });

        return grid_values;
    }

}
