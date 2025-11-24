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

} // namespace anet::viz
