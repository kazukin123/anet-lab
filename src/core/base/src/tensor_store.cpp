#include "anet/tensor_store.hpp"
#include <limits>
#include <unordered_map>
#include <vector>
#include "anet/diag.hpp"

using namespace anet;

LaneTimeTensorStore::LaneTimeTensorStore(int64_t capacity) : capacity_(capacity)
{
    if (capacity <= 0)
        ANET_SYSTEM_ERROR("Invalid tensor store capacity=" << capacity << "; expected integer > 0.");
}

void LaneTimeTensorStore::Append(const TensorDict& lanes)
{
    if (time_count_ >= capacity_)
        ANET_SYSTEM_ERROR("Tensor store time=" << time_count_ << " exceeds capacity=" << capacity_ << ".");
    if (lanes.empty())
        ANET_SYSTEM_ERROR("Invalid tensor store input; expected at least one tensor key.");

    const bool first_append = storage_.empty();
    const auto& first = lanes.begin()->second;
    if (!first.defined() || first.dim() == 0 || first.size(0) == 0)
        ANET_SYSTEM_ERROR("Invalid tensor store input key=" << lanes.begin()->first << "; expected nonempty lane axis.");
    const auto num_lanes = first_append ? first.size(0) : lane_count_;
    if (num_lanes > std::numeric_limits<int64_t>::max() / capacity_)
        ANET_SYSTEM_ERROR("Tensor store lanes=" << num_lanes << " and capacity=" << capacity_
            << "; expected their product to fit in int64_t.");
    if (!first_append && lanes.size() != storage_.size())
        ANET_SYSTEM_ERROR("Invalid tensor store key count=" << lanes.size() << "; expected " << storage_.size() << ".");

    // 入力全体を調べてから書く。失敗した intake は読み出せる範囲に入らない。
    for (const auto& [key, value] : lanes) {
        if (!value.defined() || value.dim() == 0 || value.size(0) != num_lanes
            || value.layout() != torch::kStrided)
            ANET_SYSTEM_ERROR("Invalid tensor store input key=" << key
                << "; expected strided tensor with " << num_lanes << " lanes.");
        if (first_append) continue;
        const auto stored = storage_.Get(key);
        if (!stored || stored->sizes().slice(1) != value.sizes()
            || stored->scalar_type() != value.scalar_type() || stored->device() != value.device())
            ANET_SYSTEM_ERROR("Invalid tensor store input key=" << key << " shape=" << value.sizes()
                << " dtype=" << value.scalar_type() << " device=" << value.device()
                << "; expected the key, shape, dtype and device from the first append.");
    }

    torch::NoGradGuard no_grad;
    if (first_append) {
        TensorDict allocated;
        for (const auto& [key, value] : lanes) {
            std::vector<int64_t> shape{capacity_};
            shape.insert(shape.end(), value.sizes().begin(), value.sizes().end());
            allocated.Set(key, torch::empty(shape, value.options()));
        }
        storage_ = std::move(allocated);
        lane_count_ = num_lanes;
    }
    for (const auto& [key, value] : lanes)
        storage_.At(key).select(0, time_count_).copy_(value);
    ++time_count_;
}

TensorDict LaneTimeTensorStore::Gather(std::span<const LaneTimeIndex> indices) const
{
    // たまっている範囲を確認し、時間軸と lane 軸をまとめた番号を作る。
    std::vector<int64_t> flat_indices;
    flat_indices.reserve(indices.size());
    for (const auto& index : indices) {
        if (index.time < 0 || index.time >= time_count_ || index.lane < 0 || index.lane >= lane_count_)
            ANET_SYSTEM_ERROR("Invalid tensor store index time=" << index.time << " lane=" << index.lane
                << "; expected time in [0, " << time_count_ << ") and lane in [0, " << lane_count_ << ").");
        flat_indices.push_back(index.time * lane_count_ + index.lane);
    }
    const auto selection = torch::tensor(flat_indices, torch::kLong);
    // CUDA へは固定メモリから device ごとに一度だけ番号を送る。
    torch::Tensor pinned_selection;
    std::unordered_map<torch::Device, torch::Tensor> device_selections;
    TensorDict result;
    for (const auto& [key, value] : storage_) {
        const auto device = value.device();
        // どのキーも同じ番号順で選び、元の device 上に結果を得る。
        if (device.is_cpu()) {
            result.Set(key, value.flatten(0, 1).index_select(0, selection));
            continue;
        }
        const auto [pos, inserted] = device_selections.try_emplace(device);
        if (inserted) {
            if (device.is_cuda() && !pinned_selection.defined())
                pinned_selection = selection.pin_memory();
            pos->second = device.is_cuda() ? pinned_selection.to(device, true) : selection.to(device);
        }
        result.Set(key, value.flatten(0, 1).index_select(0, pos->second));
    }
    return result;
}
