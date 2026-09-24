#include "AtariPreprocess.hpp"

#include <algorithm>
#include <utility>

#include <torch/nn/functional/upsampling.h>

#include "anet/diag.hpp"

using namespace anet::rl::env::atari;

void RollingMaxPool::Push(std::vector<uint8_t> frame)
{
    if (!latest_.empty()) previous_ = std::move(latest_);
    latest_ = std::move(frame);
}

void RollingMaxPool::Finish(std::vector<uint8_t>& output) const
{
    if (latest_.empty()) {
        ANET_SYSTEM_ERROR("RollingMaxPool requires at least one frame.");
    }
    if (previous_.empty()) {
        output = latest_;
    } else {
        PixelwiseMax(previous_, latest_, output);
    }
}

void anet::rl::env::atari::PixelwiseMax(
    const std::vector<uint8_t>& first,
    const std::vector<uint8_t>& second,
    std::vector<uint8_t>& output)
{
    if (first.size() != second.size()) {
        ANET_SYSTEM_ERROR("PixelwiseMax requires equal input sizes. first=" << first.size()
            << ", second=" << second.size() << ".");
    }

    output.resize(first.size());
    std::transform(first.begin(), first.end(), second.begin(), output.begin(),
        [](uint8_t a, uint8_t b) { return std::max(a, b); });
}

torch::Tensor anet::rl::env::atari::InterleavedRgbToChw(
    const uint8_t* source,
    int source_height,
    int source_width)
{
    if (source == nullptr || source_height <= 0 || source_width <= 0) {
        ANET_SYSTEM_ERROR("InterleavedRgbToChw requires non-null input and positive dimensions. source_height="
            << source_height << ", source_width=" << source_width << ".");
    }

    // ALEの画素単位RGB配列をHWCとして解釈し、所有権を持つCHWへ1回でコピーする。
    const auto options = torch::TensorOptions().dtype(torch::kUInt8);
    const auto input = torch::from_blob(
        const_cast<uint8_t*>(source), { source_height, source_width, 3 }, options);
    auto output = torch::empty({ 3, source_height, source_width }, options);
    output.copy_(input.permute({ 2, 0, 1 }));
    return output;
}

torch::Tensor anet::rl::env::atari::ResizeGrayscale(
    const uint8_t* source,
    int source_height,
    int source_width,
    int destination_size)
{
    if (source == nullptr || source_height <= 0 || source_width <= 0 || destination_size <= 0) {
        ANET_SYSTEM_ERROR("ResizeGrayscale requires non-null input and positive dimensions. source_height="
            << source_height << ", source_width=" << source_width
            << ", destination_size=" << destination_size << ".");
    }

    // ALEの単一グレーフレームをarea補間し、観測契約のuint8へ戻す。
    auto input = torch::from_blob(
        const_cast<uint8_t*>(source), { 1, 1, source_height, source_width },
        torch::TensorOptions().dtype(torch::kUInt8));
    auto resized = torch::nn::functional::interpolate(
        input.to(torch::kFloat32),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{ destination_size, destination_size })
            .mode(torch::kArea));
    return resized.squeeze(0).round().clamp(0, 255).to(torch::kUInt8).contiguous();
}
