#include "AtariPreprocess.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include <torch/nn/functional/upsampling.h>

#include "anet/diag.hpp"
#include "anet/profile.hpp"

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

GrayscaleResizer::GrayscaleResizer(
    int source_height, int source_width, int destination_size, const std::string& mode)
    : source_height_(source_height)
    , source_width_(source_width)
    , destination_size_(destination_size)
    , pixel_area_(mode == "pixel_area")
{
    // 設定に依存する検証と重みの準備は構築時に完了させる。
    if (source_height <= 0 || source_width <= 0 || destination_size <= 0) {
        ANET_SYSTEM_ERROR("GrayscaleResizer requires positive dimensions. source_height="
            << source_height << ", source_width=" << source_width
            << ", destination_size=" << destination_size << ". Expected dimensions > 0.");
    }
    if (mode != "adaptive_average" && mode != "pixel_area") {
        ANET_SYSTEM_ERROR("GrayscaleResizer mode='" << mode
            << "'. Expected adaptive_average or pixel_area.");
    }
    if (pixel_area_) {
        if (destination_size > std::min(source_height, source_width)) {
            ANET_SYSTEM_ERROR("GrayscaleResizer pixel_area destination_size=" << destination_size
                << ". Expected <= " << std::min(source_height, source_width) << ".");
        }
        horizontal_weights_ = MakeWeights(source_width, destination_size);
        vertical_weights_ = MakeWeights(source_height, destination_size);
        // 積算上限を寸法で決め、Atariの40,800/8,568,000は32bit、より大きい画像は64bitで保つ。
        const int64_t area = static_cast<int64_t>(source_height) * source_width;
        const size_t count = static_cast<size_t>(source_height) * destination_size;
        if (area <= std::numeric_limits<int32_t>::max() / 255) horizontal_.resize(count);
        else wide_horizontal_.resize(count);
    }
}

GrayscaleResizer::AxisWeights GrayscaleResizer::MakeWeights(int source_size, int destination_size)
{
    // 小さいカーネルは4要素へ揃える。大きい縮小率は必要数を同じ平坦配列へ格納する。
    const int maximum = static_cast<int>(std::min<int64_t>(source_size,
        (static_cast<int64_t>(source_size) + destination_size - 1) / destination_size + 1));
    AxisWeights result{.stride = maximum <= 4 ? 4 : maximum};
    // paddingは画素0・重み0に初期化され、範囲外の入力を読まない。
    result.pixels.resize(static_cast<size_t>(destination_size) * result.stride);
    for (int destination = 0; destination < destination_size; ++destination) {
        const int64_t begin = static_cast<int64_t>(destination) * source_size;
        const int64_t end = static_cast<int64_t>(destination + 1) * source_size;
        const int first = static_cast<int>(begin / destination_size);
        const int last = static_cast<int>((end + destination_size - 1) / destination_size);
        for (int input = first; input < last; ++input) {
            const int64_t overlap = std::min(end, static_cast<int64_t>(input + 1) * destination_size)
                - std::max(begin, static_cast<int64_t>(input) * destination_size);
            result.pixels[static_cast<size_t>(destination) * result.stride + input - first] = {
                .index = input, .overlap = static_cast<int>(overlap)};
        }
    }
    return result;
}

template<typename Sum, typename Pixel>
Sum GrayscaleResizer::WeightedValue(const Pixel* source, const PixelWeight* weights, int count, int stride)
{
    // Atariの通常サイズでは分岐付きの内側ループを外す。0重みを含めても整数和は同じ。
    if (count == 4) {
        return static_cast<Sum>(source[weights[0].index * stride]) * weights[0].overlap
            + static_cast<Sum>(source[weights[1].index * stride]) * weights[1].overlap
            + static_cast<Sum>(source[weights[2].index * stride]) * weights[2].overlap
            + static_cast<Sum>(source[weights[3].index * stride]) * weights[3].overlap;
    }
    Sum value = 0;
    for (int i = 0; i < count; ++i) {
        value += static_cast<Sum>(source[weights[i].index * stride]) * weights[i].overlap;
    }
    return value;
}

template<typename Sum>
torch::Tensor GrayscaleResizer::ResizePixelArea(const uint8_t* source, std::vector<Sum>& horizontal) const
{
    ANET_PROFILE_SCOPE(horizontal);
    // 作業領域は構築時に確保済み。すべての要素を上書きし、前回の画像を残さない。
    for (int y = 0; y < source_height_; ++y) {
        for (int x = 0; x < destination_size_; ++x) {
            const auto* weights = horizontal_weights_.pixels.data() + static_cast<size_t>(x) * horizontal_weights_.stride;
            horizontal[y * destination_size_ + x] = WeightedValue<Sum>(
                source + y * source_width_, weights, horizontal_weights_.stride, 1);
        }
    }
    ANET_PROFILE_SCOPE_NEXT(vertical);
    auto output = torch::empty({1, destination_size_, destination_size_},
        torch::TensorOptions().dtype(torch::kUInt8));
    auto* pixels = output.data_ptr<uint8_t>();
    // 分母と商/余りの規則を保ち、正規化と偶数丸めは最後に一度だけ行う。
    const Sum divisor = static_cast<Sum>(source_height_) * source_width_;
    for (int y = 0; y < destination_size_; ++y) {
        const auto* weights = vertical_weights_.pixels.data() + static_cast<size_t>(y) * vertical_weights_.stride;
        for (int x = 0; x < destination_size_; ++x) {
            const Sum value = WeightedValue<Sum>(horizontal.data() + x, weights,
                vertical_weights_.stride, destination_size_);
            const Sum integer = value / divisor;
            const Sum twice_remainder = 2 * (value % divisor);
            const Sum rounded = integer + (twice_remainder > divisor
                || (twice_remainder == divisor && integer % 2 != 0));
            pixels[y * destination_size_ + x] = static_cast<uint8_t>(rounded);
        }
    }
    return output;
}

torch::Tensor GrayscaleResizer::Resize(const uint8_t* source) const
{
    ANET_PROFILE_SCOPE(resize);
    if (source == nullptr) {
        ANET_SYSTEM_ERROR("GrayscaleResizer source=null. Expected a non-null grayscale image.");
    }
    if (!pixel_area_) {
        // 現行方式のfloat32 area補間とuint8への戻し方をそのまま保つ。
        auto input = torch::from_blob(
            const_cast<uint8_t*>(source), {1, 1, source_height_, source_width_},
            torch::TensorOptions().dtype(torch::kUInt8));
        auto resized = torch::nn::functional::interpolate(
            input.to(torch::kFloat32),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{destination_size_, destination_size_})
                .mode(torch::kArea));
        return resized.squeeze(0).round().clamp(0, 255).to(torch::kUInt8).contiguous();
    }

    if (!wide_horizontal_.empty()) return ResizePixelArea(source, wide_horizontal_);
    return ResizePixelArea(source, horizontal_);
}
