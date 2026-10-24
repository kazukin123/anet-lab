#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace anet::rl::env::atari {

    class RollingMaxPool final {
    public:
        void Push(std::vector<uint8_t> frame);
        void Finish(std::vector<uint8_t>& output) const;

    private:
        std::vector<uint8_t> previous_;
        std::vector<uint8_t> latest_;
    };

    void PixelwiseMax(
        const std::vector<uint8_t>& first,
        const std::vector<uint8_t>& second,
        std::vector<uint8_t>& output);

    torch::Tensor InterleavedRgbToChw(
        const uint8_t* source,
        int source_height,
        int source_width);

    class GrayscaleResizer final {
    public:
        GrayscaleResizer(int source_height, int source_width, int destination_size,
            const std::string& mode);
        /// pixel_areaは作業領域を再利用するため、同一縮小器のResizeを同時実行しない。
        torch::Tensor Resize(const uint8_t* source) const;

    private:
        struct PixelWeight {
            int index;
            int overlap;
        };
        struct AxisWeights {
            int stride = 0;
            std::vector<PixelWeight> pixels;
        };
        static AxisWeights MakeWeights(int source_size, int destination_size);
        template<typename Sum, typename Pixel>
        static Sum WeightedValue(const Pixel* source, const PixelWeight* weights, int count, int stride);
        template<typename Sum>
        torch::Tensor ResizePixelArea(const uint8_t* source, std::vector<Sum>& horizontal) const;

        int source_height_;
        int source_width_;
        int destination_size_;
        bool pixel_area_;
        AxisWeights horizontal_weights_;
        AxisWeights vertical_weights_;
        // pixel_areaの同じインスタンスを同時利用しない。入力寸法で片方だけを構築時に確保する。
        mutable std::vector<int32_t> horizontal_;
        mutable std::vector<int64_t> wide_horizontal_;
    };

} // namespace anet::rl::env::atari
