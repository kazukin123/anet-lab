#pragma once

#include <cstdint>
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

    torch::Tensor ResizeGrayscale(
        const uint8_t* source,
        int source_height,
        int source_width,
        int destination_size);

} // namespace anet::rl::env::atari
