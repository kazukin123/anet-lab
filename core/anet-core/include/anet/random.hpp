#pragma once
#include <random>
#include <cstdint>
#include <torch/torch.h>

namespace anet {

    class RandomGenerator {
    public:
        RandomGenerator();
        uint64_t AutoSeed();

        void SetSeed(uint64_t seed);
        uint64_t GetSeed() const { return seed_; }

        uint64_t RandUint64() { return engine_(); }
        size_t RandIndex(size_t max) {
            std::uniform_int_distribution<size_t> dist(0, max);
            return dist(engine_);
        }

        // デフォルト RNG（シングルトン的に利用）
        static RandomGenerator& Default();

        // コピー禁止
        RandomGenerator(const RandomGenerator&) = delete;
        RandomGenerator& operator=(const RandomGenerator&) = delete;
    private:
        uint64_t MakeRandomSeed();
    private:
        uint64_t seed_ = 0;
        std::mt19937_64 engine_;
    };

}   // namespace anet
