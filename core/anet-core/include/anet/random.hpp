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

        // [0, max] の離散乱数
        size_t RandIndex(size_t max);

        // -----------------------------------------
        // 拡張：0〜1 一様乱数
        // -----------------------------------------
        float Uniform01();

        // -----------------------------------------
        // 拡張：任意範囲 [low, high] の実数乱数
        // -----------------------------------------
        float Uniform(float low, float high);

        // -----------------------------------------
        // 拡張：任意範囲 [low, high] の整数乱数
        // -----------------------------------------
        int RandInt(int low, int high);


        /// デフォルトインスタンスを取得
        static RandomGenerator& Default();

        RandomGenerator(const RandomGenerator&) = delete;
        RandomGenerator& operator=(const RandomGenerator&) = delete;
    private:
        uint64_t MakeRandomSeed();
    private:
        uint64_t seed_ = 0;
        std::mt19937_64 engine_;
    };

    class RandomHolder {
    public:
        RandomHolder() {}
        RandomHolder(RandomGenerator* rnd) : rnd_(rnd) {}

        void SetRandomGenerator(RandomGenerator* rng) {
            rnd_ = (rng ? rng : &RandomGenerator::Default());
        }
    protected:
        RandomGenerator* rnd_ = &RandomGenerator::Default();
    };

}   // namespace anet
