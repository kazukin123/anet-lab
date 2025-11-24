#pragma once
#include <random>
#include <cstdint>
#include <torch/torch.h>

namespace anet {

    class RandomGenerator {
    public:
        RandomGenerator(bool withTorch = false, bool withCuda = false);
        RandomGenerator(uint64_t seed, bool withTorch = true, bool withCuda = true);

        uint64_t AutoSeed(bool withTorch = false, bool withCuda = false);
        void SetSeed(uint64_t seed, bool withTorch = false, bool withCuda = false);
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
        static std::shared_ptr<RandomGenerator> Default();

        RandomGenerator(const RandomGenerator&) = delete;
        RandomGenerator& operator=(const RandomGenerator&) = delete;
    private:
        uint64_t MakeRandomSeed();
    private:
        uint64_t seed_ = 0;
        std::mt19937_64 engine_;
        mutable std::mutex mutex_;
    };

    class RandomHolder {
    public:
        //RandomHolder(std::shared_ptr<RandomGenerator> rnd) : rnd_(rnd) {}
        RandomHolder(std::shared_ptr<RandomGenerator> rnd = nullptr) : rnd_(rnd == nullptr ? RandomGenerator::Default() : rnd) {}

        std::shared_ptr<RandomGenerator> GetRandomGenerator() {
            return rnd_;
        }

        void SetRandomGenerator(std::shared_ptr<RandomGenerator> rnd) {
            rnd_ = rnd;
        }
    protected:
        std::shared_ptr<RandomGenerator> rnd_;
    };

}   // namespace anet
