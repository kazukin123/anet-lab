#include "anet/random.hpp"
#include <chrono>
#include <limits>

namespace anet {

    RandomGenerator::RandomGenerator()    {
        AutoSeed();
    }

    RandomGenerator::RandomGenerator(uint64_t seed, bool withTorch, bool withCuda) {
        SetSeed(seed, withTorch, withCuda);
    }

    uint64_t RandomGenerator::AutoSeed() {
        SetSeed(MakeRandomSeed());
        return seed_;
    }

    void RandomGenerator::SetSeed(uint64_t seed, bool withTorch, bool withCuda)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        seed_ = seed;
        engine_.seed(seed_);

        // Torch CPU RNG
        if (withTorch)
            torch::manual_seed(seed_);

        // Torch CUDA RNG
        if (withCuda && torch::cuda::is_available()) {
            torch::cuda::manual_seed(seed_);
            torch::cuda::manual_seed_all(seed_);
        }
    }

    uint64_t RandomGenerator::MakeRandomSeed()
    {
        // 高精度クロック利用
        auto now = std::chrono::high_resolution_clock::now();
        auto cnt = now.time_since_epoch().count();
        return static_cast<uint64_t>(cnt);
    }

    size_t RandomGenerator::RandIndex(size_t max)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::uniform_int_distribution<size_t> dist(0, max - 1);
        return dist(engine_);
    }

    float RandomGenerator::Uniform01()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // RandUint64() を 0-1 に正規化 (double精度)
        constexpr double inv = 1.0 / (double)std::numeric_limits<uint64_t>::max();
        return static_cast<float>(engine_() * inv);
    }

    float RandomGenerator::Uniform(float low, float high)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::uniform_real_distribution<float> dist(low, high);
        return dist(engine_);
    }

    int RandomGenerator::RandInt(int low, int high)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::uniform_int_distribution<int> dist(low, high);
        return dist(engine_);
    }

    // =========================================================
    // デフォルト RNG（シングルトン的動作）
    // =========================================================

    RandomGenerator& RandomGenerator::Default()
    {
        static RandomGenerator inst;
        return inst;
    }

} // namespace anet
