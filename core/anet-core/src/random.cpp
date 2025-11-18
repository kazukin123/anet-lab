#include "anet/random.hpp"
#include <chrono>
#include <limits>

namespace anet {

    RandomGenerator::RandomGenerator()
    {
        seed_ = MakeRandomSeed();
        engine_.seed(seed_);
    }

    uint64_t RandomGenerator::AutoSeed()
    {
        seed_ = MakeRandomSeed();
        engine_.seed(seed_);
        return seed_;
    }

    void RandomGenerator::SetSeed(uint64_t seed)
    {
        seed_ = seed;
        engine_.seed(seed_);
    }

    uint64_t RandomGenerator::MakeRandomSeed()
    {
        // 高精度クロック利用
        auto now = std::chrono::high_resolution_clock::now();
        auto cnt = now.time_since_epoch().count();
        return static_cast<uint64_t>(cnt);
    }

    size_t RandomGenerator::RandIndex(size_t max) {
        std::uniform_int_distribution<size_t> dist(0, max);
        return dist(engine_);
    }

    float RandomGenerator::Uniform01()
    {
        // RandUint64() を 0-1 に正規化 (double精度)
        constexpr double inv = 1.0 / (double)std::numeric_limits<uint64_t>::max();
        return static_cast<float>(engine_() * inv);
    }

    float RandomGenerator::Uniform(float low, float high)
    {
        std::uniform_real_distribution<float> dist(low, high);
        return dist(engine_);
    }

    int RandomGenerator::RandInt(int low, int high)
    {
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
