#include "anet/random.hpp"
#include <limits>
#include <chrono>
#include <optional>
#include<torch/torch.h>

namespace anet {

    /// 初期値ホワイトニング用（seed整形用）再現性あり・高速・分布品質良好
    inline uint64_t splitmix64(uint64_t x)
    {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    inline uint64_t hash64(uint64_t x)
    {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    inline uint64_t hash64(const char* s)
    {
        uint64_t h = 0x9ae16a3b2f90404fULL;
        while (*s) {
            h ^= uint64_t(uint8_t(*s++));
            h *= 0xbf58476d1ce4e5b9ULL;
            h ^= h >> 31;
        }
        return hash64(h); // 最後に安定ミックス
    }

    inline seed_t MakeAutoSeed()
    {
        std::random_device rd;
        uint64_t raw = (uint64_t(rd()) << 32) ^ uint64_t(rd());
        return splitmix64(raw);
    }

    // =========================================================

    RandomGenerator::RandomGenerator(std::optional<seed_t> seed)
    {
        if (seed.has_value()) {
            SetSeed(*seed);
        } else {
            AutoSeed();
        }
    }

    seed_t RandomGenerator::AutoSeed()
    {
        SetSeed(MakeAutoSeed());
        return seed_;
    }

    void RandomGenerator::SetSeed(seed_t seed)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        seed_ = seed;
        engine_.seed(seed_);
    }

    seed_t RandomGenerator::RandUint64()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return engine_();
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

    std::shared_ptr<RandomGenerator> RandomGenerator::Default()
    {
        static std::shared_ptr<RandomGenerator> inst = std::make_shared<RandomGenerator>();
        return inst;
    }

    // =========================================================

    MasterSeedManager::MasterSeedManager(std::optional<seed_t> master_seed)
    {
        if (master_seed.has_value()) {
            master_seed_ = *master_seed;
        } else {
            master_seed_ = MakeAutoSeed();
        }
        //RandomGenerator::Default()->SetSeed(master_seed_);
        ApplyTorchSeed();
    }

    /// torch/cudaはグローバルなseedしか出来ないのでここで反映
    void MasterSeedManager::ApplyTorchSeed()
    {
        // Torch CPU seed
        torch::manual_seed(master_seed_);

        // Torch CUDA seed
        if (torch::cuda::is_available()) {
            torch::cuda::manual_seed(master_seed_);
            torch::cuda::manual_seed_all(master_seed_);
        }
    }

    seed_t MasterSeedManager::GetMasterSeed() const
    {
        return master_seed_;
    }

    seed_t MasterSeedManager::GetGroupSeed(const char* group_name, std::optional<seed_t> override) const
    {
        if (override.has_value()) {
            return *override;
        }

        return splitmix64(hash64(group_name) ^ master_seed_);
    }

    // =========================================================

    seed_t SeedMaker::MakeAutoSeed()
    {
        return anet::MakeAutoSeed();
    }

    SeedMaker::SeedMaker(std::optional<seed_t> group_seed)
        : base_seed_(group_seed.value_or(MakeAutoSeed()))
    {
    }

    seed_t SeedMaker::GetGroupSeed() const
    {
        return base_seed_;
    }

    seed_t SeedMaker::MakeNamedSeed(const char* name, std::optional<seed_t> override) const
    {
        return splitmix64(hash64(name) ^ base_seed_);
    }

    seed_t SeedMaker::MakeIndexedSeed(size_t index) const
    {
        return splitmix64(base_seed_ + index);
    }


    // =========================================================

    RandomHolder::RandomHolder(std::optional<seed_t> seed)
        : rnd_(seed.has_value() ?
            std::make_shared<RandomGenerator>(*seed) :
            std::make_shared<RandomGenerator>()
        )
    {
    }

} // namespace anet
