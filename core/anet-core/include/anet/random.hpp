#pragma once
#include <random>
#include <cstdint>
#include <torch/torch.h>

namespace anet {

    using seed_t = uint64_t;

    class RandomGenerator {
    public:
    public:
        RandomGenerator(std::optional<seed_t> seed = std::nullopt);

        seed_t AutoSeed();
        void SetSeed(seed_t seed);
        seed_t GetSeed() const { return seed_; }

        seed_t RandUint64();
        size_t RandIndex(size_t max);
        float Uniform01();
        float Uniform(float low, float high);
        int RandInt(int low, int high);


        /// デフォルトインスタンスを取得
        static std::shared_ptr<RandomGenerator> Default();

        RandomGenerator(const RandomGenerator&) = delete;
        RandomGenerator& operator=(const RandomGenerator&) = delete;
    private:
        seed_t seed_ = 0;
        std::mt19937_64 engine_;
        mutable std::mutex mutex_;
    };

    class MasterSeedManager {
    public:
        explicit MasterSeedManager(std::optional<seed_t> master_seed = std::nullopt);

        seed_t GetMasterSeed() const;
        seed_t GetGroupSeed(const char* group_name, std::optional<seed_t> override = std::nullopt) const;
    private:
        void ApplyTorchSeed();
    private:
        seed_t master_seed_;
    };

    class SeedMaker {
    public:
        static seed_t MakeAutoSeed();
    public:
        explicit SeedMaker(std::optional<seed_t> base_seed_ = std::nullopt);

        seed_t GetGroupSeed() const;

        seed_t MakeNamedSeed(const char* name, std::optional<seed_t> override = std::nullopt) const;
        seed_t MakeIndexedSeed(size_t index) const;
    private:
        seed_t base_seed_;
    };

    class RandomHolder {
    public:
        explicit RandomHolder(std::optional<seed_t> seed = std::nullopt);
        RandomHolder(std::shared_ptr<RandomGenerator> rnd) : rnd_(rnd) { }

        void SetSeed(seed_t seed) { rnd_->SetSeed(seed); }
        seed_t GetSeed() const { return rnd_->GetSeed(); }

        std::shared_ptr<RandomGenerator> GetRandomGenerator() { return rnd_; }
        //void SetRandomGenerator(std::shared_ptr<RandomGenerator> rnd) { rnd_ = rnd; }
    protected:
        std::shared_ptr<RandomGenerator> rnd_;
    };

}   // namespace anet
