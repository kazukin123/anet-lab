#pragma once
#include <random>
#include <cstdint>
#include <optional>
#include <mutex>
#include <unordered_map>
#include <torch/torch.h>

namespace anet {

    using seed_t = uint64_t;

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
        float Gamma(float alpha, float beta = 1.0f);    ///< ガンマ分布からのサンプリング (Dirichletノイズ用)

        inline torch::Generator GetTorchGenerator(torch::Device device)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            // deviceの正規化キー（32bit整数）を生成
            uint16_t type_val = static_cast<uint16_t>(device.type());
            int16_t idx = (device.has_index() && device.index() >= 0) ? device.index() : 0;
            uint32_t key = (static_cast<uint32_t>(type_val) << 16) | static_cast<uint16_t>(idx);

            // 整数キーによるハッシュ検索
            auto it = torch_gens_.find(key);
            if (it != torch_gens_.end()) {
                return it->second; // 見つかれば即座に返す
            }

            // 初回のみコールドパス（cpp側）に投げて生成させる
            return CreateTorchGenerator(device, key);
        }

        /// デフォルトインスタンスを取得
        static std::shared_ptr<RandomGenerator> Default();

        RandomGenerator(const RandomGenerator&) = delete;
        RandomGenerator& operator=(const RandomGenerator&) = delete;
    private:
        torch::Generator CreateTorchGenerator(torch::Device device, uint32_t key);
    private:
        mutable std::mutex mutex_;
        seed_t seed_ = 0;
        std::mt19937_64 engine_;
        std::unordered_map<uint32_t, torch::Generator> torch_gens_;
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
        explicit RandomHolder(std::optional<seed_t> seed);
        RandomHolder(std::shared_ptr<RandomGenerator> rnd) : rnd_(rnd) { }

        void SetSeed(seed_t seed) { rnd_->SetSeed(seed); }
        seed_t GetSeed() const { return rnd_->GetSeed(); }

        std::shared_ptr<RandomGenerator> GetRandomGenerator() { return rnd_; }
        //void SetRandomGenerator(std::shared_ptr<RandomGenerator> rnd) { rnd_ = rnd; }
    protected:
        std::shared_ptr<RandomGenerator> rnd_;
    };

}   // namespace anet
