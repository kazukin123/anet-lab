// ImageData.hpp
#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <torch/torch.h>
#include "anet/config.hpp"
#include "anet/thread.hpp"
#include "anet/rl.hpp"

namespace anet::img {

    using DatasetKey = std::string;

    enum class ImageCacheMode {
        Auto,
        None,
        FullRam,
    };

    struct ImageDatasetConfig {
        std::filesystem::path root_dir;
        std::filesystem::path list_txt_path;
        std::filesystem::path classes_txt_path;
        std::string suffix = ".jpg";
        int image_width = 0;
        int image_height = 0;
        ImageCacheMode cache_mode = ImageCacheMode::Auto;
        uint64_t cache_max_bytes = 4294967296ULL;

        static ImageDatasetConfig Resolve(const anet::ConfigData& config_data, const DatasetKey& key);
        bool Equals(const ImageDatasetConfig& other, std::string* first_different_field = nullptr) const;
        anet::ConfigData ToConfigData(const DatasetKey& key) const;
        std::string ToString() const;
    };

    using ResolvedImageDatasetCatalog = std::unordered_map<DatasetKey, ImageDatasetConfig>;

    ResolvedImageDatasetCatalog ResolveImageDatasetCatalog(const anet::ConfigData& config_data);

    struct ImageManifest {
        std::vector<std::filesystem::path> paths;
        std::vector<int64_t> targets;
        std::vector<std::string> class_names;
    };

    class ImageDataset {
    public:
        ImageDataset(DatasetKey key, ImageDatasetConfig config);

        const DatasetKey& GetKey() const { return key_; }
        const ImageDatasetConfig& GetConfig() const { return config_; }
        const ImageManifest& GetManifest() const { return manifest_; }
        size_t Size() const { return manifest_.paths.size(); }

        void PrepareCache();
        torch::data::Example<> Get(size_t index);

    private:
        enum class CachePrepareState { Unprepared, Preparing, Ready, Failed };
        enum class EntryState { Empty, Loading, Ready, Failed };

        torch::Tensor Decode(size_t index) const;
        uint64_t RequiredPayloadBytes() const;

        DatasetKey key_;
        ImageDatasetConfig config_;
        ImageManifest manifest_;

        mutable std::mutex mutex_;
        std::condition_variable condition_;
        CachePrepareState cache_prepare_state_ = CachePrepareState::Unprepared;
        ImageCacheMode effective_cache_mode_ = ImageCacheMode::None;
        std::exception_ptr cache_prepare_failure_;
        torch::Tensor cache_payload_;
        std::vector<EntryState> entry_states_;
        std::vector<std::exception_ptr> entry_failures_;
    };

    class ImageDatasetManager {
    public:
        static ImageDatasetManager& Instance();

        void RegisterCatalog(const ResolvedImageDatasetCatalog& catalog);
        std::shared_ptr<ImageDataset> Acquire(const DatasetKey& key);

    private:
        enum class State { Registered, Loading, Ready, Failed };
        struct Entry {
            ImageDatasetConfig config;
            State state = State::Registered;
            std::shared_ptr<ImageDataset> dataset;
            std::exception_ptr failure;
            std::condition_variable condition;
        };

        ImageDatasetManager() = default;

        std::mutex mutex_;
        std::unordered_map<DatasetKey, std::shared_ptr<Entry>> entries_;
    };

    struct TrainImageDataSourceConfig : public anet::Config {
        std::string dataset_key;
        struct {
            bool enabled = false;
            double hflip_p = 0.5;
            double rrc_scale_min = 0.7;
            double rrc_scale_max = 1.0;
            double rrc_ratio_min = 0.75;
            double rrc_ratio_max = 1.3333333;
        } augment;

        TrainImageDataSourceConfig(
            const anet::ConfigData& config_data = anet::EmptyConfigData,
            const std::string& config_prefix = "");
    };

    struct EvalImageDataSourceConfig : public anet::Config {
        std::string dataset_key;
        struct {
            std::string mode = "full";
            struct {
                int64_t size = 1;
            } rotating;
        } eval_window;

        EvalImageDataSourceConfig(
            const anet::ConfigData& config_data = anet::EmptyConfigData,
            const std::string& config_prefix = "");
    };

    struct ImageBatch {
        torch::Tensor grid;
        torch::Tensor targets;
        std::vector<uint64_t> epoch_tags;
        int64_t valid_count = 0;
        bool window_end = false;
        uint64_t completed_cycles = 0;
    };

    class ImageDataSource {
    public:
        ImageDataSource(
            TrainImageDataSourceConfig config,
            int batch_size,
            anet::seed_t seed,
            int worker_type,
            int worker_threads);
        ImageDataSource(
            EvalImageDataSourceConfig config,
            int batch_size,
            anet::seed_t seed,
            int worker_type,
            int worker_threads);
        ~ImageDataSource();

        ImageBatch NextBatch();
        void Shutdown();

        const DatasetKey& GetDatasetKey() const { return dataset_key_; }
        const std::shared_ptr<ImageDataset>& GetDataset() const { return dataset_; }
        int GetWorkerCount() const { return worker_count_; }

    private:
        struct SampleSelection {
            std::vector<size_t> indices;
            std::vector<uint64_t> epoch_tags;
            int64_t valid_count = 0;
            bool window_end = false;
            uint64_t completed_cycles = 0;
        };
    private:
        enum class Role { Train, Eval };

        ImageDataSource(
            Role role,
            std::string dataset_key,
            bool augment_enabled,
            double augment_hflip_p,
            double augment_rrc_scale_min,
            double augment_rrc_scale_max,
            double augment_rrc_ratio_min,
            double augment_rrc_ratio_max,
            std::string eval_window_mode,
            int64_t eval_window_rotating_size,
            int batch_size,
            anet::seed_t seed,
            int worker_type,
            int worker_threads);

        SampleSelection SelectTrainBatch();
        SampleSelection SelectEvalBatch();
        torch::Tensor ApplyAugment(const torch::Tensor& image, uint64_t epoch_tag, size_t dataset_index);
        torch::Tensor ApplyRandomResizedCrop(const torch::Tensor& image, std::mt19937_64& random);
    private:
        Role role_;
        std::string dataset_key_;
        bool augment_enabled_;
        double augment_hflip_p_;
        double augment_rrc_scale_min_;
        double augment_rrc_scale_max_;
        double augment_rrc_ratio_min_;
        double augment_rrc_ratio_max_;
        std::string eval_window_mode_;
        int64_t eval_window_rotating_size_;
        int batch_size_;
        anet::seed_t augment_seed_;
        std::shared_ptr<ImageDataset> dataset_;

        std::mt19937_64 sampler_random_;
        std::vector<size_t> permutation_;
        size_t cursor_ = 0;
        uint64_t cycle_ = 0;
        int64_t eval_window_progress_ = 0;
        int worker_type_;
        int worker_threads_;
        int worker_count_ = 1;
        std::shared_ptr<anet::ThreadPool> decode_pool_;
    };

} // namespace anet::img
