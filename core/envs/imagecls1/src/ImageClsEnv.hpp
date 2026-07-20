#pragma once

#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "anet/config.hpp"
#include "anet/env.hpp"
#include "ImageData.hpp"

namespace anet::rl::env {

struct ImageClsEnvConfig : public anet::Config {
    int max_steps = 100;

    ImageClsEnvConfig(
        const anet::ConfigData& config_data = anet::EmptyConfigData,
        const std::string& config_prefix = "")
        : anet::Config(config_data, "ImageClsEnv", config_prefix)
    {
        ANET_READ_CONFIG(config_data, max_steps);
        if (max_steps <= 0) {
            ANET_SYSTEM_ERROR("Invalid ImageClsEnv.max_steps=" << max_steps
                << ". Expected a positive value.");
        }
    }
};

class ImageClsEnv final : public anet::rl::BatchEnvBase {
public:
    ImageClsEnv(
        const ImageClsEnvConfig& config,
        std::unique_ptr<anet::img::ImageDataSource> source,
        anet::ConfigData config_data,
        const std::string& name,
        int num_envs,
        anet::rl::RunMode run_mode);

    anet::rl::EnvSpec GetSpec() const override { return spec_; }
    anet::rl::BatchEnvSpec GetBatchSpec() const override { return batch_spec_; }
    torch::Device GetDevice() const override { return torch::Device(torch::kCPU); }
    std::shared_ptr<const anet::rl::BatchResetResult> Reset() override;
    std::shared_ptr<const anet::rl::BatchStepResult> Step(
        std::shared_ptr<anet::rl::BatchActionInfo> action_info) override;
    void Shutdown() override;

    std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
    std::optional<std::vector<torch::Tensor>> GetTensorVector(
        const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(
        const std::string&, int64_t = -1) const override { return std::nullopt; }

private:
    anet::rl::BatchState MakeState(
        const anet::img::ImageBatch& batch,
        const torch::Tensor& done,
        const torch::Tensor& truncated,
        const torch::Tensor& episode_start) const;
    std::vector<anet::rl::AuxData> MakeAuxDataList(const anet::img::ImageBatch& batch) const;
    void UpdateMetrics(const torch::Tensor& reward);

    ImageClsEnvConfig config_;
    anet::rl::EnvSpec spec_;
    anet::rl::BatchEnvSpec batch_spec_;
    std::unique_ptr<anet::img::ImageDataSource> source_;
    anet::img::ImageBatch current_batch_;
    int step_count_ = 0;

    double accuracy_correct_ = 0.0;
    int64_t accuracy_samples_ = 0;
    float accuracy_snapshot_ = std::numeric_limits<float>::quiet_NaN();
    uint64_t active_epoch_tag_ = 0;
    bool has_active_epoch_tag_ = false;
    uint64_t epoch_count_ = 0;
};

class ImageClsEnvFactory final : public anet::rl::BatchEnvFactory {
public:
    std::string GetTargetEnvClassId() const override { return "ImageClsEnv"; }

    void ValidateConfig(
        const anet::ConfigData& config_data,
        anet::rl::RunMode run_mode,
        const std::string& config_prefix) const override;

    std::shared_ptr<anet::rl::BatchEnv> CreateBatchEnv(
        const anet::ConfigData& config_data,
        const torch::Device& device,
        const std::string& name,
        std::optional<anet::seed_t> seed = std::nullopt,
        int num_envs = 1,
        anet::rl::RunMode run_mode = anet::rl::RunMode::Train,
        const std::string& config_prefix = "") override;
};

} // namespace anet::rl::env
