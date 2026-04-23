// ImageClsEnv.hpp
#pragma once

#include <vector>
#include <memory>
#include <string>
#include <torch/torch.h>
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/rl.hpp"
#include "ImageData.hpp"

namespace anet::rl::env {

    struct ImageClsEnvConfig : public anet::Config {
        std::string root_dir = "";
        std::string train_list_txt_path = "";
        std::string eval_list_txt_path = "";
        std::string classes_txt_path = "";
        int image_width = -1;
        int image_height = -1;
        std::string suffix = ".jpg";

        int max_steps = 100; // 1エピソードあたりの分類回数（画像枚数）

        ImageClsEnvConfig(const anet::ConfigData& config_data = anet::EmptyConfigData, const std::string& config_prefix = "")
            : anet::Config(config_data, "ImageClsEnv", config_prefix)
        {
            ANET_READ_CONFIG(config_data, root_dir);
            ANET_READ_CONFIG(config_data, train_list_txt_path);
            ANET_READ_CONFIG(config_data, eval_list_txt_path);
            ANET_READ_CONFIG(config_data, classes_txt_path);
			ANET_READ_CONFIG(config_data, image_width);
			ANET_READ_CONFIG(config_data, image_height);
            ANET_READ_CONFIG(config_data, suffix);
            ANET_READ_CONFIG(config_data, max_steps);
        }
    };

    class ImageClsEnv final : public anet::rl::SingleDiscreteEnv, public anet::RandomHolder {
    public:
        ImageClsEnv(const ImageClsEnvConfig& config, std::optional<anet::seed_t> seed);

        // --- SingleDiscreteEnv インターフェースの実装 ---
        anet::rl::EnvSpec GetSpec() const override;
        std::shared_ptr<const anet::rl::SingleResetResult> Reset(anet::rl::RunMode mode) override;
        std::shared_ptr<const anet::rl::SingleStepResult> Step(int64_t action, anet::rl::RunMode mode) override;

        // --- Module (Metrics) インターフェース ---
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const { return std::nullopt; }
    private:
        anet::rl::SingleState FetchRandomImageState(anet::rl::RunMode mode);
        anet::rl::AuxData MakeAuxData() const;

    private:
        ImageClsEnvConfig config_;
        anet::rl::EnvSpec spec_;

        std::unique_ptr<anet::img::ImageDataSource> train_data_source_;
        std::unique_ptr<anet::img::ImageDataSource> eval_data_source_;

        int step_count_;
        float ep_reward_sum_;
        bool done_, truncated_, episode_start_;
        bool episode_just_ended_ = false;

        int64_t current_true_label_;

        // メトリクス用
        float last_episode_len_ = 0.0f;
        float last_reward_sum_ = 0.0f;
        float last_accuracy_ = 0.0f;
    };

    class ImageClsEnvFactory final : public anet::rl::SingleDiscreteEnvFactory {
    public:
        ImageClsEnvFactory() = default;

        std::string GetTargetEnvClassId() const override { return "ImageClsEnv"; }

        std::shared_ptr<anet::rl::SingleDiscreteEnv> CreateSingleEnv(
            const anet::ConfigData& config_data,
            const torch::Device& device,
            std::optional<anet::seed_t> seed = std::nullopt,
            const std::string& config_prefix = "") override;
    };

} // namespace anet::rl::env
