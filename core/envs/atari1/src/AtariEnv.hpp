#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "anet/config.hpp"
#include "anet/env.hpp"
#include "anet/random.hpp"

namespace ale {
    class ALEInterface;
}

namespace anet::rl::env {

    struct AtariEnvConfig : public anet::Config {
        std::string game;
        std::string rom_dir;
        int screen_size = 84;
        int frame_skip = 4;
        bool max_pool = true;
        float repeat_action_probability = 0.25f;
        int noop_max = 0;
        bool fire_reset = false;
        bool episodic_life = false;
        bool reward_clip = true;
        bool full_action_space = false;
        int mode = -1;
        int difficulty = -1;
        int max_episode_frames = 108000;
        bool retain_rgb_frame = true;
        bool display_screen = false;
        bool sound = false;

        explicit AtariEnvConfig(
            const anet::ConfigData& config_data = anet::EmptyConfigData,
            const std::string& config_prefix = "");
    };

    class AtariEnv final : public SingleDiscreteEnvBase, public anet::RandomHolder {
    public:
        AtariEnv(
            const AtariEnvConfig& config,
            const torch::Device& device,
            const std::string& name,
            std::optional<anet::seed_t> seed = std::nullopt,
            RunMode run_mode = RunMode::Train);
        ~AtariEnv() override;

        EnvSpec GetSpec() const override;
        std::shared_ptr<const SingleResetResult> Reset() override;
        std::shared_ptr<const SingleStepResult> Step(int64_t action) override;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(
            const std::string& key, int64_t index = -1) const override;

    private:
        void ValidateConfig() const;
        std::string ResolveRomPath() const;
        void ConfigureAle(const std::string& rom_path);
        void SelectActions();
        void ApplyResetActions();
        torch::Tensor CaptureObservation();
        void CaptureRgbFrame();
        AuxData MakeAuxData() const;
        SingleState MakeState(torch::Tensor grid, bool done, bool truncated, bool episode_start) const;

        AtariEnvConfig config_;
        torch::Device device_;
        std::unique_ptr<ale::ALEInterface> ale_;
        std::vector<int> action_set_;
        std::vector<std::string> action_labels_;

        float episode_score_ = 0.0f;
        int64_t episode_len_ = 0;
        int current_lives_ = 0;
        bool life_loss_pending_ = false;

        bool completion_available_ = false;
        float completed_episode_score_ = 0.0f;
        int64_t completed_episode_len_ = 0;
        int64_t completed_episode_frames_ = 0;

        std::vector<uint8_t> pooled_frame_;
        torch::Tensor rgb_frame_;
    };

    class AtariEnvFactory final : public SingleDiscreteEnvFactory {
    public:
        std::shared_ptr<SingleDiscreteEnv> CreateSingleEnv(
            const anet::ConfigData& config_data,
            const torch::Device& device,
            const std::string& name,
            std::optional<anet::seed_t> seed = std::nullopt,
            RunMode run_mode = RunMode::Train,
            const std::string& config_prefix = "") override;

        std::string GetTargetEnvClassId() const override { return "AtariEnv"; }
        std::string GetDefaultConfigPrefix() const override { return "AtariEnv"; }
    };

} // namespace anet::rl::env
