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

    /**
     * @brief 人間正規化スコアの基準表。
     *
     * 2 系統が併存しており値が実質的に異なる（例: Pong の human は 14.6 対 9.3）。
     * どちらの表で正規化したかを取り違えると文献比較が狂うため、キーを分けて両方出す。
     */
    enum class HnsBaseline {
        Dqn57,      ///< Wang et al. 2016 系 57 ゲーム表。DeepMind dqn_zoo/atari_data.py と同一値。Rainbow / IQN / Agent57 / BBF が使う現代標準
        Nature49,   ///< Mnih et al. 2015 Extended Data Table 2 の 49 ゲーム表
    };

    /**
     * @brief 人間正規化スコア HNS を % 表記（100 = 人間）で返す。
     *
     * `100 * (raw_score - random) / (human - random)`。分母は絶対値化しない
     * （dqn_zoo / Dopamine / IQN / Agent57 と同一式）。
     *
     * @param game      ALE の rom() 表記（snake_case）
     * @param raw_score reward clip 前の生スコア
     * @return 基準表に game が無ければ std::nullopt
     */
    std::optional<float> HumanNormalizedScore(
        const std::string& game, float raw_score, HnsBaseline baseline);

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
        float ApplyFireReset();     ///< fire_reset有効時にFIREを1回打ち、その報酬を返す
        torch::Tensor CaptureObservation();
        void CaptureRgbFrame();
        AuxData MakeAuxData() const;
        SingleState MakeState(torch::Tensor grid, bool done, bool truncated, bool episode_start) const;

        AtariEnvConfig config_;
        torch::Device device_;
        std::unique_ptr<ale::ALEInterface> ale_;
        std::vector<int> action_set_;
        std::vector<std::string> action_labels_;

        float game_score_ = 0.0f;
        int64_t game_len_ = 0;
        int current_lives_ = 0;
        bool life_loss_pending_ = false;

        bool completion_available_ = false;
        float completed_game_score_ = 0.0f;
        int64_t completed_game_len_ = 0;
        int64_t completed_game_frames_ = 0;

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
