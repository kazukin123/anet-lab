// LunarLanderEnv.hpp
#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <box2d/box2d.h> 
#include <torch/torch.h>
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/rl.hpp"

namespace anet::rl::env {

    constexpr float kMainEngineForce = 40.0f;   ///< メインエンジンの推力
    constexpr float kSideEngineForce = 10.0f;   ///< サイドエンジンの推力
    constexpr float kSideEngineTorque = 5.0f;   ///< サイドエンジンの回転トルク
    constexpr float kLanderRadius = 0.25f;      ///< Lander本体（円形）の半径

    constexpr int kActionNoop = 0;
    constexpr int kActionLeft = 1;
    constexpr int kActionMain = 2;
    constexpr int kActionRight = 3;

    /// LunarLander 環境の設定
    struct LunarLanderEnvConfig : public anet::Config {
        int limit_step = 1000;
        float world_half_width = 1.5f;
        float world_height = 2.0f;
        float ground_y = 0.0f;
        float gravity = -10.0f;

        bool enable_wind = false;
        float wind_power = 15.0f;       ///< Gym の WIND_POWER 相当
        float turbulence_power = 1.5f;  ///< Gym の TURBULENCE_POWER 相当

        int terrain_point_count = 21;   ///< 地形 polyline の頂点数（少なくとも 2）
        float terrain_noise_height = 0.3f; ///< 地形高さノイズの上限

        /// @todo terrain_point_count / terrain_noise_height の適切な値を検討する。

        LunarLanderEnvConfig(
            const anet::ConfigData& config_data = anet::EmptyConfigData)
            : anet::Config(config_data, "LunarLanderEnv")
        {
            ANET_READ_CONFIG(config_data, limit_step);
            ANET_READ_CONFIG(config_data, world_half_width);
            ANET_READ_CONFIG(config_data, world_height);
            ANET_READ_CONFIG(config_data, ground_y);
            ANET_READ_CONFIG(config_data, gravity);
            ANET_READ_CONFIG(config_data, enable_wind);
            ANET_READ_CONFIG(config_data, wind_power);
            ANET_READ_CONFIG(config_data, turbulence_power);
            ANET_READ_CONFIG(config_data, terrain_point_count);
            ANET_READ_CONFIG(config_data, terrain_noise_height);

            ANET_ASSERT(terrain_point_count >= 2);
        }
    };

    /// LunarLander 単一環境クラス（離散アクション）
    /// Box2D を用いて 2D 物理を簡易再現する。
    class LunarLanderEnv : public anet::rl::SingleDiscreteEnv, public anet::RandomHolder, public std::enable_shared_from_this<LunarLanderEnv> {
    public:
        LunarLanderEnv(
            const LunarLanderEnvConfig& config,
            const torch::Device& device,
            const std::optional<anet::seed_t> seed = std::nullopt);

        ~LunarLanderEnv() override;

        anet::rl::EnvSpec GetSpec() const override;
        std::shared_ptr<const anet::rl::SingleResetResult> Reset(anet::rl::RunMode mode = anet::rl::RunMode::Train) override;
        std::shared_ptr<const anet::rl::SingleStepResult> Step(int64_t action, anet::rl::RunMode mode = anet::rl::RunMode::Train) override;

    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
    private:
        class Result;
        class ResetResult;
        class StepResult;
    private:
        void buildWorld();
        void destroyWorld();

        void buildGround();
        void buildLander();

        void applyWind();
        void applyActionForce(int64_t action);

        anet::rl::SingleState makeState() const;
        float computeShaping(const anet::rl::SingleState& state) const;
        std::pair<float, float> calcReward(const anet::rl::SingleState& state, bool crashed, bool landed, int64_t action);
        bool checkCrash() const;
        bool checkLanded() const;

        anet::rl::AuxData CreateAuxData(float reward, float raw_reward) const;
    private:
        // 着陸パッドの水平区間と高さ
        struct PadInfo {
            float x1 = 0.0f;
            float x2 = 0.0f;
            float y = 0.0f;
        };
    private:
        LunarLanderEnvConfig config_;
        torch::TensorOptions float_opt_;
        torch::TensorOptions bool_opt_;

        std::unique_ptr<b2World> world_;
        b2Body* ground_body_ = nullptr;
        b2Body* lander_body_ = nullptr;
        b2Body* left_leg_body_ = nullptr;
        b2Body* right_leg_body_ = nullptr;
        b2RevoluteJoint* left_leg_joint_ = nullptr;
        b2RevoluteJoint* right_leg_joint_ = nullptr;

        std::vector<b2Vec2> terrain_points_;
        PadInfo pad_info_;

        bool body_contact_ = false;
        bool left_leg_contact_ = false;
        bool right_leg_contact_ = false;

        int step_count_ = 0;
        float last_wind_x_ = 0.0f;
        float last_wind_torque_ = 0.0f;
        int64_t wind_idx_ = 0;
        int64_t torque_idx_ = 0;

        float last_shaping_ = 0.0f;
        bool has_prev_shaping_ = false;

        // ContactListener は脚の接地を検出するために使用
        class ContactListener : public b2ContactListener {
        public:
            explicit ContactListener(LunarLanderEnv& env) : env_(env) {}
            void BeginContact(b2Contact* contact) override;
            void EndContact(b2Contact* contact) override;
        private:
            LunarLanderEnv& env_;
        };

        std::unique_ptr<ContactListener> contact_listener_;
    };

    class LunarLanderEnvFactory : public anet::rl::SingleDiscreteEnvFactory {
    public:
        LunarLanderEnvFactory() = default;

        std::string GetTargetEnvClassId() const override { return "LunarLanderEnv"; }

        std::shared_ptr<anet::rl::SingleDiscreteEnv> CreateSingleEnv(
            const anet::ConfigData& config_data,
            const torch::Device& device,
            std::optional<anet::seed_t> seed = std::nullopt) override;
    };
}
