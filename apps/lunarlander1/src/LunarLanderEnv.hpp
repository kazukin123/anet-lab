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

/// LunarLander 環境の設定
struct LunarLanderEnvConfig : public anet::Config {
    int limit_step = 1000;
    float world_half_width = 10.5f;
    float world_height = 5.0f;
    float ground_y = 0.0f;
    float gravity_y = -10.0f;

    bool enable_wind = true;
    float wind_power = 15.0f;       ///< Gym の WIND_POWER 相当
    float turbulence_power = 1.5f;  ///< Gym の TURBULENCE_POWER 相当

    int terrain_point_count = 21;   ///< 地形 polyline の頂点数（少なくとも 2）
    float terrain_noise_height = 2.0f; ///< 地形高さノイズの上限
  
    /// @todo terrain_point_count / terrain_noise_height の適切な値を検討する。

    LunarLanderEnvConfig(
        const anet::ConfigData& config_data = anet::EmptyConfigData)
        : anet::Config(config_data, "LunarLanderEnv")
    {
        ANET_READ_CONFIG(config_data, limit_step);
        ANET_READ_CONFIG(config_data, world_half_width);
        ANET_READ_CONFIG(config_data, world_height);
        ANET_READ_CONFIG(config_data, ground_y);
        ANET_READ_CONFIG(config_data, gravity_y);
        ANET_READ_CONFIG(config_data, enable_wind);
        ANET_READ_CONFIG(config_data, wind_power);
        ANET_READ_CONFIG(config_data, turbulence_power);
        ANET_READ_CONFIG(config_data, terrain_point_count);
        ANET_READ_CONFIG(config_data, terrain_noise_height);

        ANET_ASSERT(terrain_point_count >= 2);
    }
};

/// LunarLander 単一環境クラス（離散アクション）
///
/// Box2D を用いて 2D 物理を簡易再現する。
/// 現時点では下記が @todo:
/// - Gym と同等の地形ランダム生成
/// - Gym と同等の報酬設計・終端条件
class LunarLanderEnv
    : public anet::rl::SingleDiscreteEnv
    , public anet::RandomHolder
{
public:
    LunarLanderEnv(
        const LunarLanderEnvConfig& config,
        const torch::Device& device,
        const std::optional<anet::seed_t> seed = std::nullopt);

    ~LunarLanderEnv() override;

    anet::rl::EnvSpec GetSpec() const override;
    anet::rl::SingleState Reset(
        anet::rl::RunMode mode = anet::rl::RunMode::Train) override;
    anet::rl::SingleStepResult Step(
        int64_t action,
        anet::rl::RunMode mode = anet::rl::RunMode::Train) override;

    // === UI / AP 向けの補助 Getter ===

    /// 地形を構成する地面の頂点列（world 座標系、Box2D 単位）
    const std::vector<b2Vec2>& GetTerrainPolyline() const { return terrain_points_; }

    /// 着陸パッドの水平区間と高さ
    struct PadInfo {
        float x1 = 0.0f;
        float x2 = 0.0f;
        float y = 0.0f;
    };

    PadInfo GetPadInfo() const { return pad_info_; }

    /// World の描画・スケーリング向け境界
    void GetWorldBounds(
        float& min_x, float& max_x,
        float& min_y, float& max_y) const
    {
        min_x = -config_.world_half_width;
        max_x = config_.world_half_width;
        min_y = config_.ground_y;
        max_y = config_.world_height;
    }

    /// 現在の風（直近 step で適用した wind_x）
    float GetLastWindX() const { return last_wind_x_; }

public:
    std::optional<float> GetScalar(const std::string& key, int index = -1) const override;
    std::optional<torch::Tensor> GetTensor(const std::string& key, int index = -1) const override;
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int index = -1) const override;
private:
    void buildWorld();
    void destroyWorld();

    void buildGround();
    void buildLander();

    void applyWind();
    void applyActionForce(int64_t action);

    anet::rl::SingleState makeState() const;
    float calcReward(
        const anet::rl::SingleState& state,
        bool done,
        bool crashed,
        bool landed) const;
    bool checkCrash() const;
    bool checkLanded() const;

private:
    LunarLanderEnvConfig config_;
    torch::TensorOptions obs_opt_;

    std::unique_ptr<b2World> world_;
    b2Body* ground_body_ = nullptr;
    b2Body* lander_body_ = nullptr;
    b2Body* left_leg_body_ = nullptr;
    b2Body* right_leg_body_ = nullptr;
    b2RevoluteJoint* left_leg_joint_ = nullptr;
    b2RevoluteJoint* right_leg_joint_ = nullptr;

    std::vector<b2Vec2> terrain_points_;
    PadInfo pad_info_;

    bool left_leg_contact_ = false;
    bool right_leg_contact_ = false;

    int step_count_ = 0;
    float last_wind_x_ = 0.0f;

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

    std::unique_ptr<anet::rl::SingleDiscreteEnv> CreateSingleEnv(
        const anet::ConfigData& config_data,
        const torch::Device& device,
        std::optional<anet::seed_t> seed = std::nullopt) override;
};

