/*
 * LunarLander C++ Port
 * Based on work by OpenAI and Farama Foundation.
 * Licensed under the MIT License. See LICENSE file for details.
 */

#include "LunarLanderEnv.hpp"
#include <cmath>
#include <algorithm>
#include <box2d/box2d.h>
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/env.hpp"


using namespace anet::rl::env;
namespace LOG = anet::log;


// ボディの定義
constexpr float kLanderDensity = 5.0f;
constexpr float kLegDensity = 1.0f;

// 脚の定義
constexpr float kLegLength = 0.30f;
constexpr float kLegOffsetX = 0.30f;
constexpr float kPadHalfWidth = 0.60f;
//constexpr float kLegAttachAngle = 0.5236f; // 30°
constexpr float kLegAttachAngle = 0.7854f; // 45°
constexpr float kLegWidth = 0.1f;
constexpr float kLegRestAngle = 0.60f;         ///< 空中で自然に開く脚の目標角度
constexpr float kLegWeldFrequencyHz = 8.0f;    ///< 脚を目標角へ戻す角度バネの周波数。大きいほど固いバネ。
constexpr float kLegWeldDampingRatio = 1.0f;   ///< 脚の角度バネの減衰比。大きいほどビヨンビヨンしない。1.0で揺れずに目標角度に収束

// 報酬関連の定数
constexpr float kStepPenality = -0.01f;
constexpr float kCrashPenalty = -100.0f;
constexpr float kLandReward = 100.0f;


enum class GroundFixtureType : std::uintptr_t {
    Terrain = 0,
    Pad = 1
};

static bool IsPadFixture(const b2Fixture* fixture)
{
    return fixture->GetUserData().pointer ==
        static_cast<std::uintptr_t>(GroundFixtureType::Pad);
}

anet::rl::EnvSpec LunarLanderEnv::GetSpec() const
{
    // obs_spec
    anet::TensorSpec obs_spec {
        .type = anet::SpaceType::Vector,
        .shape = { config_.obs_include_action ? 8 + kActionCount : 8 },
        .dtype = torch::kFloat32,
        .num_classes = 0,   // 連続値
        .labels = { "x", "y", "vx", "vy", "angle", "v_angle", "leg_l", "leg_r" },
        .min_values = { -1.5, -0.5, -5.0, -5.0, -3.14, -5.0, 0.0, 0.0 },
        .max_values = {  1.5,  2.0,  5.0,  5.0,  3.14,  5.0, 1.0, 1.0 }
    };
    if (config_.obs_include_action) {
        obs_spec.labels.insert(obs_spec.labels.end(), { "a_noop", "a_left", "a_main", "a_right" });
        obs_spec.min_values.insert(obs_spec.min_values.end(), kActionCount, 0.0);
        obs_spec.max_values.insert(obs_spec.max_values.end(), kActionCount, 1.0);
    }

    // StateSpec
    anet::rl::StateSpec state_spec;
    state_spec.obs_spec[anet::rl::ObsKeys::kVector] = obs_spec;

    // ActionSpec
    anet::rl::ActionSpec action_spec {
        .is_discrete = true,
        .value_labels = { "do nothing", "fire left", "fire main", "fire right" },
        .dims = {
            { 0, 3, "engine" }  // [0] min, max, name
        }
    };

    // EnvSpec
    anet::rl::EnvSpec env_spec {
        .state_spec = state_spec,
        .action_spec = action_spec,
        .reward_range = { -100.0f, 100.0f }
    };

    return env_spec;
}

// ===== ContactListener =====

void LunarLanderEnv::ContactListener::BeginContact(b2Contact* contact)
{
    b2Fixture* fixture_a = contact->GetFixtureA();
    b2Fixture* fixture_b = contact->GetFixtureB();
    b2Body* body_a = fixture_a->GetBody();
    b2Body* body_b = fixture_b->GetBody();

    auto is_left_leg = [&](b2Fixture* f) {
        return f->GetBody() == env_.left_leg_body_;
        };

    auto is_right_leg = [&](b2Fixture* f) {
        return f->GetBody() == env_.right_leg_body_;
        };

    auto is_contact_target = [&](b2Fixture* f) {
        if (env_.config_.contact_mode == "ground") {
            return f->GetBody() == env_.ground_body_;
        }
        if (env_.config_.contact_mode == "pad") {
            return IsPadFixture(f);
        }
        ANET_SYSTEM_ERROR("Invalid LunarLanderEnv.contact_mode: "
            << env_.config_.contact_mode << ". Expected ground or pad.");
        return false;
        };

    // Lander本体
    if (body_a == env_.lander_body_ || body_b == env_.lander_body_) {
        env_.body_contact_ = true;
    }

    // 左脚 × contact target
    if ((is_left_leg(fixture_a) && is_contact_target(fixture_b)) ||
        (is_left_leg(fixture_b) && is_contact_target(fixture_a))) {
        env_.left_leg_contact_count_++;
        env_.left_leg_contact_ = env_.left_leg_contact_count_ > 0;
    }

    // 右脚 × contact target
    if ((is_right_leg(fixture_a) && is_contact_target(fixture_b)) ||
        (is_right_leg(fixture_b) && is_contact_target(fixture_a))) {
        env_.right_leg_contact_count_++;
        env_.right_leg_contact_ = env_.right_leg_contact_count_ > 0;
    }
}

void LunarLanderEnv::ContactListener::EndContact(b2Contact* contact)
{
    b2Fixture* fa = contact->GetFixtureA();
    b2Fixture* fb = contact->GetFixtureB();

    auto is_contact_target = [&](b2Fixture* f) {
        if (env_.config_.contact_mode == "ground") {
            return f->GetBody() == env_.ground_body_;
        }
        if (env_.config_.contact_mode == "pad") {
            return IsPadFixture(f);
        }
        ANET_SYSTEM_ERROR("Invalid LunarLanderEnv.contact_mode: "
            << env_.config_.contact_mode << ". Expected ground or pad.");
        return false;
        };

    if ((fa->GetBody() == env_.left_leg_body_ && is_contact_target(fb)) ||
        (fb->GetBody() == env_.left_leg_body_ && is_contact_target(fa))) {
        env_.left_leg_contact_count_ = std::max(0, env_.left_leg_contact_count_ - 1);
        env_.left_leg_contact_ = env_.left_leg_contact_count_ > 0;
    }

    if ((fa->GetBody() == env_.right_leg_body_ && is_contact_target(fb)) ||
        (fb->GetBody() == env_.right_leg_body_ && is_contact_target(fa))) {
        env_.right_leg_contact_count_ = std::max(0, env_.right_leg_contact_count_ - 1);
        env_.right_leg_contact_ = env_.right_leg_contact_count_ > 0;
    }
}

// ===== LunarLanderEnv::StepResult =====

class LunarLanderEnv::Result : virtual public anet::rl::SingleEnvResult {
public:
    Result(std::shared_ptr<const LunarLanderEnv> env, float reward, float raw_reward)
        : env_(env), reward_(reward), raw_reward_(raw_reward) { }
    anet::rl::AuxData GetAuxData() const override { return env_->CreateAuxData(reward_, raw_reward_); }

    virtual ~Result() = default;
protected:
    std::shared_ptr<const LunarLanderEnv> env_;
    float reward_;
    float raw_reward_;
};

class LunarLanderEnv::ResetResult : public anet::rl::SingleResetResult, public LunarLanderEnv::Result {
public:
    ResetResult(
        std::shared_ptr<const LunarLanderEnv> env, const anet::rl::SingleState state)
        : Result(env, 0.0f, 0.0f), SingleResetResult(std::move(state)) { }
};

class LunarLanderEnv::StepResult : public anet::rl::SingleStepResult, public LunarLanderEnv::Result {
public:
    StepResult(
        std::shared_ptr<const LunarLanderEnv> env, float reward, float raw_reward,
        anet::rl::SingleState next_state)
        : Result(env, reward, raw_reward), SingleStepResult(reward, std::move(next_state)) { }
};

// ===== LunarLanderEnv =====

LunarLanderEnv::LunarLanderEnv(
    const LunarLanderEnvConfig& config,
    const torch::Device& device,
    const std::string& name,
    const std::optional<anet::seed_t> seed)
    : SingleDiscreteEnvBase(name)
    , anet::RandomHolder(seed)
    , config_(config)
{
    ANET_LOG_DEBUG_PREFIXED("seed=" << this->GetSeed());

    if (auto logger = anet::MetricsLogger::Instance()) {
        logger->Log("LunarLanderEnv", config_.ToJson());
    }

    float_opt_ = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    bool_opt_ = torch::TensorOptions().dtype(torch::kBool).device(device);

    // 初期化（乱数での初期化は改めてResetで行われる想定）
    float default_start_y = config_.ground_y + config_.world_height * 0.6f;
    buildWorld(0.0f, default_start_y, 0.0f);
}

LunarLanderEnv::~LunarLanderEnv()
{
    destroyWorld();
}

void LunarLanderEnv::buildGround()
{
    terrain_points_.clear();

    const float half_w = config_.world_half_width;
    const float ground_y = config_.ground_y;

    // --- Pad 定義 ---
    pad_info_.x1 = -kPadHalfWidth;
    pad_info_.x2 = +kPadHalfWidth;
    pad_info_.y = ground_y;

    const float min_x = -half_w;
    const float max_x = +half_w;

    const int total_points = std::max(2, config_.terrain_point_count);
    const int left_points =
        std::max(2, static_cast<int>(total_points *
            (pad_info_.x1 - min_x) / (max_x - min_x)));
    const int right_points =
        std::max(2, total_points - left_points);

    // --- 左側地形 ---
    {
        const float dx =
            (pad_info_.x1 - min_x) / static_cast<float>(left_points - 1);

        for (int i = 0; i < left_points; ++i) {
            const float x = min_x + dx * static_cast<float>(i);
            const float noise =
                rnd_->Uniform(-config_.terrain_noise_height,
                    config_.terrain_noise_height);
            const float y = ground_y + noise;
            terrain_points_.emplace_back(x, y);
        }
    }

    // --- Pad（完全に水平） ---
    terrain_points_.emplace_back(pad_info_.x1, pad_info_.y);
    terrain_points_.emplace_back(pad_info_.x2, pad_info_.y);

    // --- 右側地形 ---
    {
        const float dx =
            (max_x - pad_info_.x2) / static_cast<float>(right_points - 1);

        for (int i = 1; i < right_points; ++i) {
            const float x = pad_info_.x2 + dx * static_cast<float>(i);
            const float noise =
                rnd_->Uniform(-config_.terrain_noise_height,
                    config_.terrain_noise_height);
            const float y = ground_y + noise;
            terrain_points_.emplace_back(x, y);
        }
    }

    // --- Box2D Body ---
    b2BodyDef ground_def;
    ground_def.position.Set(0.0f, 0.0f);
    ground_body_ = world_->CreateBody(&ground_def);

    // --- Terrain fixtures ---
    for (size_t i = 0; i + 1 < terrain_points_.size(); ++i) {
        const b2Vec2& p0 = terrain_points_[i];
        const b2Vec2& p1 = terrain_points_[i + 1];

        // Pad 上面は独立した pad fixture に任せる。垂直段差は terrain fixture として残す。
        const bool is_pad_top =
            p0.y == pad_info_.y &&
            p1.y == pad_info_.y &&
            p0.x >= pad_info_.x1 &&
            p1.x <= pad_info_.x2;
        if (is_pad_top) {
            continue;
        }

        b2EdgeShape edge;
        edge.SetTwoSided(p0, p1);

        b2FixtureDef fd;
        fd.shape = &edge;
        fd.density = 0.0f;
        fd.restitution = 0.1f;
        fd.friction = 0.5f;

        b2Fixture* f = ground_body_->CreateFixture(&fd);
        f->GetUserData().pointer =
            static_cast<std::uintptr_t>(GroundFixtureType::Terrain);
    }

    // --- Pad fixture（完全に独立） ---
    {
        const float pad_half_width = 0.5f * (pad_info_.x2 - pad_info_.x1);
        const float pad_thickness = 0.2f; // Pad厚み。 推奨 0.05～0.2

        b2PolygonShape pad_shape;
        pad_shape.SetAsBox(pad_half_width, pad_thickness * 0.5f,
            b2Vec2(0.5f * (pad_info_.x1 + pad_info_.x2),pad_info_.y - pad_thickness * 0.5f),
            0.0f);

        b2FixtureDef fd;
        fd.shape = &pad_shape;
        fd.density = 0.0f;
        fd.restitution = 0.0f;
        fd.friction = 2.5f;

        b2Fixture* f = ground_body_->CreateFixture(&fd);
        f->GetUserData().pointer =
            static_cast<std::uintptr_t>(GroundFixtureType::Pad);
    }
}

void LunarLanderEnv::buildLander(float init_x, float init_y, float init_angle)
{
    // 本体
    {
        b2BodyDef body_def;
        body_def.type = b2_dynamicBody;
        body_def.position.Set(init_x, init_y);
        body_def.angle = init_angle;
        body_def.linearDamping = 0.5f;
        body_def.angularDamping = 0.5f;
        lander_body_ = world_->CreateBody(&body_def);

        b2CircleShape shape;
        shape.m_radius = kLanderRadius;

        b2FixtureDef fixture_def;   /// @todo 衝突マスクで地面だけ衝突判定にする
        fixture_def.shape = &shape;
        fixture_def.density = kLanderDensity;
        fixture_def.friction = 0.3f;
        fixture_def.restitution = 0.0f;
        lander_body_->CreateFixture(&fixture_def);
    }

    // --------------------
    // 脚
    // --------------------

    b2Rot rotation(init_angle);

    // ===============================
    // 脚 共通 fixture 定義
    // ===============================

    // --- 脚本体（細長い box） ---
    b2PolygonShape leg_shape;
    leg_shape.SetAsBox(
        0.05f,                       // 半幅
        kLegLength * 0.5f,           // 半高さ
        b2Vec2(0.0f, -kLegLength * 0.5f), // 脚原点から下方向
        0.0f);

    b2FixtureDef leg_fix;
    leg_fix.shape = &leg_shape;
    leg_fix.density = 1.0f;
    leg_fix.friction = 1.5f;
    leg_fix.restitution = 0.0f;

    // --- 足裏（円） ---
    b2CircleShape foot_shape;
    foot_shape.m_radius = 0.06f;
    foot_shape.m_p.Set(0.0f, -kLegLength); // 脚先端

    b2FixtureDef foot_fix;
    foot_fix.shape = &foot_shape;
    foot_fix.density = 0.0f;
    foot_fix.friction = 2.0f;
    foot_fix.restitution = 0.0f;

    // ===============================
    // 左脚
    // ===============================
    {
        b2Vec2 offset(-kLegOffsetX, -kLanderRadius);

        b2BodyDef def;
        def.type = b2_dynamicBody;
        def.position = lander_body_->GetPosition() + b2Mul(rotation, offset);
        def.angle = lander_body_->GetAngle() + kLegAttachAngle;
        def.angularDamping = 5.0f;

        left_leg_body_ = world_->CreateBody(&def);
        left_leg_body_->CreateFixture(&leg_fix);
        left_leg_body_->CreateFixture(&foot_fix);
    }

    // ===============================
    // 右脚
    // ===============================
    {
        b2Vec2 offset(+kLegOffsetX, -kLanderRadius);

        b2BodyDef def;
        def.type = b2_dynamicBody;
        def.position = lander_body_->GetPosition() + b2Mul(rotation, offset);
        def.angle = lander_body_->GetAngle() - kLegAttachAngle;
        def.angularDamping = 5.0f;

        right_leg_body_ = world_->CreateBody(&def);
        right_leg_body_->CreateFixture(&leg_fix);
        right_leg_body_->CreateFixture(&foot_fix);
    }

    // --------------------
    // ジョイント（弱い角度バネ付き）
    // --------------------
    auto create_leg_joint = [&](b2Body* leg_body, const b2Vec2& local_anchor_a, float reference_angle) {
        b2WeldJointDef joint_def;
        joint_def.bodyA = lander_body_;
        joint_def.bodyB = leg_body;
        joint_def.localAnchorA = local_anchor_a;
        joint_def.localAnchorB.Set(0.0f, 0.0f);   // ★脚の原点＝付け根
        joint_def.referenceAngle = reference_angle;
        b2AngularStiffness(
            joint_def.stiffness,
            joint_def.damping,
            kLegWeldFrequencyHz,
            kLegWeldDampingRatio,
            lander_body_,
            leg_body);
        return static_cast<b2WeldJoint*>(world_->CreateJoint(&joint_def));
        };

    // 左脚ジョイント
    left_leg_joint_ = create_leg_joint(
        left_leg_body_,
        b2Vec2(-kLegOffsetX, -kLanderRadius),
        -kLegRestAngle);

    // 右脚ジョイント
    right_leg_joint_ = create_leg_joint(
        right_leg_body_,
        b2Vec2(+kLegOffsetX, -kLanderRadius),
        kLegRestAngle);
}

void LunarLanderEnv::applyWind()
{
    if (!config_.enable_wind) {
        return;
    }
    if (!lander_body_) {
        return;
    }

    if (shouldStopWind()) {
        last_wind_x_ = 0.0f;
        last_wind_torque_ = 0.0f;
        return;
    }

    // --- Gymに仕様に合わせる

    // 水平方向の風力を計算
    const double k = 0.01;
    const double idx_w = static_cast<double>(wind_idx_);
    float wind_val = std::tanh(std::sin(2.0 * k * idx_w) + std::sin(b2_pi * k * idx_w));        // sin(0.02 * idx) + sin(pi * 0.01 * idx)
    float wind_mag = wind_val * config_.wind_power;
    wind_idx_++;

    // 水平方向の風力を反映
    b2Vec2 force(wind_mag, 0.0f);
    lander_body_->ApplyForceToCenter(force, true);

    // 風による回転力（乱気流）を計算
    const double idx_t = static_cast<double>(torque_idx_);
    float torque_val = std::tanh(std::sin(2.0 * k * idx_t) + std::sin(b2_pi * k * idx_t));
    float torque_mag = torque_val * config_.turbulence_power;
    torque_idx_++;

    // 回転力を反映
    lander_body_->ApplyTorque(torque_mag, true);

    // 記録用 (GetScalar等で使用)
    last_wind_x_ = wind_mag;
    last_wind_torque_ = torque_mag;
}

bool LunarLanderEnv::shouldStopWind() const
{
    if (config_.wind_stop_mode == "none") {
        return false;
    }

    if (config_.wind_stop_mode == "ground") {
        return legTouchesGround(left_leg_body_) || legTouchesGround(right_leg_body_);
    }

    if (config_.wind_stop_mode == "pad") {
        return legTouchesPad(left_leg_body_) || legTouchesPad(right_leg_body_);
    }

    ANET_SYSTEM_ERROR("Invalid LunarLanderEnv.wind_stop_mode: "
        << config_.wind_stop_mode << ". Expected none, ground, or pad.");
    return false;
}

bool LunarLanderEnv::legTouchesGround(const b2Body* leg_body) const
{
    if (!leg_body || !ground_body_) {
        return false;
    }

    for (const b2ContactEdge* edge = leg_body->GetContactList(); edge; edge = edge->next) {
        b2Contact* contact = edge->contact;
        if (!contact || !contact->IsTouching()) {
            continue;
        }

        const b2Fixture* fixture_a = contact->GetFixtureA();
        const b2Fixture* fixture_b = contact->GetFixtureB();
        const b2Fixture* other_fixture = nullptr;
        if (fixture_a->GetBody() == leg_body) {
            other_fixture = fixture_b;
        } else if (fixture_b->GetBody() == leg_body) {
            other_fixture = fixture_a;
        }

        if (other_fixture && other_fixture->GetBody() == ground_body_) {
            return true;
        }
    }

    return false;
}

bool LunarLanderEnv::legTouchesPad(const b2Body* leg_body) const
{
    if (!leg_body) {
        return false;
    }

    for (const b2ContactEdge* edge = leg_body->GetContactList(); edge; edge = edge->next) {
        b2Contact* contact = edge->contact;
        if (!contact || !contact->IsTouching()) {
            continue;
        }

        const b2Fixture* fixture_a = contact->GetFixtureA();
        const b2Fixture* fixture_b = contact->GetFixtureB();
        const b2Fixture* other_fixture = nullptr;
        if (fixture_a->GetBody() == leg_body) {
            other_fixture = fixture_b;
        } else if (fixture_b->GetBody() == leg_body) {
            other_fixture = fixture_a;
        }

        if (other_fixture && IsPadFixture(other_fixture)) {
            return true;
        }
    }

    return false;
}

void LunarLanderEnv::destroyWorld()
{
    if (!world_) {
        return;
    }

    left_leg_joint_ = nullptr;
    right_leg_joint_ = nullptr;
    left_leg_body_ = nullptr;
    right_leg_body_ = nullptr;
    lander_body_ = nullptr;
    ground_body_ = nullptr;

    contact_listener_.reset();
    world_.reset();
}

void LunarLanderEnv::buildWorld(float init_x, float init_y, float init_angle)
{
    destroyWorld();

    b2Vec2 gravity(0.0f, config_.gravity);
    world_ = std::make_unique<b2World>(gravity);

    contact_listener_ = std::make_unique<ContactListener>(*this);
    world_->SetContactListener(contact_listener_.get());

    buildGround();
    buildLander(init_x, init_y, init_angle);
}

std::shared_ptr<const anet::rl::SingleResetResult> LunarLanderEnv::Reset(anet::rl::RunMode mode)
{
    ANET_PROFILE_FUNC();

    // ランダム初期値生成
    float dx = rnd_->Uniform(-config_.init.x_range, config_.init.x_range);
    float dy = rnd_->Uniform(-config_.init.y_range, config_.init.y_range);
    float init_angle = rnd_->Uniform(-config_.init.angle_range, config_.init.angle_range);
    float init_x_vel = rnd_->Uniform(-config_.init.x_velocity_range, config_.init.x_velocity_range);
    float init_y_vel = rnd_->Uniform(-config_.init.y_velocity_range, config_.init.y_velocity_range);
    float init_anglular_vel = rnd_->Uniform(-config_.init.angular_velocity_range, config_.init.angular_velocity_range);

    // ベース位置（Y座標はコンフィグ依存）
    float init_x = dx;
    float init_y = config_.ground_y + config_.world_height * 0.6f + dy;

    // ワールド再構築
    buildWorld(init_x, init_y, init_angle);

    // 状態変数初期化
    step_count_ = 0;
    last_action_ = -1;
    body_contact_ = false;
    left_leg_contact_ = false;
    right_leg_contact_ = false;
    left_leg_contact_count_ = 0;
    right_leg_contact_count_ = 0;
    has_prev_shaping_ = false;
    last_shaping_ = 0.0f;

    // 風初期化
    if (config_.enable_wind) {
        wind_idx_ = rnd_->RandInt(-9999, 9999);
        torque_idx_ = rnd_->RandInt(-9999, 9999);
    } else {
        wind_idx_ = 0;
        torque_idx_ = 0;
    }
    applyWind();

    // 本体の速度・角速度リセット
    lander_body_->SetLinearVelocity(b2Vec2(init_x_vel, init_y_vel));
    lander_body_->SetAngularVelocity(init_anglular_vel);

    // 左脚の速度・角速度リセット
    b2Vec2 left_vel = lander_body_->GetLinearVelocityFromWorldPoint(left_leg_body_->GetWorldCenter());
    left_leg_body_->SetLinearVelocity(left_vel);
    left_leg_body_->SetAngularVelocity(init_anglular_vel);

    // 右脚の速度・角速度リセット
    b2Vec2 right_vel = lander_body_->GetLinearVelocityFromWorldPoint(right_leg_body_->GetWorldCenter());
    right_leg_body_->SetLinearVelocity(right_vel);
    right_leg_body_->SetAngularVelocity(init_anglular_vel);

    // 戻り生成
    auto state = makeState();
    state.episode_start = true;
    state.done = false;
    state.truncated = false;
    auto ret = std::make_shared<ResetResult>(this->shared_from_this(), std::move(state));

    return ret;
}

void LunarLanderEnv::applyActionForce(int64_t action)
{
    if (!lander_body_) {
        return;
    }

    switch (action) {
    case kActionMain: { // main engine
        b2Vec2 force(0.0f, kMainEngineForce);
        lander_body_->ApplyForceToCenter(force, true);
        break;
    }
    case kActionLeft: { // left engine
        b2Vec2 force(+kSideEngineForce, kSideEngineForce * 0.5f);
        lander_body_->ApplyForceToCenter(force, true);
        lander_body_->ApplyTorque(-kSideEngineTorque, true);
        break;
    }
    case kActionRight: { // right engine
        b2Vec2 force(-kSideEngineForce, kSideEngineForce * 0.5f);
        lander_body_->ApplyForceToCenter(force, true);
        lander_body_->ApplyTorque(+kSideEngineTorque, true);
        break;
    }
    default:
        break;
    }
}

bool LunarLanderEnv::checkCrash() const
{
    if (!lander_body_)
        return true;

    // 本体が地面に触れたらクラッシュ
    if (body_contact_) {
        //ANET_LOG_DEBUG("crashed: body_contact. y=" << lander_body_->GetPosition().y);
        return true;
    }

    // 範囲外はクラッシュ扱い
    float x = lander_body_->GetPosition().x;
    if ((x < -config_.world_half_width) || (x > config_.world_half_width)) {
        //ANET_LOG_DEBUG("crashed x=" << x);
        return true;
    }

    // 空高く上がったらクラッシュ扱い
    //float y = lander_body_->GetPosition().y;
    //if (y > 6.0f) {
    //    return true;
    //}

    return false;
}

bool LunarLanderEnv::checkLanded() const
{
    if (!lander_body_) {
        return false;
    }

    if (config_.landing_detection_mode == "contact") {
        // 両足が付いたら着陸
        const bool legs_contact = left_leg_contact_ && right_leg_contact_;
        return legs_contact;
    }

    if (config_.landing_detection_mode == "not_awake") {
        return !lander_body_->IsAwake();
    }

    ANET_SYSTEM_ERROR("Invalid LunarLanderEnv.landing_detection_mode: "
        << config_.landing_detection_mode << ". Expected contact or not_awake.");
    return false;

    //const b2Vec2 pos = lander_body_->GetPosition();
    //const b2Vec2 vel = lander_body_->GetLinearVelocity();
    //const float raw_angle = lander_body_->GetAngle();
    //float angle = std::fmod(raw_angle + b2_pi, 2.0f * b2_pi);
    //if (angle < 0.0f) angle += 2.0f * b2_pi;
    //angle -= b2_pi;

    //const bool over_pad =
    //    (pos.x >= pad_info_.x1) && (pos.x <= pad_info_.x2) &&
    //    (std::abs(pos.y - pad_info_.y) < 0.5f);

    //const bool slow_enough =
    //    (std::abs(vel.x) < 1.0f) &&
    //    (std::abs(vel.y) < 1.0f) &&
    //    (std::abs(angle) < 0.3f);

    //return over_pad && slow_enough && legs_contact;
}

anet::rl::SingleState LunarLanderEnv::makeState() const
{

    if (!lander_body_) {
        anet::rl::SingleState s {
            .obs = anet::TensorDict(
                anet::rl::ObsKeys::kVector,
                torch::zeros({ config_.obs_include_action ? 8 + kActionCount : 8 }, float_opt_)),
            .done = true,
            .truncated = false,
            .episode_start = false
        };
        return s;
    }

    const b2Vec2 pos = lander_body_->GetPosition();
    const b2Vec2 vel = lander_body_->GetLinearVelocity();
    const float raw_angle = lander_body_->GetAngle();
    float angle = std::fmod(raw_angle + b2_pi, 2.0f * b2_pi);
    if (angle < 0.0f) angle += 2.0f * b2_pi;
    angle -= b2_pi;
    const float angular_vel = lander_body_->GetAngularVelocity();

    const float norm_x = config_.world_half_width;
    const float norm_y = config_.world_height;

    float x = pos.x / norm_x;
    float y = (pos.y - config_.ground_y) / norm_y;
    float vx = vel.x;
    float vy = vel.y;

    float left_contact = left_leg_contact_ ? 1.0f : 0.0f;
    float right_contact = right_leg_contact_ ? 1.0f : 0.0f;

    std::vector<float> obs_values = { x, y, vx, vy, angle, angular_vel, left_contact, right_contact };
    if (config_.obs_include_action) {
        obs_values.reserve(8 + kActionCount);
        for (int64_t action = 0; action < kActionCount; ++action) {
            obs_values.push_back(last_action_ == action ? 1.0f : 0.0f);
        }
    }

    anet::rl::SingleState s {
        .obs = { anet::rl::ObsKeys::kVector, torch::tensor(obs_values) },
        .done = false,
        .truncated = false,
        .episode_start = false,
    };

    return s;
}

float LunarLanderEnv::computeShaping(const anet::rl::SingleState& state) const
{
    const auto obs = state.obs.At(anet::rl::ObsKeys::kVector);

    const float x = obs[0].item<float>();   // 正規化済み x
    const float y = obs[1].item<float>();   // 正規化済み y
    const float vx = obs[2].item<float>();
    const float vy = obs[3].item<float>();
    const float angle = obs[4].item<float>();
    const float left_contact = obs[6].item<float>();
    const float right_contact = obs[7].item<float>();

    const float dist = std::sqrt(x * x + y * y);
    const float speed = std::sqrt(vx * vx + vy * vy);

    float shaping = 0.0f;
    shaping += -100.0f * dist;
    shaping += -100.0f * speed;
    shaping += -100.0f * std::abs(angle);
    shaping += 10.0f * left_contact;
    shaping += 10.0f * right_contact;

    /// @todo Gym lunar_lander.py の shaping 係数との整合を検証する。
    return shaping;
}

std::pair<float, float> LunarLanderEnv::calcReward(const anet::rl::SingleState& state, bool crashed, bool landed, int64_t action)
{
    float reward = kStepPenality;
    float raw_reward = kStepPenality;

    // shaping を計算
    const float shaping = computeShaping(state);
    //ANET_LOG_DEBUG("shaping=" << shaping);

    // shaping 差分（前回 shaping が無ければ 0）
    if (has_prev_shaping_) {
        reward += (shaping - last_shaping_);
    }
    raw_reward += shaping;
    //ANET_LOG_DEBUG("reward=" << reward);
    //ANET_LOG_DEBUG("raw_reward=" << raw_reward);

    // 次回ステップのために更新
    last_shaping_ = shaping;
    has_prev_shaping_ = true;

    // 燃料噴射ペナルティ
    if (action == kActionMain) {
        reward -= 0.3f;     // main engine fire
        raw_reward -= 0.3f;
    } else if (action == kActionLeft || action == kActionRight) {   // side engine fire
        reward -= 0.03f;
        raw_reward -= 0.03f;
    }
    //ANET_LOG_DEBUG("reward=" << reward);
    //ANET_LOG_DEBUG("raw_reward=" << raw_reward);

    // 終端ボーナス／ペナルティ
    if (crashed) {
        reward += kCrashPenalty;
        raw_reward += kCrashPenalty;
    } else if (landed) {
        reward += kLandReward;
        raw_reward += kLandReward;
    }
    //ANET_LOG_DEBUG("reward=" << reward);
    //ANET_LOG_DEBUG("raw_reward=" << raw_reward);

    /// @todo reward_range と実際の報酬スケールの整合を再確認する。
    return { reward, raw_reward };
}

std::shared_ptr<const anet::rl::SingleStepResult> LunarLanderEnv::Step(int64_t action, anet::rl::RunMode runmode)
{
    ANET_PROFILE_FUNC();

    step_count_++;
    last_action_ = action;

    applyWind();
    applyActionForce(action);

    const float time_step = 1.0f / 50.0f;
    const int vel_iter = 10;
    const int pos_iter = 2;
    world_->Step(time_step, vel_iter, pos_iter);

    // エピソード終了判定
    const bool crashed = checkCrash();
    const bool landed = checkLanded();
    bool done = crashed || landed;
    bool truncated = false;
    if (step_count_ >= config_.limit_step && !done) {
        truncated = true;
    }

    // 戻り生成
    auto state = makeState();
    state.done = done;
    state.truncated = truncated;
    const auto rewards = calcReward(state, crashed, landed, action);
    const auto result = std::make_shared<LunarLanderEnv::StepResult>(
        this->shared_from_this(), rewards.first, rewards.second, std::move(state));

    return result;
}

struct Segment {
    b2Vec2 p0;
    b2Vec2 p1;
};

static Segment getLegSegment(const b2Body* leg_body, float leg_length)
{
    const b2Vec2 root = leg_body->GetPosition(); // 付け根（ボディ原点）
    const b2Vec2 down = -leg_body->GetWorldVector(b2Vec2(0.0f, 1.0f)); // ローカル -Y

    // Box2Dの演算子が環境によって微妙なら手で掛け算するのが安全
    const b2Vec2 d(down.x * leg_length, down.y * leg_length);

    Segment s;
    s.p0 = root;        // 付け根
    s.p1 = root + d;    // 足先
    return s;
}

std::optional<float> LunarLanderEnv::GetScalar(const std::string& key, int64_t index) const
{
    ANET_ASSERT(index == -1 || index == 0);

    if (key == "wind_x") return last_wind_x_;
    if (key == "wind_torque") return last_wind_torque_;

    return std::nullopt;
}

std::optional<torch::Tensor> LunarLanderEnv::GetTensor(const std::string& key, int64_t index) const
{
    ANET_ASSERT(index == -1 || index == 0);

    if (key == "pad") {
        torch::Tensor t = torch::tensor({ pad_info_.x1, pad_info_.x2, pad_info_.y });
        return t;
    }

    if (key == "legs") {
        auto left_seg = getLegSegment(left_leg_body_, kLegLength);
        auto right_seg = getLegSegment(right_leg_body_, kLegLength);
        ANET_LOG_DEBUG_PREFIXED("lander_body=" << lander_body_->GetPosition().x << " " << lander_body_->GetPosition().y);
        ANET_LOG_DEBUG_PREFIXED("left_seg.p0=" << left_seg.p0.x << " " << left_seg.p0.y);
        ANET_LOG_DEBUG_PREFIXED("left_seg.p1=" << left_seg.p1.x << " " << left_seg.p1.y);
        ANET_LOG_DEBUG_PREFIXED("right_seg.p0=" << right_seg.p0.x << " " << right_seg.p0.y);
        ANET_LOG_DEBUG_PREFIXED("right_seg.p1=" << right_seg.p1.x << " " << right_seg.p1.y);

        torch::Tensor t = torch::tensor({
            left_seg.p0.x, left_seg.p0.y, left_seg.p1.x, left_seg.p1.y,
            right_seg.p0.x, right_seg.p0.y, right_seg.p1.x, right_seg.p1.y,
            });
        return t;
    }

    if (key == "world_bounds") {
        float min_x = -config_.world_half_width;
        float max_x = config_.world_half_width;
        float min_y = std::min(0.0f, config_.ground_y);
        float max_y = std::max(config_.world_height, config_.ground_y);
        torch::Tensor t = torch::tensor({ min_x, max_x, min_y, max_y });
        return t;
    }

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>>
LunarLanderEnv::GetTensorVector(const std::string& key, int64_t index) const
{
    ANET_ASSERT(index == -1 || index == 0);

    if (key == "terrain") {
        std::vector<torch::Tensor> out;
        out.reserve(terrain_points_.size());

        for (const auto& p : terrain_points_) {
            // 各点を (x, y) Tensor として格納
            torch::Tensor pt = torch::tensor({ p.x, p.y });
            out.push_back(pt);
        }
        return out;
    }

    return std::nullopt;
}

anet::rl::AuxData LunarLanderEnv::CreateAuxData(float reward, float raw_reward) const
{
    ANET_PROFILE_FUNC();

    anet::rl::AuxData aux;

    if (!lander_body_) {
        return aux;
    }

    // --- world 定義（描画用・不変） ---
    {
        const float min_x = -config_.world_half_width;
        const float max_x = +config_.world_half_width;
        const float min_y = config_.ground_y - config_.terrain_noise_height;
        const float max_y = config_.ground_y + config_.world_height;

        aux.emplace(
            "world",
            torch::tensor(
                { min_x, max_x, min_y, max_y },
                float_opt_));
    }
    // --- Lander 本体 ---
    {
        const b2Vec2 pos = lander_body_->GetPosition();
        const b2Vec2 vel = lander_body_->GetLinearVelocity();
        const float angle = lander_body_->GetAngle();
        const float angular_vel = lander_body_->GetAngularVelocity();

        aux.emplace(
            "lander",
            torch::tensor(
                { pos.x, pos.y, vel.x, vel.y, angle, angular_vel },
                float_opt_));
    }

    // --- 脚セグメント ---
    {
        auto left = getLegSegment(left_leg_body_, kLegLength);
        auto right = getLegSegment(right_leg_body_, kLegLength);

        aux.emplace(
            "legs",
            torch::tensor(
                {
                    left.p0.x,  left.p0.y,  left.p1.x,  left.p1.y,
                    right.p0.x, right.p0.y, right.p1.x, right.p1.y
                },
                float_opt_));
    }

    // --- パッド ---
    {
        aux.emplace(
            "pad",
            torch::tensor(
                { pad_info_.x1, pad_info_.x2, pad_info_.y },
                float_opt_));
    }

    // --- 地形（terrain polyline） ---
    {
        const size_t n = terrain_points_.size();
        if (n > 0) {
            torch::Tensor terrain = torch::empty({ static_cast<long>(n), 2 }, float_opt_);

            for (size_t i = 0; i < n; ++i) {
                terrain[i][0] = terrain_points_[i].x;
                terrain[i][1] = terrain_points_[i].y;
            }

            aux.emplace("terrain", terrain);
        }
    }

    // --- 接地フラグ（物理真値） ---
    {
        aux.emplace(
            "contacts",
            torch::tensor(
                {
                    left_leg_contact_,
                    right_leg_contact_,
                    body_contact_
                },
                bool_opt_));
    }

    // --- joint 情報 ---
    if (lander_body_ && left_leg_body_ && right_leg_body_) {
        aux.emplace(
            "joint_angle",
            torch::tensor(
                {
                    left_leg_body_->GetAngle() - lander_body_->GetAngle(),
                    right_leg_body_->GetAngle() - lander_body_->GetAngle()
                },
                float_opt_));
    }

    // --- 推力・風 ---
    {
        aux.emplace(
            "forces",
            torch::tensor(
                { last_wind_x_, last_wind_torque_ },
                float_opt_));
    }

    // --- 報酬 ---
    {
        aux.emplace(
            "rewards",
            torch::tensor(
                { reward, raw_reward },  // RL報酬、Raw報酬
                float_opt_));
    }

    //ANET_LOG_DEBUG("aux=" << anet::ToString(aux));

    return aux;
}

// ===== Factory =====

std::shared_ptr<anet::rl::SingleDiscreteEnv>
LunarLanderEnvFactory::CreateSingleEnv(const anet::ConfigData& config_data, const torch::Device& device,
    const std::string& name, std::optional<anet::seed_t> seed, const std::string& config_prefix)
{
    LunarLanderEnvConfig config(config_data, config_prefix);
    //LOG::info() << "LunarLanderEnvFactory: config_prefix=" << config_prefix << " config=" << config.ToJson();
    return std::make_shared<LunarLanderEnv>(config, device, name, seed);
}

ANET_REGISTER_ENV_FACTORY(LunarLanderEnvFactory);
