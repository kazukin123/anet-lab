#include "LunarLanderEnv.hpp"

#include <cmath>
#include <algorithm>
#include <wx/log.h>
#include <box2d/box2d.h>
#include "anet/log.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/profile.hpp"
#include "anet/env.hpp"

namespace {

    // ボディのサイズなどは Gym をざっくり参考にした値
    constexpr float kLanderDensity = 5.0f;
    constexpr float kLegDensity = 1.0f;

    // 脚の定義
    constexpr float kLegLength = 0.35f;
    constexpr float kLegOffsetX = 0.40f;
    //constexpr float kLegAttachAngle = 0.5236f; // 30°
    constexpr float kLegAttachAngle = 0.7854f; // 45°
    constexpr float kLegWidth = 0.1f;

    // 報酬関連の定数値
    constexpr float kStepPenality = -0.01f;
    constexpr float kCrashPenalty = -100.0f;
    constexpr float kLandReward = 100.0f;

    /// @todo Gym の lunar_lander.py のパラメータと整合を取る。

} // namespace

anet::rl::EnvSpec LunarLanderEnv::GetSpec() const
{
    anet::rl::StateSpec state_spec;
    state_spec.shape = { 8 };

    // ざっくりしたレンジ指定。Gym と完全一致ではない。
    state_spec.dims = {
        { {0}, -1.5f, 1.5f, "x", "" },
        { {1}, -0.5f, 2.0f, "y", "" },
        { {2}, -10.0f, 10.0f, "vx", "" },
        { {3}, -10.0f, 15.0f, "vy", "" },
        { {4}, -3.14f, 3.14f, "angle", "" },
        { {5}, -15.0f, 15.0f, "angular_velocity", "" },
        { {6}, 0.0f, 1.0f, "left_leg_contact", "" },
        { {7}, 0.0f, 1.0f, "right_leg_contact", "" }
    };

    anet::rl::ActionSpec action_spec;
    action_spec.is_discrete = true;
    action_spec.value_labels = {
        "no_op",
        "main_engine",
        "left_engine",
        "right_engine"
    };

    anet::rl::EnvSpec env_spec;
    env_spec.state_spec = state_spec;
    env_spec.action_spec = action_spec;
    env_spec.reward_range = { -100.0f, 100.0f };

    /// @todo EnvSpec.info に LunarLander 固有情報を入れる。
    return env_spec;
}

// ===== ContactListener =====

void LunarLanderEnv::ContactListener::BeginContact(b2Contact* contact)
{
    auto fixture_a = contact->GetFixtureA();
    auto fixture_b = contact->GetFixtureB();
    b2Body* body_a = fixture_a->GetBody();
    b2Body* body_b = fixture_b->GetBody();

    if (body_a == env_.lander_body_ || body_b == env_.lander_body_) {
        env_.body_contact_ = true;
    }
    if (body_a == env_.left_leg_body_ || body_b == env_.left_leg_body_) {
        env_.left_leg_contact_ = true;
    }
    if (body_a == env_.right_leg_body_ || body_b == env_.right_leg_body_) {
        env_.right_leg_contact_ = true;
    }
}

void LunarLanderEnv::ContactListener::EndContact(b2Contact* contact)
{
    auto fixture_a = contact->GetFixtureA();
    auto fixture_b = contact->GetFixtureB();
    b2Body* body_a = fixture_a->GetBody();
    b2Body* body_b = fixture_b->GetBody();

    if (body_a == env_.lander_body_ || body_b == env_.lander_body_) {
        env_.body_contact_ = false;
    }
    if (body_a == env_.left_leg_body_ || body_b == env_.left_leg_body_) {
        env_.left_leg_contact_ = false;
    }
    if (body_a == env_.right_leg_body_ || body_b == env_.right_leg_body_) {
        env_.right_leg_contact_ = false;
    }
}

// ===== LunarLanderEnv =====

LunarLanderEnv::LunarLanderEnv(
    const LunarLanderEnvConfig& config,
    const torch::Device& device,
    const std::optional<anet::seed_t> seed)
    : anet::RandomHolder(seed)
    , config_(config)
{
    ANET_LOG_DEBUG("seed=" << this->GetSeed());

    anet::MetricsLogger::Instance()->LogJson(
        "LunarLanderEnvConfig",
        config_.ToJson());
    anet::MetricsLogger::Instance()->Flush();

    obs_opt_ = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    buildWorld();
}

LunarLanderEnv::~LunarLanderEnv()
{
    destroyWorld();
}

void LunarLanderEnv::buildWorld()
{
    destroyWorld();

    b2Vec2 gravity(0.0f, config_.gravity_y);
    world_ = std::make_unique<b2World>(gravity);

    contact_listener_ = std::make_unique<ContactListener>(*this);
    world_->SetContactListener(contact_listener_.get());

    buildGround();
    buildLander();
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

void LunarLanderEnv::buildGround()
{
    terrain_points_.clear();

    const float half_w = config_.world_half_width;
    const float ground_y = config_.ground_y;

    // --- Pad 定義 ---
    pad_info_.x1 = -kLegOffsetX * 1.5f;
    pad_info_.x2 = +kLegOffsetX * 1.5f;
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

    for (size_t i = 0; i + 1 < terrain_points_.size(); ++i) {
        b2EdgeShape edge;
        edge.SetTwoSided(terrain_points_[i], terrain_points_[i + 1]);
        b2FixtureDef fd;
        fd.shape = &edge;
        fd.density = 0.0f;
        fd.restitution = 0.1f;
        fd.friction = 0.5f;
        ground_body_->CreateFixture(&fd);
    }
}

void LunarLanderEnv::buildLander()
{
    const float start_x = 0.0f;
    const float start_y = config_.ground_y + config_.world_height * 0.6f;

    // 本体
    {
        b2BodyDef body_def;
        body_def.type = b2_dynamicBody;
        body_def.position.Set(start_x, start_y);
        body_def.angle = 0.0f;
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

    const float lander_angle = lander_body_->GetAngle();

    b2PolygonShape leg_shape;
    leg_shape.SetAsBox(
        0.05f,                     // 半幅
        kLegLength * 0.5f,         // 半長
        b2Vec2(0.0f, -kLegLength * 0.5f), // ★下方向
        0.0f
    );

    b2FixtureDef leg_fix;
    leg_fix.shape = &leg_shape;
    leg_fix.density = kLegDensity;
    leg_fix.friction = 0.5f;
    leg_fix.restitution = 0.0f;

    // ---------- 左脚 ----------
    b2BodyDef left_leg_def;
    left_leg_def.type = b2_dynamicBody;
    left_leg_def.position = lander_body_->GetPosition() + b2Vec2(-kLegOffsetX, -kLanderRadius);
    left_leg_def.angle = lander_angle + kLegAttachAngle;
    left_leg_body_ = world_->CreateBody(&left_leg_def);
    left_leg_body_->CreateFixture(&leg_fix);

    // ---------- 右脚 ----------
    b2BodyDef right_leg_def;
    right_leg_def.type = b2_dynamicBody;
    right_leg_def.position = lander_body_->GetPosition() + b2Vec2(+kLegOffsetX, -kLanderRadius);
    right_leg_def.angle = lander_angle - kLegAttachAngle;
    right_leg_body_ = world_->CreateBody(&right_leg_def);
    right_leg_body_->CreateFixture(&leg_fix);

    // --------------------
    // ジョイント（固定）
    // --------------------
    b2RevoluteJointDef joint_def;
    joint_def.enableLimit = true;
    joint_def.lowerAngle = 0.0f;
    joint_def.upperAngle = 0.0f;
    joint_def.referenceAngle = 0.0f;
    joint_def.bodyA = lander_body_;
    joint_def.localAnchorB.Set(0.0f, 0.0f);   // ★脚の原点＝付け根

    // 左脚ジョイント
    joint_def.bodyB = left_leg_body_;
    joint_def.localAnchorA.Set(-kLegOffsetX, -kLanderRadius);
    left_leg_joint_ = static_cast<b2RevoluteJoint*>(world_->CreateJoint(&joint_def));

    // 右脚ジョイント
    joint_def.bodyB = right_leg_body_;
    joint_def.localAnchorA.Set(+kLegOffsetX, -kLanderRadius);
    right_leg_joint_ = static_cast<b2RevoluteJoint*>(world_->CreateJoint(&joint_def));
}

anet::rl::SingleState LunarLanderEnv::Reset(anet::rl::RunMode mode)
{
    anet::ProfileRange range("LunarLanderEnv::Reset");

    buildWorld();

    step_count_ = 0;
    last_wind_x_ = 0.0f;
    body_contact_ = false;
    left_leg_contact_ = false;
    right_leg_contact_ = false;
    has_prev_shaping_ = false;
    last_shaping_ = 0.0f;

    // 初期状態は Train / Eval で変える
    if (anet::rl::IsTrain(mode)) {
        float dx = rnd_->Uniform(-1.0f, 1.0f);
        lander_body_->SetTransform(
            b2Vec2(dx, lander_body_->GetPosition().y),
            rnd_->Uniform(-0.1f, 0.1f));
    } else {
        lander_body_->SetTransform(
            b2Vec2(0.0f, lander_body_->GetPosition().y),
            0.0f);
    }

    // 速度リセット
    lander_body_->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
    lander_body_->SetAngularVelocity(0.0f);
    left_leg_body_->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
    left_leg_body_->SetAngularVelocity(0.0f);
    right_leg_body_->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
    right_leg_body_->SetAngularVelocity(0.0f);

    auto state = makeState();
    state.episode_start = true;
    state.done = false;
    state.truncated = false;

    return state;
}

void LunarLanderEnv::applyWind()
{
    last_wind_x_ = 0.0f;

    if (!config_.enable_wind) {
        return;
    }
    if (!lander_body_) {
        return;
    }

    float w = rnd_->Uniform(-1.0f, 1.0f) * config_.wind_power;
    w += rnd_->Uniform(-1.0f, 1.0f) * config_.turbulence_power;
    last_wind_x_ = w;

    b2Vec2 force(w * 10, 0.0f);
    lander_body_->ApplyForceToCenter(force, true);
}

void LunarLanderEnv::applyActionForce(int64_t action)
{
    if (!lander_body_) {
        return;
    }

    switch (action) {
    case 1: { // main engine
        b2Vec2 force(0.0f, kMainEngineForce);
        lander_body_->ApplyForceToCenter(force, true);
        break;
    }
    case 2: { // left engine
        b2Vec2 force(-kSideEngineForce, kSideEngineForce * 0.5f);
        lander_body_->ApplyForceToCenter(force, true);
        lander_body_->ApplyTorque(-kSideEngineTorque, true);
        break;
    }
    case 3: { // right engine
        b2Vec2 force(kSideEngineForce, kSideEngineForce * 0.5f);
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

    // Gym互換：本体が地面に触れたらクラッシュ
    if (body_contact_) {
        ANET_LOG_DEBUG("crashed: body_contact. y=" << lander_body_->GetPosition().y);
        return true;
    }

    // 範囲外はクラッシュ扱い
    float x = lander_body_->GetPosition().x;
    if ((x < -config_.world_half_width) || (x > config_.world_half_width)) {
        ANET_LOG_DEBUG("crashed x=" << x);
        return true;
    }

    //ANET_LOG_DEBUG("Not crashed. x=" << x);

    return false;
}

bool LunarLanderEnv::checkLanded() const
{
    if (!lander_body_) {
        return false;
    }
    
    // 両足が付いたら着陸
    const bool legs_contact = left_leg_contact_ && right_leg_contact_;
    return (legs_contact);

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
    anet::rl::SingleState s;

    if (!lander_body_) {
        s.obs = torch::zeros({ 8 }, obs_opt_);
        s.done = true;
        s.truncated = false;
        s.episode_start = false;
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

    s.obs = torch::tensor(
        { x, y, vx, vy, angle, angular_vel, left_contact, right_contact },
        obs_opt_);
    s.done = false;
    s.truncated = false;
    s.episode_start = false;

    return s;
}

float LunarLanderEnv::computeShaping(
    const anet::rl::SingleState& state) const
{
    const auto obs = state.obs;

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
    shaping += -400.0f * dist;
    shaping += -100.0f * speed;
    shaping += -100.0f * std::abs(angle);
    shaping += 10.0f * left_contact;
    shaping += 10.0f * right_contact;

    /// @todo Gym lunar_lander.py の shaping 係数との整合を検証する。
    return shaping;
}

float LunarLanderEnv::calcReward(const anet::rl::SingleState& state, bool crashed, bool landed)
{
    float reward = kStepPenality;

    // Gym 互換の shaping を計算
    const float shaping = computeShaping(state);

    // shaping 差分（前回 shaping が無ければ 0）
    if (has_prev_shaping_) {
        reward += (shaping - last_shaping_);
    }

    // 次回ステップのために更新
    last_shaping_ = shaping;
    has_prev_shaping_ = true;

    // 終端ボーナス／ペナルティ
    if (crashed) {
        reward += kCrashPenalty;
    } else if (landed) {
        reward += kLandReward;
    }

    /// @todo Gym の燃料ペナルティ（エンジン使用量）を action から加味するか検討する。
    /// @todo reward_range と実際の報酬スケールの整合を再確認する。
    return reward;
}

anet::rl::SingleStepResult LunarLanderEnv::Step(int64_t action, anet::rl::RunMode runmode)
{
    anet::ProfileRange range("LunarLanderEnv::Step");

    step_count_++;
    
    body_contact_ = false;
    left_leg_contact_ = false;
    right_leg_contact_ = false;

    applyWind();
    applyActionForce(action);

    const float time_step = 1.0f / 50.0f;
    const int vel_iter = 6;
    const int pos_iter = 2;
    world_->Step(time_step, vel_iter, pos_iter);

    const bool crashed = checkCrash();
    const bool landed = checkLanded();

    bool done = crashed || landed;
    bool truncated = false;
    if (step_count_ >= config_.limit_step && !done) {
        truncated = true;
        done = true;
    }

    auto state = makeState();
    state.done = done;
    state.truncated = truncated;

    const float reward = calcReward(state, crashed, landed);

    anet::rl::SingleStepResult result { reward, state, CreateAux() };

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

std::optional<float> LunarLanderEnv::GetScalar(const std::string& key, int index) const
{
    /// @todo auxに移行

    ANET_ASSERT(index == -1 || index == 0);

    if (key == "wind_x") {
        return last_wind_x_;
    }

    return std::nullopt;
}

std::optional<torch::Tensor> LunarLanderEnv::GetTensor(const std::string& key, int index) const
{
    /// @todo auxに移行

    ANET_ASSERT(index == -1 || index == 0);

    if (key == "pad") {
        torch::Tensor t = torch::tensor({ pad_info_.x1, pad_info_.x2, pad_info_.y });
        return t;
    }

    if (key == "legs") {
        auto left_seg = getLegSegment(left_leg_body_, kLegLength);
        auto right_seg = getLegSegment(right_leg_body_, kLegLength);
        ANET_LOG_DEBUG("lander_body=" << lander_body_->GetPosition().x << " " << lander_body_->GetPosition().y);
        ANET_LOG_DEBUG("left_seg.p0=" << left_seg.p0.x << " " << left_seg.p0.y);
        ANET_LOG_DEBUG("left_seg.p1=" << left_seg.p1.x << " " << left_seg.p1.y);
        ANET_LOG_DEBUG("right_seg.p0=" << right_seg.p0.x << " " << right_seg.p0.y);
        ANET_LOG_DEBUG("right_seg.p1=" << right_seg.p1.x << " " << right_seg.p1.y);

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
LunarLanderEnv::GetTensorVector(const std::string& key, int index) const
{
    /// @todo auxに移行

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

std::unordered_map<std::string, torch::Tensor> LunarLanderEnv::CreateAux()
{
    std::unordered_map<std::string, torch::Tensor> aux;

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
                obs_opt_));
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
                obs_opt_));
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
                obs_opt_));
    }

    // --- パッド ---
    {
        aux.emplace(
            "pad",
            torch::tensor(
                { pad_info_.x1, pad_info_.x2, pad_info_.y },
                obs_opt_));
    }

    // --- 地形（terrain polyline） ---
    {
        const size_t n = terrain_points_.size();
        if (n > 0) {
            torch::Tensor terrain = torch::empty({ static_cast<long>(n), 2 }, obs_opt_);

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
                    left_leg_contact_ ? 1.0f : 0.0f,
                    right_leg_contact_ ? 1.0f : 0.0f
                },
                obs_opt_));
    }

    // --- joint 情報 ---
    if (left_leg_joint_ && right_leg_joint_) {
        aux.emplace(
            "joint_angle",
            torch::tensor(
                {
                    left_leg_joint_->GetJointAngle(),
                    right_leg_joint_->GetJointAngle()
                },
                obs_opt_));
    }

    // --- 推力・風 ---
    {
        aux.emplace(
            "forces",
            torch::tensor(
                { last_wind_x_ },
                obs_opt_));
    }

    return aux;
}

// ===== Factory =====

std::unique_ptr<anet::rl::SingleDiscreteEnv>
LunarLanderEnvFactory::CreateSingleEnv(
    const anet::ConfigData& config_data,
    const torch::Device& device,
    std::optional<anet::seed_t> seed)
{
    LunarLanderEnvConfig config(config_data);
    return std::make_unique<LunarLanderEnv>(config, device, seed);
}

ANET_REGISTER_ENV_FACTORY(LunarLanderEnvFactory);
