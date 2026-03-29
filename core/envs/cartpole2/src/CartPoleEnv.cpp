#include "CartPoleEnv.hpp"

#include <cmath>
#include <algorithm>
#include <random>
#include <memory>
#include <wx/log.h>
#include "anet/metrics_logger.hpp"
#include "anet/profile.hpp"
#include "anet/tensor_util.hpp"
#include "anet/env.hpp"

// 定数

//const int limit_step = 500;  // 終了条件
const float reward_scale = 1.0f;  // 2 10  20
const float done_reward = 1.0f;  // 2 10  20

const float limit_x = 2.4f;
const float limit_x_dot = 2.0f;
const float limit_theta = 90.0f; // 12.0f 90.0f
const float limit_theta_dot = 3.0f;

const float gravity = 9.8f;
const float masscart = 1.0f;
const float mass_pole = 0.10f;
const float total_mass = masscart + mass_pole;
const float length = 0.5f;
const float polemass_length = mass_pole * length;
const float force_mag = 30.0f;  // 10.0f 30.0f
const float tau = 0.02f;    //0.02f 0.01f

const float deg = (float)M_PI / 180.0f;

CartPoleEnv::CartPoleEnv(const CartPoleEnvConfig& config, const torch::Device& device, std::optional<anet::seed_t> seed)
    : RandomHolder(seed), config_(config)
{
    // パラメータ記録
    anet::MetricsLogger::Instance()->Log("CartPoleEnvConfig", config_.ToJson());
    anet::MetricsLogger::Instance()->Flush();

    obs_opt_ = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    Reset();
}

anet::rl::EnvSpec CartPoleEnv::GetSpec() const
{
    // TensorSpec
    anet::rl::TensorSpec vec_obs_spec {
        .type = anet::rl::SpaceType::Vector,
        .shape = { 4 },
        .dtype = torch::kFloat32,
        .num_classes = 0,   // 連続値
        .labels = { "x", "x_dot", "theta", "theta_dot" },
        .min_values = { -limit_x, -limit_x_dot, -limit_theta, -limit_theta_dot },
        .max_values = { limit_x,  limit_x_dot,  limit_theta,  limit_theta_dot }
    };

    // StateSpec
    anet::rl::StateSpec state_spec;
    state_spec.obs_spec[anet::rl::ObsKeys::kVector] = vec_obs_spec;

    // ActionSpec
    anet::rl::ActionSpec action_spec {
        .is_discrete = true,   // is_discreate
        .value_labels = { "right", "left"}, // value_labels
        .dims = { // dims
            { 0, 1, "force" }  // min, max, name
        }
    };

    // EnvSpec
    anet::rl::EnvSpec env_spec {
        .state_spec = state_spec,
        .action_spec = action_spec,
        .reward_range = { -1.0f, 1.0f }   // min, max
    };

    return env_spec;
}

std::shared_ptr<const anet::rl::SingleResetResult> CartPoleEnv::Reset(anet::rl::RunMode mode)
{
    anet::ProfileRange r("CartPoleEnv::Reset");

    if (anet::rl::IsTrain(mode)) {
        const float d = 0.05f;
        x_ =         rnd_->Uniform(-d, d);
        x_dot_ =     rnd_->Uniform(-d, d);
        theta_ =     rnd_->Uniform(-d, d);
        theta_dot_ = rnd_->Uniform(-d, d);
    } else {
        // 評価モードでは初期状態固定
        x_ = 0.0f;
        x_dot_ = 0.0f;
        theta_ = 0.0f;
        theta_dot_ = 0.0f;
    }
    
    //x = 0.2f;
    //x_dot = 0.2f;
    //theta = -0.05f;
    //theta_dot = 0.05;// -1.0f * 0.5;// *0.5;

    done_ = false;
    truncated_ = false;
    episode_start_ = true;
    step_count_ = 0;

    auto result = std::make_shared<anet::rl::SingleResetResult>(
        anet::rl::SingleState {
            .obs = { anet::rl::ObsKeys::kVector, torch::tensor({ x_, x_dot_, theta_, theta_dot_ }, obs_opt_) },
            .done = done_,
            .truncated = truncated_,
            .episode_start = episode_start_
        }
    );
    return result;
}

std::shared_ptr<const anet::rl::SingleStepResult> CartPoleEnv::Step(int64_t action, anet::rl::RunMode mode)
{
    anet::ProfileRange r("CartPoleEnv::Step");

    episode_start_ = false;

    // 力の符号（0右=+、1:左=-）
    float force = (action == 1) ? -force_mag : force_mag;
    //float force = force_mag;  // 動作確認用

    // 運動方程式
    float costheta = std::cos(theta_);
    float sintheta = std::sin(theta_);

    // --- 拘束反力モデル（完全拘束） ---
    bool hit_wall = false;
    if (x_ <= -limit_x && force < 0) {  // 左壁＋左向き力
        hit_wall = true;
        force = 0.0f;
        x_ = -limit_x;
        x_dot_ = 0.0f;
    } else if (x_ >= limit_x && force > 0) {  // 右壁＋右向き力
        hit_wall = true;
        force = 0.0f;
        x_ = limit_x;
        x_dot_ = 0.0f;
    }

    float temp = (force + polemass_length * theta_dot_ * theta_dot_ * sintheta) / total_mass;
    float thetaacc = (gravity * sintheta + 1 * costheta * temp) /
        (length * (4.0f / 3.0f - mass_pole * costheta * costheta / total_mass));
    float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    //thetaacc = 0;
    //thetaacc = -100;

    // リミット時、壁に押している場合は xacc を 0 に上書き
    if (hit_wall) xacc = 0.0f;

    // 更新
    x_ += tau * x_dot_;
    x_dot_ += tau * xacc;
    theta_ += tau * theta_dot_;
    theta_dot_ += tau * thetaacc;

    // clamp
    theta_dot_ = std::clamp(theta_dot_, -3.0f, 3.0f);
    x_dot_ = std::clamp(x_dot_, -2.0f, 2.0f);
    //theta_ = std::clamp(theta_, -limit_theta * deg, limit_theta * deg);

    //wxLogInfo("STEP=%d x=%f theta=%f hit_wall=%d force=%f x_dot=%f theta_dot=%f, xacc=%f thetaacc=%f",
        //step_count, x, theta, hit_wall, force, x_dot, theta_dot,xacc, thetaacc);

    // ステップ完了
    step_count_++;

    // 終了条件はステップ数のみ
    //bool done = (step_count >= 500);

    // 終了条件は下半分まで倒れたor500ステップを超えた
    float theta_deg = theta_ * 180.0f / M_PI;
    done_ = (x_ < -limit_x || x_ > limit_x || theta_deg < -limit_theta || theta_deg > limit_theta);
    //wxLogDebug("done_:[%d] a=%d x:[%f %f], theta_deg:[%f %f]", done_, action, x_, limit_x, theta_deg, limit_theta);
    //if (step_count >= limit_step) { done_ = true; }

    // 報酬: 角度安定性 + 速度安定補正
    //float reward = std::cos(theta) - 0.05f * std::abs(x_dot) - 0.01f * std::abs(theta_dot);
    //if (reward < 0.0f) reward = 0.0f;  // 安定しない場合は0報酬

    //float reward = 0.0f;
    //if (std::cos(theta) > 0.0f) {       // ポールが水平より上（-90° < θ < +90°）なら報酬を与える
        //reward = std::cos(theta) - 0.4f * std::abs(x) / x_limit;
    //}

    //// θ=0（直立）で1.0、真横で0.0、下向きでは0
    //float reward = 0.5f * (std::cos(theta) + 1.0f);  // [-1,1] → [0,1]
    //reward *= std::exp(-0.05f * std::abs(x_dot));    // 横速度で減衰（常に正）

    //float upright = std::max(0.0f, std::cos(theta));       // 立ってるほど高い
    //float stable = 1.0f / (1.0f + std::abs(theta_dot));   // 揺れが少ないほど高い
    //float reward = 10.0f * (0.5f + 0.5f * upright * stable);

    //float reward = done ? -1.0f : 1.0f;
    //float reward = done ? 0.0f : 1.0f;


    float reward = reward_scale * (1.0f
        - 0.01f * (std::abs(theta_deg) / 90.0f)   // 姿勢
        - 0.002f * (std::abs(x_) / limit_x));      // 位置

    // 終了条件ごとに分岐
    if (theta_deg < -limit_theta || theta_deg > limit_theta || x_ < -limit_x || x_ > limit_x) {
        // 倒立失敗
        reward = -done_reward;   // ← ペナルティ
    } else if (step_count_ >= config_.limit_step) {
        // 時間切れ成功
        reward = done_reward;
        truncated_ = true;
    }

    auto result = std::make_shared<anet::rl::SingleStepResult>(
        reward,
        anet::rl::SingleState {
            .obs = { anet::rl::ObsKeys::kVector, torch::tensor({ x_, x_dot_, theta_, theta_dot_ }, obs_opt_) },
            .done = done_,
            .truncated = truncated_,
            .episode_start = episode_start_
        });

    return result;
}

CartPoleEnvFactory::CartPoleEnvFactory()
{
    ;
}

std::shared_ptr<anet::rl::SingleDiscreteEnv> CartPoleEnvFactory::CreateSingleEnv(
    const anet::ConfigData& config_data,
    const torch::Device& device, std::optional<anet::seed_t> seed,
    const std::string& config_prefix)
{
    CartPoleEnvConfig config(config_data, config_prefix);
    return std::make_shared<CartPoleEnv>(config, device, seed);
}

ANET_REGISTER_ENV_FACTORY(CartPoleEnvFactory);

