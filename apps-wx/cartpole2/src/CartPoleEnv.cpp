#include "CartPoleEnv.hpp"
#include "app.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <wx/log.h>

// 定数
const int limit_step = 200;  // 終了条件
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

CartPoleEnv::CartPoleEnv(std::shared_ptr<anet::RandomGenerator> rnd) : RandomHolder(rnd)
{
    // パラメータ記録
    nlohmann::json params = {
        {"limit_step", limit_step},
    };
    anet::MetricsLogger::Instance()->LogJson("env/params", params);
    anet::MetricsLogger::Instance()->Flush();

    Reset();
}

anet::rl::EnvSpec CartPoleEnv::GetSpec() const
{
    anet::rl::StateSpec state = {
        {4},   // shape
        {      //dims
            { {0}, -limit_x - limit_x_dot * tau, limit_x + limit_x_dot * tau,     "x"},         // dims[0] coords, min, max, name, desc
            { {1}, -limit_x_dot, limit_x_dot,           "x_dot"},     // dims[1] coords, min, max, name, desc
            { {2}, -limit_theta * deg - limit_theta_dot * tau, limit_theta * deg + limit_theta_dot * tau, "theta"}, // dims[2] coords, min, max, name, desc
            { {3}, -limit_theta_dot, limit_theta_dot,           "theta_dot"}  // dims[3] coords, min, max, name, desc
        }
    };
    anet::rl::ActionSpec action = {
        true,   // is_discreate
        { "left", "right"}, // value_labels
        { // dims
            { 0, 1, "force" }  // min, max, name
        }
    };
    anet::rl::EnvSpec env_spec = {
        state,
        action,
        { -1, 1 }   //reward_range: min, max
    };

    return env_spec;
}

anet::rl::SingleState CartPoleEnv::Reset(anet::rl::RunMode mode) {
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

    return {
        torch::tensor({ x_, x_dot_, theta_, theta_dot_ }), // (4)
        done_,
        truncated_,
        episode_start_
    };
}

anet::rl::SingleStepResult CartPoleEnv::Step(int64_t action, anet::rl::RunMode mode) {
    episode_start_ = false;

    // 力の符号（1:右=+、0:左=-）
    float force = (action == 1) ? force_mag : -force_mag;
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
    } else if (step_count_ >= limit_step) {
        // 時間切れ成功
        reward = done_reward;
        truncated_ = true;
    }

    anet::rl::SingleStepResult result {
        reward,
        {
            torch::tensor({ x_, x_dot_, theta_, theta_dot_ }), // obs (4)
            done_,
            truncated_,
            episode_start_
        },
    };
    return result;
}

CartPoleEnvFactory::CartPoleEnvFactory(std::shared_ptr<anet::RandomGenerator> rnd)
    : RandomHolder(rnd)
{
    ;
}

std::unique_ptr<anet::rl::SingleDiscreteEnv> CartPoleEnvFactory::Create()
{
    return std::make_unique<CartPoleEnv>(rnd_);
}

