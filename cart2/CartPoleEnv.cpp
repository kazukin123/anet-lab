#include "CartPoleEnv.hpp"
#include <cmath>
#include <algorithm>
#include <random>

// 定数
const float gravity = 9.8f;
const float masscart = 1.0f;
const float masspole = 0.1f;
const float total_mass = masscart + masspole;
const float length = 0.5f;
const float polemass_length = masspole * length;
const float force_mag = 5.0f;
const float tau = 0.02f;    //0.02f 0.01f

CartPoleEnv::CartPoleEnv()
{
    reset();
}

torch::Tensor CartPoleEnv::reset() {
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    std::random_device rd;
    std::mt19937 gen(rd());

    x = dist(gen);
    x_dot = dist(gen);
    theta = dist(gen);
    theta_dot = dist(gen);
    step_count = 0;

    return get_state();
}

torch::Tensor CartPoleEnv::get_state() const {
    return torch::tensor({ x, x_dot, theta, theta_dot }).unsqueeze(0);
}

std::tuple<torch::Tensor, float, bool> CartPoleEnv::step(int action) {

    // カートに加える力
    float force = (action == 1) ? force_mag : -force_mag;

    // 運動方程式
    float costheta = std::cos(theta);
    float sintheta = std::sin(theta);
    float temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
    float thetaacc = (gravity * sintheta - costheta * temp) /
        (length * (4.0f / 3.0f - masspole * costheta * costheta / total_mass));
    float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    // 更新
    x += tau * x_dot;
    x_dot += tau * xacc;
    theta += tau * theta_dot;
    theta_dot += tau * thetaacc;

    // 速度・角速度に上限を設ける
    theta_dot = std::clamp(theta_dot, -3.0f, 3.0f);
    x_dot = std::clamp(x_dot, -2.0f, 2.0f);

    // x方向リミットを「終了条件ではなく物理クランプ」として実装
    const float x_limit = 2.4f;
    if (x < -x_limit) x = -x_limit;
    if (x > x_limit) x = x_limit;

    // ステップ完了
    step_count++;

    // 報酬: 角度安定性 + 速度安定補正
    //float reward = std::cos(theta) - 0.05f * std::abs(x_dot) - 0.01f * std::abs(theta_dot);
    //if (reward < 0.0f) reward = 0.0f;  // 安定しない場合は0報酬
    float reward = 0.0f;
    if (std::cos(theta) > 0.0f) {       // ポールが水平より上（-90° < θ < +90°）なら報酬を与える
        reward = std::cos(theta) - 0.01f * std::abs(x);
    }

    // 総報酬更新
    total_reward += reward;

    // 終了条件はステップ数のみ
    bool done = (step_count >= 500);

    return { get_state(), reward, done };
}

//float CartPoleEnv::get_x() const { return x; }
//float CartPoleEnv::get_theta() const { return theta; }
