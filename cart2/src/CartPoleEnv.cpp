#include "CartPoleEnv.hpp"
#include "app.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <wx/log.h>

// 定数
const float x_limit = 2.4f;
const float theta_limit = 90.0f; // 12.0f;
const float gravity = 9.8f;
const float masscart = 1.0f;
const float mass_pole = 0.10f;
const float total_mass = masscart + mass_pole;
const float length = 0.5f;
const float polemass_length = mass_pole * length;
const float force_mag = 30.0f;  // 10.0f 30.0f
const float tau = 0.02f;    //0.02f 0.01f
const float reward_scale = 1.0f;  // 2 10  20

const int max_steps = 500;  // 終了条件

CartPoleEnv::CartPoleEnv() {
    // パラメータ記録
    nlohmann::json params = {
        {"max_steps", max_steps},
    };
    wxGetApp().logJson("env/params", params);
    wxGetApp().flushMetricsLog();

    Reset();
}

torch::Tensor CartPoleEnv::Reset(anet::rl::RunMode mode) {

    if (anet::rl::IsTrain(mode)) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::random_device rd;
        std::mt19937 gen(rd());

        x = dist(gen) * 0.05f;
        x_dot = dist(gen) * 0.05f;
        theta = dist(gen) * 0.05f;
        theta_dot = dist(gen) * 0.05f;
    }else {
        // 評価モードでは初期状態固定
        x = 0;
        x_dot = 0;
        theta = 0;
        theta_dot = 0;
    }
    
    //x = 0.2f;
    //x_dot = 0.2f;
    //theta = -0.05f;
    //theta_dot = 0.05;// -1.0f * 0.5;// *0.5;

    step_count = 0;

    return GetState();
}

torch::Tensor CartPoleEnv::GetState() const {
    return torch::tensor({ x, x_dot, theta, theta_dot }).unsqueeze(0);
}

anet::rl::EnvResponse CartPoleEnv::DoStep(const torch::Tensor& action_tensor, anet::rl::RunMode mode) {
    int action = action_tensor.item<int>();

    // 力の符号（右:+、左-）
    float force = (action == 1) ? force_mag : -force_mag;
    //float force = force_mag;  // 動作確認用

    // 運動方程式
    float costheta = std::cos(theta);
    float sintheta = std::sin(theta);

    // --- 拘束反力モデル（完全拘束） ---
    bool hit_wall = false;
    if (x <= -x_limit && force < 0) {  // 左壁＋左向き力
        hit_wall = true;
        force = 0.0f;
        x = -x_limit;
        x_dot = 0.0f;
    }
    else if (x >= x_limit && force > 0) {  // 右壁＋右向き力
        hit_wall = true;
        force = 0.0f;
        x = x_limit;
        x_dot = 0.0f;
    }

    float temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
    float thetaacc = (gravity * sintheta + 1 * costheta * temp) /
        (length * (4.0f / 3.0f - mass_pole * costheta * costheta / total_mass));
    float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    //thetaacc = 0;
    //thetaacc = -100;

    // リミット時、壁に押している場合は xacc を 0 に上書き
    if (hit_wall) xacc = 0.0f;

    // 更新
    x += tau * x_dot;
    x_dot += tau * xacc;
    theta += tau * theta_dot;
    theta_dot += tau * thetaacc;

    // 速度・角速度に上限を設ける
    theta_dot = std::clamp(theta_dot, -3.0f, 3.0f);
    x_dot = std::clamp(x_dot, -2.0f, 2.0f);

    //wxLogInfo("STEP=%d x=%f theta=%f hit_wall=%d force=%f x_dot=%f theta_dot=%f, xacc=%f thetaacc=%f",
        //step_count, x, theta, hit_wall, force, x_dot, theta_dot,xacc, thetaacc);

    // ステップ完了
    step_count++;

    // 終了条件はステップ数のみ
    //bool done = (step_count >= 500);

    // 終了条件は下半分まで倒れたor500ステップを超えた
    float theta_deg = theta * 180.0f / M_PI;
    bool done = (x < -x_limit || x > x_limit || theta_deg < -theta_limit || theta_deg > theta_limit);
    if (step_count >= 500) { done = true; }

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
        - 0.002f * (std::abs(x) / x_limit));      // 位置

    // 終了条件ごとに分岐
	bool truncated = false;
    if (theta_deg < -90.0f || theta_deg > 90.0f ||
        x < -x_limit || x > x_limit) {
        // 倒立失敗
        reward = -reward_scale;   // ← ペナルティ
    }
    else if (step_count >= 500) {
        // 時間切れ成功
        reward = +reward_scale;   // ← ボーナス
        truncated = true;
    }

    return { GetState(), reward, done, truncated };
}
