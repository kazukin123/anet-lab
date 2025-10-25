#include <tuple>
#include <torch/torch.h>
#include <random>
#include <wx/log.h>

#include "anet/rl.hpp"

void evaluateEnvironment(anet::rl::Environment& env, int num_actions, int num_trials = 1000,
    float max_possible_reward = 1.0f, int max_steps_per_episode = 500)
{
    std::uniform_int_distribution<int> action_dist(0, num_actions - 1);
    std::random_device rd;
    std::mt19937 gen(rd());

    float reward_sum = 0.0f;
    float reward_sq_sum = 0.0f;
    float min_reward = std::numeric_limits<float>::max();
    float max_reward = -std::numeric_limits<float>::max();
    int short_fail_count = 0;

    for (int i = 0; i < num_trials; ++i) {
        torch::Tensor state = env.Reset();
        float total_reward = 0.0f;
        int steps = 0;

		while (true) {  // ランダム方策でエピソード完了まで実行して評価
            int action_int = action_dist(gen);
            auto action = torch::tensor({ action_int }, torch::kLong);
            auto [next_state, reward, done, truncated] = env.DoStep(action);
            total_reward += reward;
            steps++;
            if (done || truncated) break;
        }

        reward_sum += total_reward;
        reward_sq_sum += total_reward * total_reward;
        min_reward = std::min(min_reward, total_reward);
        max_reward = std::max(max_reward, total_reward);
        if (steps < 5) short_fail_count++;
    }

    float mean = reward_sum / num_trials;
    float var = reward_sq_sum / num_trials - mean * mean;
    float stddev = std::sqrt(std::max(var, 0.0f));
    float fail_rate = static_cast<float>(short_fail_count) / num_trials;

    // --- ▼ 非依存スケールに正規化 ----------------------------
    // 1. 環境固有スケールを除去：最大理論報酬で割る
    float normalized_mean = mean / (max_possible_reward * max_steps_per_episode);
    float normalized_std = stddev / (max_possible_reward * max_steps_per_episode);
    float normalized_min = min_reward / (max_possible_reward * max_steps_per_episode);
    float normalized_max = max_reward / (max_possible_reward * max_steps_per_episode);

    // --- ▼ 難易度スコア算出 ----------------------------
    // 1. 平均スコア: 0.1〜0.3が「ちょうどよい難易度」
    float mean_score = std::clamp((normalized_mean - 0.05f) / (0.35f - 0.05f), 0.0f, 1.0f);

    // 2. 変動係数 (CV) スコア: 0.3〜0.6が理想
    float cv = (normalized_mean > 1e-6f) ? (normalized_std / normalized_mean) : 10.0f;
    float cv_score = std::exp(-std::pow((cv - 0.5f) / 0.3f, 2));  // ガウス型評価

    // 3. 早期失敗率スコア：0.5〜0.8：まずまず健全
    float fail_score = 1.0f - std::clamp(fail_rate * 2.0f, 0.0f, 1.0f);

    // 4. 総合難易度（加重平均）:0.4～0.7が理想
    float difficulty_index = 0.5f * mean_score + 0.3f * cv_score + 0.2f * fail_score;
    // --------------------------------------------------------

    wxLogInfo("=== Random Policy Evaluation ===");
    wxLogInfo("Trials: %d", num_trials);
    wxLogInfo("Mean reward: %.2f (normalized: %.3f)", mean, normalized_mean);
    wxLogInfo("Stddev: %.2f (normalized: %.3f, CV=%.2f)", stddev, normalized_std, cv);
    wxLogInfo("Min: %.1f (%.3f), Max: %.1f (%.3f)", min_reward, normalized_min, max_reward, normalized_max);
    wxLogInfo("Early fail (<5 steps): %d%%", (int)(fail_rate * 100));
    wxLogInfo("Mean score: %.2f", mean_score);
    wxLogInfo("CV score: %.2f", cv_score);
    wxLogInfo("Fail score: %.2f", fail_score);
    wxLogInfo("Difficulty index: %.2f", difficulty_index);

#ifdef WX_APP_COMPATIBLE
    nlohmann::json params = {
        {"mean_reward_raw", mean},
        {"random_reward_stddev", stddev},
        {"random_reward_min", min_reward},
        {"random_reward_max", max_reward},
        {"random_fail_rate", fail_rate},
        {"normalized_mean", normalized_mean},
        {"normalized_stddev", normalized_std},
        {"mean_score", mean_score},
        //{"mean_score_comment", "0.0～0.2：極めて難しい / 0.4～0.7：適正 / 0.7以上: 容易"},
        {"cv_score", cv_score},
        //{"cv_score_comment", "0.4～0.6：理想的 / 0.7以上：運ゲー"},
        {"fail_score", fail_score},
        //{"fail_score_comment", "0.0〜0.4：初期条件が厳しすぎる / 0.5〜0.8：まずまず健全 / 0.8〜1.0：安定して挑戦できる"},
        {"difficulty_index", difficulty_index},
        //{"difficulty_comment", "0.2以下：学習不能級 / 0.2〜0.4：過酷 / 0.4〜0.7：適正 / 0.7以上：容易"}
    };
    wxGetApp().logJson("env/difficulty", params);
#endif
}

