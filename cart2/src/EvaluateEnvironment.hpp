#include <tuple>
#include <torch/torch.h>
#include <random>
#include <wx/log.h>

void evaluateEnvironment(Env& env, int num_actions, int num_trials = 100) {
    std::uniform_int_distribution<int> action_dist(0, num_actions - 1);
    std::random_device rd;
    std::mt19937 gen(rd());

    float reward_sum = 0.0f;
    float reward_sq_sum = 0.0f;
    float min_reward = std::numeric_limits<float>::max();
    float max_reward = -std::numeric_limits<float>::max();
    int short_fail_count = 0;

    for (int i = 0; i < num_trials; ++i) {
        torch::Tensor state = env.reset();
        float total_reward = 0.0f;
        int steps = 0;

		while (true) {  // ランダム方策でエピソード完了まで実行して評価
            int action = action_dist(gen);
            auto [next_state, reward, done] = env.step(action);
            total_reward += reward;
            steps++;
            if (done) break;
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

    // --- 難易度スコア算出 ---
    // 1. reward_meanが高いほど良い（50〜150を理想）
    float mean_score = std::clamp((mean - 10.0f) / (150.0f - 10.0f), 0.0f, 1.0f);
    // 2. CV（変動係数）が1.0に近いと混沌、0.3〜0.6が理想
    float cv = (mean > 1e-6f) ? (stddev / mean) : 10.0f;
    float cv_score = std::exp(-std::pow((cv - 0.5f) / 0.3f, 2));  // ガウス型評価
    // 3. 早期失敗率が低いほど良い
    float fail_score = 1.0f - std::clamp(fail_rate * 2.0f, 0.0f, 1.0f);
    // 総合難易度（加重平均）
    float difficulty_index = 0.5f * mean_score + 0.3f * cv_score + 0.2f * fail_score;

    wxLogInfo("=== Random Policy Evaluation ===");
    wxLogInfo("Trials: %d", num_trials);
    wxLogInfo("Mean reward: %.2f", mean);
    wxLogInfo("Stddev: %.2f (CV=%.2f)", stddev, cv);
    wxLogInfo("Min: %.1f, Max: %.1f", min_reward, max_reward);
    wxLogInfo("Early fail (<5 steps): %d%%", (int)(fail_rate * 100));
    wxLogInfo("Difficulty index: %.2f", difficulty_index);

#ifdef WX_APP_COMPATIBLE
    // パラメータ記録
    nlohmann::json params = {
        {"random_reward_mean",   mean},
        {"random_reward_stddev", stddev},
        {"random_reward_min",    min_reward},
        {"eps_min",              max_reward},
        {"random_fail_rate",     (float)short_fail_count / num_trials},
        {"difficulty_index",     difficulty_index},
    };
    wxGetApp().logJson("env/difficulty", params);
    //wxGetApp().logScalar("env/random_reward_mean", 0, mean);            // ランダム方策の平均報酬（≒環境の基本難易度） <10：学習不能級、10〜50：難しい、50〜150：通常、>150：容易（ランダムでも安定生存）
    //wxGetApp().logScalar("env/random_reward_stddev", 0, stddev);        // 報酬分散（低すぎると探索が不十分）大きいなら運ゲー
    //wxGetApp().logScalar("env/random_reward_min", 0, min_reward);       // 最悪ケースの報酬
    //wxGetApp().logScalar("env/random_reward_max", 0, max_reward);       // ベストケース報酬
    //wxGetApp().logScalar("env/random_fail_rate", 0, (float)short_fail_count / num_trials);      // 5step未満で終了した割合（高いと難易度高すぎ）
    //wxGetApp().logScalar("env/difficulty_index", 0, difficulty_index);  // 0.0〜0.2：学習不可能級、0.2〜0.4：過酷、0.4〜0.7：丁度よい、0.7〜1.0：安易
    //wxGetApp().flushMetricsLog();
#endif
}
