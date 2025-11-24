#include <algorithm>
#include "anet/eval_env.hpp"

namespace anet::rl {

    // =============================================================
    // ローカル関数：ActionSpecから汎用ランダム行動を生成
    // =============================================================
    static torch::Tensor RandomActionFromSpec(
        const ActionSpec& spec,
        RandomGenerator& rng)
    {
        if (spec.is_discrete) {
            int n = spec.value_labels.size();  // 離散アクション数
            ANET_ASSERT(n > 0);
            int64_t a = rng.RandInt(0, n - 1);
            return torch::tensor({ a }, torch::kInt64);
        }

        // -------- continuous action (Box) --------
        ANET_ASSERT(!spec.dims.empty());
        const int dim = (int)spec.dims.size();

        torch::Tensor act = torch::empty({ dim }, torch::kFloat32);
        float* ptr = act.data_ptr<float>();

        for (int i = 0; i < dim; ++i) {
            const auto& d = spec.dims[i];
            float r = rng.Uniform01(); // [0,1)
            ptr[i] = d.min_value + r * (d.max_value - d.min_value);
        }
        return act;
    }

    // =============================================================
    // EnvEvalResult::ToJson()
    // =============================================================
    nlohmann::json EnvEvalResult::ToJson() const
    {
        nlohmann::json j;
        j["average_return"] = average_return;
        j["average_length"] = average_length;
        j["max_return"] = max_return;
        j["min_return"] = min_return;
        return j;
    }

    // =============================================================
    // EvaluateEnvironmentDifficulty()
    // =============================================================
    EnvEvalResult EvaluateEnvironmentDifficulty(
        BatchEnvironment& env,
        int n_episodes,
        int64_t seed)
    {
        auto rng = RandomGenerator(seed, false, false);

        EnvEvalResult R;
        std::vector<float> returns;
        std::vector<int> lengths;
        returns.reserve(n_episodes);
        lengths.reserve(n_episodes);

        const ActionSpec action_spec = env.GetSpec().action_spec;

        for (int ep = 0; ep < n_episodes; ++ep) {
            BatchState s = env.Reset(RunMode::Eval1);

            float ep_ret = 0.0f;
            int   ep_len = 0;

            while (true) {
                torch::Tensor action = RandomActionFromSpec(action_spec, rng);
                BatchStepResult step = env.DoStep(action, RunMode::Eval1);

                ep_ret += step.reward.item<float>();
                ep_len++;

                bool done = step.next_state.done.item<bool>();
                bool truncated = step.next_state.truncated.item<bool>();
                if (done || truncated) break;

                s = step.next_state;
            }

            returns.push_back(ep_ret);
            lengths.push_back(ep_len);
        }

        // 集計
        if (!returns.empty()) {
            float sum_r = 0.0f;
            float sum_l = 0.0f;
            R.max_return = -1e30f;
            R.min_return = 1e30f;

            for (size_t i = 0; i < returns.size(); ++i) {
                float v = returns[i];
                sum_r += v;
                sum_l += lengths[i];
                R.max_return = std::max(R.max_return, v);
                R.min_return = std::min(R.min_return, v);
            }
            R.average_return = sum_r / returns.size();
            R.average_length = sum_l / lengths.size();
        }

        return R;
    }

} // namespace anet::rl
