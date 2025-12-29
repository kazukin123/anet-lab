#include <nlohmann/json.hpp>
#include "anet/rl.hpp"

namespace anet::rl {

    struct EnvEvalResult {
        float average_return = 0.0f;
        float average_length = 0.0f;
        float max_return = 0.0f;
        float min_return = 0.0f;

        anet::json ToJson() const;
    };

    /// @brief 環境の本質的難易度（reward/episode_length分布）を推定する
    /// 行動は ActionSpec に従う純ランダム。Agent不要。
    /// @param env 評価対象の BatchEnvironment
    /// @param n_episodes 評価エピソード数
    /// @param seed 固定したい場合の乱数シード
    /// @todo 将来: 複数モード(RandomPolicy/ZeroActionなど)追加
    EnvEvalResult EvaluateEnvironmentDifficulty(
        anet::rl::BatchEnv& env,int n_episodes = 100, std::int64_t seed = -1
    );
}   // namespace anet::rl
