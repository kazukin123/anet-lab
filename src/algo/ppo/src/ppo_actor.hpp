#pragma once

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include "anet/random.hpp"
#include "anet/rl.hpp"
#include "anet/nn.hpp"
#include "ppo_config.hpp"
#include "ppo_network_model.hpp"

namespace anet::rl::ppo {

    /// 方策から行動を選び、Learner が rollout にためる値を行動の情報（info）に載せる。
    class Actor final : public anet::rl::Actor {
    public:
        static constexpr char kLogProbKey[] = "log_prob";            ///< [lane]。選んだ行動の、選んだ時点の log 確率。
        static constexpr char kValueKey[] = "value";                 ///< [lane]。選んだ時点の観測の value。
        static constexpr char kPolicyVersionKey[] = "policy_version"; ///< CPU のスカラー（int64）。行動を選んだ方策の世代。
        static constexpr char kInputKeyPrefix[] = "input.";          ///< Network へ渡した入力のキーに付ける接頭辞。
    public:
        Actor(const ActorConfig& config, std::shared_ptr<std::shared_mutex> mutex, std::shared_ptr<NetworkModel> model,
            torch::Device device, anet::seed_t seed);
        std::shared_ptr<BatchActionInfo> MakeAction(const StepCounts& step, const anet::rl::BatchState& state,
            const MakeActionOptions& options = {}) const override;
        void Sync() override;
    private:
        const ActorConfig config_;
        std::shared_ptr<std::shared_mutex> mutex_; ///< Agent が持つ Network のロック。
        std::shared_ptr<NetworkModel> model_;      ///< Agent が所有する Resource。Learner が更新する Network を持つ。
        std::shared_ptr<anet::nn::Network> network_; ///< 行動選択に使う Network。共有か、この Actor だけの写し。
        const torch::Device device_;
        mutable anet::RandomGenerator random_; ///< この Actor だけが消費する乱数。
        uint64_t synced_version_ = 0;          ///< 写しが、最後に写した時点の方策の世代。
    };

} // namespace anet::rl::ppo
