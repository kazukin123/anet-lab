#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include "anet/config.hpp"
#include "anet/schedule.hpp"
#include "anet/nn.hpp"

namespace anet::rl::ppo {

    /// Agent の class id。設定の接頭辞（`PPOAgent.*`）でもある。
    inline constexpr char kAgentClassId[] = "PPOAgent";

    /// Actor カタログの 1 項目（`PPOAgent.actor.[<Actor キー>].*`）。
    struct ActorConfig {
        static constexpr char kActionModeSample[] = "sample";
        static constexpr char kActionModeGreedy[] = "greedy";

        std::string action_mode = kActionModeSample; ///< sample は方策から抽選し、greedy は確率が最大の行動を選ぶ。
        bool clone_model = false; ///< Network の写しを持つか。偽なら Learner と Network を共有する。
    };

    /// Learner の設定の値（`PPOAgent.learner.*`）。
    struct LearnerConfig {
        struct KlStop {
            struct Epoch {
                float threshold = 0.015f; ///< epoch の最後のミニバッチの approx KL がこれを超えたら、残りの epoch をやめる。
            };
            std::string mode = kKlStopNone;
            Epoch epoch;
        };

        static constexpr char kAdvantageNormNone[] = "none";
        static constexpr char kAdvantageNormMinibatch[] = "minibatch";
        static constexpr char kTruncationBootstrap[] = "bootstrap";
        static constexpr char kTruncationTerminal[] = "terminal";
        static constexpr char kKlStopNone[] = "none";
        static constexpr char kKlStopEpoch[] = "epoch";

        int epochs = 4;          ///< 1 つの rollout を一巡する回数 K。
        int minibatch_count = 4; ///< 1 epoch のミニバッチの数 M。
        float gamma = 0.99f;     ///< 割引率 γ。
        float gae_lambda = 0.95f; ///< GAE の λ。
        anet::ProfiledValueConfig<double> learning_rate{ .value = 2.5e-4, .min_value = 0.0 };
        double adam_eps = 1e-5;
        double weight_decay = 0.0;
        bool use_fused_optimizer = true;
        anet::ProfiledValueConfig<float> ratio_clip_range{
            .value = 0.2f, .min_value = std::numeric_limits<float>::denorm_min() }; ///< 比の clip の幅 ε。
        bool use_value_clip = true;
        float value_clip_range = 0.2f;
        anet::ProfiledValueConfig<float> entropy_coef{ .value = 0.01f, .min_value = 0.0f };
        float value_coef = 0.5f;
        bool use_grad_clip = true;
        float grad_clip_max_norm = 0.5f;
        std::string advantage_norm_mode = kAdvantageNormMinibatch;
        std::string truncation_mode = kTruncationBootstrap;
        KlStop kl_stop;
    };

    /// `PPOAgent` の設定。読み取りと、設定だけで決まる検証をコンストラクタで行う。
    struct PPOAgentConfig : public anet::Config {
        struct Rollout {
            int length = 128; ///< rollout の長さ T（intake の数）。
        };
        struct HeadPart {
            anet::nn::WeightInitConfig init;
        };
        struct Head {
            HeadPart policy;
            HeadPart value;
        };

        explicit PPOAgentConfig(const anet::ConfigData& config_data = anet::EmptyConfigData,
            const std::string& config_prefix = "");

        /// lane 数と突き合わせる検査。rollout の遷移の数がミニバッチの数で割り切れることなどを確かめる。
        void ValidateLaneCount(int64_t lane_count) const;

        Rollout rollout;
        LearnerConfig learner;
        Head head;
        std::map<std::string, ActorConfig> actor;
        anet::nn::NetworkGraphVizConfig nn_viz;
        std::string auto_load_file;
    };

} // namespace anet::rl::ppo
