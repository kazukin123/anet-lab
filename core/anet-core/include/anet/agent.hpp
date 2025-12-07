// anet/agent.hpp
#pragma once

#include <memory>
#include <optional>
#include <mutex>
#include "anet/rl.hpp"

namespace anet::rl {

    // 環境のステップに同期して更新する Agent 基底クラス
    template<typename ConfigT>
    class StepBasedAgent : public Agent, public anet::RandomHolder {
    public:
        StepBasedAgent(ConfigT config, torch::Device device,
            std::shared_ptr<anet::rl::Notifier> notifier,
            std::optional<seed_t> seed = std::nullopt)
            : RandomHolder(seed), config_(config), notifier_(notifier),device_(device)
        {
        }

        virtual ~StepBasedAgent() = default;
    protected:
        mutable std::shared_mutex mutex_;
    protected:
        // Resource（Agentが管理すべき領域）
        ConfigT config_;
        torch::Device device_;
        std::shared_ptr<anet::rl::Notifier> notifier_;

    };

} // namespace anet::rl