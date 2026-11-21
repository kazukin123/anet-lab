#include "anet/replay_experience.hpp"

#include <algorithm>
#include "anet/diag.hpp"

namespace anet::rl {

// ===========================================================================
// Queue
// ===========================================================================

void ExperienceQueue::Push(const QueueRecord& record)
{
    buffer_.push_back(record);
}

void ExperienceQueue::Pop(size_t k)
{
    if (k >= buffer_.size()) {
        buffer_.clear();
    } else {
        buffer_.erase(buffer_.begin(), buffer_.begin() + k);
    }
}

std::vector<QueueRecord> ExperienceQueue::Peek(size_t k) const
{
    size_t count = std::min(k, buffer_.size());
    return std::vector<QueueRecord>(buffer_.begin(), buffer_.begin() + count);
}

size_t ExperienceQueue::Size() const
{
    return buffer_.size();
}

void ExperienceQueue::Clear()
{
    buffer_.clear();
}


// ===========================================================================
// Concrete Controller & Builder
// ===========================================================================

namespace {

    class NStepQueueController : public ExperienceQueueController {
    public:
        explicit NStepQueueController(int n_step) : n_step_(n_step)
        {
        }

        std::vector<ExperienceSequence> ExtractSequences(ExperienceQueue& queue) override
        {
            std::vector<ExperienceSequence> sequences;
            while (queue.Size() > 0) {
                auto peeked = queue.Peek(n_step_);
                bool has_terminal = false;
                size_t term_idx = 0;

                // 系列内に終端(done/truncated)があるかチェック
                for (size_t i = 0; i < peeked.size(); ++i) {
                    if (peeked[i].done || peeked[i].truncated) {
                        has_terminal = true;
                        term_idx = i;
                        break;
                    }
                }

                if (has_terminal) {
                    // 終端までで系列を確定し、先頭1要素分だけ進める(Pop1)
                    ExperienceSequence seq(peeked.begin(), peeked.begin() + term_idx + 1);
                    sequences.push_back(seq);
                    queue.Pop(1);
                } else if (peeked.size() == static_cast<size_t>(n_step_)) {
                    // N-Step揃ったので確定
                    sequences.push_back(peeked);
                    queue.Pop(1);
                } else {
                    // まだ未来が揃っていないので待機
                    break;
                }
            }
            return sequences;
        }
    private:
        int n_step_;
    };

    class DefaultExperienceBuilder : public ReplayExperienceBuilder {
    public:


        ReplayExperience Build(const ExperienceSequence& sequence, std::span<const float> gammas) const override
        {
            ReplayExperience exp{};
            exp.target_return = 0.0f;
            exp.terminal = false;

            int n = static_cast<int>(sequence.size());

            // シーケンスの末尾がダミーなら、それは実データではないので長さに含めない
            if (sequence.back().is_dummy) {
                n -= 1;
                // ダミー起因（＝元のステップがTruncated）なので、ブートストラップを継続(terminal=false)
                exp.terminal = false;
            } else {
                // 本当のゲームオーバー
                exp.terminal = sequence.back().done;
                exp.truncated = sequence.back().truncated && !sequence.back().done;
            }

            exp.actual_n_steps = n;
            if (gammas.empty()) ANET_SYSTEM_ERROR("Replay gammas: count=0; expected at least one component.");
            exp.extra_component_returns.assign(gammas.size() - 1, 0.0f);

            // 割引報酬和の計算 (逆順)
            for (int i = n - 1; i >= 0; --i) {
                exp.target_return = sequence[i].reward + gammas.front() * exp.target_return;
                for (size_t component = 1; component < gammas.size(); ++component) {
                    exp.extra_component_returns[component - 1] = sequence[i].extra_component_rewards.at(component - 1)
                        + gammas[component] * exp.extra_component_returns[component - 1];
                }
            }
            return exp;
        }
    };


} // namespace

std::unique_ptr<ExperienceQueueController> CreateNStepQueueController(int n_step)
{
    return std::make_unique<NStepQueueController>(n_step);
}

std::unique_ptr<ReplayExperienceBuilder> CreateDefaultExperienceBuilder()
{
    return std::make_unique<DefaultExperienceBuilder>();
}

} // namespace anet::rl
