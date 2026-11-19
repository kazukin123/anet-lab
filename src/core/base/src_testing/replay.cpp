#include "anet/test/replay.hpp"

#include <limits>
#include "replay_buffer_impl.hpp"

namespace rl = anet::rl;

namespace {
    class PrioritySlot final : public rl::ReplayPriorityStore {
    public:
        float raw = std::numeric_limits<float>::quiet_NaN();
        rl::ReplayPrioritySource source = rl::ReplayPrioritySource::NONE;
        void InvalidateSlot(int64_t) override { raw = std::numeric_limits<float>::quiet_NaN(); }
        void SetRawInitialPriority(int64_t, float value, rl::ReplayPrioritySource from) override { raw = value; source = from; }
        void SetMaxInitialPriority(int64_t) override { source = rl::ReplayPrioritySource::MAX_INITIAL; }
        rl::ReplayPriorityUpdateResult ApplyLearnerPriorities(const std::vector<int64_t>&, const std::vector<float>&) override { return {}; }
        std::optional<float> GetScalar(const std::string&) const override { return std::nullopt; }
        std::optional<torch::Tensor> GetPriorityTensor(int64_t) const override { return torch::tensor(raw); }
        torch::Tensor GatherPriorityRows(const torch::Tensor&) const override { return torch::tensor({raw}); }
    };
}

anet::test::ReferenceInitialPriority anet::test::CompleteReferenceInitialPriority(
    int n_step, int actual_n_steps, float target_return, float discount,
    bool terminal, bool truncated, const rl::ReplayInitialPriorityHintRow& start_hint,
    const rl::ReplayInitialPriorityHintRow& end_hint,
    std::unique_ptr<rl::InitialPriorityEstimator> estimator)
{
    rl::ValidIndexManager index(1, 32);
    for (int step = 0; step < 16; ++step) {
        index.MarkWritten(0, step);
        index.AdvanceWriteCursor(0);
        index.MarkValid(0);
    }
    rl::InitialPriorityCompleter reference(
        { .mode = rl::ReplayInitialPriorityMode::ACTOR_APPROX, .n_step = n_step },
        1, std::move(estimator));
    reference.Enqueue(0, {
        .target_return = target_return,
        .discount = discount,
        .terminal = terminal,
        .truncated = truncated,
        .actual_n_steps = actual_n_steps,
        .start_hint = start_hint,
    });
    PrioritySlot slot;
    reference.CompleteReady(0, end_hint, actual_n_steps, index, slot);
    return { slot.raw, slot.source };
}
