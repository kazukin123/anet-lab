#include "anet/test/catch.hpp"
#include "ppo_fixture_test.hpp"
#include <algorithm>
#include <filesystem>
#include <string>
#include <utility>
#include "anet/metrics_logger.hpp"
#include "ppo_actor.hpp"

using namespace anet::rl::ppo;
using namespace anet::rl::ppo::test;

constexpr char kSharedBodyBranch[] = "shared_body";
constexpr char kMetricsRunName[] = "ppo_test";
constexpr char kCorridorEnvName[] = "corridor";
constexpr char kMetricsOutputDir[] = "out/test-tmp";

namespace {

    class NoopMetricsBackend final : public anet::IBackend {
    public:
        void Open(const std::filesystem::path&, const std::string&) override {}
        void WriteJsonl(const anet::json&) override {}
        void Flush() override {}
    };

    // lane ごとの補助の情報を持たない、Reset の結果。
    class CorridorResetResult final : public anet::rl::BatchResetResult {
    public:
        CorridorResetResult(anet::rl::BatchState state, int lane_count)
            : anet::rl::BatchResetResult(std::move(state)), lane_count_(lane_count) {}
        std::vector<anet::rl::AuxData> GetAuxDataList(int env_index = -1) const override
            { return std::vector<anet::rl::AuxData>(env_index >= 0 ? 1 : static_cast<size_t>(lane_count_)); }
    private:
        const int lane_count_;
    };

    // lane ごとの補助の情報を持たない、step の結果。
    class CorridorStepResult final : public anet::rl::BatchStepResult {
    public:
        CorridorStepResult(CorridorEnv::StepResult result, int lane_count)
            : anet::rl::BatchStepResult(result.reward, result.next_state, result.continue_state,
                static_cast<uint32_t>(lane_count), static_cast<uint32_t>(result.finished_returns.size()))
            , lane_count_(lane_count) {}
        std::vector<anet::rl::AuxData> GetAuxDataList(int env_index = -1) const override
            { return std::vector<anet::rl::AuxData>(env_index >= 0 ? 1 : static_cast<size_t>(lane_count_)); }
    private:
        const int lane_count_;
    };

} // namespace

ScopedNoopMetricsLogger::ScopedNoopMetricsLogger()
{
    anet::MetricsLogger::Reset();
    const anet::MetricsLoggerConfig logger_config{ .run_name_tmpl = kMetricsRunName };
    anet::MetricsLogger::Init(std::make_unique<NoopMetricsBackend>(), logger_config, kMetricsOutputDir);
}

ScopedNoopMetricsLogger::~ScopedNoopMetricsLogger()
{
    anet::MetricsLogger::Reset();
}

CorridorEnv::CorridorEnv(int lane_count)
    : positions_(static_cast<size_t>(lane_count), 0)
    , steps_(static_cast<size_t>(lane_count), 0)
    , returns_(static_cast<size_t>(lane_count), 0.0f)
{
}

torch::Tensor CorridorEnv::MakeObs() const
{
    const auto lane_count = static_cast<int64_t>(positions_.size());
    auto obs = torch::zeros({ lane_count, kLength }, torch::kFloat32);
    for (int64_t lane = 0; lane < lane_count; ++lane) obs[lane][positions_[static_cast<size_t>(lane)]] = 1.0f;
    return obs;
}

anet::rl::BatchState CorridorEnv::Reset()
{
    const auto lane_count = static_cast<int64_t>(positions_.size());
    std::fill(positions_.begin(), positions_.end(), 0);
    std::fill(steps_.begin(), steps_.end(), 0);
    std::fill(returns_.begin(), returns_.end(), 0.0f);
    const auto off = torch::zeros({ lane_count }, torch::kBool);
    return anet::rl::BatchState({ anet::rl::ObsKeys::kVector, MakeObs() }, off, off, torch::ones({ lane_count }, torch::kBool));
}

CorridorEnv::StepResult CorridorEnv::Step(const torch::Tensor& actions)
{
    const auto lane_count = static_cast<int64_t>(positions_.size());
    const auto cpu_actions = actions.to(torch::kCPU);
    StepResult result;
    result.reward = torch::zeros({ lane_count }, torch::kFloat32);
    auto done = torch::zeros({ lane_count }, torch::kBool);
    auto truncated = torch::zeros({ lane_count }, torch::kBool);

    // 位置を動かし、報酬と、終端・打ち切りを決める。
    for (int64_t lane = 0; lane < lane_count; ++lane) {
        const auto i = static_cast<size_t>(lane);
        positions_[i] = std::clamp(positions_[i] + (cpu_actions[lane].item<int64_t>() == 1 ? 1 : -1), 0, kLength - 1);
        ++steps_[i];
        const bool reached = positions_[i] == kLength - 1;
        const float reward = reached ? kGoalReward : kStepReward;
        returns_[i] += reward;
        result.reward[lane] = reward;
        done[lane] = reached;
        truncated[lane] = !reached && steps_[i] >= kLimitStep;
    }
    const auto off = torch::zeros({ lane_count }, torch::kBool);
    result.next_state = anet::rl::BatchState({ anet::rl::ObsKeys::kVector, MakeObs() }, done, truncated, off);

    // 終わった lane を最初の位置へ戻し、続きの状態を作る。
    auto episode_start = torch::zeros({ lane_count }, torch::kBool);
    for (int64_t lane = 0; lane < lane_count; ++lane) {
        const auto i = static_cast<size_t>(lane);
        if (!done[lane].item<bool>() && !truncated[lane].item<bool>()) continue;
        result.finished_returns.push_back(returns_[i]);
        positions_[i] = 0;
        steps_[i] = 0;
        returns_[i] = 0.0f;
        episode_start[lane] = true;
    }
    result.continue_state = anet::rl::BatchState({ anet::rl::ObsKeys::kVector, MakeObs() }, off, off, episode_start);
    return result;
}

CorridorBatchEnv::CorridorBatchEnv(int lane_count)
    : anet::rl::BatchEnvBase(kCorridorEnvName, lane_count), env_(lane_count), lane_count_(lane_count)
{
}

anet::rl::EnvSpec CorridorBatchEnv::GetSpec() const
{
    return MakeEnvSpec(CorridorEnv::kLength, CorridorEnv::kActionCount);
}

anet::rl::BatchEnvSpec CorridorBatchEnv::GetBatchSpec() const
{
    return anet::rl::BatchEnvSpec{ .num_envs = lane_count_, .num_threads = 1 };
}

torch::Device CorridorBatchEnv::GetDevice() const
{
    return torch::Device(torch::kCPU);
}

std::shared_ptr<const anet::rl::BatchResetResult> CorridorBatchEnv::Reset()
{
    return std::make_shared<CorridorResetResult>(env_.Reset(), lane_count_);
}

std::shared_ptr<const anet::rl::BatchStepResult> CorridorBatchEnv::Step(
    std::shared_ptr<anet::rl::BatchActionInfo> action_info)
{
    return std::make_shared<CorridorStepResult>(env_.Step(action_info->GetAction(torch::Device(torch::kCPU))),
        lane_count_);
}

anet::ConfigData anet::rl::ppo::test::MakeConfigData(bool shared_body)
{
    anet::test::RegisterStandardNetworkModules();
    anet::ConfigData data;
    // Body の部品。stack 軸を畳む Flatten を先頭に置く。
    data.Set("net.block.[Flatten].type", "Flatten");
    data.Set("net.block.[Hidden].type", "Linear");
    data.Set("net.block.[Hidden].linear.out_features", 16);
    data.Set("net.block.[Tanh].type", "Tanh");
    constexpr char kStructure[] = "Flatten > Hidden > Tanh";
    if (shared_body) {
        data.Set("PPOAgent.net.branch.[shared_body].bind", anet::rl::ObsKeys::kVector);
        data.Set("PPOAgent.net.branch.[shared_body].structure", kStructure);
        data.Set("PPOAgent.net.body.output.[policy_feature]", kSharedBodyBranch);
        data.Set("PPOAgent.net.body.output.[value_feature]", kSharedBodyBranch);
    } else {
        data.Set("PPOAgent.net.branch.[policy_body].bind", anet::rl::ObsKeys::kVector);
        data.Set("PPOAgent.net.branch.[policy_body].structure", kStructure);
        data.Set("PPOAgent.net.branch.[value_body].bind", anet::rl::ObsKeys::kVector);
        data.Set("PPOAgent.net.branch.[value_body].structure", kStructure);
        data.Set("PPOAgent.net.body.output.[policy_feature]", "policy_body");
        data.Set("PPOAgent.net.body.output.[value_feature]", "value_body");
    }
    // Train は Network を共有して抽選し、eval は写しを持って確率が最大の行動を選ぶ。
    data.Set("PPOAgent.actor.[train].action_mode", "sample");
    data.Set("PPOAgent.actor.[train].clone_model", false);
    data.Set("PPOAgent.actor.[eval].action_mode", "greedy");
    data.Set("PPOAgent.actor.[eval].clone_model", true);
    data.Set("PPOAgent.learner.use_fused_optimizer", false);
    return data;
}

anet::rl::EnvSpec anet::rl::ppo::test::MakeEnvSpec(int64_t obs_size, int action_count)
{
    anet::rl::EnvSpec env_spec;
    env_spec.state_spec.obs_spec[anet::rl::ObsKeys::kVector] = anet::TensorSpec{
        .type = anet::SpaceType::Vector, .shape = { obs_size }, .dtype = torch::kFloat32 };
    env_spec.action_spec.is_discrete = true;
    for (int action = 0; action < action_count; ++action) {
        env_spec.action_spec.value_labels.push_back("a" + std::to_string(action));
    }
    env_spec.reward_range = { -1.0f, 1.0f };
    return env_spec;
}

Rig anet::rl::ppo::test::MakeRig(const anet::ConfigData& config_data, const anet::rl::EnvSpec& env_spec,
    int lane_count, torch::Device device, anet::seed_t seed)
{
    Rig rig;
    rig.metrics_logger = std::make_shared<ScopedNoopMetricsLogger>();
    rig.env_spec = env_spec;
    rig.batch_env_spec = anet::rl::BatchEnvSpec{ .num_envs = lane_count, .num_threads = 1 };
    // 製品と同じく、factory が設定から Agent を作る。
    const auto agent = PPOAgentFactory().CreateAgent(env_spec, rig.batch_env_spec, device, config_data, nullptr, seed);
    rig.agent = std::static_pointer_cast<PPOAgent>(agent);
    rig.actor = rig.agent->CreateActor(anet::rl::ActorRequest{ .batch_env_spec = rig.batch_env_spec,
        .env_spec = env_spec, .device = device, .seed = seed, .actor_key = kTrainActorKey });
    rig.learner = rig.agent->CreateLearner();
    return rig;
}

anet::rl::BatchState anet::rl::ppo::test::MakeState(const torch::Tensor& vector)
{
    const auto off = torch::zeros({ vector.size(0) }, torch::kBool);
    return anet::rl::BatchState({ anet::rl::ObsKeys::kVector, vector }, off, off, off);
}

anet::rl::BatchState anet::rl::ppo::test::MakeState(const torch::Tensor& vector, const std::vector<bool>& done,
    const std::vector<bool>& truncated)
{
    const auto lane_count = vector.size(0);
    auto done_tensor = torch::zeros({ lane_count }, torch::kBool);
    auto truncated_tensor = torch::zeros({ lane_count }, torch::kBool);
    for (int64_t lane = 0; lane < lane_count; ++lane) {
        done_tensor[lane] = static_cast<bool>(done.at(static_cast<size_t>(lane)));
        truncated_tensor[lane] = static_cast<bool>(truncated.at(static_cast<size_t>(lane)));
    }
    return anet::rl::BatchState({ anet::rl::ObsKeys::kVector, vector }, done_tensor, truncated_tensor,
        torch::zeros({ lane_count }, torch::kBool));
}

torch::Tensor anet::rl::ppo::test::ReadValues(const anet::rl::Actor& actor, const torch::Tensor& vector)
{
    return actor.MakeAction({}, MakeState(vector))->GetInfo().At(Actor::kValueKey).cpu();
}

anet::rl::MakeActionOptions anet::rl::ppo::test::MakeForcedActionOptions(const torch::Tensor& actions)
{
    return anet::rl::MakeActionOptions{ .forced_actions = anet::rl::ForcedActions::ByMask(
        torch::ones({ actions.size(0) }, torch::kBool), actions) };
}

torch::Tensor anet::rl::ppo::test::ReadLogProbs(const anet::rl::Actor& actor, const torch::Tensor& vector,
    const torch::Tensor& actions)
{
    return actor.MakeAction({}, MakeState(vector), MakeForcedActionOptions(actions))->GetInfo()
        .At(Actor::kLogProbKey).cpu();
}

std::vector<anet::rl::ScalarMetricSubscription> anet::rl::ppo::test::MakeAllSubscriptions(int action_count)
{
    std::vector<anet::rl::ScalarMetricSubscription> subscriptions;
    const auto add = [&subscriptions](const std::string& key) {
        subscriptions.push_back(anet::rl::ScalarMetricSubscription{ .source_key = key,
            .event = anet::rl::EventType::LEARN, .target = anet::rl::EventField::UPDATE_RESULT });
    };
    for (const auto* key : UpdateResult::GetMinibatchStatKeys()) add(key);
    for (const auto* key : UpdateResult::GetRolloutStatKeys()) add(key);
    for (const auto* key : UpdateResult::GetHostStatKeys()) add(key);
    for (int action = 0; action < action_count; ++action) {
        add(UpdateResult::kActionProbKeyPrefix + std::to_string(action));
        add(UpdateResult::kActionFreqKeyPrefix + std::to_string(action));
    }
    return subscriptions;
}

anet::rl::BatchUpdateResultList anet::rl::ppo::test::StepOnce(Rig& rig, const anet::rl::BatchState& state,
    const torch::Tensor& reward, const anet::rl::BatchState& next_state, const anet::rl::StepCounts& counts,
    const anet::rl::MakeActionOptions& options)
{
    // Runner と同じく、行動選択 → 経験の組み立て → Learner の順に呼ぶ。経験は CPU の値で渡す。
    const auto action_info = rig.actor->MakeAction(counts, state, options);
    const anet::rl::BatchExperience experience(state.Clone(), action_info, reward.clone(), next_state.Clone());
    return rig.learner->UpdateFromBatch(counts, experience);
}
