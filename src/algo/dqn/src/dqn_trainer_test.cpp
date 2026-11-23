#include "anet/test/catch.hpp"
#include <array>
#include "anet/agent.hpp"
#include "default_dqn_agent.hpp"
#include "anet/env.hpp"
#include "anet/init.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/observers.hpp"
#include "anet/test/util.hpp"
#include "anet/trainer.hpp"
#include "anet/metrics.hpp"
#include "anet/gui.hpp"
#include "anet/test/nn.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>


namespace rl = anet::rl;

static torch::Tensor BoolTensor(const std::vector<bool>& values)
{
    auto tensor = torch::empty({ static_cast<int64_t>(values.size()) }, torch::TensorOptions().dtype(torch::kBool));
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        tensor[i].fill_(values[static_cast<size_t>(i)]);
    }
    return tensor;
}

static anet::TensorDict MakeObs(int64_t num_envs)
{
    return anet::TensorDict{ { rl::ObsKeys::kVector, torch::zeros({ num_envs, 1 }, torch::kFloat32) } };
}

static rl::BatchState MakeState(int64_t num_envs)
{
    const std::vector<bool> flags(static_cast<size_t>(num_envs), false);
    return rl::BatchState{ MakeObs(num_envs), BoolTensor(flags), BoolTensor(flags), BoolTensor(flags) };
}

static rl::BatchState MakeResetState(int64_t num_envs)
{
    const std::vector<bool> flags(static_cast<size_t>(num_envs), false);
    const std::vector<bool> starts(static_cast<size_t>(num_envs), true);
    return rl::BatchState{ MakeObs(num_envs), BoolTensor(flags), BoolTensor(flags), BoolTensor(starts) };
}

static rl::EnvSpec MakeEnvSpec()
{
    anet::TensorSpec vector_spec;
    vector_spec.type = anet::SpaceType::Vector;
    vector_spec.shape = { 1 };
    vector_spec.dtype = torch::kFloat32;

    rl::EnvSpec spec;
    spec.state_spec.obs_spec[rl::ObsKeys::kVector] = vector_spec;
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = { "noop" };
    spec.reward_range = { -1.0f, 1.0f };
    return spec;
}

namespace {
    class TestResetResult final : public rl::BatchResetResult {
    public:
        explicit TestResetResult(int64_t num_envs)
            : rl::BatchResetResult(MakeResetState(num_envs))
            , num_envs_(num_envs)
        {
        }
    
        std::vector<rl::AuxData> GetAuxDataList(int env_index = -1) const override
        {
            if (env_index >= 0) return { rl::AuxData{} };
            return std::vector<rl::AuxData>(static_cast<size_t>(num_envs_));
        }
    
    private:
        int64_t num_envs_ = 0;
    };
}

namespace {
    class TestStepResult final : public rl::BatchStepResult {
    public:
        explicit TestStepResult(int64_t num_envs)
            : rl::BatchStepResult(
                torch::zeros({ num_envs }, torch::kFloat32),
                MakeState(num_envs),
                MakeState(num_envs),
                static_cast<uint32_t>(num_envs),
                0)
            , num_envs_(num_envs)
        {
        }
    
        std::vector<rl::AuxData> GetAuxDataList(int env_index = -1) const override
        {
            if (env_index >= 0) return { rl::AuxData{} };
            return std::vector<rl::AuxData>(static_cast<size_t>(num_envs_));
        }
    
    private:
        int64_t num_envs_ = 0;
    };
}

namespace {
    class TestBatchEnv final : public rl::BatchEnvBase {
    public:
        TestBatchEnv(
            const std::string& name,
            int num_envs,
            torch::Device device,
            rl::RunMode run_mode = rl::RunMode::Train,
            rl::EpisodeScope episode_scope = rl::EpisodeScope::PER_LANE)
            : rl::BatchEnvBase(name, num_envs, run_mode)
            , batch_spec_{
                .num_envs = num_envs,
                .num_threads = 1,
                .episode_scope = episode_scope,
            }
            , device_(std::move(device))
        {
        }
    
        rl::EnvSpec GetSpec() const override { ++spec_read_count; return spec; }
    
        // 公開Runner経路で、拒否した入力の副作用と仕様の再取得を観測する。
        rl::EnvSpec spec = MakeEnvSpec();
        mutable int spec_read_count = 0;
        int reset_count = 0;
        std::vector<int64_t> actions;
        rl::BatchEnvSpec GetBatchSpec() const override { return batch_spec_; }
        torch::Device GetDevice() const override { return device_; }
    
        std::shared_ptr<const rl::BatchResetResult> Reset() override
        {
            ++reset_count;
            return std::make_shared<TestResetResult>(batch_spec_.num_envs);
        }
    
        std::shared_ptr<const rl::BatchStepResult> Step(std::shared_ptr<rl::BatchActionInfo> action_info) override
        {
            actions.push_back(action_info->GetAction()[0].item<int64_t>());
            return std::make_shared<TestStepResult>(batch_spec_.num_envs);
        }
    
        std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return std::nullopt; }
        std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }
    
    private:
        rl::BatchEnvSpec batch_spec_;
        torch::Device device_;
    };
}

namespace {
    class RunManagerNoopMetricsBackend final : public anet::IBackend {
    public:
        void Open(const std::filesystem::path&, const std::string&) override {}
        void WriteJsonl(const anet::json&) override {}
        void Flush() override {}
    };
}

namespace {
    class ScopedRunManagerMetricsLogger final {
    public:
        ScopedRunManagerMetricsLogger()
        {
            anet::MetricsLogger::Reset();
            anet::MetricsLoggerConfig logger_config;
            logger_config.run_name_tmpl = "run_manager_env_name_test";
            anet::MetricsLogger::Init(
                std::make_unique<RunManagerNoopMetricsBackend>(), logger_config, "C:/tmp");
        }
    
        ~ScopedRunManagerMetricsLogger()
        {
            anet::MetricsLogger::Reset();
        }
    };
}

TEST_CASE("Runner selects an Actor catalog entry and exposes its epsilon", "[trainer][rfd0061][actor_catalog_tracer]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    auto env = std::make_shared<TestBatchEnv>("train", 2, torch::Device(torch::kCPU));
    anet::ConfigData config;
    config.Set("DefaultDQNAgent.quantile_mode", "none");
    config.Set("DefaultDQNAgent.head.use_dueling", false);


    config.Set("DefaultDQNAgent.learner.enabled", false);
    config.Set("DefaultDQNAgent.learner.replay_capacity", 16);
    config.Set("DefaultDQNAgent.learner.replay_batch_size", 2);
    config.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);
    config.Set("DefaultDQNAgent.actor.[probe].policy.policy_type", "EpsilonGreedy");
    config.Set("DefaultDQNAgent.actor.[probe].policy.epsilon", 0.3f);
    config.Set("DefaultDQNAgent.actor.[probe].policy.epsilon.start", 0.3f);
    config.Set("DefaultDQNAgent.actor.[probe].policy.epsilon.end", 0.3f);
    // Actor カタログの実行目的を明示し、Noisy OFF の既定動作を選ぶ。
    config.Set("DefaultDQNAgent.nn_forward.[test_default].noisy.mode", "mu_only");
    config.Set("DefaultDQNAgent.actor.[probe].nn_forward_key", "test_default");
    config.Set("DefaultDQNAgent.learner.current.nn_forward_key", "test_default");
    config.Set("DefaultDQNAgent.learner.target.nn_forward_key", "test_default");
    config.Set("DefaultDQNAgent.net.branch.[features].bind", "vector");
    config.Set("DefaultDQNAgent.net.branch.[features].structure", "Flatten");
    config.Set("net.block.[Flatten].type", "Flatten");
    config.Set("DefaultDQNAgent.net.body.output.[features]", "features");
    config.Set("metrics.scalar.[probe]", "$train @train $actor epsilon");
    rl::dqn::DefaultDQNAgentFactory factory;
    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = factory.CreateAgent(env->GetSpec(), env->GetBatchSpec(), torch::kCPU, config, notifier, 123);
    const rl::ActorRequest request{
        .batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(),
        .device = torch::kCPU, .seed = 456, .actor_key = "probe"};
    auto runner = std::make_shared<rl::SerialTrainRunner>(env, agent, notifier, request);
    runner->DoStep();
    REQUIRE(runner->GetActor()->GetScalar("epsilon").has_value());
    CHECK(*runner->GetActor()->GetScalar("epsilon") == Catch::Approx(0.3f));
    const rl::ObserverFactory observers(config);
    REQUIRE(observers.GetScalarMetricDefs().size() == 1);
    CHECK(observers.GetScalarMetricDefs().front().subscription.target == rl::EventField::ACTOR);
}
