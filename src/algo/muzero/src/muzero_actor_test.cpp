#include "anet/test/catch.hpp"
#include "anet/test/nn.hpp"
#include "anet/test/util.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/agent.hpp"
#include "anet/algo/muzero.hpp"
#include "muzero_proto_agent.hpp"

constexpr char kTargetPolicyKey[] = "target_policy";

TEST_CASE("MuZero initialization keeps its first Agent factory registrations", "[init][idempotent]")
{
    anet::rl::algo::InitMuZero();
    auto& repository = anet::rl::AgentRepository::Instance();
    constexpr char kClassId[] = "MuZeroAgent";
    const auto factory = repository.GetAgentFactory(kClassId);
    REQUIRE(factory);
    CHECK_NOTHROW(anet::rl::algo::InitMuZero());
    CHECK(repository.GetAgentFactory(kClassId) == factory);
}

namespace rl = anet::rl;
constexpr auto kVectorKey = rl::ObsKeys::kVector;

static rl::EnvSpec MakeTestEnvSpec()
{
    rl::EnvSpec spec;
    spec.state_spec.obs_spec[kVectorKey] = {
        .type = anet::SpaceType::Vector, .shape = { 4 }, .dtype = torch::kFloat32,
    };
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = { "a0", "a1" };
    spec.reward_range = { -1.0f, 1.0f };
    return spec;
}

namespace {
    class NoopMetricsBackend final : public anet::IBackend {
    public:
        void Open(const std::filesystem::path&, const std::string&) override {}
        void WriteJsonl(const anet::json&) override {}
        void Flush() override {}
    };

    class ScopedNoopMetricsLogger final {
    public:
        ScopedNoopMetricsLogger()
        {
            anet::MetricsLogger::Reset();
            anet::MetricsLoggerConfig config;
            config.run_name_tmpl = "muzero_actor_test";
            anet::MetricsLogger::Init(std::make_unique<NoopMetricsBackend>(), config, "out/test-tmp");
        }

        ~ScopedNoopMetricsLogger() { anet::MetricsLogger::Reset(); }
    };
}

TEST_CASE("MuZero Actor catalog owns temperature and exploration noise", "[rfd0061][muzero][actor_catalog][rfd0170][forced_action]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    anet::ConfigData data;
    data.Set("net.block.[Tiny].type", "Linear");
    data.Set("net.block.[Tiny].linear.out_features", 4);
    data.Set("net.block.[ReLU].type", "ReLU");
    for (const auto& name : {"rep", "dyn", "pred"}) {
        const auto prefix = std::string("net.") + name;
        data.Set(prefix + ".branch.[main].structure", "Tiny > ReLU");
        data.Set(prefix + ".branch.[main].bind", std::string(name) == "rep" ? "vector"
            : std::string(name) == "dyn" ? "hidden_state, action_one_hot" : "hidden_state");
    }
    data.Set("net.rep.body.output.[features]", "main");
    data.Set("net.dyn.body.output.[hidden_state]", "main");
    data.Set("net.dyn.body.output.[reward_feature]", "main");
    data.Set("net.pred.body.output.[value_feature]", "main");
    data.Set("net.pred.body.output.[policy_feature]", "main");
    data.Set("MuZeroAgent.model.hidden_state_dim", 4);
    data.Set("MuZeroAgent.buffer.capacity", 32);
    data.Set("MuZeroAgent.mcts.num_simulations", 32);
    data.Set("MuZeroAgent.mcts.root_exploration_fraction", 1.0f);
    for (const auto& key : {"quiet", "noisy", "eval"}) {
        const auto prefix = std::string("MuZeroAgent.actor.[") + key + "]";
        data.Set(prefix + ".temp_start", std::string(key) == "eval" ? 0.0f : 2.0f);
        data.Set(prefix + ".temp_end", 0.0f);
        data.Set(prefix + ".temp_decay_steps", 200);
        data.Set(prefix + ".add_exploration_noise", std::string(key) == "noisy");
    }
    rl::muzero_proto::MuZeroAgentFactory factory;
    auto agent = factory.CreateAgent(MakeTestEnvSpec(), {2, 1}, torch::kCPU, data, nullptr, 123);
    const auto create = [&](const std::string& key) {
        return agent->CreateActor(rl::ActorRequest{.batch_env_spec = {2, 1}, .env_spec = MakeTestEnvSpec(),
            .device = torch::kCPU, .seed = 456, .actor_key = key});
    };
    auto quiet = create("quiet"), noisy = create("noisy"), eval = create("eval");
    REQUIRE(quiet->GetScalar("tau").has_value());
    CHECK(std::isnan(*quiet->GetScalar("tau")));
    CHECK_FALSE(quiet->GetScalar("unknown").has_value());
    auto flags = torch::zeros({2}, torch::kBool);
    rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({2, 4})}}, flags, flags, flags);
    rl::StepCounts counts;
    counts.exp_step = 100;
    // 新しいActorで要求 false→true→true の WARN 件数と AuxData を確認する。
    auto trace_actor = create("quiet");
    anet::test::LogCaptureGuard logs;
    const auto no_request = trace_actor->MakeAction(counts, state, {.nn_trace = false});
    logs.Flush();
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);
    const auto first_request = trace_actor->MakeAction(counts, state, {.nn_trace = true});
    logs.Flush();
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 1);
    const auto second_request = trace_actor->MakeAction(counts, state, {.nn_trace = true});
    logs.Flush();
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 1);
    for (const auto& result : {no_request, first_request, second_request}) {
        const auto& aux = result->GetAuxData();
        CHECK(std::none_of(aux.begin(), aux.end(), [](const auto& entry) {
            return entry.first.starts_with(anet::nn::kNnTracePrefix);
        }));
    }
    const auto quiet_result = quiet->MakeAction(counts, state);
    const auto noisy_result = noisy->MakeAction(counts, state);
    CHECK(*quiet->GetScalar("tau") == Catch::Approx(1.0f));
    CHECK(*noisy->GetScalar("tau") == Catch::Approx(1.0f));
    CHECK_FALSE(torch::equal(quiet_result->GetInfo().At("target_policy"), noisy_result->GetInfo().At("target_policy")));
    eval->MakeAction(counts, state);
    CHECK(*eval->GetScalar("tau") == Catch::Approx(0.0f));
    CHECK(*quiet->GetScalar("tau") == Catch::Approx(1.0f));

    const auto value = 1 - quiet_result->GetAction()[0].item<int64_t>();
    auto forced_actor = create("quiet");
    const auto forced = forced_actor->MakeAction(counts, state, {.forced_actions = rl::ForcedActions::ByLane(0, value)});
    const auto expected = quiet_result->GetAction().clone();
    expected[0].fill_(value);
    CHECK(torch::equal(forced->GetAction(), expected));
    CHECK(torch::equal(forced->GetInfo().At(kTargetPolicyKey), quiet_result->GetInfo().At(kTargetPolicyKey)));
}
