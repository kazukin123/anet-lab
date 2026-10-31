#include "anet/catch_test.hpp"
#include "anet/metrics.hpp"
#include "anet/env.hpp"
#include "anet/gui.hpp"
#include <algorithm>
#include <cmath>
#include <map>

using namespace anet::rl;
using anet::rl::gui::AppendTracePercentile;

namespace {

// DQNのキーを一切使わず、診断の公開契約だけで複数時点の確定を作る。
class DiagnosticScalars final : public anet::ModuleBase {
public:
    explicit DiagnosticScalars(std::vector<std::map<std::string, float>> rows) : rows_(std::move(rows)) {}
    std::optional<float> GetScalar(const std::string& key, int64_t lane = -1) const override
    {
        ANET_CHECK(lane >= 0 && lane < static_cast<int64_t>(rows_.size()));
        const auto found = rows_[lane].find(key);
        return found == rows_[lane].end() ? std::nullopt : std::optional(found->second);
    }
private:
    std::vector<std::map<std::string, float>> rows_;
};

class ScriptStepDiagnostics final : public AgentStepDiagnosticsProvider {
public:
    explicit ScriptStepDiagnostics(int64_t lanes) : lanes_(lanes) {}
    AgentStepDiagnosticsResult Observe(const BatchExperience&) override
    {
        AgentStepDiagnosticsResult result;
        std::vector<std::map<std::string, float>> rows;
        const auto nan = std::numeric_limits<float>::quiet_NaN();
        for (int64_t lane = 0; lane < lanes_; ++lane) {
            rows.push_back({{"entropy", static_cast<float>(10 * step_ + lane)}, {"known_nan", nan},
                {"late_a", nan}, {"late_b", nan}});
            result.pending_keys.push_back({"late_a", "late_b"});
            if (step_ >= 1) result.completed.push_back({.lane=lane, .steps_back=1,
                .resolved_keys={"late_a"}, .values=Source("late_a", 100.0f + lane)});
            if (step_ >= 2) {
                // 別の鍵の確定と同時に既に確定済みの鍵が届いても二重通知しない。
                result.completed.push_back({.lane=lane, .steps_back=2,
                    .resolved_keys={"late_a"}, .values=Source("late_a", 300.0f)});
                result.completed.push_back({.lane=lane, .steps_back=2,
                    .resolved_keys={"late_b"}, .values=Source("late_b", lane == 0 ? nan : 200.0f)});
            }
        }
        result.current = std::make_unique<DiagnosticScalars>(std::move(rows));
        ++step_;
        return result;
    }
    void Reset() override { step_ = 0; }
private:
    static std::unique_ptr<const anet::Module> Source(const std::string& key, float value)
    {
        return std::make_unique<DiagnosticScalars>(std::vector<std::map<std::string, float>>{{{key, value}}});
    }
    int64_t lanes_;
    int64_t step_ = 0;
};

class DiagnosticAgent final : public Agent {
public:
    std::unique_ptr<AgentStepDiagnosticsProvider> CreateStepDiagnosticsProvider(int64_t lanes) const override
    { return std::make_unique<ScriptStepDiagnostics>(lanes); }
    std::shared_ptr<Actor> CreateActor(const ActorRequest&) const override { return nullptr; }
    std::shared_ptr<Learner> CreateLearner() override { return nullptr; }
    torch::Device GetDevice() const override { return torch::kCPU; }
    std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }
};

TrainEvent DiagnosticEvent(int64_t lanes, step_t step)
{
    const BatchState state{anet::TensorDict{}, torch::zeros({lanes}, torch::kBool),
        torch::zeros({lanes}, torch::kBool), torch::zeros({lanes}, torch::kBool)};
    auto action = std::make_shared<BatchActionInfo>(torch::zeros({lanes}, torch::kInt64));
    const BatchExperience experience{state, action, torch::zeros({lanes}), state};
    auto result = std::make_shared<PlainBatchStepResult>(experience.reward, state, state, lanes, 0,
        std::vector<AuxData>(static_cast<size_t>(lanes)));
    return TrainEvent{{experience, nullptr, {.exp_step=step}, nullptr, {}}, nullptr, result, action};
}

} // namespace

TEST_CASE("Agent-step columns resolve independently through GetScalar across lanes", "[prd094][generic_step_completion]")
{
    anet::ConfigData config;
    const std::vector<std::string> keys{"entropy", "known_nan", "late_a", "late_b", "late_a"};
    for (size_t i = 0; i < keys.size(); ++i)
        config.Set("app.eval_trace_panel.columns.[" + std::to_string(i + 1) + "]", "$agent_step " + keys[i]);
    StepTraceBuilder builder(config, EnvSpec{}, std::make_shared<DiagnosticAgent>(), 2);
    std::vector<std::shared_ptr<StepTrace>> rows;
    std::vector<std::pair<int64_t, std::vector<size_t>>> updates;
    builder.SetHandlers([&](const auto& row, const TrainEvent&) { rows.push_back(row); },
        [&](const auto& row, std::span<const size_t> columns) {
            updates.emplace_back(row->id, std::vector<size_t>(columns.begin(), columns.end()));
        });
    builder.OnTrain(DiagnosticEvent(2, 0));
    REQUIRE(rows.size() == 2);
    CHECK(rows[0]->values[0] == 0.0f);
    CHECK(rows[1]->values[0] == 1.0f);
    CHECK(std::isnan(rows[0]->values[1]));
    CHECK(rows[0]->value_ready == std::vector<bool>{true, true, false, false, false});
    CHECK_FALSE(rows[0]->complete);
    CHECK(updates.empty());
    builder.OnTrain(DiagnosticEvent(2, 1));
    CHECK(rows[0]->value_ready == std::vector<bool>{true, true, true, false, true});
    CHECK(rows[0]->values[2] == 100.0f);
    CHECK(rows[1]->values[2] == 101.0f);
    CHECK(rows[0]->values[4] == 100.0f);
    REQUIRE(updates.size() == 2);
    CHECK(updates[0].second == std::vector<size_t>{2, 4});
    CHECK_FALSE(rows[0]->complete);
    builder.OnTrain(DiagnosticEvent(2, 2));
    CHECK(rows[0]->complete);
    CHECK(rows[1]->complete);
    CHECK(std::isnan(rows[0]->values[3]));
    CHECK(rows[1]->values[3] == 200.0f);
    CHECK(updates[3].first == rows[0]->id);
    CHECK(updates[3].second == std::vector<size_t>{3});
    CHECK(rows[0]->value_ready == std::vector<bool>{true, true, true, true, true});
    builder.Reset();
    updates.clear();
    const auto stale = rows.back();
    builder.OnTrain(DiagnosticEvent(2, 3));
    CHECK(updates.empty());
    CHECK_FALSE(rows.back()->complete);
    builder.OnTrain(DiagnosticEvent(2, 4));
    builder.OnTrain(DiagnosticEvent(2, 5));
    CHECK_FALSE(stale->complete);
    CHECK(std::none_of(updates.begin(), updates.end(), [&](const auto& update) { return update.first == stale->id; }));
}

TEST_CASE("Diagnostic providers validate unknown keys at first read and wait only for selected columns",
    "[prd094][generic_step_keys]")
{
    auto agent = std::make_shared<DiagnosticAgent>();
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[27]", "$agent_step not_provided");
    REQUIRE_NOTHROW(StepTraceColumnSpec::Parse(config, EnvSpec{}));
    StepTraceBuilder unknown(config, EnvSpec{}, agent, 2);
    CHECK_THROWS_WITH(unknown.OnTrain(DiagnosticEvent(2, 0)),
        Catch::Matchers::ContainsSubstring("column=27") && Catch::Matchers::ContainsSubstring("key='not_provided'"));

    config.Set("app.eval_trace_panel.columns.[27]", "$agent_step known_nan");
    StepTraceBuilder immediate(config, EnvSpec{}, agent, 2);
    int completed = 0;
    immediate.SetHandlers([&](const auto& row, const TrainEvent&) {
        CHECK(row->complete);
        CHECK(row->value_ready == std::vector<bool>{true});
        CHECK(std::isnan(row->values[0]));
    }, [&](const auto&, std::span<const size_t> columns) {
        ++completed;
        CHECK(columns.empty());
    });
    immediate.OnTrain(DiagnosticEvent(2, 0));
    CHECK(completed == 2);
    immediate.OnTrain(DiagnosticEvent(2, 1));
    CHECK(completed == 4);
}

TEST_CASE("Resolved column notifications update percentile windows after display eviction exactly once",
    "[prd094][generic_step_percentile]")
{
    anet::ConfigData config;
    const std::vector<std::string> keys{"entropy", "late_a", "late_b"};
    for (size_t i = 0; i < keys.size(); ++i)
        config.Set("app.eval_trace_panel.columns.[" + std::to_string(i + 1) + "]", "$agent_step " + keys[i] + " display:pct");
    StepTraceBuilder builder(config, EnvSpec{}, std::make_shared<DiagnosticAgent>(), 2);
    std::vector<std::deque<float>> windows(3);
    std::optional<int64_t> displayed;
    size_t evicted_updates = 0;
    // 表は1行だけ保持する。coreの通知と実際の百分位窓を接続し、遅延列の二重投入を検出する。
    builder.SetHandlers([&](const auto& row, const TrainEvent&) {
        displayed = row->id;
        for (size_t i = 0; i < keys.size(); ++i)
            if (row->value_ready[i]) AppendTracePercentile(windows[i], row->values[i], 10);
    }, [&](const auto& row, std::span<const size_t> columns) {
        if (row->id != displayed) ++evicted_updates;
        for (const auto col : columns) AppendTracePercentile(windows[col], row->values[col], 10);
    });
    builder.OnTrain(DiagnosticEvent(2, 0));
    CHECK(windows[0] == std::deque<float>{0, 1});
    CHECK(windows[1].empty());
    builder.OnTrain(DiagnosticEvent(2, 1));
    builder.OnTrain(DiagnosticEvent(2, 2));
    CHECK(evicted_updates == 6);
    CHECK(windows[0] == std::deque<float>{0, 1, 10, 11, 20, 21});
    CHECK(windows[1] == std::deque<float>{100, 101, 100, 101});
    CHECK(windows[2] == std::deque<float>{200});
    CHECK(AppendTracePercentile(windows[1], 100, 10) == 30.0f);
}

TEST_CASE("Unsupported diagnostics leave arbitrary agent-step columns empty", "[prd094][generic_step_keys]")
{
    // 非対応のnamespaceは、DQNの既知キー一覧に依存せず空欄として記録する。
    EnvSpec spec;
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[1]", "$agent_step entropy");
    const BatchState state{anet::TensorDict{}, torch::zeros({1}, torch::kBool),
        torch::zeros({1}, torch::kBool), torch::zeros({1}, torch::kBool)};
    auto action = std::make_shared<BatchActionInfo>(torch::tensor({0}, torch::kInt64));
    const BatchExperience experience{state, action, torch::tensor({2.0f}), state};
    auto result = std::make_shared<PlainBatchStepResult>(experience.reward, state, state, 1, 0,
        std::vector<AuxData>{});
    const TrainEvent event{{experience, nullptr, {.exp_step=1}, nullptr, {}}, nullptr, result, action};
    StepTraceBuilder builder(config, spec, nullptr, 1);
    std::shared_ptr<StepTrace> row;
    builder.SetHandlers([&](const auto& value, const TrainEvent&) { row = value; }, {});
    builder.OnTrain(event);
    REQUIRE(row);
    CHECK(std::isnan(row->values[0]));
    CHECK(row->complete);
}

TEST_CASE("Step trace columns parse ordered declarations and reject invalid contracts", "[prd094][columns]")
{
    EnvSpec spec;
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = { "NOOP", "UP" };
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[20]", "$env_step reward label:r display:pct");
    config.Set("app.eval_trace_panel.columns.[2]", "$action_info action display:name");
    auto columns = StepTraceColumnSpec::Parse(config, spec);
    REQUIRE(columns.size() == 2);
    CHECK(columns[0].number == 2);
    CHECK(columns[0].label == "action");
    CHECK(columns[0].display == "name");
    CHECK(columns[1].label == "r");
    CHECK(columns[1].display == "pct");

    // 閉じた列DSLはmetricsの制御トークンや時点の違うtargetを受け付けない。
    const std::string invalid = GENERATE(
        "$env_step reward extra", "reward", "$env_step $runner reward", "$unknown reward",
        "$agent loss", "$env lives", "$exp reward", "$update_result loss",
        "$env_step mean.reward", "$runner mean.reward", "$env_step reward label:",
        "$env_step reward label:a label:b", "$env_step reward display:value display:pct",
        "$env_step reward interval:2", "$env_step reward @train", "$env_step reward $exp_step",
        "$env_step reward display:name", "$action_info action display:pct", "$env_step reward bogus:1");
    anet::ConfigData bad;
    bad.Set("app.eval_trace_panel.columns.[1]", invalid);
    CHECK_THROWS(StepTraceColumnSpec::Parse(bad, spec));
    const std::string number = GENERATE("0", "-1", "1.5", "x", "1x", "2147483648");
    anet::ConfigData bad_number;
    bad_number.Set("app.eval_trace_panel.columns.[" + number + "]", "$env_step reward");
    CHECK_THROWS(StepTraceColumnSpec::Parse(bad_number, spec));
    spec.action_spec.is_discrete = false;
    anet::ConfigData continuous;
    continuous.Set("app.eval_trace_panel.columns.[1]", "$action_info action");
    CHECK_THROWS(StepTraceColumnSpec::Parse(continuous, spec));
}

TEST_CASE("Step trace rejects malformed column locations before recording", "[prd094][column_bounds]")
{
    EnvSpec spec;
    const std::string key = GENERATE("app.eval_trace_panel.columns.foo", "app.eval_trace_panel.columns.[1].label");
    anet::ConfigData config;
    config.Set(key, "$env_step reward");
    CHECK_THROWS(StepTraceColumnSpec::Parse(config, spec));
    anet::ConfigData empty;
    CHECK_THROWS(StepTraceBuilder(empty, spec, nullptr, -1));
}

TEST_CASE("Step trace reads continuous actions and reserves transition fields", "[prd094][trace_reader]")
{
    EnvSpec spec;
    spec.action_spec.is_discrete = false;
    spec.action_spec.dims.resize(2);
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[1]", "$action_info action.[1]");
    config.Set("app.eval_trace_panel.columns.[2]", "$env_step reward");
    config.Set("app.eval_trace_panel.columns.[3]", "$agent_step q_sa");
    const BatchState state{anet::TensorDict{}, torch::zeros({1}, torch::kBool),
        torch::zeros({1}, torch::kBool), torch::zeros({1}, torch::kBool)};
    auto action = std::make_shared<BatchActionInfo>(torch::tensor({{0.25f, -0.5f}}));
    const BatchExperience experience{state, action, torch::tensor({2.0f}), state};
    auto result = std::make_shared<PlainBatchStepResult>(experience.reward, state, state, 1, 0,
        std::vector<AuxData>{AuxData{{"reward", torch::tensor(999.0f)}}});
    const TrainEvent event{{experience, nullptr, {.exp_step=1}, nullptr, {}}, nullptr, result, action};
    StepTraceBuilder builder(config, spec, nullptr, 1);
    std::shared_ptr<StepTrace> row;
    builder.SetHandlers([&](const auto& value, const TrainEvent&) { row = value; }, {});
    builder.OnTrain(event);
    REQUIRE(row);
    CHECK(row->values[0] == -0.5f);
    CHECK(row->values[1] == 2.0f);
    CHECK(std::isnan(row->values[2]));
    config.Set("app.eval_trace_panel.columns.[4]", "$env_step missing");
    StepTraceBuilder unknown(config, spec, nullptr, 1);
    CHECK_THROWS(unknown.OnTrain(event));
}

TEST_CASE("Lane action reads reject incomplete dimension syntax", "[prd094][trace_action_dimension]")
{
    const auto info = std::make_shared<BatchActionInfo>(torch::tensor({{0.25f, -0.5f}}));
    const MetricValueContext context{.action_info=info.get()};
    CHECK(MetricValueReader::Read(context, EventField::ACTION_INFO, "action.[1]", 0) == -0.5f);
    const std::string key = GENERATE("action.[0x]", "action.[x]", "action.[-1]", "action.[0",
        "action.[2147483648]", "action.[]", "action.[0]junk");
    CHECK_THROWS_WITH(MetricValueReader::Read(context, EventField::ACTION_INFO, key, 0),
        Catch::Matchers::ContainsSubstring("expected=action.[integer dimension]"));
}

