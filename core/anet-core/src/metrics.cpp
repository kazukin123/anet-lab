// metrics.cpp
#include "anet/metrics.hpp"
#include <algorithm>
#include <charconv>
#include <limits>
#include <set>
#include "metrics_impl.hpp"
#include "anet/profile.hpp"
#include "anet/str_util.hpp"
#include "anet/util.hpp"

using namespace anet::rl;
using namespace anet::rl::metric_tokens;

std::optional<float> MetricValueReader::Read(const MetricValueContext& context,
    EventField field, const std::string& key, int64_t lane)
{
    // 結果型はModuleではないため、遷移の読み口へ直接委譲する。
    if (field == EventField::ENV_STEP) {
        return context.step_result ? context.step_result->GetScalar(key, lane) : std::nullopt;
    }
    // 呼び出し側が集約・探索・未知値の扱いを決め、ここは取得元だけを解決する。
    const anet::Module* source = nullptr;
    switch (field) {
    case EventField::AGENT: source = context.agent; break;
    case EventField::AGENT_STEP: source = context.agent_step; break;
    case EventField::RUNNER: source = context.runner; break;
    case EventField::ACTOR: source = context.runner ? context.runner->GetActor().get() : nullptr; break;
    case EventField::ENV: source = context.env ? context.env
        : (context.runner ? context.runner->GetBatchEnv().get() : nullptr); break;
    case EventField::EXPERIENCE: source = context.experience; break;
    case EventField::ACTION_INFO: source = context.action_info; break;
    case EventField::UPDATE_RESULT: source = context.update_result; break;
    default: break;
    }
    return source ? source->GetScalar(key, lane) : std::nullopt;
}

// 宣言順と各指定の出現を残し、既定値・後勝ち・診断はチャネル側へ委ねる。
std::vector<Token> anet::rl::metric_tokens::ParseMetricTokens(const std::string& definition)
{
    std::vector<Token> tokens;
    for (const auto& raw : anet::Split(definition, { " " }, true)) {
        Token token{ .raw = raw, .value = raw };
        if (raw == "@train" || raw == "@learn" || raw == "@episode_end" || raw == "@session_end") {
            token.kind = Kind::EVENT;
            token.value = raw.substr(1);
        } else if (raw == "$train_step" || raw == "$learn_step" || raw == "$episode_step"
            || raw == "$exp_step" || raw == "$update_step" || raw == "$sim_step") {
            token.kind = Kind::STEP;
            token.value = raw.substr(1);
        } else if (raw == "$actor" || raw == "$agent" || raw == "$env" || raw == "$env_step" || raw == "$runner"
            || raw == "$batch_experience" || raw == "$exp" || raw == "$batch_update_result"
            || raw == "$update_result" || raw == "$result" || raw == "$action" || raw == "$action_info") {
            token.kind = Kind::TARGET;
            token.value = raw.substr(1);
        } else if (raw == "$train" || anet::StartsWith(raw, "$eval.[")) {
            token.kind = Kind::SCOPE;
            token.value = raw == "$train" ? "train" : anet::ExtractBetween(raw, "$eval.[", "]");
        } else if (raw == "$ema") {
            token.kind = Kind::EMA;
        } else {
            const auto attr = anet::Split(raw, { ":" }, true);
            if (attr.size() == 2) {
                token.kind = Kind::ATTRIBUTE;
                token.attribute = attr[0];
                token.value = attr[1];
                if (attr[0] == "event") token.kind = Kind::EVENT;
                else if (attr[0] == "step" || attr[0] == "step_axis") token.kind = Kind::STEP;
                else if (attr[0] == "target") token.kind = Kind::TARGET;
            }
        }
        if (token.kind == Kind::EVENT) {
            if (token.value == "train") token.event = EventType::TRAIN;
            else if (token.value == "learn") token.event = EventType::LEARN;
            else if (token.value == "episode_end") token.event = EventType::EPISODE_END;
            else if (token.value == "session_end") token.event = EventType::SESSION_END;
        } else if (token.kind == Kind::STEP) {
            // 属性形は従来の短縮名を受理し、$ 形だけ train_step / learn_step を受理する。
            if (token.value == "train" || (token.attribute.empty() && token.value == "train_step")) token.step = StepAxis::TRAIN;
            else if (token.value == "learn" || (token.attribute.empty() && token.value == "learn_step")) token.step = StepAxis::LEARN;
            else if (token.value == "episode" || token.value == "episode_step") token.step = StepAxis::EPISODE;
            else if (token.value == "exp" || token.value == "exp_step") token.step = StepAxis::EXP;
            else if (token.value == "update" || token.value == "update_step") token.step = StepAxis::UPDATE;
            else if (token.value == "sim" || token.value == "sim_step") token.step = StepAxis::SIM;
        } else if (token.kind == Kind::TARGET) {
            if (token.value == "actor") token.field = EventField::ACTOR;
            else if (token.value == "agent") token.field = EventField::AGENT;
            else if (token.value == "env") token.field = EventField::ENV;
            else if (token.value == "env_step") token.field = EventField::ENV_STEP;
            else if (token.value == "runner") token.field = EventField::RUNNER;
            else if (token.value == "exp" || token.value == "batch_experience") token.field = EventField::EXPERIENCE;
            else if (token.value == "update_result" || token.value == "batch_update_result" || token.value == "result") token.field = EventField::UPDATE_RESULT;
            else if (token.value == "action_info" || token.value == "action") token.field = EventField::ACTION_INFO;
        }
        tokens.push_back(std::move(token));
    }
    return tokens;
}

std::vector<StepTraceColumnSpec> StepTraceColumnSpec::Parse(const anet::ConfigData& config, const EnvSpec& spec)
{
    // 制御トークンの字句だけをmetricsと共有し、列DSLの閉じた契約を検証する。
    std::vector<StepTraceColumnSpec> columns;
    std::set<int> numbers;
    const std::string prefix = "app.eval_trace_panel.columns.[";
    for (const auto& [name, definition] : config.Map()) {
        if (!name.starts_with("app.eval_trace_panel.columns.")) continue;
        const auto invalid = [&](const std::string& detail) {
            ANET_SYSTEM_ERROR("Invalid step trace column. config_key='" << name
                << "' value='" << definition << "' expected=" << detail);
        };
        StepTraceColumnSpec column;
        if (!name.starts_with(prefix) || !name.ends_with("]")) invalid("columns.[positive integer]");
        const auto number = name.substr(prefix.size(), name.size() - prefix.size() - 1);
        const auto parsed = std::from_chars(number.data(), number.data() + number.size(), column.number);
        if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size()
            || column.number < 1 || !numbers.insert(column.number).second) {
            invalid("a unique integer column number >= 1");
        }
        bool label_seen = false;
        bool display_seen = false;
        for (const auto& token : metric_tokens::ParseMetricTokens(definition)) {
            const auto& raw = token.raw;
            if (raw.starts_with("label:") || raw.starts_with("display:")) {
                const bool label = raw.starts_with("label:");
                auto& seen = label ? label_seen : display_seen;
                const auto value = raw.substr(raw.find(':') + 1);
                if (seen || value.empty() || value.find(':') != std::string::npos) {
                    invalid("one nonempty label/display attribute");
                }
                seen = true;
                (label ? column.label : column.display) = value;
            } else if (raw == "$runner" || raw == "$actor" || raw == "$env_step"
                || raw == "$action_info" || raw == "$agent_step") {
                if (!column.target.empty()) invalid("exactly one target");
                column.target = raw;
            } else if (token.kind == metric_tokens::Kind::KEY && !raw.starts_with("$")
                && !raw.starts_with("@") && raw.find(':') == std::string::npos) {
                if (!column.key.empty()) invalid("exactly one scalar key");
                column.key = raw;
            } else {
                invalid("$runner, $actor, $env_step, $action_info or $agent_step, one key, label:/display:");
            }
        }
        if (column.target.empty() || column.key.empty()) invalid("exactly one target and one scalar key");
        if (!label_seen) column.label = column.key;
        const auto aggregation = anet::ParseScalarAggregationKey(column.key);
        if (aggregation && !(column.target == "$runner"
            && (aggregation->base_key == "episode_return" || aggregation->base_key == "episode_steps"))) {
            invalid("aggregation only for $runner episode_return/episode_steps");
        }
        const bool action = column.target == "$action_info" && column.key == "action";
        const bool action_dimension = column.target == "$action_info" && column.key.starts_with("action.[");
        if (action && !spec.action_spec.is_discrete) invalid("action.[dimension] for continuous actions");
        if (action_dimension) {
            int dimension = -1;
            const auto digits = column.key.substr(8, column.key.size() - 9);
            const auto result = std::from_chars(digits.data(), digits.data() + digits.size(), dimension);
            if (spec.action_spec.is_discrete || !column.key.ends_with("]") || result.ec != std::errc{}
                || result.ptr != digits.data() + digits.size() || dimension < 0
                || dimension >= static_cast<int>(spec.action_spec.dims.size())) {
                invalid("an existing continuous action.[dimension]");
            }
        }
        if (!display_seen && action) column.display = "value_name";
        const bool display_valid = action
            ? (column.display == "value" || column.display == "name" || column.display == "value_name")
            : (column.display == "value" || (!action_dimension && column.display == "pct"));
        if (!display_valid) invalid("a display compatible with the column and action space");
        columns.push_back(std::move(column));
    }
    std::sort(columns.begin(), columns.end(), [](const auto& a, const auto& b) { return a.number < b.number; });
    return columns;
}

StepTraceBuilder::StepTraceBuilder(const anet::ConfigData& config, const EnvSpec& spec,
    std::shared_ptr<Agent> agent, int64_t num_lanes)
    : columns_(StepTraceColumnSpec::Parse(config, spec)), num_lanes_(num_lanes)
{
    ANET_CHECK_MSG(num_lanes > 0, "Step trace lane count must be positive. actual=" << num_lanes);
    diagnostics_ = agent ? agent->CreateStepDiagnosticsProvider(num_lanes) : nullptr;
    lane_steps_.resize(static_cast<size_t>(num_lanes), 0);
    pending_.resize(static_cast<size_t>(num_lanes));
}

void StepTraceBuilder::SetHandlers(AddedHandler added, Handler completed)
{
    added_ = std::move(added);
    completed_ = std::move(completed);
}

void StepTraceBuilder::Reset()
{
    // 再表示時は全laneの診断窓を破棄し、行IDだけは再利用しない。
    if (diagnostics_) diagnostics_->Reset();
    std::fill(lane_steps_.begin(), lane_steps_.end(), 0);
    for (auto& rows : pending_) rows.clear();
}

void StepTraceBuilder::OnTrain(const TrainEvent& event)
{
    ANET_PROFILE_FUNC();
    // 診断は一度だけbatchで進め、即時値と将来確定した値を同じ遷移へ結び付ける。
    ANET_CHECK_MSG(event.step_result != nullptr && event.step_result->reward.size(0) == num_lanes_,
        "Step trace requires a matching Step result. expected_lanes=" << num_lanes_);
    auto values = diagnostics_ ? diagnostics_->Observe(event.experience) : AgentStepDiagnosticsResult{};
    if (diagnostics_) {
        ANET_CHECK_MSG(values.current && values.pending_keys.size() == static_cast<size_t>(num_lanes_),
            "Step diagnostics lane count mismatch. expected=" << num_lanes_);
    }
    for (int64_t lane = 0; lane < num_lanes_; ++lane) {
        const MetricValueContext context{ .agent = event.agent.get(), .runner = event.runner.get(),
            .env = event.env.get(), .experience = &event.experience,
            .action_info = event.action_info.get(), .step_result = event.step_result.get(),
            .agent_step = values.current.get() };
        auto row = std::make_shared<StepTrace>();
        row->id = next_id_++;
        row->step = event.counts.exp_step;
        row->lane = lane;
        row->reward = event.step_result->GetScalar("reward", lane).value();
        row->done = event.step_result->GetScalar("done", lane).value() != 0;
        row->truncated = event.step_result->GetScalar("truncated", lane).value() != 0;
        for (const auto& column : columns_) {
            const auto field = column.target == "$runner" ? EventField::RUNNER
                : column.target == "$actor" ? EventField::ACTOR
                : column.target == "$action_info" ? EventField::ACTION_INFO
                : column.target == "$agent_step" ? EventField::AGENT_STEP : EventField::ENV_STEP;
            const auto value = field == EventField::AGENT_STEP && !diagnostics_
                ? std::optional(std::numeric_limits<float>::quiet_NaN())
                : MetricValueReader::Read(context, field, column.key, lane);
            if (!value) {
                ANET_SYSTEM_ERROR("Step trace scalar is unavailable. column=" << column.number
                    << " target='" << column.target << "' key='" << column.key << "' lane=" << lane
                    << " expected=recognized scalar key.");
            }
            row->values.push_back(*value);
            // 値がfiniteでもpendingなら待ち、既知NaNでもpendingでなければ確定する。
            const bool waiting = diagnostics_ && field == EventField::AGENT_STEP
                && std::find(values.pending_keys[lane].begin(), values.pending_keys[lane].end(), column.key)
                    != values.pending_keys[lane].end();
            row->value_ready.push_back(!waiting);
        }
        row->complete = std::all_of(row->value_ready.begin(), row->value_ready.end(), [](bool ready) { return ready; });
        if (!row->complete) pending_[lane].push_back({ .step = lane_steps_[lane], .row = row });
        if (added_) added_(row, event);
        if (row->complete && completed_) completed_(row, {});
    }
    // 保持画像を持たないcoreの行だけを待ち、遅延列を一度だけ確定する。
    for (const auto& completion : values.completed) {
        ANET_CHECK_MSG(completion.lane >= 0 && completion.lane < num_lanes_ && completion.steps_back >= 0
            && completion.values, "Invalid step diagnostic completion. lane=" << completion.lane
            << " steps_back=" << completion.steps_back << " expected=valid lane, nonnegative distance and scalar source.");
        auto& pending = pending_[completion.lane];
        const auto wanted = lane_steps_[completion.lane] - completion.steps_back;
        const auto found = std::find_if(pending.begin(), pending.end(), [&](const auto& value) {
            return value.step == wanted;
        });
        if (found == pending.end()) continue;
        auto row = found->row;
        std::vector<size_t> resolved_columns;
        for (size_t i = 0; i < columns_.size(); ++i) {
            const auto& column = columns_[i];
            if (row->value_ready[i] || column.target != "$agent_step"
                || std::find(completion.resolved_keys.begin(), completion.resolved_keys.end(), column.key)
                    == completion.resolved_keys.end()) continue;
            const auto value = completion.values->GetScalar(column.key, 0);
            if (!value) {
                ANET_SYSTEM_ERROR("Step trace completion scalar is unavailable. column=" << column.number
                    << " key='" << column.key << "' lane=" << completion.lane << " expected=recognized scalar key.");
            }
            row->values[i] = *value;
            row->value_ready[i] = true;
            resolved_columns.push_back(i);
        }
        row->complete = std::all_of(row->value_ready.begin(), row->value_ready.end(), [](bool ready) { return ready; });
        if (!resolved_columns.empty() && completed_) completed_(row, resolved_columns);
        if (row->complete) pending.erase(found);
    }
    for (auto& step : lane_steps_) ++step;
}

