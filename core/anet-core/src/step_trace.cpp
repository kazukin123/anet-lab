// step_trace.cpp
#include "anet/step_trace.hpp"
#include <sstream>
#include <array>
#include <format>
#include <cmath>
#include <charconv>
#include <set>
#include "metric_tokens.hpp"
#include "anet/str_util.hpp"
#include "anet/observers.hpp"
#include "anet/profile.hpp"

using namespace anet::rl;

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
        if (column.target == "$agent_step" && column.key != "q_sa"
            && column.key != "state_value" && column.key != "approx_priority") {
            invalid("q_sa, state_value or approx_priority for $agent_step");
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
    diagnostics_ = agent ? agent->CreateStepDiagnostics(num_lanes) : nullptr;
    lane_steps_.resize(static_cast<size_t>(num_lanes), 0);
    pending_.resize(static_cast<size_t>(num_lanes));
    has_priority_column_ = std::any_of(columns_.begin(), columns_.end(), [](const auto& column) {
        return column.target == "$agent_step" && column.key == "approx_priority";
    });
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
    // 診断は一度だけbatchで進め、Qと将来確定した値を同じ遷移へ結び付ける。
    ANET_CHECK_MSG(event.step_result != nullptr && event.step_result->reward.size(0) == num_lanes_,
        "Step trace requires a matching Step result. expected_lanes=" << num_lanes_);
    auto values = diagnostics_ ? diagnostics_->Observe(event.experience) : AgentStepDiagnosticsResult{};
    if (diagnostics_) {
        ANET_CHECK_MSG(values.current.size() == static_cast<size_t>(num_lanes_),
            "Step diagnostics lane count mismatch. expected=" << num_lanes_);
    }
    const AgentStepValues unavailable;
    for (int64_t lane = 0; lane < num_lanes_; ++lane) {
        const MetricValueContext context{ .agent = event.agent.get(), .runner = event.runner.get(),
            .env = event.env.get(), .experience = &event.experience,
            .action_info = event.action_info.get(), .step_result = event.step_result.get(),
            .agent_step = diagnostics_ ? &values.current[lane] : &unavailable };
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
            const auto value = MetricValueReader::Read(context, field, column.key, lane);
            if (!value) {
                ANET_SYSTEM_ERROR("Step trace scalar is unavailable. column=" << column.number
                    << " target='" << column.target << "' key='" << column.key << "' lane=" << lane
                    << " expected=recognized scalar key.");
            }
            row->values.push_back(*value);
        }
        row->complete = !diagnostics_ || !has_priority_column_;
        if (!row->complete) pending_[lane].push_back({ .step = lane_steps_[lane], .row = row });
        if (added_) added_(row, event);
        if (row->complete && completed_) completed_(row);
    }
    // 保持画像を持たないcoreの行だけを待ち、遅延列を一度だけ確定する。
    for (const auto& completion : values.completed) {
        ANET_CHECK(completion.lane >= 0 && completion.lane < num_lanes_ && completion.steps_back >= 0);
        auto& pending = pending_[completion.lane];
        const auto wanted = lane_steps_[completion.lane] - completion.steps_back;
        const auto found = std::find_if(pending.begin(), pending.end(), [&](const auto& value) {
            return value.step == wanted;
        });
        if (found == pending.end()) continue;
        auto row = found->row;
        for (size_t i = 0; i < columns_.size(); ++i) {
            if (columns_[i].target == "$agent_step" && columns_[i].key == "approx_priority") {
                row->values[i] = completion.approx_priority;
            }
        }
        row->complete = true;
        if (completed_) completed_(row);
        pending.erase(found);
    }
    for (auto& step : lane_steps_) ++step;
}

void TracePlayer::SetRows(std::vector<int64_t> oldest_to_newest)
{
    // 行IDで選択を追跡し、先頭への挿入で選択や位置を動かさない。
    rows_ = std::move(oldest_to_newest);
    if (rows_.empty()) {
        selected_.reset();
        playing_ = false;
        progress_ = 0;
    } else if (selected_ && std::find(rows_.begin(), rows_.end(), *selected_) == rows_.end()) {
        selected_ = rows_.front();
        progress_ = 0;
    }
}

void TracePlayer::Select(int64_t id, bool keep_playing)
{
    ANET_CHECK_MSG(std::find(rows_.begin(), rows_.end(), id) != rows_.end(),
        "Trace selection is unavailable. id=" << id << " expected=a retained row.");
    selected_ = id;
    if (!keep_playing) playing_ = false;
    progress_ = 0;
}

bool TracePlayer::CanPlay(double base_steps_per_second) const
{
    ANET_CHECK_MSG(std::isfinite(base_steps_per_second) && base_steps_per_second >= 0,
        "Invalid trace playback rate. base_rate=" << base_steps_per_second << " expected=finite nonnegative value.");
    return !rows_.empty() && base_steps_per_second > 0;
}

std::string TracePlayer::Play()
{
    if (rows_.empty()) return {};
    // Liveと最新での再生開始だけを最古へ戻す。
    if (!selected_ || (!playing_ && *selected_ == rows_.back())) selected_ = rows_.front();
    playing_ = true;
    progress_ = 0;
    return "Trace: play";
}

std::string TracePlayer::Pause()
{
    if (rows_.empty()) return {};
    if (!selected_) selected_ = rows_.back();
    playing_ = false;
    progress_ = 0;
    return "Trace: pause";
}

std::string TracePlayer::Stop()
{
    selected_.reset();
    playing_ = false;
    progress_ = 0;
    return "Trace: stop (live)";
}

std::string TracePlayer::StepForward()
{
    if (rows_.empty()) return {};
    if (!selected_) {
        selected_ = rows_.back();
        return "Trace: step";
    }
    playing_ = false;
    progress_ = 0;
    const auto found = std::find(rows_.begin(), rows_.end(), *selected_);
    if (std::next(found) == rows_.end()) return "Trace: end";
    selected_ = *std::next(found);
    return "Trace: step";
}

std::string TracePlayer::StepBack()
{
    if (rows_.empty()) return {};
    if (!selected_) {
        selected_ = rows_.back();
        return "Trace: step back";
    }
    playing_ = false;
    progress_ = 0;
    const auto found = std::find(rows_.begin(), rows_.end(), *selected_);
    if (found == rows_.begin()) return "Trace: start";
    selected_ = *std::prev(found);
    return "Trace: step back";
}

double TracePlayer::GetSpeed() const
{
    return std::ldexp(1.0, speed_index_ - 3);
}

std::string TracePlayer::GetSpeedLabel() const
{
    const std::array<const char*, 7> labels{ "x1/8", "x1/4", "x1/2", "x1", "x2", "x4", "x8" };
    return labels[speed_index_];
}

std::string TracePlayer::Faster()
{
    const bool at_end = speed_index_ == 6;
    if (!at_end) ++speed_index_;
    return "Trace: speed " + GetSpeedLabel() + (at_end ? " (max)" : "");
}

std::string TracePlayer::Slower()
{
    const bool at_end = speed_index_ == 0;
    if (!at_end) --speed_index_;
    return "Trace: speed " + GetSpeedLabel() + (at_end ? " (min)" : "");
}

std::string TracePlayer::Tick(double seconds, double base_steps_per_second)
{
    ANET_CHECK_MSG(std::isfinite(seconds) && seconds >= 0 && std::isfinite(base_steps_per_second)
        && base_steps_per_second >= 0, "Invalid trace playback clock. seconds=" << seconds
        << " base_rate=" << base_steps_per_second << " expected=finite nonnegative values.");
    if (!playing_ || !selected_ || rows_.empty()) return {};
    // 端へ到達した瞬間に止め、実行環境のfps変更にも次のtickから追随する。
    progress_ += seconds * base_steps_per_second * GetSpeed();
    while (progress_ >= 1.0) {
        progress_ -= 1.0;
        auto found = std::find(rows_.begin(), rows_.end(), *selected_);
        if (std::next(found) != rows_.end()) selected_ = *std::next(found);
        if (*selected_ == rows_.back()) {
            playing_ = false;
            progress_ = 0;
            return "Trace: end";
        }
    }
    return {};
}

float anet::rl::AppendTracePercentile(std::deque<float>& values, float value, size_t window_size)
{
    ANET_PROFILE_FUNC();
    ANET_CHECK_MSG(window_size > 0, "Trace percentile window must be positive. actual=" << window_size);
    // 不成立値は窓を進めず、有限値の直近窓へ自分を含めて中央順位を求める。
    if (!std::isfinite(value)) return std::numeric_limits<float>::quiet_NaN();
    values.push_back(value);
    if (values.size() > window_size) values.pop_front();
    size_t less = 0;
    size_t equal = 0;
    for (const float previous : values) {
        if (previous < value) ++less;
        else if (previous == value) ++equal;
    }
    return static_cast<float>(100.0 * (static_cast<double>(less) + 0.5 * equal) / values.size());
}

std::string StepTraceColumnSpec::Format(float value, const ActionSpec& actions) const
{
    // 空欄は非有限値だけに使い、大きな整数を有効数字で丸めない。
    if (!std::isfinite(value)) return {};
    if (target == "$action_info" && key == "action") {
        ANET_CHECK_MSG(actions.is_discrete && std::trunc(value) == value && value >= 0
            && value < static_cast<float>(actions.value_labels.size()),
            "Trace action is out of range. column=" << number << " value=" << value
                << " expected=[0," << actions.value_labels.size() << ").");
        const auto index = static_cast<size_t>(value);
        if (display == "name") return actions.value_labels[index];
        if (display == "value_name") return std::to_string(index) + ":" + actions.value_labels[index];
    }
    return std::trunc(value) == value ? std::format("{:.0f}", value) : std::format("{:.4g}", value);
}
