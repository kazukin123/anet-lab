// metrics.hpp
#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include "anet/config.hpp"
#include "anet/rl.hpp"

namespace anet::rl {

    /// 呼び出し中だけ有効な通知の借用。読み手は保持しない。
    struct MetricValueContext {
        const Agent* agent = nullptr; ///< 同期呼び出し元が所有
        const Runner* runner = nullptr; ///< 同期呼び出し元が所有
        const BatchEnv* env = nullptr; ///< 同期呼び出し元が所有
        const BatchExperience* experience = nullptr; ///< 同期呼び出し元が所有
        const BatchActionInfo* action_info = nullptr; ///< 同期呼び出し元が所有
        const BatchStepResult* step_result = nullptr; ///< 同期呼び出し元が所有
        const BatchUpdateResult* update_result = nullptr; ///< 同期呼び出し元が所有
        const anet::Module* agent_step = nullptr; ///< 同期呼び出し元が所有
    };

    class MetricValueReader {
    public:
        static std::optional<float> Read(const MetricValueContext& context,
            EventField field, const std::string& key, int64_t lane = -1);
    };

    struct StepTraceColumnSpec {
        int number = 0;
        std::string target;
        std::string key;
        std::string label;
        std::string display = "value";
        static std::vector<StepTraceColumnSpec> Parse(const ConfigData& config, const EnvSpec& spec);
    };

    struct StepTrace {
        int64_t id = 0;
        step_t step = 0;
        int64_t lane = 0;
        float reward = 0;
        bool done = false;
        bool truncated = false;
        bool complete = false;
        std::vector<float> values;
        std::vector<bool> value_ready; ///< NaNで確定した列もtrueにする。
    };

    class StepTraceBuilder {
    public:
        // 列indexはGetColumns()の0始まり。spanの寿命は同期通知中だけ。
        using Handler = std::function<void(const std::shared_ptr<StepTrace>&, std::span<const size_t>)>;
        StepTraceBuilder(const ConfigData& config, const EnvSpec& spec,
            std::shared_ptr<Agent> agent, int64_t num_lanes);
        using AddedHandler = std::function<void(const std::shared_ptr<StepTrace>&, const TrainEvent&)>;
        void SetHandlers(AddedHandler added, Handler completed);
        void OnTrain(const TrainEvent& event);
        void Reset();
        const std::vector<StepTraceColumnSpec>& GetColumns() const { return columns_; }
    private:
        std::vector<StepTraceColumnSpec> columns_;
        int64_t num_lanes_;
        int64_t next_id_ = 0;
        std::unique_ptr<AgentStepDiagnosticsProvider> diagnostics_;
        std::vector<int64_t> lane_steps_;
        struct PendingRow {
            int64_t step;
            std::shared_ptr<StepTrace> row;
        };
        std::vector<std::deque<PendingRow>> pending_;
        AddedHandler added_;
        Handler completed_;
    };

} // namespace anet::rl
