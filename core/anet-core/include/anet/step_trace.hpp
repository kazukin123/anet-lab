// step_trace.hpp
#pragma once

#include <functional>
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include "anet/config.hpp"
#include "anet/rl.hpp"

namespace anet::rl {

    struct StepTraceColumnSpec {
        int number = 0;
        std::string target;
        std::string key;
        std::string label;
        std::string display = "value";
        static std::vector<StepTraceColumnSpec> Parse(const ConfigData& config, const EnvSpec& spec);
        std::string Format(float value, const ActionSpec& actions) const;
    };

    struct StepTrace {
        int64_t id = 0;
        int64_t step = 0;
        int64_t lane = 0;
        float reward = 0;
        bool done = false;
        bool truncated = false;
        bool complete = false;
        std::vector<float> values;
    };

    class StepTraceBuilder {
    public:
        using Handler = std::function<void(const std::shared_ptr<StepTrace>&)>;
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
        std::unique_ptr<AgentStepDiagnostics> diagnostics_;
        std::vector<int64_t> lane_steps_;
        struct PendingRow {
            int64_t step;
            std::shared_ptr<StepTrace> row;
        };
        std::vector<std::deque<PendingRow>> pending_;
        bool has_priority_column_ = false;
        AddedHandler added_;
        Handler completed_;
    };

    float AppendTracePercentile(std::deque<float>& values, float value, size_t window_size);

    class TracePlayer {
    public:
        void SetRows(std::vector<int64_t> oldest_to_newest);
        void Select(int64_t id, bool keep_playing = false);
        bool CanPlay(double base_steps_per_second) const;
        std::string Play();
        std::string Pause();
        std::string Stop();
        std::string StepForward();
        std::string StepBack();
        std::string Faster();
        std::string Slower();
        std::string Tick(double seconds, double base_steps_per_second);
        std::optional<int64_t> GetSelected() const { return selected_; }
        bool IsPlaying() const { return playing_; }
        double GetSpeed() const;
        std::string GetSpeedLabel() const;
    private:
        std::vector<int64_t> rows_;
        std::optional<int64_t> selected_;
        bool playing_ = false;
        int speed_index_ = 3;
        double progress_ = 0;
    };

} // namespace anet::rl
