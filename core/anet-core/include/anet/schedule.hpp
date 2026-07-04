// anet/schedule.hpp

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "anet/config.hpp"
#include "anet/diag.hpp"

namespace anet {

    // ============================================================
    // ProfiledValue
    // ============================================================

    template<typename T>
    struct ProfiledValuePhaseConfig {
        std::string type = "constant";
        T value{};
        T start{};
        T end{};
        uint64_t steps = 0;
        double cycle_mult = 1.0;
    };

    template<typename T>
    struct ProfiledValueConfig {
        std::string type = "constant";
        T value{};
        T start{};
        T end{};
        uint64_t steps = 0;
        double cycle_mult = 1.0;
        std::vector<std::string> phases;
        anet::OrderedMap<std::string, ProfiledValuePhaseConfig<T>> phase;
    };

    template<typename T>
    struct ConfigReader<ProfiledValueConfig<T>> {
        static constexpr bool kEnabled = true;

        static void Read(
            Config& owner,
            const ConfigData& config_data,
            const std::string& key,
            ProfiledValueConfig<T>& value)
        {
            owner.ReadSubConfig(config_data, key, "type", value.type);
            owner.ReadSubConfig(config_data, key, "value", value.value);
            owner.ReadSubConfig(config_data, key, "start", value.start);
            owner.ReadSubConfig(config_data, key, "end", value.end);
            owner.ReadSubConfig(config_data, key, "steps", value.steps);
            owner.ReadSubConfig(config_data, key, "cycle_mult", value.cycle_mult);
            owner.ReadSubConfig(config_data, key, "phases", value.phases);

            value.phase.Clear();
            for (const auto& phase_name : value.phases) {
                ProfiledValuePhaseConfig<T> phase;
                const auto phase_key = owner.MakeTaggedSubConfigKey(key, "phase", phase_name);

                owner.ReadSubConfig(config_data, phase_key, "type", phase.type);
                owner.ReadSubConfig(config_data, phase_key, "value", phase.value);
                owner.ReadSubConfig(config_data, phase_key, "start", phase.start);
                owner.ReadSubConfig(config_data, phase_key, "end", phase.end);
                owner.ReadSubConfig(config_data, phase_key, "steps", phase.steps);
                owner.ReadSubConfig(config_data, phase_key, "cycle_mult", phase.cycle_mult);

                value.phase.Set(phase_name, phase);
            }
        }
    };

    template<typename T>
    class ProfiledValue {
        static_assert(std::is_arithmetic_v<T>, "ProfiledValue<T> requires arithmetic type T.");

    public:
        explicit ProfiledValue(ProfiledValueConfig<T> config)
            : config_(std::move(config))
        {
            ValidateConfig();
            value_ = Evaluate(0);
        }

        void Update(uint64_t step)
        {
            value_ = Evaluate(step);
        }

        T Value() const
        {
            return value_;
        }

        T Evaluate(uint64_t step) const
        {
            return EvaluateRootConfig(config_, step);
        }

        T EvaluateByIndex(size_t index, size_t count) const
        {
            if (config_.type == "constant") {
                return config_.value;
            }
            if (config_.type == "cosine_restart" || config_.type == "phased") {
                ANET_SYSTEM_ERROR("ProfiledValue: EvaluateByIndex does not support type. type=" << config_.type);
            }
            if (config_.type != "linear" && config_.type != "cosine") {
                ANET_SYSTEM_ERROR("ProfiledValue: EvaluateByIndex does not support type. type=" << config_.type);
            }
            if (count <= 1) {
                return config_.start;
            }
            const size_t clamped_index = std::min(index, count - 1);
            const double t = static_cast<double>(clamped_index) / static_cast<double>(count - 1);
            if (config_.type == "linear") {
                return Lerp(config_.start, config_.end, t);
            }

            const double weight = 0.5 * (1.0 + std::cos(kPi * t));
            return static_cast<T>(
                static_cast<double>(config_.end)
                + (static_cast<double>(config_.start) - static_cast<double>(config_.end)) * weight);
        }

    private:
        static constexpr double kPi = 3.14159265358979323846264338327950288;

        static bool IsRootType(const std::string& type)
        {
            return type == "constant"
                || type == "linear"
                || type == "cosine"
                || type == "cosine_restart"
                || type == "phased";
        }

        static bool IsPhaseType(const std::string& type)
        {
            return type == "constant"
                || type == "linear"
                || type == "cosine"
                || type == "cosine_restart";
        }

        static T Lerp(T start, T end, double t)
        {
            return static_cast<T>(
                static_cast<double>(start)
                + (static_cast<double>(end) - static_cast<double>(start)) * t);
        }

        static T EvaluateTimeBased(
            const std::string& type,
            T start,
            T end,
            uint64_t steps,
            double cycle_mult,
            uint64_t step)
        {
            if (type == "linear") {
                const double t = std::min(
                    1.0,
                    static_cast<double>(step) / static_cast<double>(steps));
                return Lerp(start, end, t);
            }

            if (type == "cosine") {
                if (step >= steps) {
                    return end;
                }
                const double t = static_cast<double>(step) / static_cast<double>(steps);
                const double weight = 0.5 * (1.0 + std::cos(kPi * t));
                return static_cast<T>(
                    static_cast<double>(end)
                    + (static_cast<double>(start) - static_cast<double>(end)) * weight);
            }

            if (type == "cosine_restart") {
                double local_step = static_cast<double>(step);
                double cycle_steps = static_cast<double>(steps);

                if (cycle_mult == 1.0) {
                    local_step = std::fmod(local_step, cycle_steps);
                } else {
                    while (local_step >= cycle_steps && cycle_steps > 1.0) {
                        local_step -= cycle_steps;
                        cycle_steps = std::max(1.0, cycle_steps * cycle_mult);
                    }
                    if (local_step >= cycle_steps) {
                        local_step = std::fmod(local_step, cycle_steps);
                    }
                }

                const double t = local_step / cycle_steps;
                const double weight = 0.5 * (1.0 + std::cos(kPi * t));
                return static_cast<T>(
                    static_cast<double>(end)
                    + (static_cast<double>(start) - static_cast<double>(end)) * weight);
            }

            ANET_SYSTEM_ERROR("ProfiledValue: unsupported time-based type. type=" << type);
            return T{};
        }

        static T EvaluatePhaseConfig(const ProfiledValuePhaseConfig<T>& config, uint64_t step)
        {
            if (config.type == "constant") {
                return config.value;
            }
            return EvaluateTimeBased(
                config.type,
                config.start,
                config.end,
                config.steps,
                config.cycle_mult,
                step);
        }

        static T EvaluatePhaseEnd(const ProfiledValuePhaseConfig<T>& config)
        {
            if (config.type == "constant") {
                return config.value;
            }
            return config.end;
        }

        T EvaluateRootConfig(const ProfiledValueConfig<T>& config, uint64_t step) const
        {
            if (config.type == "constant") {
                return config.value;
            }
            if (config.type == "linear" || config.type == "cosine" || config.type == "cosine_restart") {
                return EvaluateTimeBased(
                    config.type,
                    config.start,
                    config.end,
                    config.steps,
                    config.cycle_mult,
                    step);
            }
            if (config.type == "phased") {
                uint64_t phase_step = step;
                for (const auto& phase_name : config.phases) {
                    const auto& phase = config.phase.Get(phase_name);
                    if (phase_step < phase.steps) {
                        return EvaluatePhaseConfig(phase, phase_step);
                    }
                    phase_step -= phase.steps;
                }
                return EvaluatePhaseEnd(config.phase.Get(config.phases.back()));
            }

            ANET_SYSTEM_ERROR("ProfiledValue: unknown type. type=" << config.type);
            return T{};
        }

        void ValidatePhaseConfig(
            const std::string& parent_type, const std::string& phase_name, const ProfiledValuePhaseConfig<T>& config) const
        {
            if (!IsPhaseType(config.type)) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: invalid phase type. phase=" << phase_name
                    << " type=" << config.type
                    << " expected=constant|linear|cosine|cosine_restart");
            }
            if (parent_type == "phased" && config.steps == 0) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: phase steps must be greater than 0. phase=" << phase_name
                    << " steps=" << config.steps);
            }
            if ((config.type == "linear" || config.type == "cosine" || config.type == "cosine_restart")
                && config.steps == 0) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: steps must be greater than 0. phase=" << phase_name
                    << " type=" << config.type
                    << " steps=" << config.steps);
            }
            if (config.type == "cosine_restart" && config.cycle_mult <= 0.0) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: cycle_mult must be greater than 0. phase=" << phase_name
                    << " cycle_mult=" << config.cycle_mult);
            }
        }

        void ValidateConfig() const
        {
            if (!IsRootType(config_.type)) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: unknown type. type=" << config_.type
                    << " expected=constant|linear|cosine|cosine_restart|phased");
            }
            if ((config_.type == "linear" || config_.type == "cosine" || config_.type == "cosine_restart")
                && config_.steps == 0) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: steps must be greater than 0. type=" << config_.type
                    << " steps=" << config_.steps);
            }
            if (config_.type == "cosine_restart" && config_.cycle_mult <= 0.0) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: cycle_mult must be greater than 0. cycle_mult=" << config_.cycle_mult);
            }
            if (config_.type == "phased") {
                if (config_.phases.empty()) {
                    ANET_SYSTEM_ERROR("ProfiledValue: phases must not be empty for phased type.");
                }
                for (const auto& phase_name : config_.phases) {
                    if (config_.phase.find(phase_name) == config_.phase.end()) {
                        ANET_SYSTEM_ERROR("ProfiledValue: phase is listed but not defined. phase=" << phase_name);
                    }
                    ValidatePhaseConfig(config_.type, phase_name, config_.phase.Get(phase_name));
                }
            }
        }

        ProfiledValueConfig<T> config_;
        T value_{};
    };

} // namespace anet
