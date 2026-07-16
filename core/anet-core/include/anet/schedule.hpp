// anet/schedule.hpp

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
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
        std::optional<T> min_value; ///< 設定入出力には含めないinclusiveな下限。
        std::optional<T> max_value; ///< 設定入出力には含めないinclusiveな上限。
    };

    namespace detail {

        inline bool IsProfiledValueRootType(const std::string& type)
        {
            return type == "constant"
                || type == "linear"
                || type == "cosine"
                || type == "cosine_restart"
                || type == "phased";
        }

        inline bool IsProfiledValuePhaseType(const std::string& type)
        {
            return type == "constant"
                || type == "linear"
                || type == "cosine"
                || type == "cosine_restart";
        }

        template<typename T>
        bool IsFiniteProfiledValue(T value)
        {
            if constexpr (std::is_floating_point_v<T>) {
                return std::isfinite(value);
            }
            return true;
        }

        template<typename T>
        void ValidateProfiledValueBounds(
            const ProfiledValueConfig<T>& config,
            const std::string& key,
            T value)
        {
            if (!IsFiniteProfiledValue(value)) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: active value must be finite. key=" << key
                    << " value=" << value << " expected=finite");
            }
            if (config.min_value.has_value() && value < *config.min_value) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: active value is below min_value. key=" << key
                    << " value=" << value
                    << " min_value=" << *config.min_value
                    << " expected=>=" << *config.min_value);
            }
            if (config.max_value.has_value() && value > *config.max_value) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: active value is above max_value. key=" << key
                    << " value=" << value
                    << " max_value=" << *config.max_value
                    << " expected=<=" << *config.max_value);
            }
        }

        template<typename T>
        void ValidateProfiledValuePhaseConfig(
            const ProfiledValueConfig<T>& root_config,
            const std::string& root_key,
            const std::string& phase_name,
            const ProfiledValuePhaseConfig<T>& config)
        {
            const std::string phase_key = root_key + ".phase.[" + phase_name + "]";
            if (!IsProfiledValuePhaseType(config.type)) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: invalid phase type. key=" << phase_key + ".type"
                    << " value=" << config.type
                    << " expected=constant|linear|cosine|cosine_restart");
            }
            if (config.steps == 0) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: phase steps must be greater than 0. key=" << phase_key + ".steps"
                    << " value=" << config.steps << " expected=>=1");
            }
            if (config.type == "cosine_restart"
                && (!std::isfinite(config.cycle_mult) || config.cycle_mult <= 0.0)) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: cycle_mult must be greater than 0. key=" << phase_key + ".cycle_mult"
                    << " value=" << config.cycle_mult << " expected=>0");
            }

            // phase typeが実際に参照する値だけを共通boundsで検証する。
            if (config.type == "constant") {
                ValidateProfiledValueBounds(root_config, phase_key + ".value", config.value);
            } else {
                ValidateProfiledValueBounds(root_config, phase_key + ".start", config.start);
                ValidateProfiledValueBounds(root_config, phase_key + ".end", config.end);
            }
        }

        template<typename T>
        void ValidateProfiledValueConfig(
            const ProfiledValueConfig<T>& config,
            const std::string& root_key = "ProfiledValue")
        {
            // bounds自体を先に検証し、active fieldのエラーへ誤って見せない。
            if (config.min_value.has_value() && !IsFiniteProfiledValue(*config.min_value)) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: min_value must be finite. key=" << root_key + ".min_value"
                    << " value=" << *config.min_value << " expected=finite");
            }
            if (config.max_value.has_value() && !IsFiniteProfiledValue(*config.max_value)) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: max_value must be finite. key=" << root_key + ".max_value"
                    << " value=" << *config.max_value << " expected=finite");
            }
            if (config.min_value.has_value()
                && config.max_value.has_value()
                && *config.min_value > *config.max_value) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: min_value must not exceed max_value. key=" << root_key
                    << " min_value=" << *config.min_value
                    << " max_value=" << *config.max_value);
            }
            if (!IsProfiledValueRootType(config.type)) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: unknown type. key=" << root_key + ".type"
                    << " value=" << config.type
                    << " expected=constant|linear|cosine|cosine_restart|phased");
            }
            if ((config.type == "linear" || config.type == "cosine" || config.type == "cosine_restart")
                && config.steps == 0) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: steps must be greater than 0. key=" << root_key + ".steps"
                    << " value=" << config.steps << " expected=>=1");
            }
            if (config.type == "cosine_restart"
                && (!std::isfinite(config.cycle_mult) || config.cycle_mult <= 0.0)) {
                ANET_SYSTEM_ERROR(
                    "ProfiledValue: cycle_mult must be greater than 0. key=" << root_key + ".cycle_mult"
                    << " value=" << config.cycle_mult << " expected=>0");
            }

            // 最終typeが選択するfieldだけを検証し、dormant fieldは保持する。
            if (config.type == "constant") {
                ValidateProfiledValueBounds(config, root_key + ".value", config.value);
            } else if (config.type == "linear" || config.type == "cosine" || config.type == "cosine_restart") {
                ValidateProfiledValueBounds(config, root_key + ".start", config.start);
                ValidateProfiledValueBounds(config, root_key + ".end", config.end);
            } else {
                if (config.phases.empty()) {
                    ANET_SYSTEM_ERROR(
                        "ProfiledValue: phases must not be empty for phased type. key=" << root_key + ".phases");
                }
                for (const auto& phase_name : config.phases) {
                    if (config.phase.find(phase_name) == config.phase.end()) {
                        ANET_SYSTEM_ERROR(
                            "ProfiledValue: phase is listed but not defined. key=" << root_key + ".phases"
                            << " value=" << phase_name << " expected=defined phase");
                    }
                    ValidateProfiledValuePhaseConfig(
                        config, root_key, phase_name, config.phase.Get(phase_name));
                }
            }
        }

    } // namespace detail

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

            for (const auto& phase_name : value.phases) {
                const auto phase_key = owner.MakeTaggedSubConfigKey(key, "phase", phase_name);
                const bool has_explicit_definition =
                    owner.HasConfigValue(config_data, phase_key + ".type")
                    || owner.HasConfigValue(config_data, phase_key + ".value")
                    || owner.HasConfigValue(config_data, phase_key + ".start")
                    || owner.HasConfigValue(config_data, phase_key + ".end")
                    || owner.HasConfigValue(config_data, phase_key + ".steps")
                    || owner.HasConfigValue(config_data, phase_key + ".cycle_mult");
                const auto existing_phase = value.phase.find(phase_name);
                if (!has_explicit_definition && existing_phase == value.phase.end()) {
                    continue;
                }

                // programmaticな既定phaseを保持し、明示fieldだけをlayer順に上書きする。
                ProfiledValuePhaseConfig<T> phase = existing_phase != value.phase.end()
                    ? value.phase.Get(phase_name)
                    : ProfiledValuePhaseConfig<T>{};

                owner.ReadSubConfig(config_data, phase_key, "type", phase.type);
                owner.ReadSubConfig(config_data, phase_key, "value", phase.value);
                owner.ReadSubConfig(config_data, phase_key, "start", phase.start);
                owner.ReadSubConfig(config_data, phase_key, "end", phase.end);
                owner.ReadSubConfig(config_data, phase_key, "steps", phase.steps);
                owner.ReadSubConfig(config_data, phase_key, "cycle_mult", phase.cycle_mult);

                value.phase.Set(phase_name, phase);
            }
            detail::ValidateProfiledValueConfig(value, key);
        }
    };

    template<typename T>
    class ProfiledValue {
        static_assert(std::is_arithmetic_v<T>, "ProfiledValue<T> requires arithmetic type T.");

    public:
        explicit ProfiledValue(ProfiledValueConfig<T> config)
            : config_(std::move(config))
        {
            detail::ValidateProfiledValueConfig(config_);
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

        ProfiledValueConfig<T> config_;
        T value_{};
    };

} // namespace anet
