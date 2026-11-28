#include "dqn_config.hpp"

using namespace anet::rl::dqn;

void LearnerConfig::ValidateTrainingValues() const
{
    const auto check = [&](bool valid, const char* key, auto value, const char* expected) {
        if (!valid) {
            ANET_SYSTEM_ERROR("Invalid " << full_keys.GetFullKey(key) << "=" << value << "; expected " << expected << ".");
        }
    };
    constexpr char kPositiveFloat[] = "finite value > 0";
    constexpr char kPositiveInteger[] = "integer >= 1";
    check(std::isfinite(grad_clip_tau) && grad_clip_tau > 0.0f, "grad_clip_tau", grad_clip_tau, kPositiveFloat);
    check(std::isfinite(td_clip_value) && td_clip_value > 0.0f, "td_clip_value", td_clip_value, kPositiveFloat);
    check(update_interval >= 1, "update_interval", update_interval, kPositiveInteger);
    check(std::isfinite(quantile_huber_kappa) && quantile_huber_kappa > 0.0f,
        "quantile_huber_kappa", quantile_huber_kappa, kPositiveFloat);
    check(std::isfinite(per_alpha) && per_alpha >= 0.0f, "per_alpha", per_alpha, "finite value >= 0");
    check(std::isfinite(replay_ratio) && replay_ratio != 0.0f,
        "replay_ratio", replay_ratio, "finite value > 0, or < 0 to use update_interval");
    check(replay_batch_size >= 1, "replay_batch_size", replay_batch_size, kPositiveInteger);
}

const anet::ProfiledValueConfig<float>& LearnerConfig::GetIntrinsicGammaConfig() const
{
    return intrinsic.decomposed && intrinsic.discount_mode == kDiscountSeparate ? intrinsic.gamma : gamma;
}

std::string anet::rl::dqn::GetInitialProfiledValueKey(const anet::ProfiledValueConfig<float>& config)
{
    if (config.type == "phased") {
        const auto& phase = config.phase.Get(config.phases.front());
        return phase.full_keys.GetFullKey(phase.type == "constant" ? "value" : "start");
    }
    return config.full_keys.GetFullKey(config.type == "constant" ? "value" : "start");
}
