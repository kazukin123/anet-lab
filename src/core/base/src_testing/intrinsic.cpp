#include "anet/test/intrinsic.hpp"
#include "intrinsic_rnd.hpp"

anet::test::RndConfig anet::test::ReadRndConfig(const ConfigData& data, const std::string& prefix)
{
    const rl::intrinsic::RndNoveltyEstimatorConfig config(data, prefix);
    return {
        .multiplier_mode = config.multiplier_mode,
        .statistics_mode = config.statistics_mode,
        .multiplier_max = config.multiplier_max,
        .warmup_steps = config.warmup_steps,
        .warmup_mode = config.warmup_mode,
        .error_norm = { config.error_norm.average_mode, config.error_norm.ema_steps, config.error_norm.use_centering },
        .observation_clip_range = config.obs_norm.clip_range,
        .head_init = config.head.init,
        .adam_eps = config.adam_eps,
    };
}
