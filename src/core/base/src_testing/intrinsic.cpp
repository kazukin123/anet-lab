#include "anet/test/intrinsic.hpp"
#include "intrinsic_rnd.hpp"

anet::test::RndConfig anet::test::ReadRndConfig(const ConfigData& data, const std::string& prefix)
{
    const rl::intrinsic::RndNoveltyEstimatorConfig config(data, prefix);
    const auto& standard = config.error_norm.standardize;
    const auto& feedback = config.statistics.variance_feedback;
    const bool use_feedback = config.statistics.mode == rl::intrinsic::kStatisticsVarianceFeedback;
    return {
        .use_clipping = use_feedback ? feedback.use_clipping : standard.use_clipping,
        .statistics = config.statistics.mode,
        .multiplier_max = use_feedback ? feedback.multiplier_max : standard.multiplier_max,
        .error_warmup_steps = standard.warmup_steps,
        .predictor_warmup_steps = config.predictor.warmup_steps,
        .standardize = {.average_mode = standard.average_mode, .ema_steps = standard.ema_steps, .use_centering = standard.use_centering},
        .observation_clip_range = use_feedback ? feedback.obs_norm.clip_range : config.statistics.whiten.obs_norm.clip_range,
        .head_init = config.head.init,
        .adam_eps = config.adam_eps,
    };
}
