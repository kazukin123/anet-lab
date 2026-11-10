#include "anet/catch_test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include "anet/nn.hpp"
#include "nn_impl.hpp"
#include "dqn_based_agent.hpp"

using namespace anet::rl::dqn;

constexpr const char* kGradNorm = "grad_norm";
constexpr const char* kPerPrioClipRatio = "per_prio_clip_ratio";
constexpr const char* kPlasticitySrank = "plasticity_srank";
constexpr const char* kPlasticityWeightNormFeature = "plasticity_weight_norm_feature";

static std::shared_ptr<anet::nn::Network> MakeScalarSpectralNetwork()
{
    // 実物のNetworkを構築し、公開されているSN診断用tensorだけを変更する。
    anet::nn::InitNN();
    constexpr const char* kLinearBlock = "Linear";
    anet::ConfigData config;
    config.Set("net.block.[Linear].type", kLinearBlock);
    config.Set("net.block.[Linear].linear.out_features", 2);
    config.Set("net.block.[Linear].weight_norm.mode", "spectral");
    config.Set("net.branch.[feature].bind", anet::rl::ObsKeys::kVector);
    config.Set("net.branch.[feature].structure", kLinearBlock);
    config.Set("net.body.output.[feature]", "feature");
    const anet::TensorSpec spec{ .type = anet::SpaceType::Vector, .shape = {2}, .dtype = torch::kFloat32 };
    return anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(config),
        {{anet::rl::ObsKeys::kVector, spec}}, nullptr, 122, torch::Device(torch::kCPU));
}

TEST_CASE("Batch update scalar characterization covers all recognized keys", "[prd122][dqn][scalar]")
{
    // 固定した小さな値から、キーごとの現行の戻り値を独立に計算する。
    const float nan = std::numeric_limits<float>::quiet_NaN();
    BatchUpdateResult empty;
    BatchUpdateResult full;
    full.loss = torch::tensor(2.0f);
    full.td_error = torch::tensor({-1.0f, 3.0f});
    full.grad_norm = 7.0f;
    full.grad_norm_tensor = torch::tensor(5.0f);
    full.grad_clip_tau = 4.0f;
    full.max_q = torch::tensor({2.0f, 6.0f});
    full.max_q_real = torch::tensor({4.0f, 8.0f});
    full.q_sa = torch::tensor({1.0f, 3.0f});
    full.q_sa_real = torch::tensor({3.0f, 5.0f});
    full.q_std = torch::tensor(1.5f);
    full.q_gap = torch::tensor(2.5f);
    full.q_gap_rel = torch::tensor(0.5f);
    full.iqn_diagnostics = torch::arange(1, 8, torch::kFloat32);
    full.upper_tail_priority_spearman = 0.75f;
    full.per_is_weights = torch::tensor({1.0f, 3.0f});
    full.per_priorities = torch::tensor({2.0f, 6.0f});
    full.per_clipped_count = torch::tensor(1.0f);
    full.per_sample_initial_count = torch::tensor(3.0f);
    full.per_sample_fixed_initial_count = torch::tensor(1.0f);
    full.per_sample_max_initial_count = torch::tensor(1.0f);
    full.per_sample_actor_initial_count = torch::tensor(1.0f);
    full.per_minibatch_size = 4;
    full.per_update_result.applied_count = 3;
    full.per_update_result.stale_count = 1;
    full.per_update_result.actor_learner_pair_count = 5;
    full.per_update_result.actor_learner_positive_pair_ratio = 0.6f;
    full.per_update_result.actor_learner_ratio_median = 1.5f;
    full.per_update_result.actor_learner_log_ratio_mean = 0.2f;
    full.per_update_result.actor_learner_spearman = 0.8f;
    const std::array<std::tuple<const char*, float, float>, 43> direct{{
        {"loss", 2.0f, nan}, {"td_mean", 2.0f, nan}, {"td_std", std::sqrt(2.0f), nan},
        {kGradNorm, 7.0f, nan}, {"grad_clip_ratio", 1.0f, nan},
        {"q_max_max", 6.0f, nan}, {"q_max_mean", 4.0f, nan}, {"q_max_std", 2.0f, nan},
        {"q_sa_mean", 2.0f, nan}, {"q_max_real_max", 8.0f, nan}, {"q_max_real_mean", 6.0f, nan},
        {"q_max_real_std", 2.0f, nan}, {"q_sa_real_mean", 4.0f, nan}, {"q_std", 1.5f, nan},
        {"q_gap", 2.5f, nan}, {"q_gap_rel", 0.5f, nan},
        {"iqn_current_mc_scale", 1.0f, nan}, {"iqn_target_mc_scale", 2.0f, nan},
        {"iqn_priority_mc_ratio", 3.0f, nan}, {"iqn_first_priority_mc_ratio", 4.0f, nan},
        {"iqn_first_pair_abs_td", 5.0f, nan}, {"iqn_first_cancellation_ratio", 6.0f, nan},
        {"iqn_first_quantile_loss_norm", 7.0f, nan}, {"upper_tail_priority_spearman", 0.75f, nan},
        {"per_td_error_abs_max", 3.0f, nan}, {kPerPrioClipRatio, 0.25f, nan},
        {"per_sample_initial_ratio", 0.75f, nan}, {"per_sample_initial_count", 3.0f, 0.0f},
        {"per_sample_fixed_initial_ratio", 0.25f, nan}, {"per_sample_max_initial_ratio", 0.25f, nan},
        {"per_sample_actor_initial_ratio", 0.25f, nan}, {"per_priority_update_stale_ratio", 0.25f, nan},
        {"per_actor_learner_pair_count", 5.0f, 0.0f}, {"per_actor_learner_positive_pair_ratio", 0.6f, nan},
        {"per_actor_learner_ratio_median", 1.5f, nan}, {"per_actor_learner_log_ratio_mean", 0.2f, nan},
        {"per_actor_learner_spearman", 0.8f, nan}, {"per_prio_max", 6.0f, nan},
        {"per_batch_prio_mean", 4.0f, nan}, {"per_is_weight_mean", 2.0f, nan},
        {"per_prio_cv", std::sqrt(8.0f) / 4.0f, nan}, {"per_is_ess_ratio", 0.4f, nan},
        {"per_prio_ess_ratio", 0.4f, nan},
    }};
    const auto check = [&](const std::string& key, float expected, float missing = std::numeric_limits<float>::quiet_NaN()) {
        INFO(key);
        const auto actual = full.GetScalar(key);
        const auto absent = empty.GetScalar(key);
        REQUIRE(actual.has_value());
        REQUIRE(absent.has_value());
        if (std::isnan(expected)) CHECK(std::isnan(*actual));
        else CHECK(*actual == Catch::Approx(expected).margin(1e-6));
        if (std::isnan(missing)) CHECK(std::isnan(*absent));
        else CHECK(*absent == missing);
    };
    for (const auto& [key, present, missing] : direct) check(key, present, missing);

    // 配列で認識する群は全要素を違う値にし、添字の取り違えも検出する。
    full.skip_metrics = torch::arange(1, 8, torch::kFloat32);
    full.grad_noise.emplace();
    full.noisy_metrics.emplace();
    full.replay_fit_metrics.emplace();
    full.munchausen_diagnostics = torch::arange(1, 6, torch::kFloat32);
    full.policy_churn_metrics = torch::arange(1, 8, torch::kFloat32);
    for (size_t i = 0; i < kGradientNoiseMetricKeys.size(); ++i) (*full.grad_noise)[i] = float(i + 1);
    for (size_t i = 0; i < kNoisyMetricKeys.size(); ++i) (*full.noisy_metrics)[i] = float(i + 1);
    for (size_t i = 0; i < kReplayFitMetricKeys.size(); ++i) (*full.replay_fit_metrics)[i] = float(i + 1);
    const auto check_group = [&](const auto& keys) {
        for (size_t i = 0; i < keys.size(); ++i) check(keys[i], float(i + 1));
    };
    check_group(kSkipUpdateMetricKeys);
    check_group(kGradientNoiseMetricKeys);
    check_group(kNoisyMetricKeys);
    check_group(kReplayFitMetricKeys);
    check_group(kMunchausenMetricKeys);
    check_group(kPolicyChurnMetricKeys);
    full.attention_entropy = torch::tensor({0.25f});
    check("attention_entropy.[0]", 0.25f);
    check("attention_entropy.[99]", nan);

    // 可塑性は公開の要求と特徴量から全suffixを計算し、online/target双方で引く。
    full.plasticity_features = torch::eye(2);
    full.plasticity_target_features = torch::eye(2);
    full.plasticity_request = anet::nn::PlasticityMetricRequest::All();
    full.plasticity_target_request = anet::nn::PlasticityMetricRequest::All();
    const std::array<std::pair<const char*, float>, 9> plasticity{{
        {"srank", 2.0f}, {"srank_delta_005", 2.0f}, {"srank_delta_020", 2.0f},
        {"srank_ratio", 1.0f}, {"srank_ratio_delta_005", 1.0f}, {"srank_ratio_delta_020", 1.0f},
        {"dormant_ratio", 0.0f}, {"dead_ratio", 0.0f}, {"feature_norm", 1.0f},
    }};
    for (const auto& [suffix, expected] : plasticity) {
        check(std::string("plasticity_") + suffix, expected);
        check(std::string("plasticity_target_") + suffix, expected);
    }
    full.plasticity_weight_norms = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f});
    const std::array<const char*, 6> norms{{kPlasticityWeightNormFeature, "plasticity_weight_norm_readout",
        "plasticity_weight_norm_feature_effective", "plasticity_weight_norm_readout_effective",
        "plasticity_spectral_sigma_feature", "plasticity_spectral_sigma_readout"}};
    check_group(norms);
    CHECK_FALSE(empty.GetScalar("prd122_unknown").has_value());

    // 同じキーの値未成立と、優先順位・要求の境目も現行どおりに保つ。
    full.grad_norm.reset();
    CHECK(*full.GetScalar(kGradNorm) == 5.0f);
    full.plasticity_request = {};
    CHECK(std::isnan(*full.GetScalar(kPlasticitySrank)));
    full.per_minibatch_size = 0;
    CHECK(std::isnan(*full.GetScalar(kPerPrioClipRatio)));
}

TEST_CASE("Batch update scalar characterization covers all system error paths", "[prd122][dqn][scalar]")
{
    BatchUpdateResult result;
    SECTION("Malformed attention key") {
        CHECK_THROWS_WITH(result.GetScalar("attention_entropy.bad"), Catch::Matchers::ContainsSubstring("Invalid scalar key"));
    }
    SECTION("Munchausen pack shape") {
        result.munchausen_diagnostics = torch::ones({4});
        CHECK_THROWS_WITH(result.GetScalar(kMunchausenMetricKeys[0]), Catch::Matchers::ContainsSubstring("expected [5]"));
    }
    SECTION("Policy churn pack size") {
        result.policy_churn_metrics = torch::ones({6});
        CHECK_THROWS_WITH(result.GetScalar(kPolicyChurnMetricKeys[0]), Catch::Matchers::ContainsSubstring("invalid size"));
    }
    SECTION("Weight norm pack size") {
        result.plasticity_weight_norms = torch::ones({7});
        CHECK_THROWS_WITH(result.GetScalar(kPlasticityWeightNormFeature), Catch::Matchers::ContainsSubstring("expected=8"));
    }
    SECTION("Requested plasticity cache entry missing") {
        result.plasticity_request = anet::nn::PlasticityMetricRequest::All();
        result.plasticity_features = torch::eye(2);
        result.plasticity_metrics_cache.emplace();
        CHECK_THROWS_WITH(result.GetScalar(kPlasticitySrank), Catch::Matchers::ContainsSubstring("cache is missing"));
    }
    SECTION("Spectral sentinel has no network") {
        result.plasticity_weight_norms = torch::tensor({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f});
        CHECK_THROWS_WITH(result.GetScalar(kPlasticityWeightNormFeature), Catch::Matchers::ContainsSubstring("network handle is unavailable"));
    }
    SECTION("Spectral sentinel reports invalid sigma") {
        result.plasticity_online_network = MakeScalarSpectralNetwork();
        torch::NoGradGuard no_grad;
        result.plasticity_online_network->GetSpectralNormEntries().front().u.fill_(std::numeric_limits<float>::quiet_NaN());
        result.plasticity_weight_norms = torch::tensor({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f});
        CHECK_THROWS_WITH(result.GetScalar(kPlasticityWeightNormFeature), Catch::Matchers::ContainsSubstring("spectral normalization is invalid"));
    }
    SECTION("Spectral sentinel disagrees with valid network") {
        result.plasticity_online_network = MakeScalarSpectralNetwork();
        result.plasticity_weight_norms = torch::tensor({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f});
        CHECK_THROWS_WITH(result.GetScalar(kPlasticityWeightNormFeature), Catch::Matchers::ContainsSubstring("sentinel mismatch"));
    }
}
