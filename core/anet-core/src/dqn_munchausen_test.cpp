#include "anet/catch_test.hpp"
#include "anet/default_dqn_agent.hpp"
#include "anet/rainbow_agent.hpp"
#include "dqn_based_agent.hpp"
#include "nn_impl.hpp"
#include "nn_heads.hpp"
#include <ATen/autocast_mode.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <vector>

using namespace anet;
using namespace anet::rl;
using namespace anet::rl::dqn;

namespace {

struct SoftForward {
    bool training;
    bool grad_enabled;
    torch::Tensor values;
    torch::Tensor taus;
    torch::Tensor features;
};

// 公開Network境界でforwardの入力・mode・出力を記録し、Learner内部へtest専用APIを追加しない。
class SoftOracleHead final : public nn::NetworkHead {
public:
    SoftOracleHead(std::string mode, float scale, bool tbo, std::shared_ptr<std::vector<SoftForward>> records)
        : mode_(std::move(mode)), scale_(scale), tbo_(tbo), records_(std::move(records))
    {
        bias_ = register_parameter("bias", torch::zeros({ 1 }));
    }

    TensorDict Forward(const TensorDict& features) override
    {
        const auto input = features.At("feature");
        auto q = scale_ * input + bias_;
        if (is_training()) q = q + torch::tensor({ 0.25f, -0.125f }, q.options());
        torch::Tensor taus;
        torch::Tensor values;
        if (mode_ == "none") {
            values = q;
        } else {
            taus = mode_ == "iqn" ? features.At(nn::kKey_Taus)
                : torch::tensor({ 0.9f, 0.1f, 0.4f }, q.options()).unsqueeze(0).expand({ q.size(0), 3 });
            values = q.unsqueeze(2) + torch::stack({ 6.0f * taus, 2.0f * (1.0f - taus) }, 1);
        }
        // fixtureは既知の実空間値を出力空間へ写す。oracle側は独立したdouble式で逆変換する。
        if (tbo_) values = values.sign() * ((values.abs() + 1.0f).sqrt() - 1.0f) + 0.001f * values;
        // 通常headはFP32だが、fixtureでは低精度targetも注入し、Learner境界のFP32復帰を検証する。
        if (!is_training() && input.is_cuda()) values = values.to(torch::kBFloat16);
        records_->push_back({ is_training(), torch::GradMode::is_enabled(), values.detach().clone(),
            taus.defined() ? taus.detach().clone() : torch::Tensor(), input.detach().clone() });
        if (mode_ == "none") return TensorDict{ { "q", values } };
        return TensorDict{ { "q", values.mean(2) }, { "q_dist", values } };
    }

    std::optional<TensorDictFunction> GetTensorDictFunction(const std::string&) override { return std::nullopt; }

private:
    std::string mode_;
    float scale_;
    bool tbo_;
    std::shared_ptr<std::vector<SoftForward>> records_;
    torch::Tensor bias_;
};

class SoftIdentity final : public nn::NetworkModule {
public:
    torch::Tensor Forward(torch::Tensor input) override {
        // NetworkがheadのAMPを切る前のbody境界で、CUDA BF16が実際に有効なことを確認する。
        if (input.is_cuda()) {
            CHECK(at::autocast::is_autocast_enabled(at::kCUDA));
            CHECK(at::autocast::get_autocast_dtype(at::kCUDA) == torch::kBFloat16);
        }
        return input;
    }
};

std::shared_ptr<nn::Network> MakeSoftOracleNetwork(const std::string& mode, float scale, bool tbo,
    const std::shared_ptr<std::vector<SoftForward>>& records, torch::Device device)
{
    TensorSpecMap specs{ { ObsKeys::kVector, TensorSpec{ .type = SpaceType::Vector, .shape = { 2 }, .dtype = torch::kFloat32 } } };
    nn::NetworkConfig config;
    config.output_keys["feature"] = "feature";
    if (mode == "iqn") {
        specs[nn::kKey_Taus] = TensorSpec{ .shape = { 4 }, .dtype = torch::kFloat32 };
        config.output_keys[nn::kKey_Taus] = nn::kKey_Taus;
    }
    auto block = std::make_shared<nn::NetworkBlock>("identity", std::make_shared<SoftIdentity>());
    auto structure = std::make_shared<nn::NetworkStruct>(std::vector<std::shared_ptr<nn::NetworkBlock>>{ block });
    auto branch = std::make_shared<nn::NetworkBranch>("feature",
        std::vector<std::vector<std::string>>{ { ObsKeys::kVector } }, 1, structure);
    auto body = std::make_shared<nn::NetworkBody>(std::vector<std::shared_ptr<nn::NetworkBranch>>{ branch },
        specs, std::vector<std::string>{}, config.output_keys);
    auto network = std::make_shared<nn::Network>(config, specs, nullptr, body,
        std::make_shared<SoftOracleHead>(mode, scale, tbo, records));
    network->to(device);
    network->eval();
    return network;
}

class SoftOracleModel final : public NetworkModel {
public:
    SoftOracleModel(const std::string& mode, bool tbo, torch::Device device,
        const std::shared_ptr<std::vector<SoftForward>>& online,
        const std::shared_ptr<std::vector<SoftForward>>& target)
        : NetworkModel(NetworkModelConfig{}, MakeSoftOracleNetwork(mode, 1.0f, tbo, online, device),
            MakeSoftOracleNetwork(mode, 2.0f, tbo, target, device), 2, mode != "none") {}
};

// hard選択そのものを検証するため、既存protected境界だけをtest側で公開する。
class SoftHardSelector final : public QuantileLearnerBase {
public:
    using QuantileLearnerBase::QuantileLearnerBase;
    using QuantileLearnerBase::SelectTargetActions;
    std::shared_ptr<anet::rl::dqn::BatchUpdateResult> UpdateFromSamples(const ExperienceSamples&) override { return {}; }
};

EnvSpec SoftEnvSpec()
{
    EnvSpec spec;
    spec.state_spec.obs_spec[ObsKeys::kVector] = TensorSpec{ .type = SpaceType::Vector, .shape = { 2 }, .dtype = torch::kFloat32 };
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = { "a0", "a1" };
    spec.reward_range = { -1000.0f, 1000.0f };
    return spec;
}

ExperienceSamples SoftSamples(torch::Device device)
{
    const auto options = torch::TensorOptions().device(device);
    ExperienceSamples samples;
    samples.obs = TensorDict{ { ObsKeys::kVector, torch::tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } }, options) } };
    samples.next_state.next_obs = TensorDict{ { ObsKeys::kVector, torch::tensor({ { 0.5f, 1.0f }, { 1.5f, 2.0f } }, options) } };
    samples.actions = torch::tensor({ 0, 1 }, options.dtype(torch::kInt64));
    samples.target_returns = torch::tensor({ 0.1f, 0.2f }, options);
    samples.next_state.terminals = torch::tensor({ false, true }, options.dtype(torch::kBool));
    samples.n_steps = torch::tensor({ 3, 2 }, options.dtype(torch::kInt64));
    samples.replay_item_keys = torch::arange(2, torch::kInt64);
    samples.is_weights = torch::ones({ 2 }, options);
    return samples;
}

double OracleH(double x) { return std::copysign(std::sqrt(std::abs(x) + 1.0) - 1.0, x) + 0.001 * x; }
double OracleHInv(double x)
{
    const double t = (std::sqrt(1.0 + 0.004 * (std::abs(x) + 1.001)) - 1.0) / 0.002;
    return std::copysign(t * t - 1.0, x);
}

std::array<double, 2> OracleLogPolicy(const std::array<double, 2>& scores, double temperature)
{
    const double maximum = std::max(scores[0], scores[1]);
    const double log_z = std::log(std::exp((scores[0] - maximum) / temperature)
        + std::exp((scores[1] - maximum) / temperature));
    return { scores[0] - maximum - temperature * log_z, scores[1] - maximum - temperature * log_z };
}

} // namespace

TEST_CASE("Atari Munchausen profile resolves algorithm settings and all diagnostics", "[dqn][munchausen][munchausen_profile]")
{
    const std::string mode = GENERATE("target", "online", "online_reuse");
    const bool risk = GENERATE(false, true);
    CAPTURE(mode, risk);
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    const auto config_dir = root / "apps" / "runner" / "config";
    ConfigManagerOptions options;
    options.config_search_dirs = std::vector<std::filesystem::path>{ config_dir };
    options.overwrite_config_paths = { config_dir / "Atari.txt" };
    options.injected_config.Set("run.$", "run.@munchausen");
    options.injected_config.Set("A3.learner.munchausen.log_policy_mode", mode);
    options.injected_config.Set("A3.use_optimistic_target", risk);
    options.injected_config.Set("A3.train_policy.policy_type", "UQE");
    const ConfigManager manager((config_dir / "_main.txt").string(), nullptr, options);
    const auto data = manager.GetConfigData();
    const DefaultDQNAgentConfig config(data);
    CHECK(config.quantile_mode == "iqn");
    CHECK(config.learner.munchausen.enabled);
    CHECK(config.learner.munchausen.log_policy_mode == mode);
    CHECK(config.learner.munchausen.alpha == Catch::Approx(0.9));
    CHECK(config.learner.munchausen.entropy_tau == Catch::Approx(0.03));
    CHECK(config.learner.munchausen.clip_value_min == Catch::Approx(-1.0));
    CHECK_FALSE(config.learner.use_double_dqn);
    CHECK(config.use_optimistic_target == risk);
    CHECK(config.target_policy.policy_type == (risk ? "UQE" : "Greedy"));
    for (const auto* suffix : { "01_scaled_logp_mean", "02_scaled_logp_mean_ema", "03_clip_ratio", "04_bonus_mean",
        "05_bonus_mean_ema", "06_next_entropy", "07_soft_gap" }) {
        CHECK_FALSE(data.Get(std::string("metrics.scalar.[36_agent_munchausen/") + suffix + "]").empty());
    }
    // 解決後ConfigDataは材料キーを公開しないため、従来Runの実効設定からOFFを検証する。
    options.injected_config.Set("run.$", "run.@v5_iqn_impala_x2");
    const ConfigManager baseline_manager((config_dir / "_main.txt").string(), nullptr, options);
    const DefaultDQNAgentConfig baseline(baseline_manager.GetConfigData());
    CHECK_FALSE(baseline.learner.munchausen.enabled);
    CHECK(baseline.learner.use_double_dqn);
}

TEST_CASE("Munchausen Actor uses exactly one existing score forward", "[dqn][munchausen][actor_munchausen]")
{
    const std::string kind = GENERATE("none", "qr", "iqn");
    const bool enabled = GENERATE(false, true);
    const bool tbo = GENERATE(false, true);
    CAPTURE(kind, enabled, tbo);
    const torch::Device device(torch::kCPU);
    auto records = std::make_shared<std::vector<SoftForward>>();
    auto network = MakeSoftOracleNetwork(kind, 1.0f, tbo, records, device);
    ActionPolicyConfig policy_config;
    policy_config.quantile_mode = kind;
    policy_config.tau_rule.num_taus = 4;
    policy_config.tau_rule.sample_mode = "fixed";
    std::shared_ptr<ActionPolicy> policy = kind == "none"
        ? std::shared_ptr<ActionPolicy>(std::make_shared<EpsilonGreedyActionPolicy>(policy_config))
        : std::shared_ptr<ActionPolicy>(std::make_shared<UQEActionPolicy>(policy_config));
    const auto flags = torch::zeros({ 2 }, torch::kBool);
    const BatchState state(SoftSamples(device).obs, flags, flags, flags);
    torch::Tensor reference_hint;
    // 共通configにmodeが含まれても、同じscoreとseedから作るActor hintは変わらない。
    for (const std::string mode : { "target", "online", "online_reuse" }) {
        CAPTURE(mode);
        const auto context = std::make_shared<DefaultActionContext>(RunMode::Train, 67021);
        anet::rl::dqn::Actor actor(policy, nullptr, context, std::make_shared<std::shared_mutex>(), network, network,
            true, std::nullopt, false, ActorQHintConfig{
                .munchausen = MunchausenConfig{ .enabled = enabled, .log_policy_mode = mode, .entropy_tau = 0.7f },
                .use_tbo = tbo, .tbo_epsilon = 0.001f });
        records->clear();
        const auto result = actor.MakeAction(StepCounts{}, state);
        REQUIRE(records->size() == 1);
        const auto hint = result->GetReplayInitialPriorityHint();
        REQUIRE(hint.has_value());
        CHECK(hint->GetPayload().sizes() == torch::IntArrayRef({ 2, 3 }));
        if (enabled) CHECK(hint->GetPayload().select(1, 2).lt(0).any().item<bool>());
        else CHECK(torch::equal(hint->GetPayload().select(1, 2), torch::zeros({ 2 })));
        if (reference_hint.defined()) CHECK(torch::equal(hint->GetPayload(), reference_hint));
        reference_hint = hint->GetPayload().clone();
    }
}

TEST_CASE("K3 Actor hints carry the selected start bonus into initial priorities", "[dqn][munchausen][k3]")
{
    // 行動差し替え後のhintを公開推定器へ渡し、transport全体の列契約を検証する。
    AuxData aux;
    aux["q_values"] = torch::tensor({ { 1.0f, 4.0f } });
    aux["munchausen_terms"] = torch::tensor({ { -0.7f, -0.2f } });
    DQNActionInfo action(torch::tensor({ 1 }, torch::kInt64), {}, aux,
        ReplayInitialPriorityHint(torch::tensor({ { 4.0f, 5.0f, -0.2f } })));
    const auto replaced = action.WithAction(torch::tensor({ 0 }, torch::kInt64));
    REQUIRE(replaced->GetReplayInitialPriorityHint().has_value());
    CHECK(torch::equal(replaced->GetReplayInitialPriorityHint()->GetPayload(), torch::tensor({ { 1.0f, 5.0f, -0.7f } })));
    aux.erase("munchausen_terms");
    DQNActionInfo missing(torch::tensor({ 1 }, torch::kInt64), {}, aux,
        ReplayInitialPriorityHint(torch::tensor({ { 4.0f, 5.0f, -0.2f } })));
    CHECK_THROWS(missing.WithAction(torch::tensor({ 0 }, torch::kInt64)));

    for (const bool tbo : { false, true }) {
        LearnerConfig config;
        config.use_tbo = tbo;
        config.tbo_epsilon = 0.001f;
        config.per_eps = 0.01f;
        config.use_per_prio_clip = false;
        const auto estimator = CreateInitialPriorityEstimator(config);
        for (const bool terminal : { false, true }) {
            for (const float bonus : { 0.0f, -0.7f }) {
                const float q_sa = static_cast<float>(tbo ? OracleH(1.4) : 1.4);
                const std::array<float, 3> start{ q_sa, 99.0f, bonus };
                const std::array<float, 3> bootstrap{ 88.0f, static_cast<float>(tbo ? OracleH(3.2) : 3.2), -99.0f };
                REQUIRE(estimator->ValidateHint(start));
                const auto priority = estimator->Estimate(InitialPriorityEstimateInput{
                    .start_hint = start, .bootstrap_hint = bootstrap, .target_return = 1.25f,
                    .discount = 0.81f, .terminal = terminal, .actual_n_steps = 2,
                });
                REQUIRE(priority.has_value());
                const double raw_target = 1.25 + bonus + (terminal ? 0.0 : 0.81 * 3.2);
                CHECK(*priority == Catch::Approx(std::abs((tbo ? OracleH(raw_target) : raw_target) - q_sa) + 0.01).margin(3e-4));
            }
        }
        for (int column = 0; column < 3; ++column) {
            std::array<float, 3> invalid{ 1.0f, 2.0f, 0.0f };
            invalid[column] = std::numeric_limits<float>::quiet_NaN();
            CHECK_FALSE(estimator->ValidateHint(invalid));
            invalid[column] = std::numeric_limits<float>::infinity();
            CHECK_FALSE(estimator->ValidateHint(invalid));
        }
        const std::array<float, 2> old_hint{ 1.0f, 2.0f };
        CHECK_THROWS(estimator->ValidateHint(old_hint));
        CHECK_THROWS(DecodeActorQHint(torch::zeros({ 1, 2 })));
    }
}

TEST_CASE("Munchausen learners mix the full target distribution in every log policy mode", "[dqn][munchausen][soft_target]")
{
    const std::string kind = GENERATE("none", "qr", "iqn");
    const std::string mode = GENERATE("target", "online", "online_reuse");
    const bool tbo = GENERATE(false, true);
    const int risk_mode = GENERATE(0, 1, 2); // 平均、point UQE、tail-mean UQE
    const int scenario = GENERATE(0, 1, 2, 3); // CPU、CUDA BF16、alpha=0極限、clip下限=0
    if (kind == "none" && risk_mode != 0) return;
    if ((scenario == 1 || scenario == 3) && risk_mode != 0) return;
    if (scenario == 1 && !torch::cuda::is_available()) { SKIP("CUDA is unavailable"); }
    CAPTURE(kind, mode, tbo, risk_mode, scenario);
    const torch::Device device(scenario == 1 ? torch::kCUDA : torch::kCPU);
    torch::Tensor previous_td;
    double previous_loss = 0.0;
    for (int repeat = 0; repeat < 2; ++repeat) {
        LearnerConfig config;
        config.quantile_mode = kind;
        config.num_quantiles = 3;
        config.iqn.current_taus.num_taus = 4;
        config.iqn.target_taus.num_taus = 3;
        config.iqn.current_taus.sample_mode = "random";
        config.iqn.target_taus.sample_mode = "random";
        config.munchausen.enabled = true;
        config.munchausen.log_policy_mode = mode;
        config.munchausen.entropy_tau = 0.7f;
        config.munchausen.clip_value_min = -2.0f;
        if (scenario == 2) {
            config.munchausen.alpha = 0.0f;
            config.munchausen.entropy_tau = 1e-5f;
            // 低温極限の検証を、h逆変換の丸めによるclip境界の左右差から分離する。
            config.munchausen.clip_value_min = -1.7f;
        }
        if (scenario == 3) config.munchausen.clip_value_min = 0.0f;
        config.use_amp = scenario == 1;
        config.use_amp_bf16 = scenario == 1;
        config.use_double_dqn = false;
        config.use_tbo = tbo;
        config.tbo_epsilon = 0.001f;
        config.alpha = 0.0f;
        config.use_fused_optimizer = false;
        config.use_grad_clip = false;
        config.use_td_clip = false;
        config.use_per = false;
        config.replay_capacity = 8;
        config.replay_batch_size = 2;
        config.gamma = 0.9f;
        auto online = std::make_shared<std::vector<SoftForward>>();
        auto target = std::make_shared<std::vector<SoftForward>>();
        SoftOracleModel model(kind, tbo, device, online, target);
        RuntimeVars vars;
        ActionPolicyConfig policy_config;
        policy_config.quantile_mode = kind;
        policy_config.policy_type = repeat == 0 ? "Greedy" : "EpsilonGreedy";
        policy_config.eps_start = 0.0f;
        policy_config.eps_end = 0.0f;
        policy_config.uqe_tau_start = 0.9f;
        policy_config.uqe_tau_end = 0.4f;
        policy_config.uqe_tau_decay_steps = 100;
        policy_config.uqe_use_tail_mean = risk_mode == 2;
        std::shared_ptr<ActionPolicy> policy = risk_mode == 0
            ? std::shared_ptr<ActionPolicy>(std::make_shared<EpsilonGreedyActionPolicy>(policy_config))
            : std::shared_ptr<ActionPolicy>(std::make_shared<UQEActionPolicy>(policy_config));
        policy->OnLearn(StepCounts{ .exp_step = 50 });
        const auto spec = SoftEnvSpec();
        const auto update = [&]<typename ConcreteLearner>() {
            ConcreteLearner learner(config, model, vars, nullptr, BatchEnvSpec{ 2, 2 }, spec,
                device, 67010, policy, std::nullopt, 67011);
            online->clear();
            target->clear();
            return learner.UpdateFromSamples(SoftSamples(device));
        };
        const auto result = kind == "none" ? update.operator()<TDLearner>()
            : kind == "qr" ? update.operator()<QRLearner>() : update.operator()<IQNLearner>();
        REQUIRE(online->size() == (mode == "online" ? 2 : 1));
        REQUIRE(target->size() == 1);
        CHECK(result->td_error.scalar_type() == torch::kFloat32);
        CHECK(result->munchausen_diagnostics.scalar_type() == torch::kFloat32);
        if (scenario == 1) CHECK(target->at(0).values.scalar_type() == torch::kBFloat16);
        CHECK(online->at(0).training);
        CHECK(online->at(0).grad_enabled);
        CHECK_FALSE(target->at(0).training);
        CHECK_FALSE(target->at(0).grad_enabled);
        CHECK(target->at(0).values.size(0) == (mode == "target" ? 4 : 2));
        if (mode == "online") {
            CHECK_FALSE(online->at(1).training);
            CHECK_FALSE(online->at(1).grad_enabled);
        }
        if (kind == "iqn") {
            // 同じ公開tau生成契約を使い、消費順と余分なpolicy forwardがないことを検証する。
            RandomGenerator expected_rng(67011);
            CHECK(torch::equal(online->at(0).taus, GenerateTaus(2, 4, "random", 0.0f, 1.0f, device, expected_rng)));
            CHECK(torch::equal(target->at(0).taus, GenerateTaus(mode == "target" ? 4 : 2, 3, "random", 0.0f, 1.0f, device, expected_rng)));
            if (mode == "online") CHECK(torch::equal(online->at(1).taus, GenerateTaus(2, 4, "random", 0.0f, 1.0f, device, expected_rng)));
        }
        const auto as_distribution = [](torch::Tensor value) {
            value = value.to(torch::kCPU).to(torch::kFloat64);
            return value.dim() == 2 ? value.unsqueeze(2) : value;
        };
        const auto current = as_distribution(online->at(0).values);
        const auto bonus_values = as_distribution(mode == "target" ? target->at(0).values.narrow(0, 0, 2) : online->back().values);
        const auto next = as_distribution(mode == "target" ? target->at(0).values.narrow(0, 2, 2) : target->at(0).values);
        torch::Tensor hard_actions;
        if (kind == "qr" && risk_mode == 1 && scenario == 2) {
            SoftHardSelector selector(config, model, vars, nullptr, BatchEnvSpec{ 2, 2 }, spec,
                device, 67010, policy, std::nullopt, 67011);
            hard_actions = selector.SelectTargetActions(SoftSamples(device).next_state.next_obs);
        }
        double expected_loss = 0.0;
        std::array<double, 5> expected_diagnostics{};
        for (int b = 0; b < 2; ++b) {
            std::array<double, 2> current_scores{}, next_scores{};
            for (int a = 0; a < 2; ++a) {
                for (int k = 0; k < bonus_values.size(2); ++k) {
                    const double value = bonus_values[b][a][k].item<double>();
                    current_scores[a] += (tbo ? OracleHInv(value) : value) / bonus_values.size(2);
                }
                for (int k = 0; k < next.size(2); ++k) {
                    const double value = next[b][a][k].item<double>();
                    next_scores[a] += (tbo ? OracleHInv(value) : value) / next.size(2);
                }
            }
            // 診断の価値基準は平均Qのまま保持し、方策用スコアだけを経験分位へ差し替える。
            const auto next_means = next_scores;
            if (risk_mode != 0) {
                const auto risk_score = [&](const torch::Tensor& values, int action) {
                    std::vector<double> sorted;
                    for (int k = 0; k < values.size(2); ++k) {
                        const double value = values[b][action][k].item<double>();
                        sorted.push_back(tbo ? OracleHInv(value) : value);
                    }
                    std::sort(sorted.begin(), sorted.end());
                    const size_t index = static_cast<size_t>(0.65 * (sorted.size() - 1));
                    if (risk_mode == 1) return sorted[index];
                    double sum = 0.0;
                    for (size_t k = index; k < sorted.size(); ++k) sum += sorted[k];
                    return sum / (sorted.size() - index);
                };
                for (int a = 0; a < 2; ++a) {
                    current_scores[a] = risk_score(bonus_values, a);
                    next_scores[a] = risk_score(next, a);
                }
            }
            const auto current_log = OracleLogPolicy(current_scores, config.munchausen.entropy_tau);
            if (hard_actions.defined()) CHECK(hard_actions[b].item<int64_t>() == (next_scores[0] > next_scores[1] ? 0 : 1));
            const auto next_log = OracleLogPolicy(next_scores, config.munchausen.entropy_tau);
            const double selected_log = current_log[b];
            const double bonus = config.munchausen.alpha * std::clamp(selected_log, static_cast<double>(config.munchausen.clip_value_min), 0.0);
            expected_diagnostics[0] += selected_log / 2.0;
            expected_diagnostics[1] += (selected_log < config.munchausen.clip_value_min ? 1.0 : 0.0) / 2.0;
            expected_diagnostics[2] += bonus / 2.0;
            double soft_mean = 0.0;
            std::vector<double> targets;
            for (int k = 0; k < next.size(2); ++k) {
                double soft = 0.0;
                for (int a = 0; a < 2; ++a) {
                    const double probability = std::exp(next_log[a] / config.munchausen.entropy_tau);
                    const double value = next[b][a][k].item<double>();
                    soft += probability * ((tbo ? OracleHInv(value) : value) - next_log[a]);
                    if (k == 0) expected_diagnostics[3] -= probability * next_log[a] / config.munchausen.entropy_tau / 2.0;
                }
                soft_mean += soft / next.size(2);
                if (scenario == 2) {
                    const int hard_action = next_scores[0] > next_scores[1] ? 0 : 1;
                    const double hard_value = next[b][hard_action][k].item<double>();
                    CHECK(soft == Catch::Approx(tbo ? OracleHInv(hard_value) : hard_value).margin(2e-5));
                }
                const double raw_target = (b == 0 ? 0.1 : 0.2) + bonus + (b == 0 ? std::pow(0.9, 3) * soft : 0.0);
                targets.push_back(tbo ? OracleH(raw_target) : raw_target);
            }
            expected_diagnostics[4] += (soft_mean - std::max(next_means[0], next_means[1])) / 2.0;
            if (scenario == 2 && risk_mode == 0) CHECK(soft_mean == Catch::Approx(std::max(next_means[0], next_means[1])).margin(2e-5));
            double target_mean = 0.0;
            for (const double target_value : targets) target_mean += target_value / targets.size();
            CHECK(result->td_error[b].item<float>() == Catch::Approx(current[b][b].mean().item<double>() - target_mean).margin(3e-4));
            for (int i = 0; i < current.size(2); ++i) {
                const double tau = kind == "iqn" ? online->at(0).taus[b][i].item<double>() : (i + 0.5) / current.size(2);
                for (const double target_value : targets) {
                    const double diff = target_value - current[b][b][i].item<double>();
                    const double huber = std::abs(diff) < 1.0 ? 0.5 * diff * diff : std::abs(diff) - 0.5;
                    const double weight = kind == "none" ? 1.0 : std::abs(tau - (diff < 0.0 ? 1.0 : 0.0));
                    expected_loss += huber * weight / 2.0 / (kind == "iqn" ? targets.size() : current.size(2));
                }
            }
        }
        CHECK(result->loss.item<double>() == Catch::Approx(expected_loss).margin(5e-4));
        for (size_t i = 0; i < expected_diagnostics.size(); ++i) {
            const auto value = result->GetScalar(kMunchausenMetricKeys[i]);
            REQUIRE(value.has_value());
            CHECK(*value == Catch::Approx(expected_diagnostics[i]).margin(3e-4));
        }
        if (repeat > 0) {
            CHECK(torch::equal(result->td_error, previous_td));
            CHECK(result->loss.item<double>() == previous_loss);
        }
        previous_td = result->td_error.clone();
        previous_loss = result->loss.item<double>();
    }
}

TEST_CASE("Munchausen config validates dormant values and resolved conflicts", "[dqn][munchausen][config]")
{
    const bool enabled = GENERATE(false, true);
    ConfigData data;
    data.Set("DefaultDQNAgent.quantile_mode", "qr");
    data.Set("DefaultDQNAgent.learner.munchausen.enabled", enabled);
    data.Set("DefaultDQNAgent.learner.use_double_dqn", false);
    for (const auto& [key, value] : std::vector<std::pair<std::string, std::string>>{
        { "log_policy_mode", "invalid" }, { "alpha", "-0.01" }, { "alpha", "1.01" }, { "alpha", "nan" },
        { "entropy_tau", "0" }, { "entropy_tau", "-1" }, { "entropy_tau", "inf" },
        { "clip_value_min", "0.01" }, { "clip_value_min", "nan" } }) {
        auto invalid = data;
        const auto full_key = "DefaultDQNAgent.learner.munchausen." + key;
        invalid.Set(full_key, value);
        CHECK_THROWS_WITH(DefaultDQNAgentConfig(invalid), Catch::Matchers::ContainsSubstring("munchausen." + key));
    }
    auto double_data = data;
    double_data.Set("DefaultDQNAgent.learner.use_double_dqn", true);
    if (enabled) {
        CHECK_THROWS_WITH(DefaultDQNAgentConfig(double_data), Catch::Matchers::ContainsSubstring("munchausen.enabled=true")
            && Catch::Matchers::ContainsSubstring("use_double_dqn=true") && Catch::Matchers::ContainsSubstring("use_double_dqn=false"));
    } else CHECK_NOTHROW(DefaultDQNAgentConfig(double_data));
    for (const bool copied : { false, true }) {
        auto thompson = data;
        thompson.Set("DefaultDQNAgent.use_optimistic_target", copied);
        thompson.Set(copied ? "DefaultDQNAgent.train_policy.policy_type" : "DefaultDQNAgent.target_policy.policy_type", "ThompsonSampling");
        if (enabled) CHECK_THROWS_WITH(DefaultDQNAgentConfig(thompson), Catch::Matchers::ContainsSubstring("munchausen.enabled=true")
            && Catch::Matchers::ContainsSubstring("ThompsonSampling") && Catch::Matchers::ContainsSubstring("expected Greedy"));
        else CHECK_NOTHROW(DefaultDQNAgentConfig(thompson));
        if (copied) {
            thompson.Set("DefaultDQNAgent.target_policy.policy_type", "Greedy");
            CHECK_NOTHROW(DefaultDQNAgentConfig(thompson));
        }
    }
    data.Set("DefaultDQNAgent.use_optimistic_target", true);
    data.Set("DefaultDQNAgent.train_policy.policy_type", "UQE");
    CHECK(DefaultDQNAgentConfig(data).target_policy.policy_type == "UQE");
    CHECK_FALSE(RainbowAgentConfig(ConfigData{}).learner.munchausen.enabled);
    anet::rl::dqn::BatchUpdateResult empty;
    for (const auto* key : kMunchausenMetricKeys) {
        REQUIRE(empty.GetScalar(key).has_value());
        CHECK(std::isnan(*empty.GetScalar(key)));
    }
    CHECK_FALSE(empty.GetScalar("unknown_munchausen_key").has_value());
}

TEST_CASE("Munchausen target capture exposes only next state rows", "[dqn][munchausen][capture]")
{
    const std::string kind = GENERATE("none", "qr", "iqn");
    const std::string mode = GENERATE("target", "online", "online_reuse");
    const bool per = GENERATE(false, true);
    CAPTURE(kind, mode, per);
    const torch::Device device(torch::kCPU);
    auto online = std::make_shared<std::vector<SoftForward>>();
    auto target = std::make_shared<std::vector<SoftForward>>();
    SoftOracleModel model(kind, false, device, online, target);
    LearnerConfig config;
    config.quantile_mode = kind;
    config.num_quantiles = 3;
    config.iqn.current_taus.num_taus = 4;
    config.iqn.target_taus.num_taus = 3;
    config.munchausen.enabled = true;
    config.munchausen.log_policy_mode = mode;
    config.use_double_dqn = false;
    config.use_per = per;
    config.use_n_step = false;
    config.use_rb_prefetch = false;
    config.use_fused_optimizer = false;
    config.replay_capacity = 8;
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1;
    config.plasticity.feature_key = "feature";
    RuntimeVars vars;
    ActionPolicyConfig policy_config;
    policy_config.quantile_mode = kind;
    auto policy = std::make_shared<EpsilonGreedyActionPolicy>(policy_config);
    const auto spec = SoftEnvSpec();
    const auto update = [&]<typename ConcreteLearner>() {
        ConcreteLearner learner(config, model, vars, nullptr, BatchEnvSpec{ 2, 2 }, spec,
            device, 67030, policy, std::nullopt, 67031);
        learner.ConfigureScalarMetricSubscriptions({
            ScalarMetricSubscription{ .source_key = "plasticity_feature_norm", .event = EventType::LEARN,
                .interval = 1, .scope = RunnerScope::TRAIN },
            ScalarMetricSubscription{ .source_key = "plasticity_target_feature_norm", .event = EventType::LEARN,
                .interval = 1, .scope = RunnerScope::TRAIN },
        });
        online->clear();
        target->clear();
        const auto samples = SoftSamples(device);
        const auto flags = torch::zeros({ 2 }, torch::kBool);
        const BatchExperience experience(BatchState(samples.obs, flags, flags, torch::ones_like(flags)),
            std::make_shared<BatchActionInfo>(samples.actions), samples.target_returns,
            // truncatedは最終観測をstorageへ確定するため、1回のPushでnext特徴を含めてsample可能になる。
            BatchState(samples.next_state.next_obs, flags, torch::ones_like(flags), flags));
        return learner.UpdateFromBatch(StepCounts{ .exp_step = 2 }, experience);
    };
    const auto results = kind == "none" ? update.operator()<TDLearner>()
        : kind == "qr" ? update.operator()<QRLearner>() : update.operator()<IQNLearner>();
    REQUIRE(results.size() == 1);
    const auto result = std::dynamic_pointer_cast<const anet::rl::dqn::BatchUpdateResult>(results.front());
    REQUIRE(result);
    REQUIRE(target->size() == 1);
    const auto next_features = mode == "target" ? target->at(0).features.narrow(0, 2, 2) : target->at(0).features;
    CHECK(torch::equal(result->plasticity_target_features, next_features));
    CHECK(torch::equal(result->plasticity_features, online->at(0).features));
    CHECK(result->plasticity_target_features.sizes() == torch::IntArrayRef({ 2, 2 }));
    CHECK(std::isfinite(*result->GetScalar("plasticity_target_feature_norm")));
    for (const auto* key : kMunchausenMetricKeys) CHECK(std::isfinite(*result->GetScalar(key)));
    if (per) CHECK(result->per_priorities.numel() == 2);
}

TEST_CASE("Munchausen clip ratio counts strictly below the lower boundary", "[dqn][munchausen][clip]")
{
    MunchausenConfig config;
    config.entropy_tau = 0.03f;
    config.clip_value_min = -2.0f;
    const auto scores = torch::tensor({ { 0.0f, 2.0f }, { 0.0f, 3.0f }, { 0.0f, 2.0f } });
    const auto actions = torch::tensor({ 0, 0, 1 }, torch::kInt64);
    const auto terms = MakeMunchausenTargetTerms(scores, scores, scores, actions, config);
    CHECK(torch::allclose(terms.bonus, torch::tensor({ -1.8f, -1.8f, 0.0f })));
    CHECK(terms.diagnostics[1].item<float>() == Catch::Approx(1.0 / 3.0));
}
