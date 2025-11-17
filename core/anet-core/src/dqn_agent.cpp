#include "anet/dqn_agent.hpp"
#include <iostream>
#include <tuple>
#include <wx/log.h>
#include "nlohmann/json.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/tensor_check.hpp"
#include "anet/config.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/random.hpp"

using namespace anet::rl;

const float met_ema_decay = 0.995f;  // 平滑化係数(メトリクス用)
const float met_ema_decay_act = 0.9995f;  // 平滑化係数(メトリクス用)action_ema用
const float met_ema_decay_reward = 0.9995f;  // 平滑化係数(メトリクス用)action_ema用

namespace {
    static constexpr int64_t ANY = ANET_SHAPE_ANY;
}

// ======================================================
// QNet 定義（Impl を CPP に置く）
// ======================================================
struct anet::rl::DQNAgent::QNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };

    QNetImpl(int state_dim, int n_actions) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, 120));
        fc2 = register_module("fc2", torch::nn::Linear(120, 84));
        fc3 = register_module("fc3", torch::nn::Linear(84, n_actions));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};

// ======================================================
// DQNAgent 実装
// ======================================================
DQNAgent::DQNAgent(const DQNAgentConfig& config, anet::rl::Environment& env, int state_dim, int n_actions, torch::Device device) :
    state_dim_(state_dim),
    n_actions_(n_actions),
    policy_net(std::make_shared<QNetImpl>(state_dim, n_actions)),
    target_net(std::make_shared<QNetImpl>(state_dim, n_actions)),
    device_(device),
    config_(config),
    optimizer(policy_net->parameters(), torch::optim::AdamOptions(config_.alpha)),
    replay_buffer_(config_.replay_capacity)
{
    // 学習変数初期化
    tau_ = config_.softupdate_tau;
    epsilon = config_.eps_max;
    eps_reheat_floor_ = config_.eps_min;
    // EMA係数（半減期 → α = ln(2)/H）
    uema_alpha_ = (config_.uema_half_life > 0.0f)
        ? static_cast<float>(std::log(2.0) / config_.uema_half_life)
        : 1.0f; // 念のため（0割回避）

    // ヒートマップオブジェクトを生成
	auto nan = std::numeric_limits<float>::quiet_NaN();
    auto info = env.GetStateSpaceInfo();
    auto flags = anet::HeatMapFlags::HM_LogScaleValue | anet::HeatMapFlags::HM_AutoNormValue
		| anet::HeatMapFlags::HM_AutoScaleAxis | anet::HeatMapFlags::HM_LogScaleAxis | anet::HeatMapFlags::HM_ShowZeroLine;
    heatmap_visit1_ = anet::rl::MakeStateHeatMapPtr(info, 0, 2, 256, 256, 30000, flags | anet::HeatMapFlags::HM_SumMode);  // x vs theta → reward
    heatmap_visit2_ = anet::rl::MakeStateHeatMapPtr(info, 2, 3, 256, 256, 30000, flags | anet::HeatMapFlags::HM_SumMode);  // x vs theta → reward
    heatmap_td_     = anet::rl::MakeStateHeatMapPtr(info, 0, 2, 256, 256, 30000, flags | anet::HeatMapFlags::HM_MeanMode); // x vs theta → td
    hist_action_ = std::make_unique<anet::TimeHistogram>(
        2, 200, anet::TimeFrameMode::Scroll, flags, -1.0f, 1.0f, 0.05f);
    hist_q_ = std::make_unique<anet::TimeHistogram>(
        128, 1000, anet::TimeFrameMode::Unlimited, flags | anet::HeatMapFlags::HM_FlipY, 0.0f, nan, 0.05f);

    //TimeHistogram(int bins, int max_frames,
    //    TimeFrameMode mode = TimeFrameMode::Scroll,
    //    uint32_t flags = HM_AutoScaleAxis | HM_AutoNormValue,
    //    float alpha = 0.05f,
    //    float base_min = std::numeric_limits<float>::quiet_NaN(),
    //    float base_max = std::numeric_limits<float>::quiet_NaN()
    //);

    // NN初期化
    policy_net->to(device);
    target_net->to(device);
    target_net->eval();

    // 初期同期：policy → target
    torch::serialize::OutputArchive archive;
    policy_net->save(archive);
    torch::serialize::InputArchive in;
    std::stringstream ss;
    archive.save_to(ss);
    in.load_from(ss);
    target_net->load(in);
    target_net->eval();

    // ログ：パラメータ記録
    wxLogInfo("DQNAgent config=%s", config_.ToStdString());
    anet::MetricsLogger::Instance()->log_json("agent/params", config_.ToJson());
    anet::MetricsLogger::Instance()->flush();
}

anet::rl::ActionInfo DQNAgent::SelectAction(const torch::Tensor& state, anet::rl::RunMode mode) {
    int action_int = this->rnd->RandUint64() % n_actions_;
    torch::Tensor action = torch::tensor({ action_int }, torch::kLong).to(device_);
    return { action, torch::Tensor(), torch::Tensor() };
}

std::shared_ptr<anet::rl::UpdateResult> DQNAgent::UpdateStep(const anet::rl::Experience& exprence) {
    replay_buffer_.Push(exprence);

    if (replay_buffer_.Size() > config_.replay_batch_size) {
        auto samples = replay_buffer_.Sample(config_.replay_batch_size, device_);
        int B = config_.replay_batch_size;
        ANET_CHECK_DEVICE(samples.states, device_);
        ANET_CHECK_DEVICE(samples.actions, device_);
        ANET_CHECK_DEVICE(samples.next_states, device_);
        ANET_CHECK_DEVICE(samples.rewards, device_);
        ANET_CHECK_DEVICE(samples.dones, device_);
        ANET_CHECK_DEVICE(samples.truncateds, device_);
        ANET_ASSERT(samples.actions.dtype() == torch::kInt64); //DQNでは離散アクション
        ANET_CHECK_SHAPE(samples.states,     { B, state_dim_ }); // (B, state_dim)
        ANET_CHECK_SHAPE(samples.actions,    { B }, { B, 1 });   // (B,) or (B, 1)   DQNでは離散アクション
        ANET_CHECK_SHAPE(samples.rewards,    { B });             // (B,)
        ANET_CHECK_SHAPE(samples.next_states,{ B, state_dim_ }); // (B, state_dim)
        ANET_CHECK_SHAPE(samples.dones,      { B });             // (B,)
        ANET_CHECK_SHAPE(samples.truncateds, { B });             // (B,)

        wxLogDebug("ReplayBuffer batch OK: B=%lld", samples.states.size(0));
    }
    return std::make_shared<anet::rl::DQNUpdateResult>();
}

void DQNAgent::OnPostUpdate(const std::shared_ptr<UpdateResult>& result) {
    ;
}


