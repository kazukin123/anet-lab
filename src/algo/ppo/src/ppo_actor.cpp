#include "ppo_actor.hpp"
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "anet/profile.hpp"
#include "anet/diag.hpp"
#include "anet/distribution.hpp"

using namespace anet::rl::ppo;

Actor::Actor(const ActorConfig& config, std::shared_ptr<std::shared_mutex> mutex, std::shared_ptr<NetworkModel> model,
    torch::Device device, anet::seed_t seed)
    : config_(config)
    , mutex_(std::move(mutex))
    , model_(std::move(model))
    , device_(device)
    , random_(seed)
{
    if (config_.clone_model) {
        // 写しは Actor の private な Resource。作る間だけ、Learner の更新と重ならないよう共有のロックを持つ。
        std::shared_lock lock(*mutex_);
        network_ = model_->CloneNetwork(device_);
        synced_version_ = model_->GetPolicyVersion();
    } else {
        network_ = model_->GetNetwork();
    }
}

std::shared_ptr<anet::rl::BatchActionInfo> Actor::MakeAction(const StepCounts& step,
    const anet::rl::BatchState& state, const MakeActionOptions& options) const
{
    ANET_PROFILE_FUNC();
    // 行動選択は勾配を作らない。学習の forward は、Learner が rollout の更新でやり直す。
    torch::NoGradGuard no_grad;

    // 観測を device へ移し、Network の入力にする。Env が CPU にあるので、この転送は毎 step に 1 回要る。
    ANET_PROFILE_SCOPE(obs_to_device);
    const auto input = NetworkModel::MakeNetworkInput(state.obs.To(device_));

    // Network を 1 回通して、方策の logits と、状態の value を得る。
    // 共有の Network は Learner が書き換えるので、forward の間は共有のロックを持つ。写しはこの Actor しか触らない。
    ANET_PROFILE_SCOPE_NEXT(forward);
    std::optional<anet::ForwardCapture> capture;
    if (options.nn_trace) capture.emplace(std::vector<std::string>{ std::string(anet::nn::kNnTracePrefix) });
    std::shared_lock lock(*mutex_, std::defer_lock);
    if (!config_.clone_model) lock.lock();
    const auto outputs = network_->Forward(input, capture ? &*capture : nullptr);
    // どの重みで選んだかを、行動と一緒に記録する（Learner が、方策の古さを数える）。
    const auto policy_version = config_.clone_model ? synced_version_ : model_->GetPolicyVersion();
    if (lock.owns_lock()) lock.unlock();

    // 分布 π(·|s) から行動を選ぶ。学習の Actor は抽選（探索を兼ねる）、評価の Actor は確率が最大の行動も選べる。
    ANET_PROFILE_SCOPE_NEXT(select_action);
    const anet::rl::CategoricalDistribution distribution(outputs.At(NetworkModel::kLogitsKey));
    auto actions = config_.action_mode == ActorConfig::kActionModeGreedy
        ? distribution.GetMode()
        : distribution.Sample(random_.GetTorchGenerator(device_));
    // 強制行動は方策の選択のあとに反映し、実行される行動を Actor の出力にする。
    if (options.forced_actions) actions = options.forced_actions->Apply(actions, distribution.GetActionCount());

    // Learner へ運ぶ値を、行動の情報に載せる。どれも device の上のままで、CPU へ読み戻さない。
    //   log_prob: 実行される行動の log π_old(a|s)。更新で、新しい方策との比の分母になる。
    //   value   : V_old(s)。GAE の入力と、value の clip の基準になる。
    //   input.* : Network へ渡した入力。Learner は、これをためて更新の forward に使う（観測を送り直さない）。
    ANET_PROFILE_SCOPE_NEXT(build_action_info);
    anet::TensorDict info;
    info.Set(kLogProbKey, distribution.ComputeLogProb(actions));
    info.Set(kValueKey, outputs.At(NetworkModel::kValueKey).squeeze(1));
    info.Set(kPolicyVersionKey, torch::tensor(static_cast<int64_t>(policy_version), torch::kInt64));
    for (const auto& [key, value] : input) info.Set(std::string(kInputKeyPrefix) + key, value);
    auto action_info = std::make_shared<anet::rl::BatchActionInfo>(actions, info);
    if (capture) anet::rl::AppendNnTraceAux(action_info->GetAuxData(), *capture);
    return action_info;
}

void Actor::Sync()
{
    // 共有の Network は、Learner の更新がそのまま見えるので、写すものが無い。
    if (!config_.clone_model) return;
    ANET_PROFILE_FUNC();
    // 写しを、Learner の Network の今の重みに合わせる。評価の Runner が、評価セッションの初めに呼ぶ。
    std::shared_lock lock(*mutex_);
    model_->GetNetwork()->CopyTo(*network_);
    network_->eval();
    synced_version_ = model_->GetPolicyVersion();
}
