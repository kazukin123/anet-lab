#include "dqn_network_model.hpp"
#include <algorithm>
#include <charconv>
#include <tuple>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <set>
#include <utility>
#include <unordered_set>
#include "anet/log.hpp"
#include "anet/profile.hpp"
#include "anet/tensor_check.hpp"
#include "anet/tensor_util.hpp"
#include "anet/str_util.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/replay_experience.hpp"
#include "anet/image.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/nn_head.hpp"
#include "dqn_heads.hpp"

using namespace anet::rl::dqn;
namespace LOG = anet::log;

NetworkModel::NetworkModel(
    const NetworkModelConfig& config, const torch::Device device,
    const anet::nn::NetworkConfig& network_config, const anet::TensorSpecMap& obs_spec, int64_t n_actions, std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory,
    bool distributional, anet::seed_t network_seed, const SkipNetworkDefinition* skip_definition)
    : config_(config)
    , n_actions_(n_actions)
    , distributional_(distributional)
{
    ANET_ASSERT(n_actions_ > 0);

    // メインネットワークを作る
    online_net_ = anet::nn::NetworkBuilder::BuildNetwork(network_config, obs_spec, head_factory, network_seed, device);
    online_net_->to(device);
    online_net_->eval();

#if 0
    {
        torch::NoGradGuard no_grad;
        for (auto& param : online_net_->parameters()) {
            torch::nn::init::constant_(param, 0.01f);
        }
    }
#endif

    // メインネットワークをコピーしてターゲットネットワークを作る
    target_net_ = online_net_->Clone(device);
    target_net_->eval();

    // 有効時だけ、primary と対になる skip 網を構築する。
    if (skip_definition) {
        skip_config_ = skip_definition->config;
        torch::NoGradGuard no_grad;

        // primary の dummy forward から、Body が公開する特徴の shape を得る。
        anet::TensorDict dummy;
        for (const auto& [key, spec] : obs_spec) {
            auto shape = spec.shape;
            shape.insert(shape.begin(), 1);
            dummy.Set(key, torch::zeros(shape, torch::TensorOptions().device(device)
                .dtype(spec.IsDiscrete() || spec.dtype == torch::kUInt8 ? torch::kInt64 : spec.dtype)));
        }
        const auto output = online_net_->Forward(dummy);

        // bind 候補は obs（IQN の taus を含む）を土台にし、特徴を共有せず自前で抽出する構成も許す。
        // primary の公開出力を加え、包含型では後段で action も候補にする。
        auto available = obs_spec;
        for (const auto& [key, source] : network_config.output_keys) {
            const auto value = output.At(key);
            available[key] = anet::TensorSpec{
                .type = anet::SpaceType::Vector,
                .shape = std::vector<int64_t>(value.sizes().begin() + 1, value.sizes().end()),
                .dtype = value.scalar_type()};
        }

        // 包含型なら action を候補に足す
        if (skip_config_.network_mode == SkipNetworkMode::ActionIn) {
            available[NetworkModel::kKeyActionInput] = anet::TensorSpec{
                .type = anet::SpaceType::Vector, .shape = {}, .dtype = torch::kInt64, .num_classes = n_actions_};
        }

        // 未使用入力の WARN を避けるため、候補全体でなく実際の入力参照だけを spec にする。
        // bind 因子には branch 名も混ざるので、候補にある鍵だけを拾う。
        anet::TensorSpecMap inputs;
        bool binds_action = false;
        for (const auto& [name, branch] : skip_definition->network_config.branches) {
            for (const auto& term : branch.bind_terms) {
                for (const auto& key : term) {
                    if (available.contains(key)) inputs[key] = available.at(key);
                    if (key == NetworkModel::kKeyActionInput) binds_action = true;
                }
            }
        }
        // Body の素通し出力が参照する入力も spec へ含める。
        for (const auto& [key, source] : skip_definition->network_config.output_keys) {
            if (available.contains(source)) inputs[source] = available.at(source);
        }


        // action の bind は包含型だけで必須とし、分離型では禁止する。
        const bool action_in = skip_config_.network_mode == SkipNetworkMode::ActionIn;
        if (binds_action != action_in) {
            ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.skip.network_mode='" << (action_in ? SkipConfig::kNetworkModeActionIn : SkipConfig::kNetworkModeActionOut)
                << "': skip.net action binding=" << binds_action
                << "; expected " << (action_in ? "a branch binding action" : "no branch binding action") << ".");
        }

        // spec を確定して構築し、dummy forward で Head の rank と shape まで検証する。
        skip_online_net_ = anet::nn::NetworkBuilder::BuildNetwork(skip_definition->network_config, inputs, skip_definition->head_factory, skip_definition->seed, device);
        skip_online_net_->to(device);
        skip_online_net_->eval();

        // Head のrank/shapeも構築境界で確定させる。
        anet::TensorDict skip_dummy;
        for (const auto& [key, spec] : inputs) {
            if (output.Contains(key)) skip_dummy.Set(key, output.At(key));
            else if (dummy.Contains(key)) skip_dummy.Set(key, dummy.At(key));
            else skip_dummy.Set(key, torch::zeros({1}, torch::TensorOptions().device(device).dtype(torch::kInt64)));
        }
        skip_online_net_->Forward(skip_dummy);
        // 同じ構造・初期値から target を複製する。
        skip_target_net_ = skip_online_net_->Clone(device);
        skip_target_net_->eval();

    }

    ValidateSpectralNormSoftUpdateConfig();

    LOG::info() << "========== MODEL SHAPE DUMP ==========";
    for (const auto& pair : online_net_->named_parameters()) {
        LOG::info() << pair.key() << " : " << pair.value().sizes();
    }
    if (skip_online_net_) {
        LOG::info() << "========== SKIP MODEL SHAPE DUMP ==========";
        for (const auto& pair : skip_online_net_->named_parameters()) {
            LOG::info() << "skip." << pair.key() << " : " << pair.value().sizes();
        }
    }
    LOG::info() << "======================================";

    LOG::info() << "Number of online network parameter tensors: " << online_net_->parameters().size()
        + (skip_online_net_ ? skip_online_net_->parameters().size() : 0);
    LOG::info() << "Number of target network parameter tensors: " << target_net_->parameters().size()
        + (skip_target_net_ ? skip_target_net_->parameters().size() : 0);
}

NetworkModel::NetworkModel(
    const NetworkModelConfig& config,
    std::shared_ptr<anet::nn::Network> online_net,
    std::shared_ptr<anet::nn::Network> target_net,
    int64_t n_actions,
    bool distributional)
    : config_(config)
    , online_net_(std::move(online_net))
    , target_net_(std::move(target_net))
    , n_actions_(n_actions)
    , distributional_(distributional)
{
    ANET_ASSERT(n_actions_ > 0);
    ANET_ASSERT(online_net_ != nullptr);
    ANET_ASSERT(target_net_ != nullptr);
    if (online_net_ == target_net_) {
        ANET_SYSTEM_ERROR("NetworkModel requires distinct online and target Networks; supplied the same Network for both roles.");
    }
    online_net_->eval();
    target_net_->eval();
    ValidateSpectralNormSoftUpdateConfig();
}

void NetworkModel::ValidateSpectralNormSoftUpdateConfig() const
{
    const auto& source = config_.soft_update_tau;
    anet::detail::ValidateProfiledValueConfig(source, "model.soft_update_tau");
    // 共通boundsは常に検証し、SN固有の上限だけをsoft updateへ適用する。
    auto tau = source;
    tau.min_value = 0.0f;
    tau.max_value = 1.0f;
    if (config_.hard_update_interval <= 0 && !(tau.type == kConstantScheduleType && tau.value == 1.0f)
        && (!online_net_->GetSpectralNormEntries().empty()
            || !target_net_->GetSpectralNormEntries().empty()
            || (skip_online_net_ && !skip_online_net_->GetSpectralNormEntries().empty()))) {
        tau.max_value = anet::nn::kSpectralNormMaxSoftUpdateTau;
    }
    anet::detail::ValidateProfiledValueConfig(tau, "model.soft_update_tau");
}

anet::TensorDict NetworkModel::ForwardOnlineWithTrain(
    const anet::TensorDict& obs, anet::nn::NetworkForwardExecutor& forward,
    const anet::nn::NetworkForwardCallInfo& info, anet::nn::NetworkForwardCapture* capture) const
{
    anet::TrainingModeGuard guard(*online_net_, true);
    return forward.Forward(obs, info, capture);
}

anet::TensorDict NetworkModel::ForwardOnlineUpTo(
    const anet::TensorDict& obs, const std::string& branch_key, anet::nn::NetworkForwardCapture* capture) const
{
    torch::NoGradGuard no_grad;
    anet::TrainingModeGuard guard(*online_net_, false);
    return online_net_->ForwardUpTo(obs, branch_key, capture);
}

bool NetworkModel::IsDistributional() const
{
    return distributional_;
}

std::vector<torch::Tensor> NetworkModel::GetOnlineParameters() const
{
    return online_net_->parameters();
}

torch::OrderedDict<std::string, torch::Tensor> NetworkModel::GetOnlineNamedParameters() const
{
    return online_net_->named_parameters();
}

void NetworkModel::UpdateTarget(step_t learn_step, float tau)
{
    torch::NoGradGuard grad_guard;

    if (config_.hard_update_interval > 0) {
        if (learn_step % config_.hard_update_interval == 0) {
            HardUpdate();
        }
        return;
    }
    SoftUpdate(tau);
}

void NetworkModel::SoftUpdate(float tau)
{
    online_net_->SoftCopyTo(*target_net_, tau);
    if (skip_online_net_) skip_online_net_->SoftCopyTo(*skip_target_net_, tau);
}

void NetworkModel::HardUpdate()
{
    online_net_->CopyTo(*target_net_);
    if (skip_online_net_) skip_online_net_->CopyTo(*skip_target_net_);
}

std::optional<anet::TensorDictFunction> NetworkModel::GetTensorDictFunction(const std::string& key, const torch::Device& device)
{
    // 入力 TensorDict の device 正規化は Agent wrapper の責務。
    // ここでは key routing だけを行い、network 本体の関数を返す。
    (void)device;
    static constexpr const char* POLICY_PREFIX = "policy-net.";
    static constexpr const char* TARGET_PREFIX = "target-net.";

    // Keyの指定に応じて、対象のネットワークを切り替える
    std::shared_ptr<anet::nn::Network> net = nullptr;
    std::string subkey;

    if (anet::StartsWith(key, POLICY_PREFIX)) {
        net = online_net_;
        subkey = anet::RemovePrefix(key, POLICY_PREFIX);
    } else if (anet::StartsWith(key, TARGET_PREFIX)) {
        net = target_net_;
        subkey = anet::RemovePrefix(key, TARGET_PREFIX);
    }

    // 対象が見つからなければ nullopt
    if (!net) {
        return std::nullopt;
    }

    // 汎用 head ベース抽出へ委譲
    return net->GetTensorDictFunction(subkey);
}

int64_t NetworkModel::Save(OutputArchive& archive) const
{
	int64_t size = 0;
    size += archive.WriteTorchObject(online_net_);
    size += archive.WriteTorchObject(target_net_);
    if (skip_online_net_) {
        size += archive.WriteTorchObject(skip_online_net_);
        size += archive.WriteTorchObject(skip_target_net_);
    }
    return size;
}

int64_t NetworkModel::Load(InputArchive& archive)
{
    return LoadCheckpoint(archive, skip_online_net_ != nullptr);
}

int64_t NetworkModel::LoadCheckpoint(InputArchive& archive, bool has_skip)
{
    if (has_skip && !skip_online_net_) ANET_SYSTEM_ERROR("Checkpoint contains skip networks but skip.enabled=false.");
    int64_t size = 0;
    size += archive.ReadTorchObject(online_net_);
    size += archive.ReadTorchObject(target_net_);
    online_net_->eval();
    target_net_->eval();
    if (has_skip) {
        size += archive.ReadTorchObject(skip_online_net_);
        size += archive.ReadTorchObject(skip_target_net_);
        skip_online_net_->eval();
        skip_target_net_->eval();
    }
    return size;
}

anet::TensorDict anet::rl::dqn::MakeSkipInput(const anet::TensorDict& obs, const anet::TensorDict& primary,
    const torch::Tensor& actions, bool stop_gradient)
{
    // obs と公開特徴を合わせ、stop のときだけ primary への勾配を切る。
    auto input = obs;
    for (const auto& [key, value] : primary) input.Set(key, stop_gradient ? value.detach() : value);
    // action_in は [B] int64 の行動を bind し、Body が one-hot へ展開する。
    input.Set(NetworkModel::kKeyActionInput, actions);
    return input;
}

anet::TensorDict anet::rl::dqn::ReadSkipValues(const anet::TensorDict& output, const anet::TensorDict& primary,
    const torch::Tensor& actions, const SkipConfig& config)
{
    ANET_PROFILE_FUNC();
    const bool distributed = output.Contains(SkipHeadFactory::kKeyQDist);
    auto values = output.At(distributed ? SkipHeadFactory::kKeyQDist : SkipHeadFactory::kKeyQ);
    const auto batch = actions.size(0);
    // 分離型の [B,A,J(,K)] は返す行動で絞り、包含型と同じ [B,J(,K)] に揃える。
    if (config.network_mode == SkipNetworkMode::ActionOut) {
        const auto index = distributed
            ? actions.view({batch, 1, 1, 1}).expand({batch, 1, config.max_length, values.size(-1)})
            : actions.view({batch, 1, 1}).expand({batch, 1, config.max_length});
        values = values.gather(1, index).squeeze(1);
    }
    if (config.head.use_dueling) {
        const auto base = primary.At(distributed ? "q_dist" : "q").detach();
        const auto index = distributed ? actions.view({batch, 1, 1}).expand({batch, 1, base.size(-1)}) : actions.view({batch, 1});
        const auto selected = base.gather(1, index).squeeze(1);
        values = selected.unsqueeze(1) + values - values.narrow(1, 0, 1);
    }
    anet::TensorDict result;
    if (distributed) result.Set("q_dist", values);
    result.Set("q", distributed ? values.mean(-1) : values);
    return result;
}
