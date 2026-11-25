#include "ppo_agent.hpp"
#include <filesystem>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/diag.hpp"
#include "anet/metrics_logger.hpp"
#include "ppo_actor.hpp"

using namespace anet::rl::ppo;
namespace LOG = anet::log;

constexpr char kNetworkSeedName[] = "network";
constexpr char kLearnerSeedName[] = "learner";
constexpr char kNetworkStructureTag[] = "net.structure";
constexpr char kNetworkDetailTag[] = "net.detail";
constexpr char kNetworkConfigTag[] = "net.body";
constexpr char kNetworkConfigSuffix[] = ".net";

PPOAgent::PPOAgent(const PPOAgentConfig& config, const anet::nn::NetworkConfig& network_config,
    const anet::rl::EnvSpec& env_spec, const anet::rl::BatchEnvSpec& batch_env_spec, torch::Device device,
    std::optional<anet::seed_t> seed)
    : anet::rl::AgentBase(device, batch_env_spec, env_spec, seed), config_(config)
{
    ANET_PROFILE_FUNC();

    // 行動は離散で、2 つ以上であることを確かめる。連続行動は、分布と行動の受け渡しが別に要る。
    if (!env_spec.action_spec.is_discrete || n_actions_ < 2) {
        ANET_SYSTEM_ERROR(kAgentClassId << " requires a discrete action space with at least 2 actions; got is_discrete="
            << env_spec.action_spec.is_discrete << " actions=" << n_actions_ << ".");
    }
    // lane 数と突き合わせる設定の検査を、Network を作る前に済ませる。
    config_.ValidateLaneCount(num_envs_);

    LOG::info() << kAgentClassId << " config=" << config_.ToString();
    anet::MetricsLogger::Instance()->Log(config_);

    // seed の準備。Network の初期化と、Learner の並べ替えで、別の系列を使う。
    anet::SeedMaker seed_maker(GetSeed());
    const auto network_seed = seed_maker.MakeNamedSeed(kNetworkSeedName);
    const auto learner_seed = seed_maker.MakeNamedSeed(kLearnerSeedName);

    // Network（Agent の Resource）を作り、構造を記録する。
    model_ = std::make_shared<NetworkModel>(network_config, env_spec.state_spec.obs_spec, n_actions_,
        config_.head.policy.init, config_.head.value.init, device_, network_seed);
    anet::MetricsLogger::Instance()->Log(kNetworkConfigTag, network_config.ToJson());
    const auto structure_view = model_->GetNetwork()->MakeGraphViz(anet::nn::NetworkGraphVizConfig{});
    anet::MetricsLogger::Instance()->Log(kNetworkStructureTag, *structure_view);
    const auto detail_view = model_->GetNetwork()->MakeGraphViz(config_.nn_viz);
    anet::MetricsLogger::Instance()->Log(kNetworkDetailTag, *detail_view);

    // Learner は optimizer を持つので、保存と読込の対象として Agent が持つ。
    learner_ = std::make_shared<Learner>(config_.learner, config_.rollout.length, num_envs_, n_actions_, mutex_,
        model_, device_, learner_seed);

    // 保存したファイルの指定があれば、Network と optimizer を読む。
    if (!config_.auto_load_file.empty()) {
        LOG::info() << kAgentClassId << " auto-load file=" << config_.auto_load_file;
        LoadCheckpoint(config_.auto_load_file);
    }
}

std::shared_ptr<anet::rl::Actor> PPOAgent::CreateActor(const ActorRequest& request) const
{
    ANET_PROFILE_FUNC();
    env_spec_.CheckSameStateActionSpec(request.env_spec);
    // Actor の設定は、カタログの項目を Actor キーで引く。未定義のキーは、定義済みのキーを添えて止める。
    const auto& actor_config = FindActorConfig(config_.actor, request.actor_key, config_.GetConfigPrefix());
    // Network を共有する Actor は、Agent と同じ device でしか動かせない。
    ValidateActorDevice(actor_config.clone_model, request.device);
    return std::make_shared<Actor>(actor_config, mutex_, model_, request.device, request.seed);
}

std::shared_ptr<anet::rl::Learner> PPOAgent::CreateLearner()
{
    return learner_;
}

void PPOAgent::ConfigureScalarMetricSubscriptions(const std::vector<ScalarMetricSubscription>& subscriptions)
{
    std::unique_lock lock(*mutex_);
    learner_->ConfigureScalarMetricSubscriptions(subscriptions);
}

int64_t PPOAgent::Save(anet::OutputArchive& archive) const
{
    ANET_PROFILE_FUNC();
    // 更新の途中の重みを書かないよう、共有のロックを持つ。
    std::shared_lock lock(*mutex_);
    int64_t size = 0;
    size += archive.Write(anet::ArchiveHeader(kCheckpointFormat));
    size += archive.Write(config_.ToString());
    size += model_->Save(archive);
    size += learner_->Save(archive);
    LOG::info() << kAgentClassId << " serialized. total_size=" << size;
    return size;
}

void PPOAgent::LoadCheckpoint(const std::string& filename)
{
    ANET_PROFILE_FUNC();
    std::ifstream stream(filename, std::ios::binary);
    if (!stream) {
        ANET_SYSTEM_ERROR("Failed to open " << config_.GetFullKey("auto_load_file") << "='" << filename
            << "' (cwd=" << std::filesystem::current_path().string() << ").");
    }
    anet::InputArchive archive(stream, filename);

    // 形式の名前を確かめる。別の Agent や、別の版のファイルは読まない。
    anet::ArchiveHeader header;
    archive.Read(header);
    if (header.GetInfo() != kCheckpointFormat) {
        ANET_SYSTEM_ERROR("Incompatible checkpoint " << config_.GetFullKey("auto_load_file") << "='" << filename
            << "': format='" << header.GetInfo() << "'; expected '" << kCheckpointFormat << "'.");
    }
    std::string saved_config;
    archive.Read(saved_config);

    // Network と optimizer は Actor・Learner と共有するので、排他のロックの中で読む。
    std::unique_lock lock(*mutex_);
    model_->Load(archive);
    learner_->Load(archive);
}

std::shared_ptr<anet::rl::Agent> PPOAgentFactory::CreateAgent(const EnvSpec& env_spec,
    const BatchEnvSpec& batch_env_spec, const torch::Device& device, const anet::ConfigData& config_data,
    std::shared_ptr<anet::rl::Notifier> notifier, std::optional<anet::seed_t> seed) const
{
    const PPOAgentConfig config(config_data);
    const anet::nn::NetworkConfig network_config(config_data, GetTargetAgentClassId() + kNetworkConfigSuffix);
    return std::make_shared<PPOAgent>(config, network_config, env_spec, batch_env_spec, device, seed);
}
