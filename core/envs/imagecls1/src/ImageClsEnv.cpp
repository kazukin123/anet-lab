#include "ImageClsEnv.hpp"

#include <limits>

#include "anet/profile.hpp"

using namespace anet::rl::env;

namespace {

anet::seed_t ResolveSeed(std::optional<anet::seed_t> seed)
{
    return seed.value_or(anet::SeedMaker::MakeAutoSeed());
}

} // namespace

ImageClsEnv::ImageClsEnv(
    const ImageClsEnvConfig& config,
    std::unique_ptr<anet::img::ImageDataSource> source,
    anet::ConfigData config_data,
    const std::string& name,
    int num_envs,
    anet::rl::RunMode run_mode)
    : BatchEnvBase(name, num_envs, run_mode, std::move(config_data))
    , config_(config)
    , batch_spec_({ num_envs, 1 })
    , source_(std::move(source))
{
    batch_spec_.num_threads = source_->GetWorkerCount();
    const auto& dataset = source_->GetDataset();
    const auto& dataset_config = dataset->GetConfig();
    const auto& class_names = dataset->GetManifest().class_names;

    spec_.state_spec.obs_spec[anet::rl::ObsKeys::kGrid] = anet::TensorSpec{
        .type = anet::SpaceType::Grid,
        .shape = { 3, dataset_config.image_height, dataset_config.image_width },
        .dtype = torch::kUInt8,
    };
    spec_.state_spec.obs_spec[anet::rl::ObsKeys::kVector] = anet::TensorSpec{
        .type = anet::SpaceType::Vector,
        .shape = { 1 },
        .dtype = torch::kInt64,
        .num_classes = static_cast<int64_t>(class_names.size()),
    };
    spec_.action_spec.is_discrete = true;
    spec_.action_spec.value_labels = class_names;
}

anet::rl::BatchState ImageClsEnv::MakeState(
    const anet::img::ImageBatch& batch,
    const torch::Tensor& done,
    const torch::Tensor& truncated,
    const torch::Tensor& episode_start) const
{
    anet::TensorDict obs;
    obs.Set(anet::rl::ObsKeys::kGrid, batch.grid);
    obs.Set(anet::rl::ObsKeys::kVector, batch.targets.unsqueeze(1));
    return anet::rl::BatchState(obs, done, truncated, episode_start);
}

std::vector<anet::rl::AuxData> ImageClsEnv::MakeAuxDataList(
    const anet::img::ImageBatch& batch) const
{
    std::vector<anet::rl::AuxData> result;
    result.reserve(static_cast<size_t>(batch_spec_.num_envs));
    for (int64_t lane = 0; lane < batch_spec_.num_envs; ++lane) {
        anet::rl::AuxData aux;
        aux["step_count"] = torch::tensor(step_count_, torch::kInt32);
        aux["current_true_label"] = batch.targets[lane].clone();
        aux["valid"] = torch::tensor(lane < batch.valid_count, torch::kBool);
        result.push_back(std::move(aux));
    }
    return result;
}

std::shared_ptr<const anet::rl::BatchResetResult> ImageClsEnv::Reset()
{
    ANET_PROFILE_SCOPE(ImageClsEnv_Reset);
    step_count_ = 0;
    current_batch_ = source_->NextBatch();
    const auto options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    auto state = MakeState(
        current_batch_,
        torch::zeros({ batch_spec_.num_envs }, options),
        torch::zeros({ batch_spec_.num_envs }, options),
        torch::ones({ batch_spec_.num_envs }, options));
    return std::make_shared<const anet::rl::PlainBatchResetResult>(
        std::move(state), MakeAuxDataList(current_batch_));
}

void ImageClsEnv::UpdateMetrics(const torch::Tensor& reward)
{
    if (anet::rl::IsEval(GetRunMode())) {
        accuracy_correct_ += reward.narrow(0, 0, current_batch_.valid_count).sum().item<double>();
        accuracy_samples_ += current_batch_.valid_count;
        epoch_count_ += current_batch_.completed_cycles;
        if (current_batch_.window_end) {
            accuracy_snapshot_ = accuracy_samples_ == 0
                ? std::numeric_limits<float>::quiet_NaN()
                : static_cast<float>(accuracy_correct_ / static_cast<double>(accuracy_samples_));
            accuracy_correct_ = 0.0;
            accuracy_samples_ = 0;
        }
        return;
    }

    uint64_t observed_boundaries = 0;
    for (int64_t lane = 0; lane < current_batch_.valid_count; ++lane) {
        const auto epoch_tag = current_batch_.epoch_tags[static_cast<size_t>(lane)];
        if (!has_active_epoch_tag_) {
            active_epoch_tag_ = epoch_tag;
            has_active_epoch_tag_ = true;
        } else if (epoch_tag != active_epoch_tag_) {
            accuracy_snapshot_ = static_cast<float>(accuracy_correct_ / static_cast<double>(accuracy_samples_));
            accuracy_correct_ = 0.0;
            accuracy_samples_ = 0;
            active_epoch_tag_ = epoch_tag;
            ++epoch_count_;
            ++observed_boundaries;
        }
        accuracy_correct_ += reward[lane].item<double>();
        ++accuracy_samples_;
    }
    if (current_batch_.completed_cycles > observed_boundaries) {
        accuracy_snapshot_ = static_cast<float>(accuracy_correct_ / static_cast<double>(accuracy_samples_));
        accuracy_correct_ = 0.0;
        accuracy_samples_ = 0;
        has_active_epoch_tag_ = false;
        epoch_count_ += current_batch_.completed_cycles - observed_boundaries;
    }
}

std::shared_ptr<const anet::rl::BatchStepResult> ImageClsEnv::Step(
    std::shared_ptr<anet::rl::BatchActionInfo> action_info)
{
    // current batchだけを採点し、padding laneを集計とtransition数から除外する。
    ANET_PROFILE_SCOPE(reward);
    const auto actions = action_info->GetAction(torch::Device(torch::kCPU));
    ANET_ASSERT_DTYPE_MSG(actions, torch::kInt64, "ImageClsEnv actions must be int64.");
    ANET_ASSERT_SHAPE(actions, { batch_spec_.num_envs });

    auto reward = actions.eq(current_batch_.targets).to(torch::kFloat32);
    if (current_batch_.valid_count < batch_spec_.num_envs) {
        reward.narrow(0, current_batch_.valid_count,
            batch_spec_.num_envs - current_batch_.valid_count).zero_();
    }
    UpdateMetrics(reward);
    const auto scored_aux_data = MakeAuxDataList(current_batch_);

    // train episodeまたはeval windowの境界を既存BatchEnv flagへ翻訳する。
    ++step_count_;
    const bool episode_end = anet::rl::IsEval(GetRunMode())
        ? current_batch_.window_end : step_count_ >= config_.max_steps;
    // 境界通知後もRunnerが継続できるよう、次batchをfresh storageで先読みする。
    ANET_PROFILE_SCOPE_NEXT(next_batch);
    auto next_batch = source_->NextBatch();

    // evalはlane 0だけを代表episodeとし、trainは全laneを同時終端させる。
    ANET_PROFILE_SCOPE_NEXT(build_result);
    const auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    auto done = torch::zeros({ batch_spec_.num_envs }, bool_options);
    if (episode_end) {
        if (anet::rl::IsEval(GetRunMode())) done[0].fill_(true);
        else done.fill_(true);
    }
    auto truncated = torch::zeros({ batch_spec_.num_envs }, bool_options);
    auto next_episode_start = torch::zeros({ batch_spec_.num_envs }, bool_options);
    auto continue_episode_start = torch::zeros({ batch_spec_.num_envs }, bool_options);
    if (episode_end) {
        if (anet::rl::IsEval(GetRunMode())) continue_episode_start[0].fill_(true);
        else continue_episode_start.fill_(true);
    }
    auto next_state = MakeState(next_batch, done, truncated, next_episode_start);
    auto continue_state = MakeState(
        next_batch,
        torch::zeros({ batch_spec_.num_envs }, bool_options),
        torch::zeros({ batch_spec_.num_envs }, bool_options),
        continue_episode_start);

    const auto n_transitions = static_cast<uint32_t>(current_batch_.valid_count);
    const auto n_episode_end = episode_end
        ? static_cast<uint32_t>(anet::rl::IsEval(GetRunMode()) ? 1 : batch_spec_.num_envs)
        : 0U;
    current_batch_ = std::move(next_batch);
    if (episode_end) step_count_ = 0;

    return std::make_shared<const anet::rl::PlainBatchStepResult>(
        std::move(reward), std::move(next_state), std::move(continue_state),
        n_transitions, n_episode_end, scored_aux_data);
}

void ImageClsEnv::Shutdown()
{
    source_->Shutdown();
}

std::optional<float> ImageClsEnv::GetScalar(const std::string& key, int64_t index) const
{
    if (index != -1) {
        ANET_SYSTEM_ERROR("ImageClsEnv scalar is global and does not accept an index. key='"
            << key << "' index=" << index);
    }
    if (key == "accuracy") return accuracy_snapshot_;
    if (key == "epoch_count") return static_cast<float>(epoch_count_);
    ANET_SYSTEM_ERROR("Unknown ImageClsEnv scalar key='" << key
        << "'. Expected accuracy or epoch_count.");
    return std::nullopt;
}

void ImageClsEnvFactory::ValidateConfig(
    const anet::ConfigData& config_data,
    anet::rl::RunMode run_mode,
    const std::string& config_prefix) const
{
    // catalogとSource schemaだけを検証し、manifest/decode I/OはAcquireまで遅延する。
    const auto catalog = anet::img::ResolveImageDatasetCatalog(config_data);
    anet::img::ImageDatasetManager::Instance().RegisterCatalog(catalog);
    const ImageClsEnvConfig env_config(config_data, config_prefix);
    const anet::img::TrainImageDataSourceConfig train_config(config_data, config_prefix);
    const anet::img::EvalImageDataSourceConfig eval_config(config_data, config_prefix);
    (void)env_config;
    (void)run_mode;

    const auto validate_dataset_reference = [&catalog](const char* role, const std::string& dataset_key) {
        if (catalog.contains(dataset_key)) return;
        std::string available;
        for (const auto& [key, config] : catalog) {
            (void)config;
            if (!available.empty()) available += ",";
            available += key;
        }
        ANET_SYSTEM_ERROR("Unknown ImageClsEnv DatasetKey. role='" << role
            << "' dataset_key='" << dataset_key << "' available_keys='" << available << "'");
    };
    validate_dataset_reference("train", train_config.dataset_key);
    validate_dataset_reference("eval", eval_config.dataset_key);
}

std::shared_ptr<anet::rl::BatchEnv> ImageClsEnvFactory::CreateBatchEnv(
    const anet::ConfigData& config_data,
    const torch::Device& device,
    const std::string& name,
    std::optional<anet::seed_t> seed,
    int num_envs,
    anet::rl::RunMode run_mode,
    const std::string& config_prefix)
{
    if (!device.is_cpu()) {
        ANET_SYSTEM_ERROR("ImageClsEnv supports CPU device only. requested_device='"
            << device.str() << "'");
    }
    ValidateConfig(config_data, run_mode, config_prefix);
    const auto catalog = anet::img::ResolveImageDatasetCatalog(config_data);
    auto& dataset_manager = anet::img::ImageDatasetManager::Instance();
    dataset_manager.RegisterCatalog(catalog);

    const ImageClsEnvConfig env_config(config_data, config_prefix);
    const anet::img::TrainImageDataSourceConfig train_config(config_data, config_prefix);
    const anet::img::EvalImageDataSourceConfig eval_config(config_data, config_prefix);
    const anet::rl::BatchEnvBuilderConfig builder_config(config_data);

    // 標準Train/Eval sourceのmanifestはEnv構築時に両方検証し、decode/cacheだけを使用時まで遅延する。
    const auto train_dataset = dataset_manager.Acquire(train_config.dataset_key);
    const auto eval_dataset = dataset_manager.Acquire(eval_config.dataset_key);

    // 実際に注入されたEnv設定と参照dataset設定を、子を含む不変snapshotへ統合する。
    anet::ConfigData effective_config = builder_config.GetScopedConfigData();
    effective_config.MergeFromChecked(env_config.GetScopedConfigData());
    effective_config.MergeFromChecked(train_config.GetScopedConfigData());
    effective_config.MergeFromChecked(eval_config.GetScopedConfigData());
    effective_config.MergeFromChecked(train_dataset->GetConfig().ToConfigData(train_config.dataset_key));
    effective_config.MergeFromChecked(eval_dataset->GetConfig().ToConfigData(eval_config.dataset_key));

    const auto resolved_seed = ResolveSeed(seed);
    std::unique_ptr<anet::img::ImageDataSource> source;
    if (anet::rl::IsTrain(run_mode)) {
        source = std::make_unique<anet::img::ImageDataSource>(
            train_config, num_envs, resolved_seed, builder_config.worker_type, builder_config.worker_threads);
    } else {
        source = std::make_unique<anet::img::ImageDataSource>(
            eval_config, num_envs, resolved_seed, builder_config.worker_type, builder_config.worker_threads);
    }
    return std::make_shared<ImageClsEnv>(
        env_config, std::move(source), std::move(effective_config), name, num_envs, run_mode);
}
