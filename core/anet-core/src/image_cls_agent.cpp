// image_cls_agent.cpp

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include "anet/image_cls_agent.hpp"
#include "anet/log.hpp"
#include "anet/profile.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/nn_util.hpp"
#include "nn_heads.hpp"

using namespace anet::rl::img_cls;
namespace LOG = anet::log;


// ======================================================
// Actor (データ運び屋 兼 GUI可視化用推論)
// ======================================================

ImageClsActor::ImageClsActor(
    const ImageClsAgentConfig& config,
    std::shared_ptr<std::shared_mutex> mutex,
    std::shared_ptr<anet::nn::Network> network,
    anet::rl::RunMode run_mode,
    torch::Device device,
    std::shared_ptr<anet::nn::Network> src_network)
    : config_(config)
    , run_mode_(run_mode)
    , mutex_(mutex)
    , network_(network)
    , src_network_(src_network ? src_network : network)
    , device_(device)
{
}

std::shared_ptr<anet::rl::BatchActionInfo> ImageClsActor::MakeAction(
    const anet::rl::StepCounts& step, const anet::rl::BatchState& state) const
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;

    // 推論準備
    ANET_PROFILE_SCOPE(lock_wait);
    std::shared_lock lock(*mutex_, std::defer_lock);
    if (network_ == src_network_) {
        lock.lock();
    }
    
    // Forward
    ANET_PROFILE_SCOPE_NEXT(obs_to_device);
    auto obs = state.obs.To(device_);
    anet::TensorDict trace;
    anet::TraceSink sink = anet::rl::MakeActionTraceSink(trace);
    ANET_PROFILE_SCOPE_NEXT(forward);
    anet::TensorDict outputs;
    {
        anet::Autocast autocast_guard(
            device_,
            config_.bf16.enabled && config_.bf16.actor,
            torch::kBFloat16);
        outputs = network_->Forward(obs, sink);
    }

    // 推論後処理
    if (lock.owns_lock()) {
        lock.unlock();
    }

	// argmaxで推論結果を得る
    auto logits = outputs.At("logits");
    ANET_PROFILE_SCOPE_NEXT(action_to_cpu);
    auto action = logits.argmax(1).to(torch::kCPU);
    //ANET_LOG_DEBUG("action=" << anet::ToString(action));

    // 可視化用に全クラスの確率分布を info に入れる
    anet::TensorDict info;
    ANET_PROFILE_SCOPE_NEXT(info_to_cpu);
    info.Set("probs", torch::softmax(logits, 1).to(torch::kCPU));

    // BatchActionInfo(action, info, aux) の形式で返却
    ANET_PROFILE_SCOPE_NEXT(build_action_info);
    auto action_info = std::make_shared<anet::rl::BatchActionInfo>(action, info);
    anet::rl::AppendTraceAux(action_info->GetAuxData(), trace);
    return action_info;
}

void ImageClsActor::Sync()
{
    if (network_ != src_network_) {
        // Clone済み actor だけ、学習側の network から現在の重みを取り込む
        std::shared_lock lock(*mutex_);
        src_network_->CopyTo(*network_);
        network_->eval();
    }
}


// ======================================================
// Learner (純粋な教師あり学習のコア)
// ======================================================

ImageClsLearner::ImageClsLearner(
    const ImageClsAgentConfig& config,
    std::shared_ptr<std::shared_mutex> mutex,
    std::shared_ptr<anet::nn::Network> network,
    std::shared_ptr<anet::ProfiledValue<double>> learning_rate,
    torch::Device device,
    std::optional<seed_t> seed)
    : anet::RandomHolder(seed)
    , config_(config)
    , mutex_(mutex)
    , network_(network)
    , learning_rate_(learning_rate)
    , device_(device)
{
    // Optimizerを構築
    if (!learning_rate_) {
        ANET_SYSTEM_ERROR("ImageClsLearner: learning_rate must not be null.");
    }
    auto opt_options = torch::optim::AdamWOptions(learning_rate_->Value()).weight_decay(config_.weight_decay);
    LOG::verbose() << "ImageClsLearner optimizer: lr=" << opt_options.lr()
        << " weight_decay=" << opt_options.weight_decay()
        << " fused=" << config_.use_fused_optimizer;
    if (config_.use_fused_optimizer) {
        optimizer_ = std::make_unique<anet::FusedAdamW>(network_->parameters(), opt_options);
    } else {
        optimizer_ = std::make_unique<torch::optim::AdamW>(network_->parameters(), opt_options);
    }
}

struct ImageClsLearner::MixResult {
    torch::Tensor targets_b;
    double lambda = 1.0;
    std::string mode = "none";
};

struct ImageClsLearner::CutMixBox {
    int64_t x1 = 0;
    int64_t y1 = 0;
    int64_t x2 = 0;
    int64_t y2 = 0;
};

double ImageClsLearner::SampleBeta(double alpha)
{
    // alpha が無効な場合は混合なしの lambda として扱う
    if (alpha <= 0.0) {
        return 1.0;
    }

    // 対称 Beta 分布を 2 つの gamma sample から構成する
    auto& rng = *GetRandomGenerator();
    const double a = rng.Gamma(static_cast<float>(alpha), 1.0f);
    const double b = rng.Gamma(static_cast<float>(alpha), 1.0f);
    const double sum = a + b;

    // 極小値で比率を作れない場合は混合なしへ戻す
    if (sum <= std::numeric_limits<double>::min()) {
        return 1.0;
    }
    return a / sum;
}

bool ImageClsLearner::Bernoulli(double probability)
{
    // 端の確率は RNG を消費せず確定させる
    if (probability <= 0.0) return false;
    if (probability >= 1.0) return true;

    // 中間確率だけ Learner の RNG で一様サンプリングする
    auto& rng = *GetRandomGenerator();
    return rng.Uniform01() < probability;
}

ImageClsLearner::CutMixBox ImageClsLearner::SampleCutMixBox(int64_t height, int64_t width, double lambda)
{
    // lambda から差し替え領域の幅と高さを決める
    const double cut_ratio = std::sqrt(std::max(0.0, 1.0 - lambda));
    const int64_t cut_w = static_cast<int64_t>(std::round(width * cut_ratio));
    const int64_t cut_h = static_cast<int64_t>(std::round(height * cut_ratio));

    // 領域中心を画像内から一様に選ぶ
    auto& rng = *GetRandomGenerator();
    const int64_t cx = rng.RandInt(0, static_cast<int>(width - 1));
    const int64_t cy = rng.RandInt(0, static_cast<int>(height - 1));
    const int64_t left_w = cut_w / 2;
    const int64_t top_h = cut_h / 2;

    // 画像境界へクランプした矩形を返す
    return CutMixBox{
        .x1 = std::clamp<int64_t>(cx - left_w, 0, width),
        .y1 = std::clamp<int64_t>(cy - top_h, 0, height),
        .x2 = std::clamp<int64_t>(cx + (cut_w - left_w), 0, width),
        .y2 = std::clamp<int64_t>(cy + (cut_h - top_h), 0, height),
    };
}

static torch::Tensor MixImageTensor(torch::Tensor grid, torch::Tensor paired_grid, double lambda)
{
    // 一度 float へ上げて batch 全体を lambda で線形混合する
    auto mixed = grid.to(torch::kFloat32).mul(lambda)
        .add(paired_grid.to(torch::kFloat32).mul(1.0 - lambda));

    // 既存の uint8 grid 契約を保つため、丸めて元 dtype へ戻す
    if (grid.scalar_type() == torch::kUInt8 || grid.scalar_type() == torch::kByte) {
        return mixed.round().clamp_(0, 255).to(grid.scalar_type());
    }
    return mixed.to(grid.scalar_type());
}

ImageClsLearner::MixResult ImageClsLearner::ApplyMix(anet::TensorDict& obs, const torch::Tensor& targets)
{
    ANET_PROFILE_FUNC();

    ANET_PROFILE_SCOPE(select_mode);
    auto& rng = *GetRandomGenerator();
    // 既定は混合なしとして、target_b も元ラベルを指す
    MixResult mix{
        .targets_b = targets,
    };

    // 設定・batch size・適用確率で混合を bypass する
    const int64_t batch_size = targets.size(0);
    if (!config_.mixup.enabled || batch_size < 2 || !Bernoulli(config_.mixup.prob)) {
        return mix;
    }

    // 有効な alpha から、この batch で使える混合方式を決める
    const bool can_mixup = config_.mixup.mixup_alpha > 0.0;
    const bool can_cutmix = config_.mixup.cutmix_alpha > 0.0;
    if (!can_mixup && !can_cutmix) {
        return mix;
    }

    // Forward へ渡す grid batch を取得し、target batch と揃っていることを確認する
    auto grid = obs.At(anet::rl::ObsKeys::kGrid);
    if (grid.size(0) != batch_size) {
        ANET_SYSTEM_ERROR("ImageClsAgent mixup batch size mismatch. key=" << anet::rl::ObsKeys::kGrid
            << " grid_batch=" << grid.size(0) << " target_batch=" << batch_size);
    }

    // Mixup と CutMix の両方が使える場合だけ switch_prob で選択する
    bool use_cutmix = false;
    if (can_mixup && can_cutmix) {
        use_cutmix = Bernoulli(config_.mixup.switch_prob);
    } else {
        use_cutmix = can_cutmix;
    }

    const double alpha = use_cutmix ? config_.mixup.cutmix_alpha : config_.mixup.mixup_alpha;
    mix.lambda = SampleBeta(alpha);

    // current mini-batch 内で partner を作り、batch size は変えない
    ANET_PROFILE_SCOPE_NEXT(pair_batch);
    auto gen = rng.GetTorchGenerator(device_);
    auto perm = torch::randperm(batch_size, gen, torch::TensorOptions().dtype(torch::kInt64).device(device_));
    mix.targets_b = targets.index_select(0, perm);
    auto paired_grid = grid.index_select(0, perm);

    if (use_cutmix) {
        ANET_PROFILE_SCOPE_NEXT(cutmix_patch);
        // CutMix は NCHW 画像の patch 差し替えとして適用する
        if (grid.dim() != 4) {
            ANET_SYSTEM_ERROR("ImageClsAgent CutMix requires 4D NCHW grid. actual=" << grid.sizes());
        }
        const int64_t height = grid.size(2);
        const int64_t width = grid.size(3);
        if (height <= 0 || width <= 0) {
            ANET_SYSTEM_ERROR("ImageClsAgent CutMix requires positive image size. actual=" << grid.sizes());
        }

        const auto box = SampleCutMixBox(height, width, mix.lambda);
        if (box.x2 > box.x1 && box.y2 > box.y1) {
            // 同じ bbox だけ partner batch からコピーし、grid を差し替える
            using torch::indexing::Slice;
            grid.index({ Slice(), Slice(), Slice(box.y1, box.y2), Slice(box.x1, box.x2) })
                .copy_(paired_grid.index({ Slice(), Slice(), Slice(box.y1, box.y2), Slice(box.x1, box.x2) }));
            obs.Set(anet::rl::ObsKeys::kGrid, grid);
        }

        // clamped bbox の実面積で loss 用 lambda を補正する
        const double area = static_cast<double>(box.x2 - box.x1) * static_cast<double>(box.y2 - box.y1);
        mix.lambda = 1.0 - area / static_cast<double>(height * width);
        mix.mode = "cutmix";
        return mix;
    }

    // Mixup は batch 全体を線形混合した grid へ差し替える
    ANET_PROFILE_SCOPE_NEXT_FROM(mixup_blend, pair_batch);
    obs.Set(anet::rl::ObsKeys::kGrid, MixImageTensor(grid, paired_grid, mix.lambda));
    mix.mode = "mixup";
    return mix;
}

static torch::Tensor CrossEntropyLoss(
    const torch::Tensor& logits, const torch::Tensor& targets,
    const torch::nn::functional::CrossEntropyFuncOptions& options)
{
    return torch::nn::functional::cross_entropy(logits, targets, options);
}

static void SetAdamWLearningRate(torch::optim::Optimizer& optimizer, double learning_rate)
{
    for (auto& group : optimizer.param_groups()) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
        options.lr(learning_rate);
    }
}

anet::rl::BatchUpdateResultList ImageClsLearner::UpdateFromBatch(
    const anet::rl::StepCounts& step,
    const anet::rl::BatchExperience& experiences)
{
    ANET_PROFILE_FUNC();

    // バッチデータの取得と前処理
    // 環境からByte型(0-255)で送られてくる画像を Float32(0.0-1.0) に正規化
    //auto grid = experiences.state.obs.At(anet::rl::ObsKeys::kGrid);
    //auto images = grid.to(device_).to(torch::kFloat32).div(255.0);
    //ANET_LOG_DEBUG("images=" << anet::ToString(images));

    ANET_PROFILE_SCOPE(targets_to_device);
    auto vector = experiences.state.obs.At(anet::rl::ObsKeys::kVector);
    auto targets = vector.to(device_).squeeze(-1).to(torch::kInt64);
    //ANET_LOG_DEBUG("targets=" << anet::ToString(targets));

    torch::Tensor logits;
    torch::Tensor loss;
    MixResult mix;
    double current_learning_rate = learning_rate_->Value();

    {
        // ネットワーク更新準備 (排他ロック)
        ANET_PROFILE_SCOPE_NEXT(lock_wait);
        std::unique_lock lock(*mutex_);
        anet::TrainingModeGuard train_guard(*network_, true);
        ANET_PROFILE_SCOPE_NEXT(zero_grad);
        optimizer_->zero_grad();

        // Forward直前に batch 単位の Mixup/CutMix を適用
        ANET_PROFILE_SCOPE_NEXT(obs_to_device);
        auto obs = experiences.state.obs.To(device_);
        ANET_PROFILE_SCOPE_NEXT(mix);
        mix = ApplyMix(obs, targets);

        // Forward推論
        ANET_PROFILE_SCOPE_NEXT(forward);
        anet::TensorDict outputs;
        {
            anet::Autocast autocast_guard(
                device_,
                config_.bf16.enabled && config_.bf16.learner,
                torch::kBFloat16);
            outputs = network_->Forward(obs);
        }

        // 出力ロジットの取得
        logits = outputs.At("logits");

        // Loss計算 (交差エントロピー + ラベルスムージング)
        ANET_PROFILE_SCOPE_NEXT(loss);
        auto loss_opts = torch::nn::functional::CrossEntropyFuncOptions().label_smoothing(config_.label_smoothing);
        loss = CrossEntropyLoss(logits, targets, loss_opts);
        if (mix.lambda < 1.0) {
            loss = loss.mul(mix.lambda)
                .add(CrossEntropyLoss(logits, mix.targets_b, loss_opts).mul(1.0 - mix.lambda));
        }

        // 誤差逆伝播
        ANET_PROFILE_SCOPE_NEXT(backward);
        loss.backward();

        // 勾配クリッピングを追加
        ANET_PROFILE_SCOPE_NEXT(grad_clip);
        torch::nn::utils::clip_grad_norm_(network_->parameters(), config_.grad_clip_max_norm);

        // Optimizerステップ
        ANET_PROFILE_SCOPE_NEXT(lr_update);
        learning_rate_->Update(step.exp_step);
        current_learning_rate = learning_rate_->Value();
        SetAdamWLearningRate(*optimizer_, current_learning_rate);
        ANET_PROFILE_SCOPE_NEXT(optimizer_step);
        optimizer_->step();
    }

    // メトリクスを計算
    // 確率が一番高いインデックスを予測クラスとして正解と比較
    ANET_PROFILE_SCOPE(metrics);
    auto metric_logits = logits.detach();
    auto preds = metric_logits.argmax(/*dim=*/1);
    auto accuracy = (preds == targets).to(torch::kFloat32).mean();
    auto probs = torch::softmax(metric_logits, /*dim=*/1);
    auto target_prob_a = probs.gather(/*dim=*/1, targets.unsqueeze(1)).squeeze(1);
    auto target_prob_b = probs.gather(/*dim=*/1, mix.targets_b.unsqueeze(1)).squeeze(1);
    auto target_prob_mix = target_prob_a.mul(mix.lambda).add(target_prob_b.mul(1.0 - mix.lambda));
    const double eps = config_.label_smoothing;
    const double class_count = static_cast<double>(logits.size(1));
    const double ceiling = (1.0 - eps)
        * (mix.lambda * mix.lambda + (1.0 - mix.lambda) * (1.0 - mix.lambda))
        + eps / class_count;

    // 結果の返却
    auto result = std::make_shared<ImageClsUpdateResult>();
    result->loss = loss.detach();
    result->accuracy = accuracy;
    result->target_prob_mix_norm = target_prob_mix.mean().div(ceiling);
    result->accuracy_either = (preds == targets)
        .logical_or(preds == mix.targets_b)
        .to(torch::kFloat32)
        .mean();
    result->pred_max_prob = std::get<0>(probs.max(/*dim=*/1)).mean();
    if (mix.mode != "none") {
        result->same_class_pair_ratio = (targets == mix.targets_b)
            .to(torch::kFloat32)
            .mean();
    } else {
        result->same_class_pair_ratio = torch::zeros(
            {},
            torch::TensorOptions().dtype(torch::kFloat32).device(targets.device()));
    }

    // 処理ログ
    ANET_PROFILE_SCOPE_NEXT(verbose_log);
    if (config_.learn_log_interval > 0 && step.learn_step % config_.learn_log_interval == 0) {
        LOG::verbose() << "ImageClsLearner update"
            << " exp_step=" << step.exp_step
            << " learn_step=" << step.learn_step
            //<< " batch=" << targets.size(0)
            << " loss=" << result->GetScalar("loss", -1).value_or(std::numeric_limits<float>::quiet_NaN())
            << " accuracy=" << result->GetScalar("accuracy", -1).value_or(std::numeric_limits<float>::quiet_NaN())
            << " target_prob_mix_norm="
            << result->GetScalar("target_prob_mix_norm", -1).value_or(std::numeric_limits<float>::quiet_NaN())
            << " accuracy_either="
            << result->GetScalar("accuracy_either", -1).value_or(std::numeric_limits<float>::quiet_NaN())
            << " pred_max_prob="
            << result->GetScalar("pred_max_prob", -1).value_or(std::numeric_limits<float>::quiet_NaN())
            << " same_class_pair_ratio="
            << result->GetScalar("same_class_pair_ratio", -1).value_or(std::numeric_limits<float>::quiet_NaN())
            << " learning_rate=" << current_learning_rate
            << " mix_mode=" << mix.mode;
    }

    return { result };
}

int64_t ImageClsLearner::Save(anet::OutputArchive& archive) const
{
    ANET_PROFILE_FUNC();

    int64_t size = 0;
    size += archive.WriteTorchObject(*optimizer_);
    return size;
}

int64_t ImageClsLearner::Load(anet::InputArchive& archive)
{
    ANET_PROFILE_FUNC();

    int64_t size = 0;
    size += archive.ReadTorchObject(*optimizer_);
    return size;
}


// ======================================================
// Agent
// ======================================================

ImageClsAgent::ImageClsAgent(
    const ImageClsAgentConfig& config,
    const anet::nn::NetworkConfig& network_config,
    const anet::rl::EnvSpec& env_spec,
    const anet::rl::BatchEnvSpec& batch_env_spec,
    torch::Device device, std::optional<seed_t> seed)
    : anet::rl::AgentBase(device, batch_env_spec, env_spec, seed), config_(config)
{
    mutex_ = std::make_shared<std::shared_mutex>();
    learning_rate_ = std::make_shared<anet::ProfiledValue<double>>(config_.learning_rate);

    // ログ：パラメータ
    LOG::info() << "ImageClsAgent config=" << config_.ToString();
    anet::MetricsLogger::Instance()->Log(config_);

    // NN構築
    const int64_t num_classes = env_spec.action_spec.GetNumActions();
    if (num_classes <= 0) {
        ANET_SYSTEM_ERROR("ImageClsAgent requires positive action count for logits head. actual=" << num_classes);
    }
    anet::nn::WeightInitConfig head_init_config;
    head_init_config.mode = "he";
    auto head_factory = std::make_shared<anet::nn::LinearHeadFactory>(num_classes, "logits", head_init_config);
    network_ = anet::nn::NetworkBuilder::BuildNetwork(
        network_config,
        env_spec.state_spec.obs_spec,
        head_factory,
        device_);
	network_->to(device_);
    network_->eval();
    anet::MetricsLogger::Instance()->Log("net.body", network_config.ToJson());

    LOG::info() << "Number of Main Network parameters: " << network_->parameters().size();
    LOG::info() << "========== MODEL SHAPE DUMP ==========";
    for (const auto& pair : network_->named_parameters()) {
        LOG::info() << pair.key() << " : " << pair.value().sizes();
    }
    LOG::info() << "======================================";

    // Network グラフ可視化
    {
        auto structure_view = network_->MakeGraphViz(anet::nn::NetworkGraphVizConfig{});
        anet::MetricsLogger::Instance()->Log("net.structure", *structure_view);
        auto detail_view = network_->MakeGraphViz(config_.nn_viz);
        anet::MetricsLogger::Instance()->Log("net.detail", *detail_view);
    }

    // Learner は optimizer state 復元対象なので Agent と同じ lifetime で保持する
    anet::SeedMaker seed_maker(GetSeed());
    const auto learner_seed = seed_maker.MakeNamedSeed("learner");
    learner_ = std::make_shared<ImageClsLearner>(config_, mutex_, network_, learning_rate_, device_, learner_seed);

    // 暫定 checkpoint 指定がある場合は network/optimizer を復元する
    if (!config_.auto_load_file.empty()) {
        LOG::info() << "ImageClsAgent auto-load file=" << config_.auto_load_file;
        LoadNetwork(config_.auto_load_file);
    }

    // ログ記録
    anet::MetricsLogger::Instance()->Log(config);
    LOG::info() << "ImageClsAgent initialized. config=" << config_.ToString();
}

int64_t ImageClsAgent::Save(anet::OutputArchive& archive) const
{
    ANET_PROFILE_FUNC();

    ANET_PROFILE_SCOPE(lock_wait);
    std::shared_lock lock(*mutex_);
    int64_t total_size = 0;

    // ヘッダ
    ANET_PROFILE_SCOPE_NEXT(header);
    anet::ArchiveHeader header("ImageClsAgent");
    const auto header_size = archive.Write(header);
    total_size += header_size;

    // Config
    ANET_PROFILE_SCOPE_NEXT(config);
    const auto config_size = archive.Write(config_.ToString());
    total_size += config_size;

    // Network
    ANET_PROFILE_SCOPE_NEXT(network);
    const auto network_size = archive.WriteTorchObject(network_);
    total_size += network_size;

    // Learner(optimizer)
    ANET_PROFILE_SCOPE_NEXT(learner);
    const auto learner_size = learner_->Save(archive);
    total_size += learner_size;
    ANET_PROFILE_SCOPE_END(learner);

    LOG::info() << "ImageClsAgent Serialized. total_size=" << anet::FormatWithCommas(total_size)
        << " config_size=" << anet::FormatWithCommas(config_size)
        << " network_size=" << anet::FormatWithCommas(network_size)
        << " learner_size=" << anet::FormatWithCommas(learner_size);

    return total_size;
}

void ImageClsAgent::LoadNetwork(const std::string& filename)
{
    ANET_PROFILE_FUNC();

    ANET_PROFILE_SCOPE(header);
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        LOG::info() << "cwd=" << std::filesystem::current_path();
        ANET_SYSTEM_ERROR("Failed to open file for loading: " << filename);
    }
    anet::InputArchive in(ifs, filename);

    // Header
    anet::ArchiveHeader header;
    in.Read(header);
    LOG::verbose() << "ImageClsAgent::LoadNetwork: Archive Header: "
        << header.kMagicWord << " " << header.kFormatVersion << " " << header.info;
    if (header.GetInfo() != "ImageClsAgent") {
        ANET_SYSTEM_ERROR("ImageClsAgent::LoadNetwork: incompatible archive info. file=" << filename
            << " info=" << header.GetInfo() << " expected=ImageClsAgent");
    }

    // Config
    ANET_PROFILE_SCOPE_NEXT(config);
    std::string config_str;
    const auto config_size = in.Read(config_str);
    LOG::info() << "ImageClsAgent::LoadNetwork: config_size=" << config_size;
    LOG::verbose() << "ImageClsAgent::LoadNetwork: config_str=\n" << config_str;

    // Network と Learner は Actor/Learner と共有するため、同じ排他境界で復元する
    ANET_PROFILE_SCOPE_NEXT(lock_wait);
    std::unique_lock lock(*mutex_);
    ANET_PROFILE_SCOPE_NEXT(network);
    const auto network_size = in.ReadTorchObject(network_);
    network_->eval();
    ANET_PROFILE_SCOPE_NEXT(learner);
    const auto learner_size = learner_->Load(in);
    ANET_PROFILE_SCOPE_END(learner);

    LOG::info() << "ImageClsAgent::LoadNetwork: network_size=" << network_size
        << " learner_size=" << learner_size;
}

std::shared_ptr<anet::rl::Actor> ImageClsAgent::CreateActor(
    const anet::rl::BatchEnvSpec& batch_env_spec, anet::rl::RunMode run_mode, bool clone_model, std::optional<torch::Device> device) const
{
    const auto actor_device = device.value_or(device_);
    auto actor_network = network_;
    if (clone_model) {
        // Clone は source network の重みを読むため、Learner 更新と同じ mutex で保護する
        std::shared_lock lock(*mutex_);
        actor_network = network_->Clone(actor_device);
        actor_network->eval();
    }

    // Actorの生成
    return std::make_shared<ImageClsActor>(config_, mutex_, actor_network, run_mode, actor_device, network_);
}

std::shared_ptr<anet::rl::Learner> ImageClsAgent::CreateLearner()
{
    // Agent 所有の Learner を返し、optimizer state を run 断面保存の対象に保つ
    return learner_;
}

std::optional<float> ImageClsAgent::GetScalar(const std::string& key, int64_t index) const
{
    if (key == "learning_rate") {
        std::shared_lock lock(*mutex_);
        return static_cast<float>(learning_rate_->Value());
    }

    return std::nullopt;
}


// ======================================================
// Factory
// ======================================================

std::shared_ptr<anet::rl::Agent> anet::rl::img_cls::ImageClsAgentFactory::CreateAgent(
    const anet::rl::EnvSpec& env_spec,
    const anet::rl::BatchEnvSpec& batch_env_spec,
    const torch::Device& device,
    const anet::ConfigData& config_data,
    std::shared_ptr<anet::rl::Notifier> notifier,
    std::optional<anet::seed_t> seed) const
{
    ImageClsAgentConfig config(config_data);
    anet::nn::NetworkConfig net_config(config_data);
    return std::make_shared<ImageClsAgent>(config, net_config, env_spec, batch_env_spec, device, seed);
}
