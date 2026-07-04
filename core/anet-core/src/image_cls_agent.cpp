// image_cls_agent.cpp

#include "anet/image_cls_agent.hpp"
#include "anet/log.hpp"
#include "anet/profile.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/nn_util.hpp"

using namespace anet::rl::img_cls;
namespace LOG = anet::log;


static void SetAdamWLearningRate(torch::optim::Optimizer& optimizer, double learning_rate)
{
    for (auto& group : optimizer.param_groups()) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
        options.lr(learning_rate);
    }
}


// ======================================================
// Actor (データ運び屋 兼 GUI可視化用推論)
// ======================================================

ImageClsActor::ImageClsActor(
    std::shared_ptr<std::shared_mutex> mutex,
    std::shared_ptr<anet::nn::Network> network,
    anet::rl::RunMode run_mode,
    torch::Device device)
    : mutex_(mutex), network_(network), run_mode_(run_mode), device_(device)
{
}

std::shared_ptr<anet::rl::BatchActionInfo> ImageClsActor::MakeAction(
    const anet::rl::StepCounts& step, const anet::rl::BatchState& state) const
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;

    // 推論準備
    std::shared_lock lock(*mutex_);
    
    // Forward
    auto obs = state.obs.To(device_);
    anet::TensorDict trace;
    anet::TraceSink sink = anet::rl::MakeActionTraceSink(trace);
    auto outputs = network_->Forward(obs, sink);

    // 推論後処理
    lock.unlock();

	// argmaxで推論結果を得る
    auto logits = outputs.At("logits");
    auto action = logits.argmax(1).to(torch::kCPU);
    //ANET_LOG_DEBUG("action=" << anet::ToString(action));

    // 可視化用に全クラスの確率分布を info に入れる
    anet::TensorDict info;
    info.Set("probs", torch::softmax(logits, 1).to(torch::kCPU));

    // BatchActionInfo(action, info, aux) の形式で返却
    auto action_info = std::make_shared<anet::rl::BatchActionInfo>(action, info);
    anet::rl::AppendTraceAux(action_info->GetAuxData(), trace);
    return action_info;
}


// ======================================================
// Learner (純粋な教師あり学習のコア)
// ======================================================

ImageClsLearner::ImageClsLearner(
    const ImageClsAgentConfig& config,
    std::shared_ptr<std::shared_mutex> mutex,
    std::shared_ptr<anet::nn::Network> network,
    std::shared_ptr<anet::ProfiledValue<double>> learning_rate,
    torch::Device device)
    : config_(config)
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
    optimizer_ = std::make_unique<torch::optim::AdamW>(network_->parameters(), opt_options);
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

    auto vector = experiences.state.obs.At(anet::rl::ObsKeys::kVector);
    auto targets = vector.to(device_).squeeze(-1).to(torch::kInt64);
    //ANET_LOG_DEBUG("targets=" << anet::ToString(targets));

    torch::Tensor logits;
    torch::Tensor loss;

    {
        // ネットワーク更新準備 (排他ロック)
        std::unique_lock lock(*mutex_);
        anet::TrainingModeGuard train_guard(*network_, true);
        optimizer_->zero_grad();

        // Forward推論
        auto obs = experiences.state.obs.To(device_);
        auto outputs = network_->Forward(obs);

        // 出力ロジットの取得
        logits = outputs.At("logits");

        // Loss計算 (交差エントロピー + ラベルスムージング)
        auto loss_opts = torch::nn::functional::CrossEntropyFuncOptions().label_smoothing(config_.label_smoothing);
        loss = torch::nn::functional::cross_entropy(logits, targets, loss_opts);

        // 誤差逆伝播
        loss.backward();

        // 勾配クリッピングを追加
        torch::nn::utils::clip_grad_norm_(network_->parameters(), config_.grad_clip_max_norm);

        // Optimizerステップ
        learning_rate_->Update(step.exp_step);
        const double current_learning_rate = learning_rate_->Value();
        SetAdamWLearningRate(*optimizer_, current_learning_rate);
        optimizer_->step();
    }

    // メトリクス (Accuracy) の計算
    // 確率が一番高いインデックスを予測クラスとして正解と比較
    auto preds = logits.argmax(/*dim=*/1);
    float accuracy = (preds == targets).to(torch::kFloat32).mean().item<float>();

    // 結果の返却
    auto result = std::make_shared<ImageClsUpdateResult>();
    result->loss = loss.item<float>();
    result->accuracy = accuracy;

    return { result };
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
    network_ = anet::nn::NetworkBuilder::BuildNetwork(network_config, env_spec.state_spec.obs_spec, nullptr, device_);
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

    // ログ記録
    anet::MetricsLogger::Instance()->Log(config);
    LOG::info() << "ImageClsAgent initialized. config=" << config_.ToString();
}

std::shared_ptr<anet::rl::Actor> ImageClsAgent::CreateActor(
    const anet::rl::BatchEnvSpec& batch_env_spec, anet::rl::RunMode run_mode, bool clone_model, std::optional<torch::Device> device) const
{
    // Actorの生成
    return std::make_shared<ImageClsActor>(mutex_, network_, run_mode, device.value_or(device_));
}

std::shared_ptr<anet::rl::Learner> ImageClsAgent::CreateLearner()
{
    // Learnerの生成
    return std::make_shared<ImageClsLearner>(config_, mutex_, network_, learning_rate_, device_);
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
