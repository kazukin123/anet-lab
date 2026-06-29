// ImageClsEnv.cpp

#include "ImageClsEnv.hpp"

#include <algorithm>
#include <cmath>

#include "anet/metrics_logger.hpp"
#include "anet/profile.hpp"
#include "anet/env.hpp"

using namespace anet::rl::env;
namespace LOG = anet::log;

// ----------------------------------------------------
// ImageClsEnv Results (AuxData保持用)
// ----------------------------------------------------

class ImageClsResetResult : public anet::rl::SingleResetResult {
public:
    ImageClsResetResult(anet::rl::SingleState state, anet::rl::AuxData aux)
        : anet::rl::SingleResetResult(std::move(state)), aux_(std::move(aux)) {}
    anet::rl::AuxData GetAuxData() const override { return aux_; }
private:
    anet::rl::AuxData aux_;
};

class ImageClsStepResult : public anet::rl::SingleStepResult {
public:
    ImageClsStepResult(float reward, anet::rl::SingleState next_state, anet::rl::AuxData aux)
        : anet::rl::SingleStepResult(reward, std::move(next_state)), aux_(std::move(aux)) {}
    anet::rl::AuxData GetAuxData() const override { return aux_; }
private:
    anet::rl::AuxData aux_;
};

// ----------------------------------------------------
// ImageClsEnv
// ----------------------------------------------------

ImageClsEnv::ImageClsEnv(const ImageClsEnvConfig& config, std::optional<anet::seed_t> seed)
    : anet::RandomHolder(seed), config_(config)
{
    // Train用データソースの生成
    LOG::verbose() << "Loading train:" << config_.train_list_txt_path;
    train_data_source_ = std::make_unique<anet::img::ImageDataSource>(
        config_.root_dir, config_.train_list_txt_path, config_.classes_txt_path, config_.image_width, config.image_height, config_.suffix);

    // Eval用データソースの生成
    LOG::verbose() << "Loading test:" << config_.eval_list_txt_path;
    eval_data_source_ = std::make_unique<anet::img::ImageDataSource>(
        config_.root_dir, config_.eval_list_txt_path, config_.classes_txt_path, config_.image_width, config.image_height, config_.suffix);

	// 分類クラスを取得
    auto claeses = train_data_source_->GetClassLabelList();

	// 画像情報のObservation定義
    anet::TensorSpec obs_grid_spec {
        .type = anet::SpaceType::Grid,
        .shape = { 3, config_.image_height, config_.image_width },
        .dtype = torch::kUInt8
    };

    // 正しい分類クラスIDを含むObservation (離散、スカラー)
    anet::TensorSpec obs_vector_spec {
        .type = anet::SpaceType::Vector,
        .shape = { 1 }, // Scaler
        .dtype = torch::kInt64,
		.num_classes = static_cast<int64_t>(claeses.size()),  // 分類クラス数
    };

    // StateSpec
	spec_.state_spec.obs_spec[anet::rl::ObsKeys::kGrid] = obs_grid_spec;
    spec_.state_spec.obs_spec[anet::rl::ObsKeys::kVector] = obs_vector_spec;

	// ActionSpec
    spec_.action_spec.is_discrete = true;
    spec_.action_spec.value_labels = claeses;

    LOG::verbose() << "ImageClsEnv initialized.";
}

anet::rl::EnvSpec ImageClsEnv::GetSpec() const
{
    return spec_;
}

anet::rl::SingleState ImageClsEnv::FetchRandomImageState(anet::rl::RunMode mode)
{
    // モードに応じてデータソースを切り替え
    const bool is_eval = anet::rl::IsEval(mode);
    auto* source = is_eval ? eval_data_source_.get() : train_data_source_.get();

    // ランダムサンプリング
    size_t data_size = source->size().value();
    size_t rand_idx = rnd_->RandUint64() % data_size;
    auto example = source->get(rand_idx);
    current_true_label_ = example.target.item<int64_t>();

    auto image = example.data;
    if (!is_eval && config_.augment.enabled) {
        image = ApplyTrainAugment(image);
    }

    // Observationを生成
    anet::TensorDict obs;
    obs.Set(anet::rl::ObsKeys::kGrid, image);
    obs.Set(anet::rl::ObsKeys::kVector, example.target.clone());

    anet::rl::SingleState state {
        .obs = std::move(obs),
        .done = done_,
        .truncated = truncated_,
        .episode_start = episode_start_,
    };
    //ANET_LOG_DEBUG("state=" << state.ToString());

    return state;
}

torch::Tensor ImageClsEnv::ApplyTrainAugment(const torch::Tensor& image)
{
    ANET_PROFILE_SCOPE(augment);

    auto result = ApplyRandomResizedCrop(image);
    if (config_.augment.hflip_p > 0.0 && rnd_->Uniform01() < config_.augment.hflip_p) {
        result = result.flip({ 2 });
    }
    return result.contiguous();
}

torch::Tensor ImageClsEnv::ApplyRandomResizedCrop(const torch::Tensor& image)
{
    const int64_t in_h = image.size(1);
    const int64_t in_w = image.size(2);
    const int64_t out_h = config_.image_height;
    const int64_t out_w = config_.image_width;
    const double area = static_cast<double>(in_h * in_w);
    const double log_ratio_min = std::log(config_.augment.rrc_ratio_min);
    const double log_ratio_max = std::log(config_.augment.rrc_ratio_max);

    int64_t crop_h = in_h;
    int64_t crop_w = in_w;
    int64_t top = 0;
    int64_t left = 0;
    bool sampled = false;

    // torchvision の RandomResizedCrop と同じく、指定範囲から妥当な crop 矩形を複数回試す。
    for (int attempt = 0; attempt < 10; ++attempt) {
        const double target_area = area * rnd_->Uniform(
            static_cast<float>(config_.augment.rrc_scale_min),
            static_cast<float>(config_.augment.rrc_scale_max));
        const double aspect_ratio = std::exp(rnd_->Uniform(
            static_cast<float>(log_ratio_min),
            static_cast<float>(log_ratio_max)));
        const int64_t candidate_w = static_cast<int64_t>(std::round(std::sqrt(target_area * aspect_ratio)));
        const int64_t candidate_h = static_cast<int64_t>(std::round(std::sqrt(target_area / aspect_ratio)));

        if (candidate_w > 0 && candidate_w <= in_w && candidate_h > 0 && candidate_h <= in_h) {
            crop_w = candidate_w;
            crop_h = candidate_h;
            top = rnd_->RandInt(0, static_cast<int>(in_h - crop_h));
            left = rnd_->RandInt(0, static_cast<int>(in_w - crop_w));
            sampled = true;
            break;
        }
    }

    if (!sampled) {
        const double in_ratio = static_cast<double>(in_w) / static_cast<double>(in_h);
        if (in_ratio < config_.augment.rrc_ratio_min) {
            crop_w = in_w;
            crop_h = static_cast<int64_t>(std::round(static_cast<double>(crop_w) / config_.augment.rrc_ratio_min));
        } else if (in_ratio > config_.augment.rrc_ratio_max) {
            crop_h = in_h;
            crop_w = static_cast<int64_t>(std::round(static_cast<double>(crop_h) * config_.augment.rrc_ratio_max));
        }
        crop_w = std::min(crop_w, in_w);
        crop_h = std::min(crop_h, in_h);
        top = (in_h - crop_h) / 2;
        left = (in_w - crop_w) / 2;
    }

    auto cropped = image.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(top, top + crop_h),
        torch::indexing::Slice(left, left + crop_w)
    });

    if (crop_h == out_h && crop_w == out_w) {
        return cropped.contiguous();
    }

    auto resized = torch::nn::functional::interpolate(
        cropped.unsqueeze(0).to(torch::kFloat32),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{ out_h, out_w })
            .mode(torch::kBilinear)
            .align_corners(false));
    return resized.squeeze(0).round().clamp(0, 255).to(torch::kUInt8).contiguous();
}

anet::rl::AuxData ImageClsEnv::MakeAuxData() const
{
    anet::rl::AuxData aux;
    aux["step_count"] = torch::tensor(step_count_, torch::kInt32);
    aux["current_true_label"] = torch::tensor(current_true_label_, torch::kInt64);
    return aux;
}

std::shared_ptr<const anet::rl::SingleResetResult> ImageClsEnv::Reset(anet::rl::RunMode mode)
{
    //episode_just_ended_ = false;
    step_count_ = 0;
    ep_reward_sum_ = 0.0f;
    done_ = false;
    truncated_ = false;
    episode_start_ = true;

    // モードを渡して画像を取得
    auto state = FetchRandomImageState(mode);
    state.done = false;
    state.truncated = false;

    // 戻り値は const SingleResetResult の shared_ptr にする
    return std::make_shared<const ImageClsResetResult>(std::move(state), MakeAuxData());
}

std::shared_ptr<const anet::rl::SingleStepResult> ImageClsEnv::Step(int64_t action, anet::rl::RunMode mode)
{
    // 初期化＆エピソードステップ更新
    episode_start_ = false;
    step_count_++;

	// 報酬計算 (正解なら1.0、そうでなければ0.0)
    float reward = (action == current_true_label_) ? 1.0f : 0.0f;
    ep_reward_sum_ += reward;

	// エピソード終了判定
    done_ = (step_count_ >= config_.max_steps);

	// 次の状態を生成
    auto next_state = FetchRandomImageState(mode);
	next_state.done = done_;
	next_state.truncated = truncated_;

    // メトリクス情報更新
    if (done_) {
        episode_just_ended_ = true;
        last_episode_len_ = static_cast<float>(step_count_);
        last_reward_sum_ = ep_reward_sum_;
        last_accuracy_ = ep_reward_sum_ / static_cast<float>(step_count_);
    } else {
        //episode_just_ended_ = false;
    }

    auto result = std::make_shared<ImageClsStepResult>(reward, std::move(next_state), MakeAuxData());
    return result;
}

std::optional<float> ImageClsEnv::GetScalar(const std::string& key, int64_t index) const
{
    const float nan = std::numeric_limits<float>::quiet_NaN();

    if (key == "episode_len") {
        if (!episode_just_ended_) return nan;
        return last_episode_len_;
    }
    if (key == "reward_sum") {
        if (!episode_just_ended_) return nan;
        return last_reward_sum_;
    }
    if (key == "accuracy") {
        if (!episode_just_ended_) return nan;
        return last_accuracy_;
    }
    return std::nullopt;
}


// ----------------------------------------------------
// ImageClsEnvFactory
// ----------------------------------------------------

std::shared_ptr<anet::rl::SingleDiscreteEnv> ImageClsEnvFactory::CreateSingleEnv(
        const anet::ConfigData& config_data, const torch::Device& device, std::optional<anet::seed_t> seed,const std::string& config_prefix)
{
    ImageClsEnvConfig config(config_data, config_prefix);
    return std::make_shared<ImageClsEnv>(config, seed); // CPU固定
}
