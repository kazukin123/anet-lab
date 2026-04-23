// ImageClsEnv.cpp

#include "ImageClsEnv.hpp"
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
    auto* source = anet::rl::IsEval(mode) ? eval_data_source_.get() : train_data_source_.get();

    // ランダムサンプリング
    size_t data_size = source->size().value();
    size_t rand_idx = rnd_->RandUint64() % data_size;
    auto example = source->get(rand_idx);
    current_true_label_ = example.target.item<int64_t>();

	//torch::TensorOptions opt = torch::TensorOptions().dtype(torch::kUInt8);
 //   auto ones = torch::ones({ 3, 16, 16 }, opt);

    // Observationを生成
    anet::TensorDict obs;
    //obs.Set(anet::rl::ObsKeys::kGrid, ones);
    obs.Set(anet::rl::ObsKeys::kGrid, example.data);
    obs.Set(anet::rl::ObsKeys::kVector, example.target.clone());

    anet::rl::SingleState state {
        .obs = std::move(obs)
    };
    //ANET_LOG_DEBUG("state=" << state.ToString());

    return state;
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
