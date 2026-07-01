// observers.cpp

#include "anet/observers.hpp"
#include <limits>
#include <wx/log.h>
#include "anet/profile.hpp"
#include "anet/str_util.hpp"
#include "anet/tensor_util.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/log.hpp"
#include "anet/env.hpp"
#include "anet/trainer.hpp"


using namespace anet::rl;
namespace LOG = anet::log;


static float ResolveMin(bool override_flag, float override_v, std::optional<float> probe_v, float fallback)
{
    if (override_flag) return override_v;
    if (probe_v.has_value()) return *probe_v;
    return fallback;
}

static float ResolveMax(bool override_flag, float override_v, std::optional<float> probe_v, float fallback)
{
    if (override_flag) return override_v;
    if (probe_v.has_value()) return *probe_v;
    return fallback;
}

HeatMapVectorObserver::HeatMapVectorObserver(
    const std::string& tag,
    const HeatMapObserverConfig& config,
    std::shared_ptr<VectorProbe> x_probe,
    std::shared_ptr<VectorProbe> y_probe,
    std::shared_ptr<VectorProbe>  value_probe)
    : TaggedTrainObserver(tag)
    , config_(config)
    , x_probe_(x_probe), y_probe_(y_probe), value_probe_(value_probe)
{

    // Probe の min/max と config の override から HeatMap 範囲決定
    float xmin = ResolveMin(config_.override_xmin, config_.xmin, x_probe_->GetMin(), 0.0f);
    float xmax = ResolveMax(config_.override_xmax, config_.xmax, x_probe_->GetMax(), 1.0f);
    float ymin = ResolveMin(config_.override_ymin, config_.ymin, y_probe_->GetMin(), 0.0f);
    float ymax = ResolveMax(config_.override_ymax, config_.ymax, y_probe_->GetMax(), 1.0f);

    //HeatMap(int width, int height, float x_min = 0.0f, float x_max = 1.0f,
    //    float y_min = 0.0f, float y_max = 1.0f, size_t max_points = 0,
    //    uint32_t flags = HM_Default);

    heatmap_ = std::make_unique<anet::HeatMap>(
        config_.width,
        config_.height,
        xmin, xmax, ymin, ymax,
        config_.max_points,
        config_.flags
    );
}

void HeatMapVectorObserver::OnTrain(const TrainEvent& event)
{
    ANET_PROFILE_FUNC();

    // 実行判定
    auto step = event.counts.GetByAxis(anet::rl::StepAxis::TRAIN);
    if (config_.log_interval <= 0) return;
    if (step % config_.log_interval != 0) return;

    ANET_PROFILE_SCOPE(get_vector);
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // 生成： xv, yv, vv
    auto xv = x_probe_->GetVector(event);
    auto yv = y_probe_->GetVector(event);
    auto vv = value_probe_->GetVector(event);

    // 揃ってなかったらスキップ
    if (!xv.has_value() || !yv.has_value() || !vv.has_value())
        return;

    // サイズチェック
    if (xv->size() != yv->size() || xv->size() != vv->size()) {
        LOG::warn() << "HeatMapVectorObserver: size mismatch. x=" << x_probe_->GetName()
            << " y=" << y_probe_->GetName() << " v=" << value_probe_->GetName()
            << "x.size=" << xv->size() << " y.size=" << yv->size() << " v.size=" << vv->size();
        return;
    }

    // データ追加
    ANET_PROFILE_SCOPE_NEXT(add_data_batch);
    heatmap_->Reset();  // 毎回バッファクリア（最新のRB内容だけ描画）
    heatmap_->AddDataBatch(*xv, *yv, *vv);

    captured_step_ = step;

    // 画像保存
    ANET_PROFILE_SCOPE_NEXT(log_image);
    MetricsLogger::Instance()->Log(
        tag_,
        step,
        *heatmap_,
        config_.image_width,
        config_.image_height
    );
}

anet::rl::ImageData HeatMapVectorObserver::GetImageData(int width, int height)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto image = heatmap_->Render(width, height);
    return { image, captured_step_ };
}

// -------------------------------------------------------------

TimeHistogramObserver::TimeHistogramObserver(
    const std::string& tag, const TimeHistogramObserverConfig& config,
    std::shared_ptr<VectorProbe> probe)
	: TaggedLearnObserver(tag)
    , config_(config), probe_(probe)
{
    histogram_ = std::make_unique<anet::TimeHistogram>(
        config_.bins, config_.max_frames, config_.mode, config_.flags, config_.base_min, config_.base_max, config_.alpha);
}

void TimeHistogramObserver::OnLearn(const LearnEvent& event)
{
    ANET_PROFILE_FUNC();

    /// @todo メトリクスのSTEP軸を指定できるようにする
    auto step = event.counts.GetByAxis(anet::rl::StepAxis::LEARN);

    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Probeで vectorを取得
    ANET_PROFILE_SCOPE(get_vector);
    auto values = probe_->GetVector(event);
    if (values.has_value()) {
        histogram_->AddBatch(*values);
    }

    // フレーム更新判定
    bool is_frame_updated = false;
    if (config_.frame_interval > 0 && step % config_.frame_interval == 0) {
        histogram_->NextFrame();
        is_frame_updated = true;
    }

    //  ログ出力（フレーム更新があった時だけチェックする）
    if (is_frame_updated && config_.log_interval > 0) {
        // A. ログ頻度がフレームより高い (log < frame) → 毎回出す（間引きようがないため）
        // B. ログ頻度が低い (log >= frame) → ステップがログ間隔と合う時だけ出す
        if (config_.log_interval <= config_.frame_interval || step % config_.log_interval == 0) {
            ANET_PROFILE_SCOPE_NEXT(log_image);
            captured_step_ = step;
            MetricsLogger::Instance()->Log(tag_, step, *histogram_, config_.image_width, config_.image_height);
        }
    }
}

anet::rl::ImageData TimeHistogramObserver::GetImageData(int width, int height)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto image = histogram_->Render(width, height);
    return { image, captured_step_ };
}

// -------------------------------------------------------------

MultiPairHeatMapObserver::MultiPairHeatMapObserver(
    const std::string& tag,
    const HeatMapObserverConfig& config,
    const std::vector<std::shared_ptr<VectorProbe>>& axis_probes,
    std::shared_ptr<VectorProbe> value_probe)
    : TaggedTrainObserver(tag)
    , config_(config), axis_probes_(axis_probes), value_probe_(value_probe)
{
    ANET_ASSERT(axis_probes_.size() >= 2);

    const size_t m = axis_probes_.size();
    for (size_t i = 0; i < m; i++) {
        LOG::info() << "MultiPairHeatMapObserver: "
            << tag_ << " axis_probes[" << i << "] "
            << axis_probes_[i]->GetName()
            << "(" << axis_probes_[i]->GetMin().value_or(0.0f)
                   << axis_probes_[i]->GetMax().value_or(0.0f) << ")";
    }
    int plot_cnt = 0;
    for (int i = 0; i < m; i++) {
        auto x_name = axis_probes_[i]->GetName();
        for (int j = i + 1; j < m; j++) {
            auto y_name = axis_probes_[j]->GetName();
            LOG::info() << "MultiPairHeatMapObserver: "
                << tag_
                << " axis_pattern[" << plot_cnt << "]"
                << " x=[" << i << "]" << x_name
                << " y=[" << j << "]" << y_name;
            plot_cnt++;
        }
    }

    // HeatMap 生成
    heatmap_ = std::make_unique<anet::HeatMap>(
        config_.width, config_.height,
        0.0f, 1.0f,
        0.0f, 1.0f,
        config_.max_points * plot_cnt,
        config_.flags);

}

inline float Normalize01(
    float v,
    const std::optional<float>& min_opt,
    const std::optional<float>& max_opt,
    float fallback_min = -1.0f,
    float fallback_max = 1.0f)
{
    float mn = min_opt.value_or(fallback_min);
    float mx = max_opt.value_or(fallback_max);

    // min==max の場合は 0 に押しつぶす（ゼロ除算回避）
    if (mx - mn < 1e-9f) return 0.0f;

    float t = (v - mn) / (mx - mn);

    // 0〜1 に clamp
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return t;
}

void MultiPairHeatMapObserver::OnTrain(const TrainEvent& event)
{
    ANET_PROFILE_FUNC();

    auto step = event.counts.GetByAxis(anet::rl::StepAxis::TRAIN);

    // --- 値ベクトル取得 ---
    ANET_PROFILE_SCOPE(get_vector);
    
    auto vv = value_probe_->GetVector(event);
    if (!vv || vv->empty()) return;

    const size_t m = axis_probes_.size();
    if (m < 2) return;

    // --- 準備: バッファの再利用 ---
    static thread_local std::vector<float> batch_x;
    static thread_local std::vector<float> batch_y;
    static thread_local std::vector<float> batch_v;

    // スレッドローカル変数の容量確保（前回のサイズを維持）
    size_t estimated_size = vv->size();
    if (batch_x.capacity() < estimated_size) batch_x.reserve(estimated_size);
    if (batch_y.capacity() < estimated_size) batch_y.reserve(estimated_size);
    if (batch_v.capacity() < estimated_size) batch_v.reserve(estimated_size);

    ANET_PROFILE_SCOPE_NEXT(process_pairs);

    // --- 全プローブペア i < j をスキャン ---
    for (size_t i = 0; i < m; i++) {
        auto xv = axis_probes_[i]->GetVector(event);
        if (!xv || xv->empty()) continue;

        auto xmin_opt = axis_probes_[i]->GetMin();
        auto xmax_opt = axis_probes_[i]->GetMax();
        if (!xmin_opt || !xmax_opt) continue; // 範囲未定ならスキップ

        float xmin = *xmin_opt;
        float xmax = *xmax_opt;
        float x_range = xmax - xmin;
        float x_scale = (std::fabs(x_range) > 1e-6f) ? (1.0f / x_range) : 0.0f;

        for (size_t j = i + 1; j < m; j++) {
            auto yv = axis_probes_[j]->GetVector(event);
            if (!yv || yv->empty()) continue;

            auto ymin_opt = axis_probes_[j]->GetMin();
            auto ymax_opt = axis_probes_[j]->GetMax();
            if (!ymin_opt || !ymax_opt) continue;   // 範囲未定ならスキップ

            float ymin = *ymin_opt;
            float ymax = *ymax_opt;
            float y_range = ymax - ymin;
            float y_scale = (std::fabs(y_range) > 1e-6f) ? (1.0f / y_range) : 0.0f;

            size_t n = std::min({ xv->size(), yv->size(), vv->size() });
            if (n == 0) continue;

            // バッファのリセット
            batch_x.clear();
            batch_y.clear();
            batch_v.clear();

            // --- データ変換ループ ---
            const float* px = xv->data();
            const float* py = yv->data();
            const float* pv = vv->data();

            for (size_t k = 0; k < n; k++) {
                // 事前計算した係数で乗算 (除算回避)
                float x_norm = (px[k] - xmin) * x_scale;
                float y_norm = (py[k] - ymin) * y_scale;

                batch_x.push_back(x_norm);
                batch_y.push_back(y_norm);
                batch_v.push_back(pv[k]);
            }

            // 一括追加
            heatmap_->AddDataBatch(batch_x, batch_y, batch_v);
        }
    }

    // --- 画像保存 ---
    ANET_PROFILE_SCOPE_NEXT(log_image);
    if (config_.log_interval > 0 && step % config_.log_interval == 0) {
        MetricsLogger::Instance()->Log(
            tag_, step,
            *heatmap_,
            config_.image_width,
            config_.image_height);
    }
}

// -------------------------------------------------------------

SweepedHeatMapObserver::SweepedHeatMapObserver(
    const std::string& tag,
    const SweepedHeatMapObserverConfig& config,
    std::shared_ptr<ISweepInputGenerator> input_gen,
    TensorDictFunction tensor_fn,
    std::shared_ptr<ISweepOutputExtractor> output_ext,
    const std::unordered_map<std::string, std::string>& scalar_tag_label_map
    )
    : TaggedLearnObserver(tag), config_(config),
    input_gen_(input_gen), tensor_fn_(std::move(tensor_fn)), output_ext_(output_ext)
{
    scalar_label_tag_map_ = MakeReverseMap(scalar_tag_label_map);

    // Observer → InputGenerator に GridSize 希望を渡す
    input_gen_->ApplyGridSize(config_.grid_width, config_.grid_height);

    // InputGenerator が決定した GridSize を取得
    auto [in_gw, in_gh] = input_gen_->GetGridSize();

    // OutputExtractor（従属）に確定値を伝える
    output_ext_->ApplyGridSize(in_gw, in_gh);

    // 整合性チェック
    auto [out_gw, out_gh] = output_ext_->GetGridSize();
    ANET_ASSERT_MSG(in_gw == out_gw && in_gh == out_gh,
        "InputGenerator / OutputExtractor GridSize mismatch."
    );
    ANET_ASSERT(in_gw > 0);
    ANET_ASSERT(in_gh > 0);

    // 保存
    grid_w_ = in_gw;
    grid_h_ = in_gh;

    // HeatMap 構築
    heatmap_ = std::make_unique<anet::HeatMap>(
        grid_w_,
        grid_h_,
        0.0f, grid_w_,  // x-min, x-max → 0.0〜grid_wスケール
        0.0f, grid_h_,  // y-min, y-max → 0.0〜grid_hスケール
        0,              // max_points（内部deque制限なし）
        config_.flags
    );
}

void SweepedHeatMapObserver::OnLearn(const LearnEvent& event)
{
    ANET_PROFILE_FUNC();

    /// @todo メトリクスのSTEP軸を指定できるようにする？

    // 実行判定
    auto step = event.counts.learn_step;
    if (config_.log_interval <= 0) return;
    if (step % config_.log_interval != 0) return;

    ANET_PROFILE_SCOPE(render);
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // データ採取
    auto result = Render();
    captured_step_ = step;
    const auto& extract_result = result.first;
    const auto& scalars_cpu = result.second;

    // 画像ログ出力
    MetricsLogger::Instance()->Log(
        tag_,
        step,
        *heatmap_,
        config_.image_width,
        config_.image_height);
    ANET_LOG_DEBUG("LogImage() done. tag=" << tag_);

    // Scalarログ出力
    ANET_PROFILE_SCOPE_NEXT(log_scalar);
    for (int i = 0; i < extract_result.labels.size(); i++) {
        auto result_label = extract_result.labels[i];
        auto tag_itr = scalar_label_tag_map_.find(result_label);
        if (tag_itr != scalar_label_tag_map_.end()) {
            auto scalar_tag = tag_itr->second;
            auto scalar_value = scalars_cpu[i].item<float>();
            MetricsLogger::Instance()->LogScalar(scalar_tag, step, scalar_value);
        }
    }
}

anet::rl::ImageData SweepedHeatMapObserver::GetImageData(int width, int height)
{
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto image = heatmap_->Render(width, height);
    return { image, captured_step_ };
}

std::pair<ExtractResult, std::vector<torch::Tensor>> SweepedHeatMapObserver::Render()
{
    const int64_t grid_num = static_cast<int64_t>(grid_w_) * static_cast<int64_t>(grid_h_);

    // 入力バッチ生成（GPU 上）
    ANET_PROFILE_SCOPE(build);
    anet::TensorDict batch_in = input_gen_->BuildInputTensor();
    ANET_LOG_DEBUG("batch_in=" << batch_in.ToDefString());

    // NN 適用（GPU 上）
    ANET_PROFILE_SCOPE_NEXT(nn);
    anet::TensorDict batch_out_dict = tensor_fn_(batch_in);
    const auto batch_out_opt = batch_out_dict.Get(config_.output_key);
    if (!batch_out_opt.has_value()) {
        ANET_SYSTEM_ERROR(
            "SweepedHeatMapObserver: output_key not found. tag=" << tag_
            << " output_key=" << config_.output_key);
    }
    torch::Tensor batch_out = *batch_out_opt;
    ANET_ASSERT_SHAPE(batch_out, { grid_num, ANET_SHAPE_ENDANY });
    ANET_LOG_DEBUG("batch_out[" << config_.output_key << "]=" << anet::ToDefString(batch_out));

    // リクエストするLabelをscalar_tag_label_map_からsetに詰める
    std::unordered_set<std::string> req_label_set(scalar_label_tag_map_.size());
    //wxLogDebug("tag=%s req_label_set.size=%lld", heatmap_tag_.c_str(), req_label_set.size());
    for (const auto& kv : scalar_label_tag_map_) {
        req_label_set.insert(kv.first);
        //wxLogDebug("tag=%s label=%s", heatmap_tag_, kv.second);
    }

    // 出力から値抽出（GPU 上, [W*H]）
    ANET_PROFILE_SCOPE_NEXT(extract);
    ExtractResult extract_result = output_ext_->Extract(batch_out, req_label_set);
    ANET_LOG_DEBUG("grid_values=" << anet::ToDefString(extract_result.grid) << " tag=" << tag_);
    ANET_ASSERT_SHAPE(extract_result.grid, { grid_num });
    ANET_ASSERT_DTYPE(extract_result.grid, torch::kFloat32);
    ANET_ASSERT(extract_result.labels.size() == extract_result.scalars.size());

    // CPU へ一括転送
    ANET_PROFILE_SCOPE_NEXT(transfer);
    torch::Tensor grid_cpu = extract_result.grid.to(torch::kCPU);
    ANET_ASSERT_SHAPE(grid_cpu, { grid_num });
    ANET_ASSERT_DTYPE(grid_cpu, torch::kFloat32);
    float* data = grid_cpu.data_ptr<float>();
    std::vector<torch::Tensor> scalars_cpu;
    for (auto& t : extract_result.scalars) scalars_cpu.push_back(t.to(torch::kCPU));
    ANET_LOG_DEBUG("Extract done.");

    // HeatMapデータ設定
    ANET_PROFILE_SCOPE_NEXT(log_image);
    heatmap_->SetGridValues(data, grid_w_, grid_h_);
    ANET_LOG_DEBUG("SetGridValues() done.");

    return { extract_result, scalars_cpu };
}

// -------------------------------------------------------------

EpisodeEvalObserver::EpisodeEvalObserver(
    std::shared_ptr<EvalRunner> eval_runner,
    int eval_interval,
    bool use_background)
    : eval_runner_(std::move(eval_runner))
    , eval_interval_(eval_interval)
    , use_background_(use_background)
{
    ANET_CHECK(eval_runner_ != nullptr);

    // バックグラウンド有効時のみスレッドプール生成
    if (use_background_) {
        eval_pool_ = std::make_unique<anet::PinnedThreadPool>(1, "EpisodeEvalObserver");
    }
}

EpisodeEvalObserver::~EpisodeEvalObserver()
{
    // スレッドが動いていたら待つ
    if (eval_future_.valid()) {
        eval_future_.wait();
    }

    // スレッド終わり
    if (eval_pool_) {
        eval_pool_->Stop();
    }
}

void EpisodeEvalObserver::RunEvaluationEpisode(const StepCounts& event_counts)
{
    eval_runner_->Sync();
    do {
        eval_runner_->DoStep(event_counts);
    } while (!eval_runner_->LastStepHadEpisodeEnd());
}

void EpisodeEvalObserver::OnLearn(const LearnEvent& event)
{
    ANET_PROFILE_FUNC();

    auto step = event.counts.GetByAxis(anet::rl::StepAxis::LEARN);

    // 評価エピソードを終端まで回す
    if (eval_interval_ > 0 && step % eval_interval_ == 0) {
        if (use_background_) {
            // 前回の評価がまだ終わっていなければ、ここで完了までブロックして待つ
            if (eval_future_.valid()) {
                eval_future_.wait();
            }

            // スレッドに評価エピソード実行処理を投げる
            const StepCounts event_counts = event.counts;
            eval_future_ = eval_pool_->EnqueueFuture(0, [this, event_counts]() {
                try {
                    this->RunEvaluationEpisode(event_counts);
                } catch (const std::exception& e) {
                    LOG::error() << "EpisodeEvalObserver Error: " << e.what();
                }
                });
        } else {
            // フォアグラウンド実行
            RunEvaluationEpisode(event.counts);
        }
    }
}

std::string EpisodeEvalObserver::ToString() const
{
    return "EpisodeEvalObserver[" + eval_runner_->GetName() + "]";
}

// -------------------------------------------------------------

FunctionTrainObserver::FunctionTrainObserver(Fn fn, std::optional<std::string> name)
    : fn_(std::move(fn))
    , name_(name.has_value() ? std::string("FunctionTrainObserver[") + *name + "]" : "FunctionTrainObserver")
{
}

FunctionLearnObserver::FunctionLearnObserver(Fn fn, std::optional<std::string> name)
    : fn_(std::move(fn))
    , name_(name.has_value() ? std::string("FunctionLearnObserver[") + *name + "]" : "FunctionLearnObserver")
{
}


// ===========================================================================
// Conv2dVisualizationObserver
// ===========================================================================

Conv2dVisualizationObserver::Conv2dVisualizationObserver(
    const std::string& tag, int episode_interval, const Conv2dVisualizerConfig& vis_config)
    : TaggedTrainObserver(tag)
    , episode_interval_(episode_interval)
    , visualizer_(vis_config)
    , is_recording_(false)
{
    LOG::info() << "Conv2dVisualizationObserver() tag=" << tag << " channels_per_row=" << vis_config.channels_per_row;
}

void Conv2dVisualizationObserver::OnTrain(const TrainEvent& event)
{
    ANET_PROFILE_FUNC();

    auto step = event.counts.GetByAxis(anet::rl::StepAxis::TRAIN);
    const auto& state = event.experience.state;
    const auto& next_state = event.experience.next_state;

    // ---  録画の開始判定 ＆ フェイルセーフ ---

    if (state.IsEpisodeStart()) {
        anet::log::info() << "Conv2dVisualizationObserver: Episode start detected. episode_count=" << local_episode_count_;

        // 事故対策: もし録画中なのに新しいエピソードが始まってしまったら強制クローズ
        if (is_recording_) {
            anet::log::warn() << "Conv2dVisualizationObserver: force-ended due to unexpected episode_start. Previous episode may not have finished cleanly.";
            is_recording_ = false;
        }

        // 今回のエピソードが録画対象かチェック
        if (episode_interval_ > 0 && (local_episode_count_ % episode_interval_ == 0)) {
            is_recording_ = true;
            anet::log::info() << "Conv2dVisualizationObserver: Visualization started. train_step=" << event.counts.train_step << " learn_step=" << event.counts.learn_step << " tag=" << tag_;
        }

        // エピソード数インクリメント
        local_episode_count_++;
    }

    // --- 録画中でなければ即リターン ---
    if (!is_recording_) return;

    // --- 画像化と保存 ---
    if (event.action_info) {
        auto dict = anet::rl::ExtractNnTrace(event.action_info->GetAuxData());

        if (!dict.empty()) {
            auto vis_result = visualizer_.Visualize(step, dict);
            wxImage image = vis_result.first;

            if (image.IsOk()) {
                MetricsLogger::Instance()->Log(tag_, step, image);

                std::lock_guard<std::mutex> lock(image_mutex_);
                last_image_.image = image;
                last_image_.step = step;
            }

            if (is_first_record_) {
                auto json = vis_result.second;
                MetricsLogger::Instance()->Log(tag_, json);
                is_first_record_ = false;
            }
        }
    }

    // --- 録画の終了判定 ---
    if (next_state.IsDone() || next_state.IsTruncated()) {
        if (is_recording_) {
            anet::log::info() << "Conv2dVisualizationObserver: Visualization ended. train_step=" << event.counts.train_step << " learn_step=" << event.counts.learn_step << " tag=" << tag_;
        }
        is_recording_ = false;
    }
}

ImageData Conv2dVisualizationObserver::GetImageData(int width, int height)
{
    std::lock_guard<std::mutex> lock(image_mutex_);

    // まだ画像が生成されていない場合は空を返す
    if (!last_image_.image.IsOk()) {
        return last_image_;
    }

    wxImage img = last_image_.image;

    // 要求されたサイズが現在のサイズと異なる場合はリサイズ
    if (width > 0 && height > 0 && (width != img.GetWidth() || height != img.GetHeight())) {
        img = img.Scale(width, height, wxIMAGE_QUALITY_NEAREST);
    }

    return { img, last_image_.step };
}

// ===========================================================================
// MetricsLogObserverBase
// ===========================================================================

///  @todo MetricsLogObserverBaseをProbe化

MetricsLogObserverBase::MetricsLogObserverBase(
    const std::string& tag,
    const std::string& key, anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
    int interval, bool is_ema, float ema_alpha, std::optional<float> clip)
    : TaggedObserver(tag), key_(key), step_axis_(step_axis), event_field_(event_field)
    , interval_(interval), is_ema_(is_ema), val_ema_(ema_alpha), clip_(clip)
{
    ;
}

MetricsLogTrainObserver::MetricsLogTrainObserver(const std::string& tag, const std::string& key,
    anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
    int interval, bool is_ema, float ema_alpha, std::optional<float> clip)
    : MetricsLogObserverBase(tag, key, step_axis, event_field, interval, is_ema, ema_alpha, clip)
{
    ;
}

MetricsLogLearnObserver::MetricsLogLearnObserver(const std::string& tag, const std::string& key,
    anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
    int interval, bool is_ema, float ema_alpha, std::optional<float> clip)
    : MetricsLogObserverBase(tag, key, step_axis, event_field, interval, is_ema, ema_alpha, clip)
{
    ;
}

MetricsLogEpisodeEndObserver::MetricsLogEpisodeEndObserver(const std::string& tag, const std::string& key,
    anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
    int interval, bool is_ema, float ema_alpha, std::optional<float> clip)
    : MetricsLogObserverBase(tag, key, step_axis, event_field, interval, is_ema, ema_alpha, clip)
{
    ;
}

/// BATCH_UPDATE_RESULT以外用のメトリクス情報取得処理
MetricsLogObserverBase::MetricsData MetricsLogObserverBase::GetMetricsData(
    const StepCounts& counts,
    std::shared_ptr<const Agent> agent,
    std::shared_ptr<const Runner> runner,
    std::shared_ptr<const BatchEnv> env,
    const BatchExperience* experience,
    std::shared_ptr<const BatchActionInfo> action_info,
    EventField event_field)
{
    ANET_CHECK(event_field != EventField::UPDATE_RESULT);

    // Scalar取得対象
    const anet::Module* target = nullptr;

    // 指定に従ってScalar取得対象を取得
    switch (event_field) {
    case anet::rl::EventField::EXPERIENCE:
        target = experience;
        break;
    case anet::rl::EventField::AGENT:
        target = agent.get();
        break;
    case anet::rl::EventField::RUNNER:
        target = runner.get();
        break;
    case anet::rl::EventField::ENV:
        if (env != nullptr) {
            target = env.get();
        } else if (runner != nullptr) {
            target = runner->GetBatchEnv().get(); // Runner経由で取得
        }
        break;
    case anet::rl::EventField::ACTION_INFO:
        if (action_info == nullptr) {
            ANET_SYSTEM_ERROR("MetricsLogObserverBase: $action_info is only available for TrainEvent.");
        }
        target = dynamic_cast<const anet::Module*>(action_info.get());
        if (target == nullptr) {
            auto step = counts.GetByAxis(this->step_axis_);
            return MetricsData{ step, std::numeric_limits<float>::quiet_NaN() };
        }
        break;
    default:
        ANET_SYSTEM_ERROR("Unknown event field: " << static_cast<int>(event_field));
        break;
    }
    ANET_CHECK(target != nullptr);

    // step取得
    auto step = counts.GetByAxis(this->step_axis_);

    // 値取得
    std::optional<float> value;
    value = target->GetScalar(this->key_);

    // 結果生成
    MetricsData ret{ step, value };
    return ret;
}

/// BATCH_UPDATE_RESULT専用のメトリクス情報取得処理
MetricsLogObserverBase::MetricsDataList MetricsLogObserverBase::GetMetricsDataListFromUpdateResultList(
    const StepCounts& counts,
    const BatchUpdateResultList* update_result_list)
{
    MetricsDataList ret;

    // 取得元のBatchUpdateResultList
    auto learn_step = counts.learn_step;

    // 空の場合は空
    if (update_result_list == nullptr || update_result_list->empty())
        return ret;
    
    if (step_axis_ == StepAxis::LEARN) {
        // StepAxis::LEARNの場合、UpdateResultの一件毎にメトリクス情報
        for (const auto& update_result : *update_result_list) {
            // メトリクス情報取得
            auto scaler = update_result->GetScalar(this->key_);
            MetricsData data{ learn_step, scaler };
            ret.push_back(data);

            // カウント進める
            learn_step++;
        }
    } else {
        // StepAxis::LEARN以外の場合は平均値

        float sum = 0.0f;
        int count = 0;
        for (const auto& update_result : *update_result_list) {
            auto scaler = update_result->GetScalar(this->key_);
            if (!scaler.has_value()) continue;
            sum += *scaler;
            count++;
        }
        if (count > 0) {
            // step取得
            auto step = counts.GetByAxis(this->step_axis_);

            // 平均値として値算出
            float mean = sum / static_cast<float>(count);

            // メトリクス情報追加
            MetricsData data{ step, mean };
            ret.push_back(data);
        }
    }

    return ret;
}

MetricsLogObserverBase::MetricsDataList MetricsLogObserverBase::GetMetricsDataList(
    const StepCounts& counts,
    std::shared_ptr<const Agent> agent,
    std::shared_ptr<const Runner> runner,
    std::shared_ptr<const BatchEnv> env,
    const BatchExperience* experience,
    const BatchUpdateResultList* update_result_list,
    std::shared_ptr<const BatchActionInfo> action_info)
{
    MetricsDataList ret;

    if (event_field_.has_value()) {
        // 対象フィールドが指定されている場合

        if (*event_field_ == anet::rl::EventField::UPDATE_RESULT) {
            // UpdateResultList用メソッドでメトリクス情報を取得
            auto data_list = GetMetricsDataListFromUpdateResultList(counts, update_result_list);
            ret = std::move(data_list);
        } else {
            // その他メソッドでメトリクス情報を取得
            auto data = GetMetricsData(counts, agent, runner, env, experience, action_info, *event_field_);
            ret.push_back(data);
        }
    } else {
        // 対象フィールドが指定されていない場合、順番に試す

        // BatchUpdateResultList
        auto data_list = GetMetricsDataListFromUpdateResultList(counts, update_result_list);
        if (!data_list.empty()) {
            ret = std::move(data_list);
        } else {
            // Agent
            auto data = GetMetricsData(counts, agent, runner, env, experience, action_info, anet::rl::EventField::AGENT);
            if (data.second.has_value()) {
                ret.push_back(data);
            } else if (experience != nullptr) {
                // BatchExperience
                auto data = GetMetricsData(counts, agent, runner, env, experience, action_info, anet::rl::EventField::EXPERIENCE);
                if (data.second.has_value()) {
                    ret.push_back(data);
                } else {
                    // Runner
                    auto data = GetMetricsData(counts, agent, runner, env, experience, action_info, anet::rl::EventField::RUNNER);
                    if (data.second.has_value()) {
                        ret.push_back(data);
                    } else {
                        // Env
                        auto data = GetMetricsData(counts, agent, runner, env, experience, action_info, anet::rl::EventField::ENV);
                        if (data.second.has_value()) {
                            ret.push_back(data);
                        }
                    }
                }
            } else {
                // Runner
                auto data = GetMetricsData(counts, agent, runner, env, experience, action_info, anet::rl::EventField::RUNNER);
                if (data.second.has_value()) {
                    ret.push_back(data);
                } else {
                    // Env
                    auto data = GetMetricsData(counts, agent, runner, env, experience, action_info, anet::rl::EventField::ENV);
                    if (data.second.has_value()) {
                        ret.push_back(data);
                    }
                }
            }
        }
    }

    return ret;
}

void MetricsLogObserverBase::OnUpdate(const UpdateEvent& event)
{
    OnGenericUpdate(
        event.counts,
        event.agent,
        event.runner,
        event.runner != nullptr ? event.runner->GetBatchEnv() : nullptr,
        &event.experience,
        &event.update_result_list);
}

void MetricsLogObserverBase::OnTrainUpdate(const TrainEvent& event)
{
    OnGenericUpdate(
        event.counts,
        event.agent,
        event.runner,
        event.runner != nullptr ? event.runner->GetBatchEnv() : event.env,
        &event.experience,
        &event.update_result_list,
        event.action_info);
}

void MetricsLogObserverBase::OnGenericUpdate(
    const StepCounts& counts,
    std::shared_ptr<const Agent> agent,
    std::shared_ptr<const Runner> runner,
    std::shared_ptr<const BatchEnv> env,
    const BatchExperience* experience,
    const BatchUpdateResultList* update_result_list,
    std::shared_ptr<const BatchActionInfo> action_info)
{
    // メトリクスの出力データをリストとして取得
    auto metrics_list = GetMetricsDataList(counts, agent, runner, env, experience, update_result_list, action_info);

    // 取れたメトリクスを順に試す
    for (const auto& metrics : metrics_list) {
        const auto& step = metrics.first;
        const auto& value_opt = metrics.second;

        // メトリクス出力値
        std::optional<float> final_value;

        // EMA更新（出力しない場合も更新、クリッピング前の値で更新）
        if (value_opt.has_value()) {
            if (is_ema_) {
                val_ema_.Update(*value_opt);
                if (val_ema_.IsInitialized())
                    final_value = val_ema_.Value();
            } else {
                final_value = *value_opt;
            }
        } else {
            // メトリクス見つからない
            LOG::warn() << "MetricsLogObserverBase::OnUpdate(): value not found. tag=" << tag_ << " key=" << key_ << " step=" << step;
            continue;
        }

        // メトリクス出力間隔チェック
        if (step % interval_ != 0) continue;

        //  値が利用可能かチェック (EMA初期化前など)
        if (!final_value.has_value()) continue;

        // NaNやInFは出さないチェック (出力する最終値を見る)
        if (!std::isfinite(*final_value)) continue;

        // クリッピング
        if (clip_.has_value() && final_value.has_value())
            final_value = std::clamp<float>(*final_value, -*clip_, *clip_);

        // メトリクス出力
        MetricsLogger::Instance()->LogScalar(tag_, step, *final_value);
    }
}


// ===========================================================================
// GraphVizObserver
// ===========================================================================

GraphVizObserver::GraphVizObserver(
    const std::string& tag, int step_interval, int episode_interval, const std::string& provider_key,
    std::optional<anet::rl::EventField> event_field)
    : TaggedTrainObserver(tag)
    , step_interval_(step_interval), episode_interval_(episode_interval)
    , provider_key_(provider_key), event_field_(event_field)
{
    LOG::info() << "GraphVizObserver() tag=" << tag
        << " step_interval=" << step_interval_
        << " episode_interval=" << episode_interval_;
}

const anet::graphviz::GraphVizProvider* GraphVizObserver::FindProvider(const TrainEvent& event) const
{
    // event_field に応じて Provider を返す
    if (event_field_.has_value() && *event_field_ == anet::rl::EventField::ACTION_INFO) {
        if (event.experience.action) {
            return event.experience.action.get();
        }
    }

    // 今後 Agent 等から可視化情報を取る場合はここに追記
    // if (*event_field_ == anet::rl::EventField::AGENT) { return event.agent.get(); }

    return nullptr;
}

void GraphVizObserver::OnTrain(const TrainEvent& event)
{
    ANET_PROFILE_FUNC();

    auto step = event.counts.GetByAxis(anet::rl::StepAxis::TRAIN);
    const auto& state = event.experience.state;
    const auto& next_state = event.experience.next_state;

    // エピソード録画の開始・終了判定
    if (episode_interval_ > 0) {
        if (state.IsEpisodeStart()) {
            if (is_recording_) {
                LOG::warn() << "GraphVizObserver: force-ended due to unexpected episode_start.";
                is_recording_ = false;
            }
            if (local_episode_count_ % episode_interval_ == 0) {
                is_recording_ = true;
                LOG::info() << "GraphVizObserver: Episode capture started. episode=" << local_episode_count_;
            }
            local_episode_count_++;
        }
    }

    // 出力すべきタイミングかどうかの判定
    bool should_output = false;

    // パターンA: エピソード録画中
    if (is_recording_) {
        should_output = true;
    }
    // パターンB: 通常のステップインターバル
    else if (step_interval_ > 0 && step % step_interval_ == 0) {
        should_output = true;
    }

    if (!should_output) return;

    // グラフの生成と出力
    const auto* provider = FindProvider(event);
    if (!provider) return;
    auto graph = provider->CreateGraph(provider_key_, 0);
    if (!graph) return;

    // グラフ出力
    anet::MetricsLogger::Instance()->Log(tag_, step, *graph);

    // エピーソード録画終了判定
    if (is_recording_ && (next_state.IsDone() || next_state.IsTruncated())) {
        LOG::info() << "GraphVizObserver: Episode capture ended. step=" << step;
        is_recording_ = false;
    }
}


// ===========================================================================
// ObserverFactory
// ===========================================================================

static constexpr const char* CONFIG_KEY_METRICS_SCALAR_PREFIX = "metrics.scalar.[";
static constexpr const char* CONFIG_KEY_METRICS_SCALAR_SUFFIX = "]";

static constexpr const char* CONFIG_KEY_METRICS_GRAPH_PREFIX = "metrics.graph.[";
static constexpr const char* CONFIG_KEY_METRICS_GRAPH_SUFFIX = "]";

ObserverFactory::ObserverFactory(const ConfigData& config_data)
{
    ConfigData::MapType config_map = config_data.Map();
    
    // metrics.scalar.[tag] = <expr>  <$axis>  <@EventType>  <EventField>

    for (const auto& kv : config_map) {
        const std::string& config_key = kv.first;
        const std::string& config_value = kv.second;

        auto scalar_metrics_tag = anet::ExtractBetween(
            config_key, CONFIG_KEY_METRICS_SCALAR_PREFIX, CONFIG_KEY_METRICS_SCALAR_SUFFIX);

		if (!scalar_metrics_tag.empty())
        {
            ANET_LOG_DEBUG("ObserverFactory: scalar: key=" << scalar_metrics_tag  << " value=" << config_value);
            auto values = anet::Split(config_value, { " " }, true);

            // メトリクス定義情報を取得
            std::optional<std::string> key_opt;
            std::optional<anet::rl::EventType> event_opt;
            std::optional<anet::rl::RunnerScope> runner_scope_opt;
            std::string eval_name;
            std::optional<anet::rl::StepAxis> step_axis_opt;
			std::optional<anet::rl::EventField> field_opt;
            int interval = 1;
            bool is_ema = false;
			float ema_alpha = 0.01;
            std::optional<float> clip;

            for (auto v : values) {
                if (v == "@train") {
                    event_opt = EventType::TRAIN;
                } else if (v == "@learn") {
                    event_opt = EventType::LEARN;   /// @todo TRAINとLEARN以外の EventType も対応
                } else if (v == "@episode_end") {
                    event_opt = EventType::EPISODE_END;
                } else if (v == "$train_step") {
                    step_axis_opt = StepAxis::TRAIN;
                } else if (v == "$learn_step") {
                    step_axis_opt = StepAxis::LEARN;
                } else if (v == "$episode_step") {
                    step_axis_opt = StepAxis::EPISODE;
                } else if (v == "$exp_step") {
                    step_axis_opt = StepAxis::EXP;
                } else if (v == "$update_step") {
                    step_axis_opt = StepAxis::UPDATE;
                } else if (v == "$sim_step") {
                    step_axis_opt = StepAxis::SIM;
                } else if (v == "$agent") {
                    field_opt = EventField::AGENT;
                } else if (v == "$env") {
                    field_opt = EventField::ENV;
                } else if (v == "$batch_experience" || v == "$exp") {
                    field_opt = EventField::EXPERIENCE;
                } else if (v == "$batch_update_result" || v == "$update_result" || v == "$result") {
                    field_opt = EventField::UPDATE_RESULT;
                } else if (v == "$runner") {
                    field_opt = EventField::RUNNER;
                } else if (v == "$action" || v == "$action_info") {
                    field_opt = EventField::ACTION_INFO;
                } else if (v == "$train") {
                    runner_scope_opt = RunnerScope::TRAIN;
                } else if (anet::StartsWith(v, "$eval.[")) {
                    auto parsed_eval_name = anet::ExtractBetween(v, "$eval.[", "]");
                    if (parsed_eval_name.empty()) {
                        ANET_SYSTEM_ERROR("ObserverFactory: invalid eval runner scope. config_key=" << config_key
                            << " config_value=" << config_value << " token=" << v);
                    }
                    runner_scope_opt = RunnerScope::EVAL;
                    eval_name = parsed_eval_name;
                } else if (v == "$ema") {
                    is_ema = true;
                } else {
					auto attr_kv = anet::Split(v, { ":" }, true);
                    if (attr_kv.size() == 2) {
                        auto attr_key = attr_kv[0];
                        auto attr_val = attr_kv[1];
                        if (attr_key == "key") {
                            key_opt = attr_val;
                        } else if (attr_key == "event") {
                            if (attr_val == "train")
                                event_opt = EventType::TRAIN;
                            else if (attr_val == "learn")
                                event_opt = EventType::LEARN;
                            else if (attr_val == "episode_end")
                                event_opt = EventType::EPISODE_END;
                            else {
                                LOG::warn() << "Unknown event value. config_key=" << config_key
                                    << " config_value=" << config_value << " attr_val=" << attr_val;
                            }
                        } else if (attr_key == "step" || attr_key == "step_axis") {
                            if (attr_val == "train")
                                step_axis_opt = StepAxis::TRAIN;
                            else if (attr_val == "learn")
                                step_axis_opt = StepAxis::LEARN;
                            else if (attr_val == "episode_step" || attr_val == "episode")
                                step_axis_opt = StepAxis::EPISODE;
                            else if (attr_val == "exp_step" || attr_val == "exp")
                                step_axis_opt = StepAxis::EXP;
                            else if (attr_val == "update_step" || attr_val == "update")
                                step_axis_opt = StepAxis::UPDATE;
                            else if (attr_val == "sim_step" || attr_val == "sim")
                                step_axis_opt = StepAxis::SIM;
                            else {
                                LOG::warn() << "Unknown step value. config_key=" << config_key
                                    << " config_value=" << config_value << " attr_val=" << attr_val;
                            }
                        } else if (attr_key == "target") {
                            if (attr_val == "agent")
                                field_opt = EventField::AGENT;
                            else if (attr_val == "env")
                                field_opt = EventField::ENV;
                            else if (attr_val == "batch_experience" || attr_val == "exp")
                                field_opt = EventField::EXPERIENCE;
                            else if (attr_val == "batch_update_result" || attr_val == "update_result" || attr_val == "result")
                                field_opt = EventField::UPDATE_RESULT;
                            else if (attr_val == "runner")
                                field_opt = EventField::RUNNER;
                            else if (attr_val == "action" || attr_val == "action_info")
                                field_opt = EventField::ACTION_INFO;
                            else {
                                LOG::warn() << "Unknown target value. config_key=" << config_key
                                    << " config_value=" << config_value << " attr_val = " << attr_val;
                            }
                        } else if (attr_key == "interval") {
							interval = std::stoi(attr_val);
                        } else if (attr_key == "ema_alpha") {
                            is_ema = true;
                            ema_alpha = std::stof(attr_val);
                        } else if (attr_key == "clip") {
                            clip = std::stof(attr_val);
                        } else {
                            LOG::warn() << "Unknown attribute key. config_key=" << config_key
                                << " config_value=" << config_value << " attr_key=" << attr_key;
                        }
                    } else {
                        key_opt = v;
                    }
				}
            }
//            # metrics.scalar.[tag] = key [@event] [$target] [$step] [$ema] [interval:N] [ema_alpha:A]
//              #   @event   : (default: @train) @learn || @train
//              #   $step    : $train_step || $learn_step || $episode_step || $exp_step || $update_step || $sim_step
//              #   $target : $update_result || $agent || $exp || $trainer
//                metrics.scalar.min.[10_train / 10_total_reward] = train_reward_ema $ema field=trainer event:train interval:10 ema_alpha:0.01

            if (!key_opt.has_value()) {
                LOG::error() << "ObserverFactory: key not found. config_key=" << config_key << " config_value=" << config_value;
                continue;
            }

            auto key = *key_opt;
            auto event = event_opt.value_or(EventType::TRAIN);
            auto runner_scope = runner_scope_opt.value_or(RunnerScope::TRAIN);

            // 不整合チェック
            const bool is_eval_action_info_train =runner_scope == RunnerScope::EVAL && event == EventType::TRAIN && field_opt == EventField::ACTION_INFO;
            if (runner_scope == RunnerScope::EVAL && event != EventType::EPISODE_END && !is_eval_action_info_train) {
                ANET_SYSTEM_ERROR("ObserverFactory: $eval.[name] scope is only supported with @episode_end or @train $action_info. "
                    << "config_key=" << config_key << " config_value=" << config_value);
            }
            if (field_opt == EventField::ACTION_INFO && event != EventType::TRAIN) {
                ANET_SYSTEM_ERROR("ObserverFactory: $action_info scalar metrics are only supported with @train. "
                    << "config_key=" << config_key << " config_value=" << config_value);
            }
            if (event == EventType::EPISODE_END && field_opt.has_value()
                && (*field_opt == EventField::EXPERIENCE || *field_opt == EventField::UPDATE_RESULT)) {
                ANET_SYSTEM_ERROR("ObserverFactory: @episode_end does not support $exp or $update_result. "
                    << "config_key=" << config_key << " config_value=" << config_value);
            }

			// step_axisの決定
            anet::rl::StepAxis step_axis;
            if (step_axis_opt.has_value()) {
                step_axis = *step_axis_opt;
            } else {
                // stepの指定がない場合はeventに合わせて決定。
                //step_axis = (event == EventType::TRAIN) ? StepAxis::TRAIN : StepAxis::LEARN;
                if (event == EventType::TRAIN) {// || event == EventType::EPISODE_END) {
                    step_axis = StepAxis::TRAIN;
                } else {
                    step_axis = StepAxis::EXP; // LearnEventやEpisodeEndEventはデフォルトでEXPステップベース
                }
            }

            switch (event) {
            case EventType::TRAIN:
                {
                    auto train_obs = std::make_shared<MetricsLogTrainObserver>(scalar_metrics_tag, key, step_axis, field_opt
                        ,interval, is_ema, ema_alpha, clip);
                    train_observers_.push_back({ runner_scope, eval_name, train_obs });
                }
                break;
            case EventType::LEARN:
                {
                    auto learn_obs = std::make_shared<MetricsLogLearnObserver>(scalar_metrics_tag, key, step_axis, field_opt
                        , interval, is_ema, ema_alpha, clip);
                    learn_observers_.push_back({ runner_scope, eval_name, learn_obs });
                }
                break;
            case EventType::EPISODE_END:
                {
                    auto episode_end_obs = std::make_shared<MetricsLogEpisodeEndObserver>(scalar_metrics_tag, key, step_axis, field_opt
                        , interval, is_ema, ema_alpha, clip);
                    episode_end_observers_.push_back({ runner_scope, eval_name, episode_end_obs });
                }
                break;
            }
        }
 
        // ===================================================================
        // GraphVizObserver のパースと生成
        // ===================================================================
        auto graph_metrics_tag = anet::ExtractBetween(
            config_key, CONFIG_KEY_METRICS_GRAPH_PREFIX, CONFIG_KEY_METRICS_GRAPH_SUFFIX);

        if (!graph_metrics_tag.empty())
        {
            ANET_LOG_DEBUG("ObserverFactory: graph: key=" << graph_metrics_tag << " value=" << config_value);
            auto values = anet::Split(config_value, { " " }, true);

            std::optional<std::string> key_opt;
            std::optional<anet::rl::EventField> field_opt;
            int step_interval = -1;    // デフォルト無効
            int episode_interval = -1; // デフォルト無効

            for (auto v : values) {
                if (v == "$action" || v == "$action_info") {
                    field_opt = EventField::ACTION_INFO;
                } else {
                    auto attr_kv = anet::Split(v, { ":" }, true);
                    if (attr_kv.size() == 2) {
                        auto attr_key = attr_kv[0];
                        auto attr_val = attr_kv[1];

                        if (attr_key == "key") {
                            key_opt = attr_val;
                        } else if (attr_key == "target") {
                            if (attr_val == "action" || attr_val == "action_info") field_opt = EventField::ACTION_INFO;
                            else {
                                LOG::warn() << "Unknown target value for graph. attr_val=" << attr_val;
                            }
                        } else if (attr_key == "interval" || attr_key == "step_interval") {
                            step_interval = std::stoi(attr_val);
                        } else if (attr_key == "episode_interval" || attr_key == "eps_interval") {
                            episode_interval = std::stoi(attr_val);
                        } else {
                            LOG::warn() << "Unknown attribute key for graph. attr_key=" << attr_key;
                        }
                    } else {
                        if (v.rfind("@", 0) != 0 && v.rfind("$", 0) != 0) {
                            key_opt = v;
                        }
                    }
                }
            }

            if (!key_opt.has_value()) {
                LOG::error() << "ObserverFactory: graph key not found. config_key=" << config_key << " config_value=" << config_value;
                continue;
            }

            // 万が一どちらも指定されていなければ、安全のためstep_interval=1000にしておくなど適宜調整
            if (step_interval <= 0 && episode_interval <= 0) {
                step_interval = 1000;
            }

            // GraphVizObserverのインスタンス化と登録
            auto graph_obs = std::make_shared<GraphVizObserver>(
                graph_metrics_tag, step_interval, episode_interval, *key_opt, field_opt);
            train_observers_.push_back({ RunnerScope::TRAIN, "", graph_obs });

            // 次のループへ（スカラー処理等との重複を避ける）
            continue;
        }

        /// @todo HeatMap系Observerに対応

	}
}
