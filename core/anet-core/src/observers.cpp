// observers.cpp

#include "anet/observers.hpp"
#include <chrono>
#include <limits>
#include <unordered_set>
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

    // interval <= 0 は「無効」を表すため、そのときは gate を持たない
    if (config_.frame_interval > 0)
        frame_gate_.emplace(static_cast<uint64_t>(config_.frame_interval));
    if (config_.log_interval > 0)
        log_gate_.emplace(static_cast<uint64_t>(config_.log_interval));
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
    if (frame_gate_ && frame_gate_->ShouldFire(step)) {
        histogram_->NextFrame();
        is_frame_updated = true;
    }

    //  ログ出力（フレーム更新があった時だけチェックする）
    if (is_frame_updated && log_gate_) {
        // A. ログ頻度がフレームより高い (log < frame) → 毎回出す（間引きようがないため）
        // B. ログ頻度が低い (log >= frame) → ログ間隔のバケットを跨いだ時だけ出す
        if (config_.log_interval <= config_.frame_interval || log_gate_->ShouldFire(step)) {
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

    // interval <= 0 は「無効」を表すため、そのときは gate を持たない
    if (config_.log_interval > 0)
        log_gate_.emplace(static_cast<uint64_t>(config_.log_interval));
}

void SweepedHeatMapObserver::OnLearn(const LearnEvent& event)
{
    ANET_PROFILE_FUNC();

    /// @todo メトリクスのSTEP軸を指定できるようにする？

    // 実行判定
    auto step = event.counts.learn_step;
    if (!log_gate_) return;
    if (!log_gate_->ShouldFire(step)) return;

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
    , use_background_(use_background)
{
    ANET_CHECK(eval_runner_ != nullptr);

    // interval <= 0 は「無効」を表すため、そのときは gate を持たない
    if (eval_interval > 0)
        eval_gate_.emplace(static_cast<uint64_t>(eval_interval));

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

void EpisodeEvalObserver::RunEvaluationSession(const StepCounts& event_counts)
{
    eval_runner_->RunSession(event_counts);
}

void EpisodeEvalObserver::RethrowCompletedBackgroundEval()
{
    if (!eval_future_.valid()) return;
    if (eval_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;

    // std::future は get() で初めて worker 側の例外を呼び出し元へ再送出する
    eval_future_.get();
}

void EpisodeEvalObserver::WaitBackgroundEval()
{
    if (!eval_future_.valid()) return;

    // 次の評価を投入する前に、前回評価の完了と失敗を必ず回収する
    eval_future_.get();
}

void EpisodeEvalObserver::OnLearn(const LearnEvent& event)
{
    ANET_PROFILE_FUNC();

    if (use_background_) {
        RethrowCompletedBackgroundEval();
    }

    auto step = event.counts.GetByAxis(anet::rl::StepAxis::LEARN);

    // 評価エピソードを終端まで回す
    if (eval_gate_ && eval_gate_->ShouldFire(step)) {
        if (use_background_) {
            // 前回の評価がまだ終わっていなければ、ここで完了までブロックして待つ
            WaitBackgroundEval();

            // スレッドに評価エピソード実行処理を投げる
            const StepCounts event_counts = event.counts;
            eval_future_ = eval_pool_->EnqueueFuture(0, [this, event_counts]() {
                try {
                    this->RunEvaluationSession(event_counts);
                } catch (const std::exception& e) {
                    LOG::fatal() << this->ToString() << " RunEvaluationSession failed: " << e.what();
                    throw;
                } catch (...) {
                    LOG::fatal() << this->ToString() << " RunEvaluationSession failed: unknown exception";
                    throw;
                }
                });
        } else {
            // フォアグラウンド実行
            RunEvaluationSession(event.counts);
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
    , is_ema_(is_ema), gate_(ValidateMetricsInterval(tag, interval))
    , val_ema_(ema_alpha), clip_(clip)
{
    ;
}

uint64_t MetricsLogObserverBase::ValidateMetricsInterval(const std::string& tag, int interval)
{
    if (interval < 1) {
        ANET_SYSTEM_ERROR(
            "MetricsLogObserverBase interval is invalid: tag=" << tag
            << " interval=" << interval << " expected=1 or greater");
    }
    return static_cast<uint64_t>(interval);
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

MetricsLogSessionEndObserver::MetricsLogSessionEndObserver(const std::string& tag, const std::string& key,
    anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
    int interval, bool is_ema, float ema_alpha, std::optional<float> clip)
    : MetricsLogObserverBase(tag, key, step_axis, event_field, interval, is_ema, ema_alpha, clip)
{
    ;
}

MetricsLogTraceObserver::MetricsLogTraceObserver(const std::string& tag,
    std::vector<std::string> keys, StepAxis step_axis, EventField field)
    : TaggedObserver(tag), keys_(std::move(keys)), step_axis_(step_axis), field_(field)
{
}

void MetricsLogTraceObserver::OnEpisodeEnd(const EpisodeEndEvent& event)
{
    ANET_PROFILE_FUNC();
    
    // 対象を一度だけ解決し、次の Step が確定値を消す前に宣言順で読む。
    const anet::Module* target = nullptr;
    const char* target_name = "unknown";
    switch (field_) {
    case EventField::ENV:
        target_name = "env";
        target = event.env != nullptr ? event.env.get()
            : (event.runner != nullptr ? event.runner->GetBatchEnv().get() : nullptr);
        break;
    case EventField::AGENT: target_name = "agent"; target = event.agent.get(); break;
    case EventField::RUNNER: target_name = "runner"; target = event.runner.get(); break;
    default: break;
    }
    anet::json data = anet::json::object();
    for (const auto& key : keys_) {
        const auto value = target != nullptr ? target->GetScalar(key, event.env_index) : std::nullopt;
        if (!value.has_value()) {
            ANET_SYSTEM_ERROR("Trace metric key is unavailable. tag='" << tag_
                << "' key='" << key << "' lane=" << event.env_index << " target=" << target_name
                << " expected=recognized scalar key.");
        }
        data[key] = *value;
    }
    // 一つの episode を原子的な JSONL 行にし、非有限値の null 化は既存 JSON writer に任せる。
    anet::MetricsLogger::Instance()->LogTrace(tag_, event.counts.GetByAxis(step_axis_), event.env_index, data);
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
MetricsLogObserverBase::UpdateResultMetricsLookup MetricsLogObserverBase::GetMetricsDataListFromUpdateResultList(
    const StepCounts& counts,
    const BatchUpdateResultList* update_result_list)
{
    UpdateResultMetricsLookup ret;

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
            if (scaler.has_value()) {
                ret.recognized = true;
            }
            MetricsData data{ learn_step, scaler };
            ret.data_list.push_back(data);

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
            // key認識は有限値の成立可否と分離し、既知NaNから他sourceへ探索させない。
            ret.recognized = true;
            if (!std::isfinite(*scaler)) continue;
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
            ret.data_list.push_back(data);
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
            auto lookup = GetMetricsDataListFromUpdateResultList(counts, update_result_list);
            ret = std::move(lookup.data_list);
        } else {
            // その他メソッドでメトリクス情報を取得
            auto data = GetMetricsData(counts, agent, runner, env, experience, action_info, *event_field_);
            ret.push_back(data);
        }
    } else {
        // 対象フィールドが指定されていない場合、順番に試す

        // BatchUpdateResultList
        auto lookup = GetMetricsDataListFromUpdateResultList(counts, update_result_list);
        if (lookup.recognized) {
            // 既知keyなら有限な出力がなくてもsource探索をここで終了する。
            ret = std::move(lookup.data_list);
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
            if (!std::isfinite(*value_opt)) continue;
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

        // メトリクス出力間隔チェック（bucket-crossing）
        if (!gate_.ShouldFire(step)) continue;

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

namespace metrics_def_names {

    /// config の `$xxx_step` token と同じ表記を返す。
    /// anet::rl::toString(StepAxis) は "train" / "exp" を返すため、そちらとは別に持つ。
    static std::string StepAxisToken(anet::rl::StepAxis axis)
    {
        switch (axis) {
        case StepAxis::TRAIN:    return "train_step";
        case StepAxis::EXP:      return "exp_step";
        case StepAxis::UPDATE:   return "update_step";
        case StepAxis::LEARN:    return "learn_step";
        case StepAxis::EPISODE:  return "episode_step";
        case StepAxis::SIM:      return "sim_step";
        }
        return "unknown";
    }

    static std::string EventToken(anet::rl::EventType event)
    {
        switch (event) {
        case EventType::TRAIN:       return "train";
        case EventType::LEARN:       return "learn";
        case EventType::EPISODE_END: return "episode_end";
        case EventType::SESSION_END: return "session_end";
        default:                     return "unknown";
        }
    }

    static std::string FieldToken(const std::optional<anet::rl::EventField>& field)
    {
        if (!field.has_value()) return "";
        switch (*field) {
        case EventField::AGENT:         return "agent";
        case EventField::ENV:           return "env";
        case EventField::EXPERIENCE:    return "exp";
        case EventField::UPDATE_RESULT: return "update_result";
        case EventField::RUNNER:        return "runner";
        case EventField::ACTION_INFO:   return "action_info";
        default:                        return "unknown";
        }
    }

    /// step counter を所有する Runner を返す。
    /// EvalRunner::DoStep() は @train 系 event へ自分の step_counts_ を載せ、
    /// @episode_end / @session_end へは呼び出し元 (train runner) の event_counts を載せる
    /// (trainer.cpp の EvalRunner::RunSession を参照)。
    /// このため同じ $eval.[name] $exp_step でも event によって座標系が変わる。
    static std::string OwningRunner(
        anet::rl::RunnerScope scope, anet::rl::EventType event, const std::string& eval_name)
    {
        if (scope == RunnerScope::EVAL && event == EventType::TRAIN) return eval_name;
        return "train";
    }
}

anet::json anet::rl::ScalarMetricDefsToJson(
    const std::vector<ObserverFactory::ScalarMetricDef>& defs)
{
    anet::json payload = anet::json::object();
    for (const auto& def : defs) {
        payload[def.tag] = {
            {"step_axis", def.step_axis},
            {"runner", def.runner},
            {"scope", def.scope == RunnerScope::EVAL ? "eval" : "train"},
            {"eval_name", def.scope == RunnerScope::EVAL ? anet::json(def.eval_name) : anet::json(nullptr)},
            {"eval_episodes", def.scope == RunnerScope::EVAL && def.eval_episodes
                ? anet::json(*def.eval_episodes) : anet::json(nullptr)},
            {"num_envs", def.scope == RunnerScope::EVAL && def.num_envs
                ? anet::json(*def.num_envs) : anet::json(nullptr)},
            {"event", def.event},
            {"target", def.target.empty() ? anet::json(nullptr) : anet::json(def.target)},
            {"source_key", def.source_key},
            {"ema_alpha", def.has_ema ? anet::json(def.ema_alpha) : anet::json(nullptr)},
            {"interval", def.interval},
            {"clip", def.clip ? anet::json(*def.clip) : anet::json(nullptr)},
        };
    }
    return payload;
}

anet::json anet::rl::TraceMetricDefsToJson(const std::vector<ObserverFactory::TraceMetricDef>& defs)
{
    anet::json payload = anet::json::object();
    // 座標系の所有者と購読先を分け、eval 条件と順序付きキー列を tag ごとに残す。
    for (const auto& def : defs) {
        payload[def.tag] = {{"step_axis", def.step_axis}, {"runner", def.runner},
            {"scope", def.scope == RunnerScope::EVAL ? "eval" : "train"},
            {"eval_name", def.scope == RunnerScope::EVAL ? anet::json(def.eval_name) : anet::json(nullptr)},
            {"eval_episodes", def.scope == RunnerScope::EVAL && def.eval_episodes
                ? anet::json(*def.eval_episodes) : anet::json(nullptr)},
            {"num_envs", def.scope == RunnerScope::EVAL && def.num_envs
                ? anet::json(*def.num_envs) : anet::json(nullptr)},
            {"event", def.event}, {"target", def.target}, {"keys", def.keys}};
    }
    return payload;
}

namespace metric_tokens {
    enum class Kind { KEY, EVENT, STEP, TARGET, SCOPE, EMA, ATTRIBUTE };
    struct Token {
        Kind kind = Kind::KEY;
        std::string raw;
        std::string value;
        std::string attribute;
        std::optional<EventType> event;
        std::optional<StepAxis> step;
        std::optional<EventField> field;
    };

    // 宣言順と各指定の出現を残し、既定値・後勝ち・診断はチャネル側へ委ねる。
    static std::vector<Token> ParseMetricTokens(const std::string& definition)
    {
        std::vector<Token> tokens;
        for (const auto& raw : anet::Split(definition, { " " }, true)) {
            Token token{ .raw = raw, .value = raw };
            if (raw == "@train" || raw == "@learn" || raw == "@episode_end" || raw == "@session_end") {
                token.kind = Kind::EVENT;
                token.value = raw.substr(1);
            } else if (raw == "$train_step" || raw == "$learn_step" || raw == "$episode_step"
                || raw == "$exp_step" || raw == "$update_step" || raw == "$sim_step") {
                token.kind = Kind::STEP;
                token.value = raw.substr(1);
            } else if (raw == "$agent" || raw == "$env" || raw == "$runner"
                || raw == "$batch_experience" || raw == "$exp" || raw == "$batch_update_result"
                || raw == "$update_result" || raw == "$result" || raw == "$action" || raw == "$action_info") {
                token.kind = Kind::TARGET;
                token.value = raw.substr(1);
            } else if (raw == "$train" || anet::StartsWith(raw, "$eval.[")) {
                token.kind = Kind::SCOPE;
                token.value = raw == "$train" ? "train" : anet::ExtractBetween(raw, "$eval.[", "]");
            } else if (raw == "$ema") {
                token.kind = Kind::EMA;
            } else {
                const auto attr = anet::Split(raw, { ":" }, true);
                if (attr.size() == 2) {
                    token.kind = Kind::ATTRIBUTE;
                    token.attribute = attr[0];
                    token.value = attr[1];
                    if (attr[0] == "event") token.kind = Kind::EVENT;
                    else if (attr[0] == "step" || attr[0] == "step_axis") token.kind = Kind::STEP;
                    else if (attr[0] == "target") token.kind = Kind::TARGET;
                }
            }
            if (token.kind == Kind::EVENT) {
                if (token.value == "train") token.event = EventType::TRAIN;
                else if (token.value == "learn") token.event = EventType::LEARN;
                else if (token.value == "episode_end") token.event = EventType::EPISODE_END;
                else if (token.value == "session_end") token.event = EventType::SESSION_END;
            } else if (token.kind == Kind::STEP) {
                // 属性形は従来の短縮名を受理し、$ 形だけ train_step / learn_step を受理する。
                if (token.value == "train" || (token.attribute.empty() && token.value == "train_step")) token.step = StepAxis::TRAIN;
                else if (token.value == "learn" || (token.attribute.empty() && token.value == "learn_step")) token.step = StepAxis::LEARN;
                else if (token.value == "episode" || token.value == "episode_step") token.step = StepAxis::EPISODE;
                else if (token.value == "exp" || token.value == "exp_step") token.step = StepAxis::EXP;
                else if (token.value == "update" || token.value == "update_step") token.step = StepAxis::UPDATE;
                else if (token.value == "sim" || token.value == "sim_step") token.step = StepAxis::SIM;
            } else if (token.kind == Kind::TARGET) {
                if (token.value == "agent") token.field = EventField::AGENT;
                else if (token.value == "env") token.field = EventField::ENV;
                else if (token.value == "runner") token.field = EventField::RUNNER;
                else if (token.value == "exp" || token.value == "batch_experience") token.field = EventField::EXPERIENCE;
                else if (token.value == "update_result" || token.value == "batch_update_result" || token.value == "result") token.field = EventField::UPDATE_RESULT;
                else if (token.value == "action_info" || token.value == "action") token.field = EventField::ACTION_INFO;
            }
            tokens.push_back(std::move(token));
        }
        return tokens;
    }
}

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
            const auto tokens = metric_tokens::ParseMetricTokens(config_value);

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

            // scalar は既存の後勝ち・既定値・WARN 診断を保つ。
            for (const auto& token : tokens) {
                using metric_tokens::Kind;
                switch (token.kind) {
                case Kind::EVENT:
                    if (token.event) event_opt = token.event;
                    else LOG::warn() << "Unknown event value. config_key=" << config_key
                        << " config_value=" << config_value << " attr_val=" << token.value;
                    break;
                case Kind::STEP:
                    if (token.step) step_axis_opt = token.step;
                    else LOG::warn() << "Unknown step value. config_key=" << config_key
                        << " config_value=" << config_value << " attr_val=" << token.value;
                    break;
                case Kind::TARGET:
                    if (token.field) field_opt = token.field;
                    else LOG::warn() << "Unknown target value. config_key=" << config_key
                        << " config_value=" << config_value << " attr_val = " << token.value;
                    break;
                case Kind::SCOPE:
                    if (token.raw == "$train") runner_scope_opt = RunnerScope::TRAIN;
                    else {
                        if (token.value.empty()) {
                            ANET_SYSTEM_ERROR("ObserverFactory: invalid eval runner scope. config_key=" << config_key
                                << " config_value=" << config_value << " token=" << token.raw);
                        }
                        runner_scope_opt = RunnerScope::EVAL;
                        eval_name = token.value;
                    }
                    break;
                case Kind::EMA: is_ema = true; break;
                case Kind::ATTRIBUTE:
                    if (token.attribute == "key") key_opt = token.value;
                    else if (token.attribute == "interval") interval = std::stoi(token.value);
                    else if (token.attribute == "ema_alpha") { is_ema = true; ema_alpha = std::stof(token.value); }
                    else if (token.attribute == "clip") clip = std::stof(token.value);
                    else LOG::warn() << "Unknown attribute key. config_key=" << config_key
                        << " config_value=" << config_value << " attr_key=" << token.attribute;
                    break;
                case Kind::KEY: key_opt = token.raw; break;
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

            // eval の個体完了とセッション集約を混同する束縛を拒否する。
            if (runner_scope == RunnerScope::EVAL && event == EventType::EPISODE_END) {
                ANET_SYSTEM_ERROR("eval scalar metrics fire once per evaluation session; replace @episode_end with @session_end. "
                    << "config_key=" << config_key << " config_value=" << config_value);
            }
            if (runner_scope == RunnerScope::TRAIN && event == EventType::SESSION_END) {
                ANET_SYSTEM_ERROR("ObserverFactory: @session_end requires $eval.[name] scope. "
                    << "config_key=" << config_key << " config_value=" << config_value);
            }
            // 不整合チェック
            const bool is_eval_action_info_train =runner_scope == RunnerScope::EVAL && event == EventType::TRAIN && field_opt == EventField::ACTION_INFO;
            if (runner_scope == RunnerScope::EVAL && event != EventType::SESSION_END && !is_eval_action_info_train) {
                ANET_SYSTEM_ERROR("ObserverFactory: $eval.[name] scope is only supported with @session_end or @train $action_info. "
                    << "config_key=" << config_key << " config_value=" << config_value);
            }
            if (field_opt == EventField::ACTION_INFO && event != EventType::TRAIN) {
                ANET_SYSTEM_ERROR("ObserverFactory: $action_info scalar metrics are only supported with @train. "
                    << "config_key=" << config_key << " config_value=" << config_value);
            }
            if ((event == EventType::EPISODE_END || event == EventType::SESSION_END) && field_opt.has_value()
                && (*field_opt == EventField::EXPERIENCE || *field_opt == EventField::UPDATE_RESULT)) {
                ANET_SYSTEM_ERROR("ObserverFactory: episode/session end does not support $exp or $update_result. "
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

            // 解析側が設定から再導出しなくて済むよう、解決済み定義をそのまま控える。
            scalar_metric_defs_.push_back(ScalarMetricDef{
                .tag = scalar_metrics_tag,
                .step_axis = metrics_def_names::StepAxisToken(step_axis),
                .runner = metrics_def_names::OwningRunner(runner_scope, event, eval_name),
                .event = metrics_def_names::EventToken(event),
                .target = metrics_def_names::FieldToken(field_opt),
                .source_key = key,
                .has_ema = is_ema,
                .ema_alpha = ema_alpha,
                .interval = interval,
                .clip = clip,
                .scope = runner_scope,
                .eval_name = eval_name,
                .subscription = ScalarMetricSubscription{
                    .source_key = key,
                    .event = event,
                    .target = field_opt,
                    .interval = interval,
                    .scope = runner_scope,
                    .eval_name = eval_name,
                },
                });

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
            case EventType::SESSION_END:
                {
                    auto session_end_obs = std::make_shared<MetricsLogSessionEndObserver>(scalar_metrics_tag, key, step_axis, field_opt
                        , interval, is_ema, ema_alpha, clip);
                    session_end_observers_.push_back({ runner_scope, eval_name, session_end_obs });
                }
                break;
            }
        }
 
        const auto trace_tag = anet::ExtractBetween(config_key, "metrics.trace.[", "]");
        if (!trace_tag.empty()) {
            std::vector<std::string> keys;
            std::optional<EventField> field;
            StepAxis axis = StepAxis::EXP;
            RunnerScope scope = RunnerScope::TRAIN;
            std::string eval_name;
            bool event_seen = false, target_seen = false, scope_seen = false, step_seen = false;
            std::unordered_set<std::string> seen_keys;
            auto invalid = [&](const std::string& token, const std::string& expected) {
                ANET_SYSTEM_ERROR("Invalid trace declaration. config_key=" << config_key
                    << " token='" << token << "' expected=" << expected << ".");
            };
            auto unique = [&](bool& seen, const metric_tokens::Token& token) {
                if (seen) invalid(token.raw, "one specification per event, target, scope or step axis");
                seen = true;
            };
            // 出現順に検証し、同値や別表記の重複・後続上書きも拒否する。
            for (const auto& token : metric_tokens::ParseMetricTokens(config_value)) {
                using metric_tokens::Kind;
                switch (token.kind) {
                case Kind::EVENT:
                    unique(event_seen, token);
                    if (token.event != EventType::EPISODE_END) {
                        invalid(token.raw, "trace supports @episode_end only in this version");
                    }
                    break;
                case Kind::TARGET:
                    unique(target_seen, token);
                    if (token.field != EventField::ENV && token.field != EventField::RUNNER && token.field != EventField::AGENT) {
                        invalid(token.raw, "$env, $runner or $agent");
                    }
                    field = token.field;
                    break;
                case Kind::STEP:
                    unique(step_seen, token);
                    if (!token.step) invalid(token.raw, "a recognized step axis");
                    axis = *token.step;
                    break;
                case Kind::SCOPE:
                    unique(scope_seen, token);
                    if (token.raw != "$train") {
                        if (token.value.empty() || token.raw != "$eval.[" + token.value + "]") {
                            invalid(token.raw, "$train or $eval.[name]");
                        }
                        scope = RunnerScope::EVAL;
                        eval_name = token.value;
                    }
                    break;
                case Kind::EMA:
                case Kind::ATTRIBUTE:
                    invalid(token.raw, "bare scalar keys without EMA, clip, interval or key attributes");
                    break;
                case Kind::KEY:
                    if (token.raw.starts_with("@") || token.raw.starts_with("$") || token.raw.find(':') != std::string::npos) {
                        invalid(token.raw, "a recognized control token or a bare scalar key");
                    }
                    if (token.raw.starts_with("mean.") || token.raw.starts_with("max.")
                        || token.raw.starts_with("min.") || token.raw.starts_with("std.")) {
                        invalid(token.raw, "an individual scalar key without aggregation prefix");
                    }
                    if (!seen_keys.insert(token.raw).second) invalid(token.raw, "unique scalar keys");
                    keys.push_back(token.raw);
                    break;
                }
            }
            if (!event_seen) invalid("<missing>", "an explicit @episode_end event");
            if (!target_seen) invalid("<missing>", "an explicit $env, $runner or $agent target");
            if (keys.empty()) invalid("<missing>", "at least one bare scalar key");
            trace_metric_defs_.push_back(TraceMetricDef{
                .tag = trace_tag, .step_axis = metrics_def_names::StepAxisToken(axis),
                .runner = metrics_def_names::OwningRunner(scope, EventType::EPISODE_END, eval_name),
                .event = "episode_end", .target = metrics_def_names::FieldToken(field),
                .keys = keys, .scope = scope, .eval_name = eval_name });
            episode_end_observers_.push_back({ scope, eval_name,
                std::make_shared<MetricsLogTraceObserver>(trace_tag, std::move(keys), axis, *field) });
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
	}
}
