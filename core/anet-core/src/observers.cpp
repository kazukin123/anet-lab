#include "anet/observers.hpp"
#include <wx/log.h>
#include "anet/profile.hpp"
#include "anet/str_util.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/env.hpp"

using namespace anet::rl;


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
    anet::ProfileRange r("HeatMapVectorObserver::OnPostUpdate");

	/// @todo メトリクスのSTEP軸を指定できるようにする
	auto step = event.counts.GetByAxis(anet::rl::StepAxis::TRAIN);    

    // 生成： xv, yv, vv
    auto xv = x_probe_->GetVector(event);
    auto yv = y_probe_->GetVector(event);
    auto vv = value_probe_->GetVector(event);

    // 揃ってなかったらスキップ
    if (!xv.has_value() || !yv.has_value() || !vv.has_value())
        return;
    if (xv->size() != yv->size() || xv->size() != vv->size())
        return;

    // データ追加
    const size_t n = xv->size();

    for (size_t i = 0; i < n; i++) {
        float x = (*xv)[i];
        float y = (*yv)[i];
        float v = (*vv)[i];
        heatmap_->AddData(x, y, v);
        //ANET_ASSERT(x >= heatmap_->x_min_ && x <= heatmap_->x_max_);
        //ANET_ASSERT(y >= heatmap_->y_min_ && y <= heatmap_->y_max_);
        //wxLogDebug("HeatMapVectorObserver::OnPostUpdate v=%f x=%f y=%f", v, sx, theta_deg);
    }

    if (step % config_.log_interval == 0) {
        MetricsLogger::Instance()->LogImage(
            tag_,
            step,
            *heatmap_,
            config_.image_width,
            config_.image_height
        );
    }

    return;
}

TimeHistogramObserver::TimeHistogramObserver(
    const std::string& tag, const TimeHistogramObserverConfig& config,
    std::shared_ptr<VectorProbe> probe)
	: TaggedTrainObserver(tag)
    , config_(config), probe_(probe)
{
    histogram_ = std::make_unique<anet::TimeHistogram>(
        config_.bins, config_.max_frames, config_.mode, config_.flags, config_.base_min, config_.base_max, config_.alpha);
}

void TimeHistogramObserver::OnTrain(const TrainEvent& event)
{
	/// @todo メトリクスのSTEP軸を指定できるようにする
    auto step = event.counts.GetByAxis(anet::rl::StepAxis::TRAIN);

    // Probeで vectorを取得
    auto values = probe_->GetVector(event);
    if (values.has_value()) {
        histogram_->AddBatch(*values);
    }

    // フレーム更新
    if (step % config_.frame_interval == 0) {
        histogram_->NextFrame();
    }

    // ログ出力
    if (step % config_.log_interval == 0) {
        MetricsLogger::Instance()->LogImage(tag_, step, *histogram_, config_.image_width, config_.image_height);
    }

    return;
}

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
        wxLogInfo("MultiPairHeatMapObserver: %s axis_probes: [%d] %s (%f %f)",
            tag_,
            static_cast<int>(i), axis_probes_[i]->GetName().value(),
            axis_probes_[i]->GetMin().value_or(0.0f),
            axis_probes_[i]->GetMax().value_or(0.0f));
    }
    int plot_cnt = 0;
    for (int i = 0; i < m; i++) {
        auto x_name = axis_probes_[i]->GetName();
        for (int j = i + 1; j < m; j++) {
            auto y_name = axis_probes_[j]->GetName();
            wxLogInfo("MultiPairHeatMapObserver: %s axis_patters: [%d] x=[%d]%s y=[%d]%s",
                tag_,
                plot_cnt,
                i, (x_name.has_value() ? (*x_name).c_str() : ""),
                j, (y_name.has_value() ? (*y_name).c_str() : ""));
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
    anet::ProfileRange r("MultiPairHeatMapObserver::OnPostUpdate");

	/// @todo メトリクスのSTEP軸を指定できるようにする
    auto step = event.counts.GetByAxis(anet::rl::StepAxis::TRAIN);

    // 値ベクトル
    auto vv = value_probe_->GetVector(event);
    if (!vv) return;

    // 全プローブペア i<j をスキャン
    const size_t m = axis_probes_.size();

    for (size_t i = 0; i < m; i++) {
        auto xv = axis_probes_[i]->GetVector(event);
        if (!xv) continue;

        auto xmin = axis_probes_[i]->GetMin();
        auto xmax = axis_probes_[i]->GetMax();

        for (size_t j = i + 1; j < m; j++) {
            auto yv = axis_probes_[j]->GetVector(event);
            if (!yv) continue;

            auto ymin = axis_probes_[j]->GetMin();
            auto ymax = axis_probes_[j]->GetMax();

            size_t n = std::min({ xv->size(), yv->size(), vv->size() });

            for (size_t k = 0; k < n; k++) {
                float x_raw = (*xv)[k];
                float y_raw = (*yv)[k];
                float v_raw = (*vv)[k];

                // (0〜1) 正規化
                float x_norm = Normalize01(x_raw, xmin, xmax);
                float y_norm = Normalize01(y_raw, ymin, ymax);

                heatmap_->AddData(x_norm, y_norm, v_raw);
            }
        }
    }

    if (step % config_.log_interval == 0) {
        MetricsLogger::Instance()->LogImage(
            tag_, step,
            *heatmap_,
            config_.image_width,
            config_.image_height);
    }

    return;
}

SweepedHeatMapObserver::SweepedHeatMapObserver(
    const std::string& tag,
    const SweepedHeatMapObserverConfig& config,
    std::shared_ptr<ISweepInputGenerator> input_gen,
    TensorFunction tensor_fn,
    std::shared_ptr<ISweepOutputExtractor> output_ext,
    const std::unordered_map<std::string, std::string>& scalar_tag_label_map)
    : TaggedTrainObserver(tag), config_(config),
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

void SweepedHeatMapObserver::OnTrain(const TrainEvent& event)
{
    anet::ProfileRange r("SweepedHeatMapObserver::OnPostUpdate");

	/// @todo メトリクスのSTEP軸を指定できるようにする
    auto step = event.counts.GetByAxis(anet::rl::StepAxis::TRAIN);

    if (step % config_.log_interval != 0) return;

    const int64_t grid_num = static_cast<int64_t>(grid_w_) * static_cast<int64_t>(grid_h_);

    // 入力バッチ生成（GPU 上）
    torch::Tensor batch_in = input_gen_->BuildInputTensor();
    ANET_CHECK_SHAPE(batch_in, { grid_num, ANET_SHAPE_ENDANY });
    wxLogDebug(
        "SweepedHeatMapObserver::OnPostUpdate() batch_in=%s",
        anet::ToDefString(batch_in));

    // NN 適用（GPU 上）
    torch::Tensor batch_out = tensor_fn_(batch_in);
    ANET_CHECK_SHAPE(batch_out, { grid_num, ANET_SHAPE_ENDANY });
    wxLogDebug(
        "SweepedHeatMapObserver::OnPostUpdate() batch_out=%s",
        anet::ToDefString(batch_out));

    // リクエストするLabelをscalar_tag_label_map_からsetに詰める
    std::unordered_set<std::string> req_label_set(scalar_label_tag_map_.size());
    //wxLogDebug("tag=%s req_label_set.size=%lld", heatmap_tag_.c_str(), req_label_set.size());
    for (const auto& kv : scalar_label_tag_map_) {
        req_label_set.insert(kv.first);
        //wxLogDebug("tag=%s label=%s", heatmap_tag_, kv.second);
    }

    // 出力から値抽出（GPU 上, [W*H]）
    ExtractResult extract_result = output_ext_->Extract(batch_out, req_label_set);
    wxLogDebug(
        "SweepedHeatMapObserver::OnPostUpdate() grid_values=%s  tag=%s",
        anet::ToDefString(extract_result.grid), tag_);
    ANET_CHECK_SHAPE(extract_result.grid, { grid_num });
    ANET_CHECK_DTYPE(extract_result.grid, torch::kFloat32);
    ANET_ASSERT(extract_result.labels.size() == extract_result.scalars.size());

    // CPU へ一括転送
    torch::Tensor grid_cpu = extract_result.grid.to(torch::kCPU);
    ANET_CHECK_SHAPE(grid_cpu, { grid_num });
    ANET_CHECK_DTYPE(grid_cpu, torch::kFloat32);
    float* data = grid_cpu.data_ptr<float>();
    std::vector<torch::Tensor> scalars_cpu;
    for (auto& t : extract_result.scalars) scalars_cpu.push_back(t.to(torch::kCPU));
    wxLogDebug("SweepedHeatMapObserver::OnPostUpdate() Extract done.");

    // HeatMapデータ設定
    heatmap_->SetGridValues(data, grid_w_, grid_h_);
    wxLogDebug("SweepedHeatMapObserver::OnPostUpdate() SetGridValues() done.");

    // 画像ログ出力
    MetricsLogger::Instance()->LogImage(
        tag_,
        step,
        *heatmap_,
        config_.image_width,
        config_.image_height);
    wxLogDebug("SweepedHeatMapObserver::OnPostUpdate() LogImage() done. tag=%s", tag_);

    // Scalarログ出力
    for (int i = 0; i < extract_result.labels.size(); i++) {
        auto result_label = extract_result.labels[i];
        auto tag_itr = scalar_label_tag_map_.find(result_label);
        if (tag_itr != scalar_label_tag_map_.end()) {
            auto scalar_tag = tag_itr->second;
            auto scalar_value = scalars_cpu[i].item<float>();
            MetricsLogger::Instance()->LogScalar(scalar_tag, step, scalar_value);
        }
    }

    return;
}

EpisodeEvalObserver::EpisodeEvalObserver(
    ReportFunction report_function,
    std::shared_ptr<anet::rl::SingleDiscreteEnvFactory> eval_env_factory,
	const ConfigData& config_data,
    const torch::Device& device,
    anet::rl::RunMode runmode, int log_interval, int eval_inerval)
    : report_function_(std::move(report_function))
    , runmode_(runmode), log_interval_(log_interval), eval_interval_(eval_inerval)
{
    env_ = std::make_unique<VectorizedDiscreteBatchEnv>(config_data, eval_env_factory, 1, device);
}

void EpisodeEvalObserver::OnLearn(const LearnEvent& event)
{
    anet::ProfileRange r("EpisodeEvalObserver::OnPostUpdate");

	/// @todo メトリクスのSTEP軸を指定できるようにする
    auto step = event.counts.GetByAxis(anet::rl::StepAxis::LEARN);

    // 評価エピソードを終端まで回す
    if (step % eval_interval_ == 0) {
		StepCounts counts;
        auto state = env_->Reset(runmode_);
        auto eps_total_reward = 0.0f;
        bool done = false;
        bool truncated = false;
        do {
            auto action = event.agent->MakeAction(counts, state, runmode_);
            auto env_result = env_->Step(action.action);
            eps_total_reward += env_result.reward.mean().item<float>();
            state = env_result.continue_state;
            done = env_result.next_state.IsDone();
            truncated = env_result.next_state.IsTruncated();
                
            counts.train_step++;
        } while (!done && !truncated);

        this->report_function_(eps_total_reward);
    }
}

std::string EpisodeEvalObserver::ToString() const
{
    auto mode_str = anet::rl::ToString(runmode_);
    return std::string("EpisodeEvalObserver[") + mode_str + "]";
}

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

// -------------------------------------------------------------

MetricsLogObserverBase::MetricsLogObserverBase(
    const std::string& tag,
    const std::string& key, anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field)
    : TaggedObserver(tag), key_(key), step_axis_(step_axis), event_field_(event_field)
{
    ;
}

MetricsLogTrainObserver::MetricsLogTrainObserver(const std::string& tag, const std::string& key,
    anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field)
    : MetricsLogObserverBase(tag, key, step_axis, event_field)
{
    ;
}

MetricsLogLearnObserver::MetricsLogLearnObserver(const std::string& tag, const std::string& key,
    anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field)
    : MetricsLogObserverBase(tag, key, step_axis, event_field)
{
    ;
}

std::optional<float> MetricsLogObserverBase::GetScalar(const UpdateEvent& event, anet::rl::EventField event_field)
{
    // Scalar取得対象
    const anet::DataExporter* target = nullptr;

    // 指定に従ってScalar取得対象を取得
    switch (event_field) {
    case anet::rl::EventField::BATCH_EXPERIENCE:
        target = &event.batch_exp;
        break;
    case anet::rl::EventField::AGENT:
        target = event.agent.get();
        break;
    case anet::rl::EventField::BATCH_UPDATE_RESULT:
        target = event.update_result.get();
        break;
    case anet::rl::EventField::TRAINER:
        target = &event.trainer;
        break;
    }

    if (target == nullptr)
        return std::nullopt;

    // Scalar値取得
    auto scalar_value = target->GetScalar(this->key_);
    return scalar_value;
}

void MetricsLogObserverBase::OnUpdate(const UpdateEvent& event)
{
    std::optional<float> value;
    if (event_field_.has_value()) {
        value = GetScalar(event, *event_field_);
    } else {
        value = GetScalar(event, anet::rl::EventField::BATCH_UPDATE_RESULT);
        if (!value.has_value()) value = GetScalar(event, anet::rl::EventField::AGENT);
        if (!value.has_value()) value = GetScalar(event, anet::rl::EventField::BATCH_EXPERIENCE);
        if (!value.has_value()) value = GetScalar(event, anet::rl::EventField::TRAINER);
    }

    if (!value.has_value()) {
        wxLogWarning("MetricsLogObserverBase::OnUpdate(): value not found. tag=%s key=%s",
            tag_, key_);
	}

    // Metricsログ出力
    if (value.has_value()) {
        MetricsLogger::Instance()->LogScalar(
            this->tag_,
            event.counts.GetByAxis(this->step_axis_),
            *value);
    }

    return;
}

// -------------------------------------------------------------

static constexpr const char* CONFIG_KEY_METRICS_SCALAR_PREFIX = "metrics.scalar.[";
static constexpr const char* CONFIG_KEY_METRICS_SCALAR_SUFFIX = "]";

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
            wxLogDebug("ObserverFactory: scalar: key=%s value=%s", scalar_metrics_tag, config_value);
            auto values = anet::Split(config_value, { " " }, true);

            // メトリクス定義情報を取得
            std::optional<std::string> key_opt;
            std::optional<anet::rl::EventType> event_opt;
            std::optional<anet::rl::StepAxis> step_axis_opt;
			std::optional<anet::rl::EventField> field_opt;
            for (auto v : values) {
                if (v == "@train") {
                    event_opt = EventType::TRAIN;
                } else if (v == "@learn") {
                    event_opt = EventType::LEARN;   /// @todo TRAINとLEARN以外の EventType も対応
                } else if (v == "&train_step" || v == "&train") {
                    step_axis_opt = StepAxis::TRAIN;
                } else if (v == "&learn_step" || v == "&learn_step") {
                    step_axis_opt = StepAxis::LEARN;
                } else if (v == "&episode_step" || v == "&episode") {
                    step_axis_opt = StepAxis::EPISODE;
                } else if (v == "&exp_step" || v == "&exp") {
                    step_axis_opt = StepAxis::EXP;
                } else if (v == "&update_step" || v == "&update_step") {
                    step_axis_opt = StepAxis::UPDATE;
                } else if (v == "&sim_step" || v == "&sim") {
                    step_axis_opt = StepAxis::SIM;
                } else if (v == "$agent") {
                    field_opt = EventField::AGENT;
                } else if (v == "$batch_experience" || v == "$exp") {
                    field_opt = EventField::BATCH_EXPERIENCE;
                } else if (v == "$batch_update_result" || v == "$update_result" || v == "$result") {
                    field_opt = EventField::BATCH_UPDATE_RESULT;
                } else if (v == "$trainer") {
                    field_opt = EventField::TRAINER;
                } else {
                    key_opt = v;
				}
            }

            if (!key_opt.has_value()) {
                wxLogError("ObserverFactory: key not found. config_key=%s config_value=%s",
                    config_value, config_value);
                continue;
            }

            auto key = *key_opt;
            auto event = event_opt.value_or(EventType::TRAIN);
			auto step_axis = step_axis_opt.value_or(
                (*event_opt == EventType::LEARN) ? StepAxis::LEARN : StepAxis::TRAIN);

            switch (event) {
            case EventType::TRAIN:
                {
                    auto train_obs = std::make_shared<MetricsLogTrainObserver>(scalar_metrics_tag, key, step_axis, field_opt);
                    train_observers_.push_back(train_obs);
                }
                break;
            case EventType::LEARN:
                {
                    auto learn_obs = std::make_shared<MetricsLogLearnObserver>(scalar_metrics_tag, key, step_axis, field_opt);
                    learn_observers_.push_back(learn_obs);
                }
                break;
            }
        }
 
        /// @todo intervalを設定化
		/// @todo EMAを設定化
        /// @todo HeatMap系Observerに対応
	}
}

