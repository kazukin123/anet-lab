#include "anet/observers.hpp"
#include <wx/log.h>
#include "anet/profile.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/env.hpp"

namespace anet::rl {

    void MetricsLogObserver::OnPostUpdate(
        int step,
        std::shared_ptr<Agent> agent,
        const anet::rl::BatchExperience& experiences,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result
    )
    {
        auto map = result->GetMetricsMap();
        for (const auto& [tag, value] : map) {
            MetricsLogger::Instance()->LogScalar(tag, step, value);
        }
    }

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
        : config_(config), tag_(tag),
        x_probe_(x_probe), y_probe_(y_probe), value_probe_(value_probe)
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

    void HeatMapVectorObserver::OnPostUpdate(
        int step,
        std::shared_ptr<Agent> agent,
        const anet::rl::BatchExperience& batch_exp,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result)
    {
        anet::ProfileRange r("HeatMapVectorObserver::OnPostUpdate");

        // 生成： xv, yv, vv
        auto xv = x_probe_->GetVector(step, agent, batch_exp, result);
        auto yv = y_probe_->GetVector(step, agent, batch_exp, result);
        auto vv = value_probe_->GetVector(step, agent, batch_exp, result);

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
    }

    MultiPairHeatMapObserver::MultiPairHeatMapObserver(
        const std::string& tag,
        const HeatMapObserverConfig& config,
        const std::vector<std::shared_ptr<VectorProbe>>& axis_probes,
        std::shared_ptr<VectorProbe> value_probe)
        : tag_(tag), config_(config), axis_probes_(axis_probes), value_probe_(value_probe)
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

    void MultiPairHeatMapObserver::OnPostUpdate(
        int step,
        std::shared_ptr<Agent> agent,
        const BatchExperience& batch_exp,
        std::shared_ptr<const BatchUpdateResult> result)
    {
        anet::ProfileRange r("MultiPairHeatMapObserver::OnPostUpdate");

        // 値ベクトル
        auto vv = value_probe_->GetVector(step, agent, batch_exp, result);
        if (!vv) return;

        // 全プローブペア i<j をスキャン
        const size_t m = axis_probes_.size();

        for (size_t i = 0; i < m; i++) {
            auto xv = axis_probes_[i]->GetVector(step, agent, batch_exp, result);
            if (!xv) continue;

            auto xmin = axis_probes_[i]->GetMin();
            auto xmax = axis_probes_[i]->GetMax();

            for (size_t j = i + 1; j < m; j++) {
                auto yv = axis_probes_[j]->GetVector(step, agent, batch_exp, result);
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
    }

    SweepedHeatMapObserver::SweepedHeatMapObserver(
        const std::string& heatmap_tag,
        const SweepedHeatMapObserverConfig& config,
        std::shared_ptr<ISweepInputGenerator> input_gen,
        TensorFunction tensor_fn,
        std::shared_ptr<ISweepOutputExtractor> output_ext,
        const std::unordered_map<std::string, std::string>& scalar_tag_label_map)
        : heatmap_tag_(heatmap_tag), config_(config),
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

    void SweepedHeatMapObserver::OnPostUpdate(
        int step,
        std::shared_ptr<Agent> agent,
        const anet::rl::BatchExperience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result)
    {
        anet::ProfileRange r("SweepedHeatMapObserver::OnPostUpdate");

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
            anet::ToDefString(extract_result.grid), heatmap_tag_);
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
            heatmap_tag_,
            step,
            *heatmap_,
            config_.image_width,
            config_.image_height);
        wxLogDebug("SweepedHeatMapObserver::OnPostUpdate() LogImage() done. tag=%s", heatmap_tag_);

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
    }

    EpisodeEvalObserver::EpisodeEvalObserver(
        const std::string& tag,
        std::shared_ptr<anet::rl::SingleDiscreteEnvFactory> eval_env_factory,
		const ConfigData& config_data,
        const torch::Device& device,
        anet::rl::RunMode runmode, int log_interval, int eval_inerval, float ema_decay)
        : tag_(tag), runmode_(runmode), log_interval_(log_interval), eval_interval_(eval_inerval), eval_total_reward_(ema_decay)
    {
        env_ = std::make_unique<VectorizedDiscreteBatchEnv>(config_data, eval_env_factory, 1, device);
    }

    void EpisodeEvalObserver::OnPostUpdate(
        int step,
        std::shared_ptr<Agent> agent,
        const BatchExperience& batch_exp,
        std::shared_ptr<const BatchUpdateResult> result)
    {
        anet::ProfileRange r("EpisodeEvalObserver::OnPostUpdate");

        // 評価エピソードを終端まで回す
        if (step % eval_interval_ == 0) {
            auto state = env_->Reset(runmode_);
            auto eps_total_reward = 0.0f;
            bool done = false;
            bool truncated = false;
            do {
                auto action = agent->MakeAction(state, runmode_);
                auto env_result = env_->Step(action.action);
                eps_total_reward += env_result.reward.mean().item<float>();
                state = env_result.continue_state;
                done = env_result.next_state.IsDone();
                truncated = env_result.next_state.IsTruncated();
            } while (!done && !truncated);
            eval_total_reward_.Update(eps_total_reward);
        }

        // 評価エピソードのトータル報酬をメトリクスとして出力
        if (step % log_interval_ == 0) {
            MetricsLogger::Instance()->LogScalar(tag_, step, eval_total_reward_.Value());
        }
    }

}