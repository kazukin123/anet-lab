#include <wx/log.h>
#include "anet/observers.hpp"
#include "anet/metrics_logger.hpp"

namespace anet::rl {

    void MetricsLogObserver::OnPostUpdate(
        int step,
        const anet::rl::BatchExperience& experiences,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result
    )
    {
        auto map = result->GetMetricsMap();
        for (const auto& [tag, value] : map) {
            MetricsLogger::Instance()->LogScalar(tag, step, value);
        }
    }

    static float ResolveMin(
        bool override_flag, float override_v, std::optional<float> probe_v, float fallback)
    {
        if (override_flag) return override_v;
        if (probe_v.has_value()) return *probe_v;
        return fallback;
    }

    static float ResolveMax(
        bool override_flag, float override_v, std::optional<float> probe_v, float fallback)
    {
        if (override_flag) return override_v;
        if (probe_v.has_value()) return *probe_v;
        return fallback;
    }

    HeatMapObserver::HeatMapObserver(
        const std::string& tag,
        const HeatMapObserverConfig& config,
        std::shared_ptr<IFloatProbe> x_probe,
        std::shared_ptr<IFloatProbe> y_probe,
        std::shared_ptr<IFloatProbe>  value_probe)
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

    void HeatMapObserver::OnPostUpdate(
        int step,
        const anet::rl::BatchExperience& batch_experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result)
    {
        // N 環境ぶんループ（Probe が Update を受けてから TryGetFloat）
        //const int N = experience.state.size(0);

        auto exp_list = batch_experience.ToExperienceList();
        for (auto exp : exp_list) {
            // 生成： xv, yv, vv
            auto xv = x_probe_->GetFloat(step, exp, result);
            auto yv = y_probe_->GetFloat(step, exp, result);
            auto vv = value_probe_->GetFloat(step, exp, result);

            // 値が揃ってなかったらスキップ
            if (!xv.has_value() || !yv.has_value() || !vv.has_value())
                continue;

            // データ追加
            heatmap_->AddData(*xv, *yv, *vv);
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

    SweepedHeatMapObserver::SweepedHeatMapObserver(
        const std::string& tag,
        const SweepedHeatMapObserverConfig& config,
        std::shared_ptr<ISweepInputGenerator> input_gen,
        TensorFunction tensor_fn,
        std::shared_ptr<ISweepOutputExtractor> output_ext)
        : tag_(tag), config_(config),
        input_gen_(input_gen),
        tensor_fn_(std::move(tensor_fn)),
        output_ext_(output_ext)
    {
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
        const anet::rl::BatchExperience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result)
    {
        if (step % config_.log_interval != 0) return;

        const int64_t grid_num = static_cast<int64_t>(grid_w_) * static_cast<int64_t>(grid_h_);

        // -----------------------------
        // 入力バッチ生成（GPU 上）
        // -----------------------------
        torch::Tensor batch_in = input_gen_->BuildInputTensor();
        ANET_CHECK_SHAPE(batch_in, { grid_num, ANET_SHAPE_ENDANY });
        wxLogDebug(
            "SweepedHeatMapObserver::OnPostUpdate() batch_in=%s",
            anet::ToDefString(batch_in));

        // -----------------------------
        // NN 適用（GPU 上）
        // -----------------------------
        torch::Tensor batch_out = tensor_fn_(batch_in);
        ANET_CHECK_SHAPE(batch_out, { grid_num, ANET_SHAPE_ENDANY });
        wxLogDebug(
            "SweepedHeatMapObserver::OnPostUpdate() batch_out=%s",
            anet::ToDefString(batch_out));

        // -----------------------------
        // 出力から値抽出（GPU 上, [W*H]）
        // -----------------------------
        torch::Tensor grid_values = output_ext_->ExtractValue(batch_out);
        ANET_CHECK_SHAPE(grid_values, { grid_num });
        ANET_CHECK_DTYPE(grid_values, torch::kFloat32);
        wxLogDebug(
            "SweepedHeatMapObserver::OnPostUpdate() grid_values=%s",
            anet::ToDefString(grid_values));

        // -----------------------------
        // CPU へ一括転送して HeatMap に投入
        // -----------------------------
        torch::Tensor grid_cpu = grid_values.to(torch::kCPU);
        ANET_CHECK_SHAPE(grid_cpu, { grid_num });
        ANET_CHECK_DTYPE(grid_cpu, torch::kFloat32);
        float* data = grid_cpu.data_ptr<float>();
        heatmap_->SetGridValues(data, grid_w_, grid_h_);
        wxLogDebug("SweepedHeatMapObserver::OnPostUpdate() SetGridValues() done.");

        // -----------------------------
        // 画像ログ出力
        // -----------------------------
        MetricsLogger::Instance()->LogImage(
            tag_,
            step,
            *heatmap_,
            config_.image_width,
            config_.image_height);
        wxLogDebug("SweepedHeatMapObserver::OnPostUpdate() LogImage() done.");
    }


/*
    static float ResolveAxis(
        bool override_flag,
        float override_v,
        float fallback)
    {
        if (override_flag) return override_v;
        return fallback;
    }

    SweepedHeatMapObserver::SweepedHeatMapObserver(
        const std::string& tag,
        const SweepedHeatMapObserverConfig& config,
        SweepInputGenerator* input_gen,
        SweepOutputExtractor* output_ext,
        ForwardFn forward_fn)
        : tag_(tag),
        config_(config),
        forward_fn_(forward_fn)
    {
        input_gen_.reset(input_gen);
        output_ext_.reset(output_ext);

        float xmin = ResolveAxis(config_.override_xmin, config_.xmin, config_.xmin);
        float xmax = ResolveAxis(config_.override_xmax, config_.xmax, config_.xmax);
        float ymin = ResolveAxis(config_.override_ymin, config_.ymin, config_.ymin);
        float ymax = ResolveAxis(config_.override_ymax, config_.ymax, config_.ymax);

        grid_ = std::make_unique<anet::SweepedHeatMap>(
            config_.width,
            config_.height,
            xmin, xmax,
            ymin, ymax
        );
    }

    void SweepedHeatMapObserver::OnPostUpdate(
        int step,
        const anet::rl::BatchExperience& experiences,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result)
    {
        if (step % config_.log_interval != 0)
            return;

        const int W = config_.width;
        const int H = config_.height;

        std::vector<float> xs(W);
        std::vector<float> ys(H);

        for (int ix = 0; ix < W; ix++) {
            xs[ix] = config_.xmin +
                (config_.xmax - config_.xmin) * (float(ix) / float(W - 1));
        }

        for (int iy = 0; iy < H; iy++) {
            ys[iy] = config_.ymin +
                (config_.ymax - config_.ymin) * (float(iy) / float(H - 1));
        }

        // NNをバッチ駆動
        torch::Tensor batch_input = input_gen_->BuildBatchInput(xs, ys, config_.x_index, config_.y_index);
        torch::Tensor batch_output = forward_fn_(batch_input);
        torch::Tensor value_grid = output_ext_->ExtractValue(batch_output, H, W);

        // HeatMapに反映
        grid_->SetValues(value_grid);

        MetricsLogger::Instance()->LogImage(
            tag_,
            step,
            *grid_,
            config_.image_width,
            config_.image_height
        );
    }
*/
}