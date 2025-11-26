// image_log.cpp （抜粋）

#include "anet/probe.hpp"

namespace anet {

    std::optional<float> MetricsScalarProbe::GetFloat(
        int step,
        const anet::rl::Experience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    {
        return result->GetScalar(key_);  ///< @todo GetScalar 実装と合わせる
    }

    std::optional<float> StaticScalarProbe::GetFloat(
        int,
        const anet::rl::Experience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    {
        return value_;
    }

    //void TensorInputProbe::SetTensor(const torch::Tensor& t)
    //{
    //    latest_tensor_ = t;
    //    cached_value_.reset();

    //    if (!latest_tensor_.defined()) return;

    //    auto flat = latest_tensor_.flatten();
    //    const auto numel = flat.numel();

    //    if (index_ < 0 || index_ >= numel) return;

    //    cached_value_ = flat[index_].item<float>();
    //}

    //std::optional<float> TensorInputProbe::GetFloat(
    //    int step,
    //    const anet::rl::Experience& experience,
    //    std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    //{
    //    // 現状は SetTensor 経由でのみ更新する設計
    //    /// @todo 必要に応じて BatchExperience/BUR から Tensor を取得する経路を追加
    //}

    std::optional<float> FunctionFloatProbe::GetFloat(
        int step,
        const anet::rl::Experience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    {
        return fn_(step, experience, std::move(result));
    }

     StateAxisProbe::StateAxisProbe(int state_index, const anet::rl::StateSpec* spec, bool for_next_state)
        : state_index_(state_index), for_next_state_(for_next_state)
    {
        // EnvSpec の state_spec から min/max を取得
        if (spec != nullptr && state_index >= 0 && state_index < (int)spec->CalcFlattenSize()) {
            // 指定されたindexの定義情報を取得
            const anet::rl::StateDimInfo* s = spec->FindDim(state_index);

            // 定義情報を取得出来たらmin/maxをセット
            if (s != nullptr) {
                min_ = s->min_value;
                max_ = s->max_value;
            }
        }
    }

    std::optional<float> StateAxisProbe::GetFloat(
        int step,
        const anet::rl::Experience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    {
        const auto& flat = experience.next_state.Flatten();
        ANET_CHECK_SHAPE(flat, { ANET_SHAPE_ANY });     // (N)
        int64_t numel = flat.numel();
        if (state_index_ < 0 || state_index_ >= numel) return std::nullopt;

        return flat[state_index_].item<float>();
    }

    RewardProbe::RewardProbe(const anet::rl::EnvSpec* spec)
    {
        if (spec != nullptr) {
            min_ = spec->reward_range.first;
            max_ = spec->reward_range.second;
        }
    }

    std::optional<float> RewardProbe::GetFloat(
        int step,
        const anet::rl::Experience& experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result) const
    {
        return experience.reward;
    }

    RLStateSweepProcessor::RLStateSweepProcessor(
        const anet::rl::StateSpec& state_spec,
        int x_index,
        int y_index,
        ValueExtractFunction value_extract_fn,
        const torch::Device& device,
        std::optional<torch::Tensor> base_state,
        std::optional<float> x_min_override, std::optional<float> x_max_override,
        std::optional<float> y_min_override, std::optional<float> y_max_override)
        : state_spec_(state_spec)
        , device_(device)
        , x_index_(x_index)
        , y_index_(y_index)
        , value_extract_fn_(value_extract_fn)
    {
        int64_t flat_size = state_spec_.CalcFlattenSize();

        if (base_state.has_value()) {
            base_flatten_ = base_state.value().clone();
        } else {
            base_flatten_ = torch::zeros({ flat_size }, torch::kFloat32);
        }

        // min/max：StateSpec から
        const auto* dx = state_spec_.FindDim(x_index_);
        const auto* dy = state_spec_.FindDim(y_index_);

        float xs_min = dx ? dx->min_value : -1.0f;
        float xs_max = dx ? dx->max_value : 1.0f;
        float ys_min = dy ? dy->min_value : -1.0f;
        float ys_max = dy ? dy->max_value : 1.0f;

        x_min_overridden_ = x_min_override.has_value();
        x_max_overridden_ = x_max_override.has_value();
        y_min_overridden_ = y_min_override.has_value();
        y_max_overridden_ = y_max_override.has_value();

        x_min_ = x_min_override.value_or(xs_min);
        x_max_ = x_max_override.value_or(xs_max);
        y_min_ = y_min_override.value_or(ys_min);
        y_max_ = y_max_override.value_or(ys_max);
    }

    // ===========================================================
    // ISweepInputGenerator
    // ===========================================================

    void RLStateSweepProcessor::ApplyGridSize(int width, int height)
    {
        if (width > 0) grid_w_ = width;
        if (height> 0) grid_h_ = height;
    }

    std::pair<int, int> RLStateSweepProcessor::GetGridSize() const
    {
        return { grid_w_, grid_h_ };
    }

    torch::Tensor RLStateSweepProcessor::BuildInputTensor()
    {
        ANET_ASSERT(grid_w_ > 0);
        ANET_ASSERT(grid_h_ > 0);

        const int64_t flat_size = base_flatten_.size(0);
        const int64_t grid_num = static_cast<int64_t>(grid_w_) * static_cast<int64_t>(grid_h_);

        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

        // base_flatten_ を device に移動し、全セル分に複製
        torch::Tensor base = base_flatten_.to(opts);
        torch::Tensor batch = base.unsqueeze(0).repeat({ grid_num, 1 });

        // X軸補間値作成
        torch::Tensor xs;
        if (grid_w_ > 1) {
            xs = torch::linspace(0.0f, 1.0f, grid_w_, opts);
        } else {
            xs = torch::zeros({ 1 }, opts);
        }
        torch::Tensor xv = x_min_ + xs * (x_max_ - x_min_);
        xv = xv.repeat({ grid_h_ });

        // Y軸補間値作成
        torch::Tensor ys;
        if (grid_h_ > 1) {
            ys = torch::linspace(0.0f, 1.0f, grid_h_, opts);
        } else {
            ys = torch::zeros({ 1 }, opts);
        }
        torch::Tensor yv = y_min_ + ys * (y_max_ - y_min_);
        yv = yv.repeat_interleave(grid_w_);

        // 行インデックス [0 .. grid_num-1]
        torch::Tensor idx = torch::arange(
            grid_num, torch::TensorOptions().dtype(torch::kLong).device(device_));

        // 指定された state index へ X/Y 値を上書き
        batch.index_put_({ idx, static_cast<int64_t>(x_index_) }, xv);
        batch.index_put_({ idx, static_cast<int64_t>(y_index_) }, yv);

        // Shape 検証
        ANET_CHECK_SHAPE(batch, { grid_num, flat_size });

        return batch;  // [W*H, flat_size] on device_
    }

    int64_t RLStateSweepProcessor::GetFlattenSize() const
    {
        return base_flatten_.size(0);
    }

    ExtractResult RLStateSweepProcessor::Extract(const torch::Tensor& batched_out,
        const std::unordered_set<std::string>& required_labels)
    {
        const int64_t grid_num = static_cast<int64_t>(grid_w_) * static_cast<int64_t>(grid_h_);
        ANET_CHECK_SHAPE(batched_out, { grid_num, ANET_SHAPE_ENDANY });

        auto extract_result = value_extract_fn_(batched_out, required_labels);
        ANET_CHECK_SHAPE(extract_result.grid, { grid_num });

        return extract_result;
    }

    //struct ExtractResult {
    //    torch::Tensor grid;                          // HeatMap 用
    //    std::vector<std::string> labels;             // それぞれの scalar 名
    //    std::vector<torch::Tensor> scalars;          // 個別 scalar 値（GPU Tensor）
    //};

    // ==== Extractors

    namespace extractor {

        // -------

        template <class Map, class Key>
        bool map_contains(const Map& m, const Key& k) {
            return m.find(k) != m.end();
        }

        void push_scalar(std::vector<std::string>& labels, std::vector<torch::Tensor>& scalars,
            const std::string& label, const torch::Tensor& scalar_tensor)
        {
            labels.push_back(label);
            scalars.push_back(scalar_tensor);
        }

        // -------

        ExtractResult MaxExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req) { // t: [W*H, out_dim]
            auto grid = std::get<0>(t.max(1)); // [W*H]
            auto max = grid.max();
            return { grid,
                { "max" },
                { max }
            };
        }
        ExtractResult MeanExtractor(
            const torch::Tensor& t, const std::unordered_set<std::string>& req) {
            return { t.mean(1) }; // [W*H]
        }
        ExtractResult IndexExtractor(
            const torch::Tensor& t, const std::unordered_set<std::string>& req, int idx) {
            return { t.index({ torch::indexing::Slice(), idx }) };
        }
        ExtractResult DiffIndexExtractor(
            const torch::Tensor& t, const std::unordered_set<std::string>& req, int plus_idx, int minus_idx) {
            using namespace torch::indexing;
            auto plus_val = t.index({ Slice(), plus_idx });
            auto minus_va = t.index({ Slice(), minus_idx });
            return { plus_val - minus_va };
        }
        ExtractResult PairDiffExtractor(
            const torch::Tensor& t, const std::unordered_set<std::string>& req, int n_actions) {
            using namespace torch::indexing;
            ANET_CHECK_SHAPE(t, { ANET_SHAPE_ANY, n_actions * 2 });
            auto q_online = t.index({ Slice(), Slice(0, n_actions) });               // [N, n_actions]
            auto q_target = t.index({ Slice(), Slice(n_actions, n_actions * 2) });   // [N, n_actions]
            auto diff = (q_online - q_target).abs().mean(1);                         // [N]
            return { diff };
        }
        ExtractResult QdeltaQmaxCombined(
            const torch::Tensor& t, const std::unordered_set<std::string>& req,
            int n_actions,
            float qdelta_scale,      // ex: 0.5f
            float qmax_threshold)    // ex: 20.0f
        {
            using namespace torch::indexing;

            auto q_online = t.index({ Slice(), Slice(0, n_actions) });
            auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

            // Qdelta = |online - target| の平均
            auto qdelta = (q_online - q_target).abs().mean(1);       // [N]

            // Qmax = |online| の最大値
            auto qmax = std::get<0>(q_online.abs().max(1));       // [N]

            // 正規化（0〜1）
            auto qdelta_norm = (qdelta / qdelta_scale).clamp(0.0f, 1.0f);
            auto qmax_norm = (qmax / qmax_threshold).clamp(0.0f, 1.0f);

            // 合成
            auto combined = qdelta_norm * qmax_norm;
            return { combined };
        }
        // =============================
        // QdeltaQmaxCombinedAuto
        // -----------------
        //    Qdelta 高 × Qmax 高
        //    → 発散で地形が壊れて target 追従不能の領域
        //    Qdelta 高 × Qmax 低
        //    → target追従不足（でも発散ではない）
        //    Qdelta 低 × Qmax 高
        //    → Qの発散だけが起きている領域（target が遅れて青くなる前兆）
        //    両方低
        //    → 安定
        // -----------------
        //    発散領域（本当に最悪の赤）
        //    → 真っ赤に浮かび上がる
        //    （Qdelta_norm ≈ 1＆Qmax_norm ≈ 1）
        //    Qmax が高いのに Qdelta はまだ小さい（発散初期段階）
        //    → 暗オレンジ色に現れる
        //    → 崩壊の前兆が見える
        //    赤いけれど発散ではない Qdelta の赤
        //    → 黄色〜緑程度で止まる
        //    Qdelta が高いが Qmax はまだ青い（target追従だけ遅れ）
        //    → 緑〜黄に現れる
        // =============================
        ExtractResult QdeltaQmaxCombinedAuto(
            const torch::Tensor& t, const std::unordered_set<std::string>& req,
            int n_actions)
        {
            using namespace torch::indexing;

            auto q_online = t.index({ Slice(), Slice(0, n_actions) });
            auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

            auto qdelta = (q_online - q_target).abs().mean(1);
            auto qmax = std::get<0>(q_online.abs().max(1));

            // GPU 上の max（scalar-tensor）
            auto qdelta_max = qdelta.max();  // shape [], device same as qdelta
            auto qmax_max = qmax.max();    // shape [], device same as qmax

            // EPS を GPU Tensor で作る
            auto eps = torch::full({}, 1e-6, qdelta.options());   // shape=[]

            // GPU同士の除算 → 完全GPU処理
            auto qdelta_norm = qdelta / (qdelta_max + eps);
            auto qmax_norm = qmax / (qmax_max + eps);

            return { qdelta_norm * qmax_norm };
        }
        /// QDELTA × |Qdiff|
        ExtractResult QdeltaQdiffCombinedAuto(
            const torch::Tensor& t, const std::unordered_set<std::string>& req,
            int n_actions,
            int action_index_a,
            int action_index_b)
        {
            using namespace torch::indexing;

            ANET_ASSERT(t.dim() == 2);
            ANET_ASSERT(n_actions > 0);
            ANET_ASSERT(t.size(1) == n_actions * 2);
            ANET_ASSERT(action_index_a >= 0 && action_index_a < n_actions);
            ANET_ASSERT(action_index_b >= 0 && action_index_b < n_actions);

            // t: [N, 2 * n_actions] = [q_online, q_target]
            auto q_online = t.index({ Slice(), Slice(0, n_actions) });
            auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

            // Qdelta(s) = mean_a |Q_online(s,a) - Q_target(s,a)|
            auto qdelta = (q_online - q_target).abs().mean(1);  // [N]

            // Qdiff(s) = Q_online(s, b) - Q_online(s, a)
            auto qdiff = q_online.index({ Slice(), action_index_b })
                - q_online.index({ Slice(), action_index_a }); // [N]
            auto qdiff_abs = qdiff.abs();                      // [N]

            // GPU 上での max（0 次元 Tensor）
            auto qdelta_max = qdelta.max();    // []
            auto qdiff_max = qdiff_abs.max();  // []

            // EPS を GPU Tensor で生成
            auto eps = torch::full({}, 1e-6f, t.options());

            // 自動正規化（0〜1）
            auto qdelta_norm = qdelta / (qdelta_max + eps);
            auto qdiff_norm = qdiff_abs / (qdiff_max + eps);

            // 合成: target 乖離 × 境界ゆらぎ
            auto combined = qdelta_norm * qdiff_norm;          // [N]

            return { combined };
        }
        ExtractResult BoundaryMaskFromQdiffAuto(
            const torch::Tensor& t, const std::unordered_set<std::string>& req,
            int n_actions,
            int action_index_a,
            int action_index_b)
        {
            using namespace torch::indexing;

            ANET_ASSERT(t.dim() == 2);
            ANET_ASSERT(n_actions > 0);
            ANET_ASSERT(t.size(1) == n_actions);
            ANET_ASSERT(action_index_a >= 0 && action_index_a < n_actions);
            ANET_ASSERT(action_index_b >= 0 && action_index_b < n_actions);

            // t: [N, 2 * n_actions] = [q_online, q_target]
            auto q_online = t.index({ Slice(), Slice(0, n_actions) });

            // Qdiff(s) = Q_online(s, b) - Q_online(s, a)
            auto qdiff = q_online.index({ Slice(), action_index_b }) -
                q_online.index({ Slice(), action_index_a });  // [N]
            auto qdiff_abs = qdiff.abs();                             // [N]

            // GPU 上で max を取得（0 次元 Tensor）
            auto qdiff_max = qdiff_abs.max();                         // []

            // EPS を GPU Tensor として生成
            auto eps = torch::full({}, 1e-6f, t.options());

            // 正規化: 0〜1 （境界からの距離）
            auto qdiff_norm = qdiff_abs / (qdiff_max + eps);          // [N], 0〜1

            // 境界強度: 境界付近ほど 1.0、遠いほど 0.0
            auto boundary_strength = 1.0f - qdiff_norm;               // [N]

            return { boundary_strength };
        }

        ExtractResult BoundaryMaskedQdeltaAuto(
            const torch::Tensor& t, const std::unordered_set<std::string>& req,
            int n_actions,
            int action_index_a,
            int action_index_b)
        {
            using namespace torch::indexing;

            ANET_ASSERT(t.dim() == 2);
            ANET_ASSERT(n_actions > 0);
            ANET_ASSERT(t.size(1) == n_actions * 2);
            ANET_ASSERT(action_index_a >= 0 && action_index_a < n_actions);
            ANET_ASSERT(action_index_b >= 0 && action_index_b < n_actions);

            // t: [N, 2 * n_actions] = [q_online, q_target]
            auto q_online = t.index({ Slice(), Slice(0, n_actions) });
            auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

            // Qdelta(s) = mean_a |Q_online - Q_target|
            auto qdelta = (q_online - q_target).abs().mean(1);   // [N]

            // Qdiff(s) = Q_online(s,b) - Q_online(s,a)
            auto qdiff = q_online.index({ Slice(), action_index_b })
                - q_online.index({ Slice(), action_index_a });    // [N]
            auto qdiff_abs = qdiff.abs();

            // 正規化用 max
            auto qdelta_max = qdelta.max();
            auto qdiff_max = qdiff_abs.max();

            auto eps = torch::full({}, 1e-6f, t.options());

            // 0〜1 に normalize
            auto qdelta_norm = qdelta / (qdelta_max + eps);
            auto qdiff_norm = qdiff_abs / (qdiff_max + eps);

            // 境界強度（1が境界付近）
            auto boundary_strength = 1.0f - qdiff_norm;          // [N]

            // HeatMap 用（正規化された combined）
            auto combined = boundary_strength * qdelta_norm;     // [N]

            std::vector<std::string> labels;
            std::vector<torch::Tensor> scalars;

            if (map_contains(req, "raw_qdelta_mean")) {
                push_scalar(labels, scalars, "raw_qdelta_mean", qdelta.mean());
            }
            if (map_contains(req, "raw_qdelta_max")) {
                push_scalar(labels, scalars, "raw_qdelta_max", qdelta.max());
            }
            if (map_contains(req, "raw_boundary_mean")) {
                push_scalar(labels, scalars, "raw_boundary_mean", boundary_strength.mean());
            }
            if (map_contains(req, "boundary_area")) {
                push_scalar(labels, scalars, "boundary_area", (boundary_strength > 0.5f).sum());
            }
            if (map_contains(req, "combined_mean")) {
                push_scalar(labels, scalars, "combined_mean", combined.mean());
            }
            if (map_contains(req, "combined_max")) {
                push_scalar(labels, scalars, "combined_max", combined.max());
            }

            return { combined, labels, scalars };
        }
    }

}
