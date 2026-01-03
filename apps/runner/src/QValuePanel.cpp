// QValuePanel.cpp
#include "QValuePanel.hpp"

#include <algorithm>
#include <wx/dcbuffer.h>
#include "anet/config.hpp"
#include "anet/observers.hpp"
#include "anet/heat_map.hpp"


//static constexpr int kRowHeight = 24;   ///< 行の高さ（GridとHeatMapの同期用）
static constexpr int kActionNameColWidth = 100;
static constexpr int kActionNameColMinWidth = 50;
static constexpr int kColWidth = 80;
static constexpr int kColMinWidth = 70;
static constexpr bool kHistDefaultCheck = true;
static constexpr bool kAdvDefaultCheck = true;
static constexpr bool kLogScaleDefaultCheck = true;


// ----------------------------------------------------------------------------
// QValuePanel::Config
// ----------------------------------------------------------------------------

struct QValuePanel::Config : public anet::Config {
    int row_height = 24;
    bool use_hist_smooth = true;
    float smooth_radius = 4.0f;
    int hist_bins = 510;
    float hist_range_k = 1.0;

    QValuePanel::Config(const anet::ConfigData& config_data) : anet::Config("QValuePanel")
    {
        ANET_READ_CONFIG(config_data, row_height);
        ANET_READ_CONFIG(config_data, hist_bins);
        ANET_READ_CONFIG(config_data, hist_range_k);
        ANET_READ_CONFIG(config_data, use_hist_smooth);
        ANET_READ_CONFIG(config_data, smooth_radius);
    }
};


// ----------------------------------------------------------------------------
// QValueHeatMapPanel
// ----------------------------------------------------------------------------

QValueHeatMapPanel::QValueHeatMapPanel(wxWindow* parent)
    : anet::rl::gui::Panel(parent, wxID_ANY)
{
    // ちらつき防止: 背景スタイルを設定
    SetBackgroundStyle(wxBG_STYLE_PAINT);

    // Bind形式でのイベント設定
    Bind(wxEVT_PAINT, &QValueHeatMapPanel::OnPaint, this);
    Bind(wxEVT_ERASE_BACKGROUND, &QValueHeatMapPanel::OnEraseBackground, this);
}

//void QValueHeatMapPanel::UpdateHeatMap(const wxImage& heatmap_image, int y_offset, int target_height)
void QValueHeatMapPanel::UpdateHeatMap(const wxImage& heatmap_image, int y_offset, int target_height, float guide_line_pos)
{
    heatmap_image_ = heatmap_image;
    y_offset_ = y_offset;
    target_height_ = target_height;
    guide_line_pos_ = guide_line_pos; // 位置を保存
    Refresh(); // OnPaintをトリガー
}

void QValueHeatMapPanel::OnEraseBackground(wxEraseEvent& event)
{
    // 【重要】ダブルバッファリング時のちらつき防止
    // デフォルト処理（背景を白で塗りつぶす等）を行わせないために
    // 何もせず、event.Skip() も呼ばない。
}

void QValueHeatMapPanel::OnPaint(wxPaintEvent& event)
{
    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    if (!heatmap_image_.IsOk()) return;

    // パネルの現在のクライアント幅を取得
    int width = GetClientSize().GetWidth();

    // 幅が0以下や画像がない場合は描画しない
    if (width > 0) {
        // ウィンドウ幅に合わせて画像をリサイズ
        //wxImage scaled_img = heatmap_image_.Scale(width, target_height_);
        //wxImage scaled_img = heatmap_image_.Scale(width, target_height_, wxIMAGE_QUALITY_HIGH);
        wxImage scaled_img = heatmap_image_.Scale(width, target_height_, wxIMAGE_QUALITY_NEAREST);

        // 描画
        dc.DrawBitmap(wxBitmap(scaled_img), 0, y_offset_, true);

        // センターライン（ガイドライン）の描画
        if (guide_line_pos_ >= 0.0f && guide_line_pos_ <= 1.0f) {
            int line_x = static_cast<int>(guide_line_pos_ * width);

            // シアン色 (0, 255, 255), 太さ2px
            wxPen pen(wxColour(0, 255, 255), 2, wxPENSTYLE_SOLID);
            dc.SetPen(pen);

            // 縦線を引く
            dc.DrawLine(line_x, y_offset_, line_x, y_offset_ + target_height_);
        }
    }
}

// ----------------------------------------------------------------------------
// QValuePanel Implementation
// ----------------------------------------------------------------------------

QValuePanel::QValuePanel(wxWindow* parent, const anet::ConfigData& config_data)
    : anet::rl::gui::Panel(parent, wxID_ANY)
{
    // Config
    config_ = std::make_unique<QValuePanel::Config>(config_data);

    // ガウス分布風の重みテーブルを事前に作っておく（高速化）
    if (config_->use_hist_smooth && config_->smooth_radius > 0) {
        hist_weights_.resize(config_->smooth_radius * 2 + 1);
        float sigma = static_cast<float>(config_->smooth_radius) / 2.0f; // σ調整
        for (int r = -config_->smooth_radius; r <= config_->smooth_radius; ++r) {
            // e^(-x^2 / 2σ^2)
            float w = std::exp(-(r * r) / (2 * sigma * sigma));
            hist_weights_[r + config_->smooth_radius] = w;
        }
    }

    // UIレイアウト初期化
    InitLayout();

    // 必要に応じてパネル全体のリサイズイベントもBind可能
    Bind(wxEVT_SIZE, &QValuePanel::OnSize, this);
    Bind(wxEVT_CLOSE_WINDOW, &QValuePanel::OnCloseWindow, this);
}

void QValuePanel::InitLayout()
{
    auto* main_sizer = new wxBoxSizer(wxHORIZONTAL);

    // ====================================================================
    // 左側: Grid
    // ====================================================================
    grid_ = new wxGrid(this, wxID_ANY);
    SetupGrid();
    main_sizer->Add(grid_, 0, wxEXPAND | wxALL, 5);

    // ====================================================================
    // 右側: 縦並び (ヘッダーエリア + ヒートマップエリア)
    // ====================================================================
    auto* right_sizer = new wxBoxSizer(wxVERTICAL);

    // --------------------------------------------------
    // 右上: ヘッダーエリア (チェックボックス配置)
    // --------------------------------------------------
    auto* header_panel = new wxPanel(this, wxID_ANY);
    header_panel->SetMinSize(wxSize(-1, config_->row_height));
    header_panel->SetMaxSize(wxSize(-1, config_->row_height));

    auto* header_sizer = new wxBoxSizer(wxHORIZONTAL);

    // Hist CheckBox
    hist_check_ = new wxCheckBox(header_panel, wxID_ANY, "Hist");
    hist_check_->SetValue(kHistDefaultCheck);
    hist_check_->Bind(wxEVT_CHECKBOX, &QValuePanel::OnCheck, this);
    header_sizer->Add(hist_check_, 0, wxALIGN_CENTER_VERTICAL | wxALL, 0);

    // Advantage CheckBox
    adv_check_ = new wxCheckBox(header_panel, wxID_ANY, "Advantage");
    adv_check_->SetValue(kAdvDefaultCheck);
    adv_check_->Bind(wxEVT_CHECKBOX, &QValuePanel::OnCheck, this);
    header_sizer->Add(adv_check_, 0, wxALIGN_CENTER_VERTICAL | wxALL, 0);

    // Log Scale CheckBox
    log_scale_check_ = new wxCheckBox(header_panel, wxID_ANY, "Log Scale");
    log_scale_check_->SetValue(kLogScaleDefaultCheck);
    log_scale_check_->Bind(wxEVT_CHECKBOX, &QValuePanel::OnCheck, this);
    header_sizer->Add(log_scale_check_, 0, wxALIGN_CENTER_VERTICAL | wxALL, 0);

    header_panel->SetSizer(header_sizer);
    right_sizer->Add(header_panel, 0, wxEXPAND | wxTOP | wxRIGHT, 5);

    // --------------------------------------------------
    // 右下: HeatMapパネル
    // --------------------------------------------------
    heatmap_panel_ = new QValueHeatMapPanel(this);
    right_sizer->Add(heatmap_panel_, 1, wxEXPAND | wxBOTTOM | wxRIGHT, 5);

    main_sizer->Add(right_sizer, 1, wxEXPAND, 0);
    SetSizer(main_sizer);
}

void QValuePanel::SetupGrid()
{
    grid_->CreateGrid(0, 5);
    grid_->SetColLabelValue(0, "Action");
    grid_->SetColLabelValue(1, "Mean");
    grid_->SetColLabelValue(2, "Std");
    grid_->SetColLabelValue(3, "Max");
    grid_->SetColLabelValue(4, "Min");

    grid_->SetDefaultRowSize(config_->row_height, true);
    grid_->SetColLabelSize(config_->row_height); // ヘッダー高さ
    grid_->SetRowLabelSize(30);
    grid_->DisableDragRowSize(); // 高さ同期のため固定
    grid_->EnableEditing(false);
    //grid_->SetSelectionMode(wxGrid::wxGridSelectNone);
    //grid_->SetCellHighlightPenWidth(0);
    //grid_->SetCellHighlightROPenWidth(0);
    grid_->SetColSize(0, kActionNameColWidth);
    grid_->SetColMinimalWidth(0, kActionNameColMinWidth);

    grid_->SetDefaultCellAlignment(wxALIGN_LEFT, wxALIGN_CENTER);
    for (int i = 1; i < 5; ++i) {
        wxGridCellAttr* align_attr = new wxGridCellAttr();
        align_attr->SetAlignment(wxALIGN_RIGHT, wxALIGN_CENTER);
        grid_->SetColAttr(i, align_attr);
        grid_->SetColSize(i, kColWidth);
        grid_->SetColMinimalWidth(i, kColMinWidth);
    }

    // セル選択を無効化
    grid_->DisableOverlaySelection();
    grid_->Bind(wxEVT_GRID_SELECT_CELL, [this](wxGridEvent& e) {
            e.Veto();              // 現在セルにしない（枠も動かない）
            grid_->ClearSelection();
        });
}

void QValuePanel::Initialize(const anet::rl::ActionSpec& action_spec, std::shared_ptr<anet::rl::Notifier> notifier)
{
    // アクション名
    action_names_.insert(action_names_.begin(), action_spec.value_labels.begin(), action_spec.value_labels.end());

    // Detach用にNotifierを保持
    notifier_ = notifier;

    // Notifier
    this->observer_ = notifier->Attach<anet::rl::FunctionTrainObserver>(
        [this](const anet::rl::TrainEvent& event)
        {
            /// @todo Eval想定で毎Step反映固定なのを再検討

            // データ作る
            auto data = CreateData(event);

            // データがなかったら反映しない
            if (!data.has_value()) return;

            // データ反映
            ApplyData(*data);
        },
        "QValuePanel");
}

std::optional<QValueData> QValuePanel::CreateData(const anet::rl::TrainEvent& event)
{
    QValueData data;
    const auto& aux_data = event.action_info.GetAuxData();

    auto q_values_itr = aux_data.find("q_values");
    if (q_values_itr == aux_data.end()) return std::nullopt;
    torch::Tensor q_values_plain = q_values_itr->second;

    auto q_quantiles_itr = aux_data.find("q_quantiles");
    torch::Tensor q_values = q_values_plain;
    if (q_quantiles_itr != aux_data.end()) q_values = q_quantiles_itr->second;

    ANET_LOG_DEBUG("q_values=" << anet::ToString(q_values));

    ANET_CHECK(q_values.size(0) > 0);
    //ANET_ASSERT_SHAPE(q_values, { ANET_SHAPE_ANY, static_cast<int64_t>(action_names_.size()), ANET_SHAPE_ENDANY });
    auto first_env_q = q_values[0]; // (N, A)
    ANET_LOG_DEBUG("first_env_q=" << anet::ToString(first_env_q));
    ANET_CHECK(first_env_q.size(0) == action_names_.size());
    auto n_actions = first_env_q.size(0);

    // 統計値生成
    data.stats.resize(n_actions);
    for (int i = 0; i < n_actions; i++) {
        data.stats[i].mean = first_env_q[i].mean().item<float>();
        data.stats[i].std_dev = first_env_q[i].std().item<float>();
        data.stats[i].max = first_env_q[i].max().item<float>();
        data.stats[i].min = first_env_q[i].min().item<float>();
    }

    int width = 0;
    if (first_env_q.sizes().size() == 1) { // PlainQ
        width = 1;
        data.xv.resize(n_actions);
        data.yv.resize(n_actions);
        data.vv.resize(n_actions);
    } else if (first_env_q.sizes().size() == 2) { // QR
        width = first_env_q.size(1);
        auto data_size = n_actions * width;
        data.xv.resize(data_size);
        data.yv.resize(data_size);
        data.vv.resize(data_size);
    }

    // HeatMap用データ生成
    if (first_env_q.sizes().size() == 1) { // PlainQ
        for (int y = 0; y < n_actions; y++) {
            auto val = first_env_q[y].item<float>();
            data.xv[y] = 0;
            data.yv[y] = y;
            data.vv[y] = val;
        }
    } else if (first_env_q.sizes().size() == 2) { // QR
        int i = 0;
        for (int y = 0; y < n_actions; y++) {
            for (int x = 0; x < width; x++) {
                auto val = first_env_q[y][x].item<float>();
                data.xv[i] = x;
                data.yv[i] = y;
                data.vv[i] = val;
                i++;
            }
        }
    }

    // データ設定
    data.width = width;
    data.height = n_actions;

    return data;
}

void QValuePanel::ApplyData(const QValueData& data)
{
    data_store_.Update(data);
    Update();
}

void QValuePanel::Update()
{
    auto data_opt = data_store_.Get();
    if (!data_opt.has_value()) return;

    // データコピー
    QValueData data = *data_opt;
    if (data.vv.empty()) return;

    // UI状態取得
    bool is_hist = IsHistogram();
    bool is_adv = IsAdvantage();
    bool is_log = IsLogScale();

    // --------------------------------------------------------
    // 0. 統計量計算 (GridとHeatMap共通で使用)
    // --------------------------------------------------------
    double sum = std::accumulate(data.vv.begin(), data.vv.end(), 0.0);
    double global_mean = sum / data.vv.size();

    // 分散・標準偏差計算
    double sq_sum = std::inner_product(data.vv.begin(), data.vv.end(), data.vv.begin(), 0.0);
    double global_stdev = std::sqrt(sq_sum / data.vv.size() - global_mean * global_mean);


    // --------------------------------------------------------
    // 1. Grid更新 (Advantage対応)
    // --------------------------------------------------------
    int current_rows = grid_->GetNumberRows();
    int new_rows = static_cast<int>(action_names_.size());


    // 最大のMeanを持つアクションを特定する
    float max_mean_val = -std::numeric_limits<float>::infinity();
    int max_mean_idx = -1;

    for (int i = 0; i < new_rows; ++i) {
        // 統計データが揃っていない場合のガード
        if (i < data.stats.size()) {
            if (data.stats[i].mean > max_mean_val) {
                max_mean_val = data.stats[i].mean;
                max_mean_idx = i;
            }
        }
    }

    grid_->BeginBatch();

    if (new_rows > current_rows) {
        grid_->AppendRows(new_rows - current_rows);
    } else if (new_rows < current_rows) {
        grid_->DeleteRows(new_rows, current_rows - new_rows);
    }

    // Grid表示用のオフセット値
    float grid_offset = is_adv ? static_cast<float>(global_mean) : 0.0f;

    const wxColour kHighlightColor(255, 225, 225); // 薄い赤 (ご希望通り)
    const wxColour kNormalColor(*wxWHITE);         // 通常色

    for (int i = 0; i < new_rows; ++i) {
        grid_->SetRowLabelValue(i, wxString::Format("%d", i));
        grid_->SetCellValue(i, 0, action_names_[i]);

        // AdvantageONなら、平均値を引いた値を表示して確認できるようにする
        // Std(ばらつき)はシフトしても変わらないのでそのまま
        grid_->SetCellValue(i, 1, wxString::Format("%.3f", data.stats[i].mean - grid_offset));
        grid_->SetCellValue(i, 2, wxString::Format("%.3f", data.stats[i].std_dev));
        grid_->SetCellValue(i, 3, wxString::Format("%.3f", data.stats[i].max - grid_offset));
        grid_->SetCellValue(i, 4, wxString::Format("%.3f", data.stats[i].min - grid_offset));

        wxColour bg_color = (i == max_mean_idx) ? kHighlightColor : kNormalColor;
        for (int col = 0; col < 5; ++col) {
            grid_->SetCellBackgroundColour(i, col, bg_color);
        }
    }

    grid_->EndBatch();
    grid_->AutoSizeColumns(false);
    grid_->InvalidateBestSize();
    Layout();


    // --------------------------------------------------------
    // 2. HeatMapデータ加工プロセス
    // --------------------------------------------------------

    // [Phase A] 値の変換 (Advantage)
    float shift_val = 0.0f;
    if (is_adv) {
        shift_val = static_cast<float>(global_mean);
        for (auto& val : data.vv) {
            val -= shift_val;
        }
        // global_mean は中心化したので 0.0 として扱う
        global_mean = 0.0;
    }

    // ==========================================
    // [Phase B] ヒストグラム化 (Histogram)
    // ==========================================
    int n_bins = config_->hist_bins;
    float guide_line_ratio = -1.0f; // ガイドライン位置 (0.0~1.0, 負なら非表示)

    if (is_hist) {
        // ----------------------------------------------------
        // 1. 現在のフレームでの有効範囲を計算 (Mean ± Xσ)
        // ----------------------------------------------------
        float range_k = config_->hist_range_k;// 1.0f;
        float current_min = static_cast<float>(global_mean - range_k * global_stdev);
        float current_max = static_cast<float>(global_mean + range_k * global_stdev);

        // Min/Maxも考慮してデータが存在する範囲を含める
        auto mm = std::minmax_element(data.vv.begin(), data.vv.end());
        if (current_min > *mm.first) current_min = *mm.first;
        if (current_max < *mm.second) current_max = *mm.second;

        // ----------------------------------------------------
        // 2. 累積範囲 (カメラ) の更新
        //    一度広がった範囲は狭めない（カメラ固定）
        // ----------------------------------------------------
        if (current_min < accumulated_min_) accumulated_min_ = current_min;
        if (current_max > accumulated_max_) accumulated_max_ = current_max;

        // Advantageモードの場合は、0を中心に対称にする（左右のバランスを保つため）
        if (is_adv) {
            float abs_max = std::max(std::abs(accumulated_min_), std::abs(accumulated_max_));
            accumulated_min_ = -abs_max;
            accumulated_max_ = abs_max;
        }

        // 最終的なヒストグラム範囲
        float hist_min = accumulated_min_;
        float hist_max = accumulated_max_;

        // 幅が極小の場合の安全策
        if (std::abs(hist_max - hist_min) < 1e-4) {
            hist_max = hist_min + 1.0f;
        }

        // ----------------------------------------------------
        // 3. センターライン (0の位置) の計算
        // ----------------------------------------------------
        // Advantageなら 0.0 は平均値の位置。Rawなら絶対値の0。
        float zero_point = 0.0f;
        if (zero_point >= hist_min && zero_point <= hist_max) {
            guide_line_ratio = (zero_point - hist_min) / (hist_max - hist_min);
        }

        // ----------------------------------------------------
        // 4. 投票 (Voting)
        // ----------------------------------------------------
        int n_actions = data.height;

        std::vector<float> new_xv, new_yv, new_vv;
        int total_points = n_actions * n_bins;
        new_xv.reserve(total_points);
        new_yv.reserve(total_points);
        new_vv.assign(total_points, 0.0f);

        float inv_range = 1.0f / (hist_max - hist_min);

        for (size_t i = 0; i < data.vv.size(); ++i) {
            int action_idx = static_cast<int>(data.yv[i]);
            float val = data.vv[i];

            float norm = (val - hist_min) * inv_range;
            int center_bin_idx = static_cast<int>(std::floor(norm * n_bins));

            // Clamp (画面外の値を両端に寄せる)
            if (center_bin_idx < 0) center_bin_idx = 0;
            if (center_bin_idx >= n_bins) center_bin_idx = n_bins - 1;

            if (config_->use_hist_smooth) {
                // ★本格スムージング (KDE: Kernel Density Estimation)
                // そのビンを中心に、左右へ重みを配分する
                int base_idx = action_idx * n_bins;

                for (int r = -config_->smooth_radius; r <= config_->smooth_radius; ++r) {
                    int target_bin = center_bin_idx + r;

                    // 範囲チェック
                    if (target_bin >= 0 && target_bin < n_bins) {
                        // 重みを加算
                        new_vv[base_idx + target_bin] += hist_weights_[r + config_->smooth_radius];
                    }
                }
            } else {
                // スムージングなし (点描画)
                new_vv[action_idx * n_bins + center_bin_idx] += 1.0f;
            }
        }

        // 座標データ生成
        for (int y = 0; y < n_actions; ++y) {
            for (int x = 0; x < n_bins; ++x) {
                new_xv.push_back(static_cast<float>(x));
                new_yv.push_back(static_cast<float>(y));
            }
        }
        data.xv = new_xv;
        data.yv = new_yv;
        data.vv = new_vv;
        data.width = n_bins;
    }

    // ==========================================
    // [Phase C] 色付けのための手動正規化 (Manual Normalization)
    // ==========================================
    float disp_min = 0.0f;
    float disp_max = 1.0f;

    if (is_hist) {
        //  色の基準(disp_max)を決める際、「両端のビン（外れ値のゴミ箱）」を除外して中央部分のピークを探す

        disp_min = 0.0f;
        disp_max = 1.0f; // fallback

        // ビンデータの中から最大値を探す (ただし両端 0, n_bins-1 は無視)
        float max_val_in_center = 0.0f;

        // データ構造は [Action0のBin0..N, Action1のBin0..N, ...]
        for (int i = 0; i < data.vv.size(); ++i) {
            int bin_idx = i % n_bins;
            // 両端のビンは無視
            if (bin_idx == 0 || bin_idx == n_bins - 1) continue;

            if (data.vv[i] > max_val_in_center) {
                max_val_in_center = data.vv[i];
            }
        }

        // もし中央にデータが全くない場合は、全体の最大値を使う（真っ黒回避）
        if (max_val_in_center < 1e-6) {
            auto mm = std::minmax_element(data.vv.begin(), data.vv.end());
            max_val_in_center = *mm.second;
        }

        disp_max = max_val_in_center;

        // LogScale対応
        if (is_log) {
            for (auto& val : data.vv) val = std::log1p(val);
            disp_max = std::log1p(disp_max);
        }
    } else {
        // Raw / Advantage (変更なし)
        if (is_adv) {
            auto mm = std::minmax_element(data.vv.begin(), data.vv.end());
            float abs_max = std::max(std::abs(*mm.first), std::abs(*mm.second));
            disp_min = -abs_max;
            disp_max = abs_max;
        } else {
            auto mm = std::minmax_element(data.vv.begin(), data.vv.end());
            disp_min = *mm.first;
            disp_max = *mm.second;
        }
    }

    // 正規化実行
    if (std::abs(disp_max - disp_min) > 1e-6) {
        float inv_range = 1.0f / (disp_max - disp_min);
        for (auto& val : data.vv) {
            val = (val - disp_min) * inv_range;
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
        }
    } else {
        std::fill(data.vv.begin(), data.vv.end(), 0.5f);
    }

    // 描画
    uint32_t flags = anet::HeatMapFlags::HM_SumMode;
    anet::HeatMap heat_map(data.width, data.height, 0, data.width, 0, data.height, 0, flags);
    heat_map.AddDataBatch(data.xv, data.yv, data.vv);
    auto heatmap_image = heat_map.Render();

    int total_rows = static_cast<int>(action_names_.size());
    int target_height = total_rows * config_->row_height;
    heatmap_panel_->UpdateHeatMap(heatmap_image, 0, target_height, guide_line_ratio);
    //heatmap_panel_->UpdateHeatMap(heatmap_image, 0, target_height);
}

//void QValuePanel::Update()
//{
//    auto data_opt = data_store_.Get();
//    if (!data_opt.has_value()) return;
//    auto data = *data_opt;
//
//    bool log_scale = IsLogScale();
//
//    // Grid更新処理
//    int current_rows = grid_->GetNumberRows();
//    int new_rows = static_cast<int>(action_names_.size());
//
//    grid_->BeginBatch(); // 再描画抑制
//
//    if (new_rows > current_rows) {
//        grid_->AppendRows(new_rows - current_rows);
//    } else if (new_rows < current_rows) {
//        grid_->DeleteRows(new_rows, current_rows - new_rows);
//    }
//
//    for (int i = 0; i < new_rows; ++i) {
//        grid_->SetRowLabelValue(i, wxString::Format("%d", i));
//        grid_->SetCellValue(i, 0, action_names_[i]);
//        grid_->SetCellValue(i, 1, wxString::Format("%.3f", data.stats[i].mean));
//        grid_->SetCellValue(i, 2, wxString::Format("%.3f", data.stats[i].std_dev));
//        grid_->SetCellValue(i, 3, wxString::Format("%.3f", data.stats[i].max));
//        grid_->SetCellValue(i, 4, wxString::Format("%.3f", data.stats[i].min));
//    }
//
//    grid_->EndBatch();
//    grid_->AutoSizeColumns(false);
//    grid_->InvalidateBestSize();
//    this->Layout();
//
//    // UI状態取得
//    bool is_hist = IsHistogram();
//    bool is_adv = IsAdvantage();
//    bool is_log = IsLogScale();
//
//    // --------------------------------------------------------
//    // 2. HeatMapデータ加工プロセス
//    // --------------------------------------------------------
//
//    if (data.vv.empty()) return;
//
//    // 全体の統計量計算 (Advantage計算やHist範囲決定に使用)
//    double sum = std::accumulate(data.vv.begin(), data.vv.end(), 0.0);
//    double global_mean = sum / data.vv.size();
//
//    // 分散・標準偏差計算
//    double sq_sum = std::inner_product(data.vv.begin(), data.vv.end(), data.vv.begin(), 0.0);
//    double global_stdev = std::sqrt(sq_sum / data.vv.size() - global_mean * global_mean);
//
//
//    // ==========================================
//    // [Phase A] 値の変換 (Advantage)
//    // ==========================================
//    if (is_adv) {
//        // 平均を引く (中心化)
//        for (auto& val : data.vv) {
//            val -= static_cast<float>(global_mean);
//        }
//        // 中心化したので、基準となるMeanは0になる
//        global_mean = 0.0;
//    }
//
//    // ==========================================
//    // [Phase B] ヒストグラム化 (Histogram)
//    // ==========================================
//    if (is_hist) {
//        // 表示範囲決定: Mean ± 2.5σ (ここは変更なしでOK)
//        // ※分布の裾野を見たい場合は 2.5 -> 3.0 や 4.0 に広げても良いです
//        float range_k = 3.0f;
//        float hist_min = static_cast<float>(global_mean - range_k * global_stdev);
//        float hist_max = static_cast<float>(global_mean + range_k * global_stdev);
//
//        // GridのMin/Max情報も参考にして、明らかにデータが存在する範囲にクリップ
//        auto mm = std::minmax_element(data.vv.begin(), data.vv.end());
//        // ただし、Advantageの場合は0中心を見たいので、あまり極端にMin/Maxに合わせすぎない方が良い場合もあります
//        // ここでは「計算範囲」と「実データ」の共通部分をとるロジックにします
//        if (hist_min > *mm.first) hist_min = *mm.first;   // 左端を実データまで広げるか、あるいは固定するか
//        if (hist_max < *mm.second) hist_max = *mm.second; // 右端を広げるか
//
//        // ※ 見やすさ優先なら、実データのMinMaxに合わせず、σ範囲で切って
//        //    はみ出した分を端に寄せる（以下のClamp処理）のがベストです。
//        //    今回は「σ範囲固定」に戻します。
//        hist_min = static_cast<float>(global_mean - range_k * global_stdev);
//        hist_max = static_cast<float>(global_mean + range_k * global_stdev);
//
//        if (std::abs(hist_max - hist_min) < 1e-4) {
//            hist_max = hist_min + 1.0f;
//        }
//
//        // ビン数設定
//        int n_actions = data.height;
//        int n_bins = 510; // 解像度
//
//        std::vector<float> new_xv, new_yv, new_vv;
//        int total_points = n_actions * n_bins;
//        new_xv.reserve(total_points);
//        new_yv.reserve(total_points);
//        new_vv.assign(total_points, 0.0f);
//
//        // 投票
//        float inv_range = 1.0f / (hist_max - hist_min);
//
//        for (size_t i = 0; i < data.vv.size(); ++i) {
//            int action_idx = static_cast<int>(data.yv[i]);
//            float val = data.vv[i];
//
//            // 正規化位置計算
//            float norm = (val - hist_min) * inv_range;
//            int bin_idx = static_cast<int>(std::floor(norm * n_bins));
//
//            // 【修正】範囲外を捨てずに、両端に集約する (Clamp)
//            if (bin_idx < 0) bin_idx = 0;
//            if (bin_idx >= n_bins) bin_idx = n_bins - 1;
//
//            new_vv[action_idx * n_bins + bin_idx] += 1.0f;
//        }
//
//        // --- 以下変更なし ---
//        // 座標生成
//        for (int y = 0; y < n_actions; ++y) {
//            for (int x = 0; x < n_bins; ++x) {
//                new_xv.push_back(static_cast<float>(x));
//                new_yv.push_back(static_cast<float>(y));
//            }
//        }
//        data.xv = new_xv;
//        data.yv = new_yv;
//        data.vv = new_vv;
//        data.width = n_bins;
//    }
//
//
//    // ==========================================
//    // [Phase C] 色付けのための手動正規化 (Manual Normalization)
//    // ==========================================
//    // ★対策1: HeatMapのAutoNormを使わず、ここで 0.0～1.0 に値を整える
//
//    // 表示したい値の範囲を決定
//    float disp_min = 0.0f;
//    float disp_max = 1.0f;
//
//    if (is_hist) {
//        // ヒストグラムの場合: 0 ～ 最大頻度
//        auto mm = std::minmax_element(data.vv.begin(), data.vv.end());
//        disp_min = 0.0f;
//        disp_max = *mm.second; // 最大頻度
//
//        // LogScale対応
//        if (is_log) {
//            // log(1 + x) 等で変換
//            for (auto& val : data.vv) {
//                val = std::log1p(val);
//            }
//            disp_max = std::log1p(disp_max);
//        }
//
//    } else {
//        // Raw / Advantage の場合
//        if (is_adv) {
//            // Advantage: 0を中心に、絶対値の最大幅をとる (対称レンジ)
//            // 例: -5 ～ +5 -> 0.0 ～ 1.0 (0が0.5になる)
//            auto mm = std::minmax_element(data.vv.begin(), data.vv.end());
//            float abs_max = std::max(std::abs(*mm.first), std::abs(*mm.second));
//            disp_min = -abs_max;
//            disp_max = abs_max;
//        } else {
//            // Raw (Value): 単純なMin-Max
//            auto mm = std::minmax_element(data.vv.begin(), data.vv.end());
//            disp_min = *mm.first;
//            disp_max = *mm.second;
//        }
//    }
//
//    // 正規化実行 (0.0 ～ 1.0 にマップ)
//    if (std::abs(disp_max - disp_min) > 1e-6) {
//        float inv_range = 1.0f / (disp_max - disp_min);
//        for (auto& val : data.vv) {
//            val = (val - disp_min) * inv_range;
//            // クリップ
//            if (val < 0.0f) val = 0.0f;
//            if (val > 1.0f) val = 1.0f;
//        }
//    } else {
//        std::fill(data.vv.begin(), data.vv.end(), 0.5f); // 範囲がない場合はグレー
//    }
//
//
//    // --------------------------------------------------------
//    // 描画
//    // --------------------------------------------------------
//    uint32_t flags = 0;    //  HM_AutoNormValue は指定しない (既に0-1に正規化済みのため)
//    //if (log_scale) flags |= anet::HeatMapFlags::HM_LogScaleValue;     // LogScaleも自前でやったので指定しない
//
//    anet::HeatMap heat_map(data.width, data.height, 0, data.width, 0, data.height, 0, flags);
//    heat_map.AddDataBatch(data.xv, data.yv, data.vv);
//    auto heatmap_image = heat_map.Render();
//
//    //  HeatMapパネル更新
//    int total_rows = static_cast<int>(action_names_.size());
//    int target_height = total_rows * kRowHeight;
//    heatmap_panel_->UpdateHeatMap(heatmap_image, 0, target_height);
//}

bool QValuePanel::IsHistogram() const
{
    return hist_check_->GetValue();
}

bool QValuePanel::IsAdvantage() const
{
    return adv_check_->GetValue();
}

bool QValuePanel::IsLogScale() const
{
    return log_scale_check_->GetValue();
}

void QValuePanel::OnCheck(wxCommandEvent& event)
{
    // モードが変わったら範囲をリセットしないと、
    // Advantage(-5~+5) と Raw(-100~+1000) が混ざって見えなくなる
    bool current_adv = IsAdvantage();
    bool current_hist = IsHistogram();
    bool current_log = IsLogScale();

    if (current_adv != last_is_adv_ || current_hist != last_is_hist_ || current_log != last_is_log_) {
        ResetRange();
        last_is_adv_ = current_adv;
        last_is_hist_ = current_hist;
        last_is_log_ = current_log;
    }
    Update();
}

void QValuePanel::ResetRange() {
    accumulated_min_ = std::numeric_limits<float>::max();
    accumulated_max_ = std::numeric_limits<float>::lowest();
}

void QValuePanel::OnSize(wxSizeEvent& event)
{
    Layout(); // Sizerの再計算
    event.Skip();
}

void QValuePanel::OnCloseWindow(wxCloseEvent& event)
{
    if (notifier_ && observer_) {
        notifier_->Detach(observer_);
    }
}

