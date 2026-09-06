// QValuePanel.cpp
#include "QValuePanel.hpp"

#include <algorithm>
#include <utility>
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
static constexpr bool kLogScaleDefaultCheck = false;
static constexpr bool kAutoRangeDefaultCheck = true;


// ----------------------------------------------------------------------------
// QValuePanel::Config
// ----------------------------------------------------------------------------

struct QValuePanel::Config : public anet::Config {
    int row_height = 24;
    bool use_hist_smooth = true;
    float smooth_radius = 4.0f;
    int hist_bins = 510;
    float hist_range_k = 1.0;

    Config(const anet::ConfigData& config_data) : anet::Config("QValuePanel")
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
    SetBackgroundColour(wxColour(0, 0, 0));

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
    Refresh(false); // OnPaintをトリガー
}

void QValueHeatMapPanel::SetScrollOffsetY(int scroll_y)
{
    scroll_y = std::max(0, scroll_y);
    if (scroll_y_ == scroll_y) return;

    scroll_y_ = scroll_y;
    Refresh(false);
    Update();
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
    wxSize client_size = GetClientSize();

    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.SetBrush(wxBrush(GetBackgroundColour()));
    dc.DrawRectangle(0, 0, client_size.GetWidth(), client_size.GetHeight());

    if (!heatmap_image_.IsOk()) return;

    // パネルの現在のクライアント幅を取得
    int width = client_size.GetWidth();

    // 幅が0以下や画像がない場合は描画しない
    if (width > 0 && target_height_ > 0) {
        // ウィンドウ幅に合わせて画像をリサイズ
        //wxImage scaled_img = heatmap_image_.Scale(width, target_height_);
        //wxImage scaled_img = heatmap_image_.Scale(width, target_height_, wxIMAGE_QUALITY_HIGH);
        wxImage scaled_img = heatmap_image_.Scale(width, target_height_, wxIMAGE_QUALITY_NEAREST);
        int draw_y = y_offset_ - scroll_y_;

        // 描画
        dc.DrawBitmap(wxBitmap(scaled_img), 0, draw_y, false);

        // センターライン（ガイドライン）の描画
        if (guide_line_pos_ >= 0.0f && guide_line_pos_ <= 1.0f) {
            int line_x = static_cast<int>(guide_line_pos_ * width);

            // シアン色 (0, 255, 255), 太さ2px
            wxPen pen(wxColour(0, 255, 255), 2, wxPENSTYLE_SOLID);
            dc.SetPen(pen);

            // 縦線を引く
            dc.DrawLine(line_x, draw_y, line_x, draw_y + target_height_);
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
    main_sizer->Add(grid_, 0, wxEXPAND, 0);

    // ====================================================================
    // 右側: 縦並び (ヘッダーエリア + ヒートマップエリア)
    // ====================================================================
    auto* right_sizer = new wxBoxSizer(wxVERTICAL);

    // --------------------------------------------------
    // 右上: ヘッダーエリア (チェックボックス配置)
    // --------------------------------------------------
    header_panel_ = new wxPanel(this, wxID_ANY);

    auto* header_sizer = new wxBoxSizer(wxHORIZONTAL);

    // Hist CheckBox
    hist_check_ = new wxCheckBox(header_panel_, wxID_ANY, "Hist");
    hist_check_->SetValue(kHistDefaultCheck);
    hist_check_->Bind(wxEVT_CHECKBOX, &QValuePanel::OnCheck, this);
    header_sizer->Add(hist_check_, 0, wxALIGN_CENTER_VERTICAL | wxALL, 0);

    // Advantage CheckBox
    adv_check_ = new wxCheckBox(header_panel_, wxID_ANY, "Advantage");
    adv_check_->SetValue(kAdvDefaultCheck);
    adv_check_->Bind(wxEVT_CHECKBOX, &QValuePanel::OnCheck, this);
    header_sizer->Add(adv_check_, 0, wxALIGN_CENTER_VERTICAL | wxALL, 0);

    // Log Scale CheckBox
    log_scale_check_ = new wxCheckBox(header_panel_, wxID_ANY, "Log Scale");
    log_scale_check_->SetValue(kLogScaleDefaultCheck);
    log_scale_check_->Bind(wxEVT_CHECKBOX, &QValuePanel::OnCheck, this);
    header_sizer->Add(log_scale_check_, 0, wxALIGN_CENTER_VERTICAL | wxALL, 0);

    // Auto Range CheckBox
    auto_range_check_ = new wxCheckBox(header_panel_, wxID_ANY, "Auto Range");
    auto_range_check_->SetValue(kAutoRangeDefaultCheck);
    auto_range_check_->Bind(wxEVT_CHECKBOX, &QValuePanel::OnCheck, this);
    header_sizer->Add(auto_range_check_, 0, wxALIGN_CENTER_VERTICAL | wxALL, 0);

	// Reset Range Button
    reset_range_button_ = new wxButton(header_panel_, wxID_ANY, "Reset Range", wxDefaultPosition, wxDefaultSize, wxBU_EXACTFIT);
    reset_range_button_->Bind(wxEVT_BUTTON, &QValuePanel::OnResetRangeClick, this);
    header_sizer->Add(reset_range_button_, 0, wxALIGN_CENTER_VERTICAL | wxALL, 0);

    // ヘッダパネルを追加
    header_panel_->SetSizer(header_sizer);
    const int header_height = std::max(config_->row_height, header_sizer->CalcMin().GetHeight());
    header_panel_->SetMinSize(wxSize(-1, header_height));
    header_panel_->SetMaxSize(wxSize(-1, header_height));
    grid_->SetColLabelSize(header_height);
    right_sizer->Add(header_panel_, 0, wxEXPAND, 0);

    // --------------------------------------------------
    // 右下: HeatMapパネル
    // --------------------------------------------------
    heatmap_panel_ = new QValueHeatMapPanel(this);
    right_sizer->Add(heatmap_panel_, 1, wxEXPAND, 0);

    main_sizer->Add(right_sizer, 1, wxEXPAND, 0);
    SetSizer(main_sizer);
    Layout();
    RefreshLayoutChildren();
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
    grid_->SetColSize(0, kActionNameColWidth);
    grid_->SetColMinimalWidth(0, kActionNameColMinWidth);
    grid_->ShowScrollbars(wxSHOW_SB_NEVER, wxSHOW_SB_DEFAULT);  // 横スクロールバーを強制非表示に

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
    grid_->Bind(wxEVT_GRID_CELL_LEFT_CLICK, &QValuePanel::OnGridActionClick, this);
    grid_->Bind(wxEVT_GRID_LABEL_LEFT_CLICK, &QValuePanel::OnGridActionClick, this);

    grid_->Bind(wxEVT_SCROLLWIN_TOP, &QValuePanel::OnGridScroll, this);
    grid_->Bind(wxEVT_SCROLLWIN_BOTTOM, &QValuePanel::OnGridScroll, this);
    grid_->Bind(wxEVT_SCROLLWIN_LINEUP, &QValuePanel::OnGridScroll, this);
    grid_->Bind(wxEVT_SCROLLWIN_LINEDOWN, &QValuePanel::OnGridScroll, this);
    grid_->Bind(wxEVT_SCROLLWIN_PAGEUP, &QValuePanel::OnGridScroll, this);
    grid_->Bind(wxEVT_SCROLLWIN_PAGEDOWN, &QValuePanel::OnGridScroll, this);
    grid_->Bind(wxEVT_SCROLLWIN_THUMBTRACK, &QValuePanel::OnGridScroll, this);
    grid_->Bind(wxEVT_SCROLLWIN_THUMBRELEASE, &QValuePanel::OnGridScroll, this);
    grid_->Bind(wxEVT_MOUSEWHEEL, &QValuePanel::OnGridMouseWheel, this);
    grid_->Bind(wxEVT_KEY_UP, &QValuePanel::OnGridKeyUp, this);

    if (auto* grid_window = grid_->GetGridWindow()) {
        grid_window->Bind(wxEVT_MOUSEWHEEL, &QValuePanel::OnGridMouseWheel, this);
        grid_window->Bind(wxEVT_KEY_UP, &QValuePanel::OnGridKeyUp, this);
    }
}

void QValuePanel::Initialize(std::shared_ptr<anet::rl::RunManager> run_manager, std::shared_ptr<anet::rl::EvalRunner> runner)
{
    runner_ = runner;

    // アクション名
    auto action_spec = runner->GetBatchEnv()->GetSpec().action_spec;
    action_names_.insert(action_names_.begin(), action_spec.value_labels.begin(), action_spec.value_labels.end());

	// Observer生成&登録
    auto notifier = run_manager->GetNotifier();
    this->observer_ = notifier->AttachScoped<anet::rl::FunctionTrainObserver>(
        runner,
        [this](const anet::rl::TrainEvent& event)
        {
            // データ作る
            auto data = CreateData(event);

            // データがなかったら反映しない
            if (!data.has_value()) return;

            // データ反映 (保存は毎Step、画面更新はApplyData内でフレーム1回に集約される)
            ApplyData(*data);
        },
        "QValuePanel");

    // Detach用にNotifierを保持
    notifier_ = notifier;
}

void QValuePanel::SetActionHandler(std::function<void(int64_t)> action_handler)
{
    action_handler_ = std::move(action_handler);
}

std::optional<QValueData> QValuePanel::CreateData(const anet::rl::TrainEvent& event)
{
    ANET_PROFILE_FUNC();

    const auto& aux_data = event.action_info->GetAuxData();
    auto q_values_itr = aux_data.find("q_values");
    if (q_values_itr == aux_data.end()) return std::nullopt;

    torch::Tensor q_values_plain = q_values_itr->second;
    auto q_quantiles_itr = aux_data.find("q_quantiles");
    auto full_q_quantiles_itr = aux_data.find("full_q_quantiles");
    torch::Tensor q_values = q_values_plain;
    
    // 観測用full分布があれば優先し、未生成時は従来のrisk分布・scalar Qへ順にfallbackする。
    if (full_q_quantiles_itr != aux_data.end() && full_q_quantiles_itr->second.defined()) {
        q_values = full_q_quantiles_itr->second;
    } else if (q_quantiles_itr != aux_data.end() && q_quantiles_itr->second.defined()) {
        q_values = q_quantiles_itr->second;
    }

    auto raw_actions_itr = aux_data.find("raw_actions");
    torch::Tensor raw_actions;
    if (raw_actions_itr != aux_data.end()) raw_actions = raw_actions_itr->second;

    //ANET_LOG_DEBUG("q_values=" << anet::ToString(q_values));

    ANET_CHECK(q_values.size(0) > 0);
    //ANET_ASSERT_SHAPE(q_values, { ANET_SHAPE_ANY, static_cast<int64_t>(action_names_.size()), ANET_SHAPE_ENDANY });
    // 要素ごとの .item<float>() は毎回GPU→CPU同期が走り非常に重いため、
    // 先頭env分だけ一括でCPUへ転送し、以降は生ポインタで読む(同期は実質この1回)
    const torch::Tensor first_env_q =
        q_values[0].detach().to(torch::kCPU, torch::kFloat).contiguous(); // PlainQ:(A,) / QR:(A,W)
    ANET_CHECK(first_env_q.size(0) == action_names_.size());
    const auto n_actions = first_env_q.size(0);

    // PlainQ(A,) と QR(A,W) を幅widthの行列として統一的に扱う
    const int width = (first_env_q.dim() == 2) ? static_cast<int>(first_env_q.size(1)) : 1;
    const float* q_ptr = first_env_q.data_ptr<float>();

    QValueData data;

    // Action表示データ設定
    if (raw_actions.defined()) {
        // 人間が指示したかもしれない実際に実行したActionではなく、Agentが選択したActionを画面表示データに設定
        data.selected_action = raw_actions[0].item<int64_t>();
    } else {
        // raw_actionが見れない場合は仕方ないので実際に実行したActionを画面表示データに設定
        data.selected_action = event.action_info->GetAction(torch::kCPU)[0].item<int64_t>();
    }

    // 統計値生成 (Tensor reduction + .item() の繰り返しはGPU同期が多発するため、CPU上で1パス集計)
    data.stats.resize(n_actions);
    for (int i = 0; i < n_actions; i++) {
        const float* row = q_ptr + static_cast<size_t>(i) * width;
        double sum = 0.0;
        double sq_sum = 0.0;
        float min_val = row[0];
        float max_val = row[0];
        for (int x = 0; x < width; x++) {
            const float v = row[x];
            sum += v;
            sq_sum += static_cast<double>(v) * v;
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
        const double mean = sum / width;
        // torch::Tensor::std() に合わせた不偏分散(N-1)。width=1 のときは 0 扱い
        const double var = (width > 1) ? std::max(0.0, (sq_sum - width * mean * mean) / (width - 1)) : 0.0;
        data.stats[i].mean = static_cast<float>(mean);
        data.stats[i].std_dev = static_cast<float>(std::sqrt(var));
        data.stats[i].max = max_val;
        data.stats[i].min = min_val;
    }

    // HeatMap用データ生成 (PlainQ/QRとも行優先の格子値として一括コピー)
    data.vv.assign(q_ptr, q_ptr + static_cast<size_t>(n_actions) * width);

    // データ設定
    data.width = width;
    data.height = n_actions;

    return data;
}

void QValuePanel::ApplyData(const QValueData& data)
{
    ANET_PROFILE_FUNC();

    // 最新断面を保存
    data_store_.Update(data);

    // 画面更新はイベントループへ1回だけ予約する
    // (タイマーハンドラ内で毎Step呼ばれても、Update()の実行はフレームあたり1回に集約される)
    if (!update_pending_.exchange(true)) {
        CallAfter([this] {
            update_pending_.store(false);
            Update();
        });
    }
}

void QValuePanel::Update()
{
    ANET_PROFILE_FUNC();

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
    // 統計量計算 (GridとHeatMap共通で使用)
    // --------------------------------------------------------
    double sum = std::accumulate(data.vv.begin(), data.vv.end(), 0.0);
    double global_mean = sum / data.vv.size();

    // 分散・標準偏差計算
    double sq_sum = std::inner_product(data.vv.begin(), data.vv.end(), data.vv.begin(), 0.0);
    double global_stdev = std::sqrt(sq_sum / data.vv.size() - global_mean * global_mean);


    // --------------------------------------------------------
    // Grid更新 (Advantage対応)
    // --------------------------------------------------------
    int current_rows = grid_->GetNumberRows();
    int new_rows = static_cast<int>(action_names_.size());
    const bool rows_changed = (new_rows != current_rows);

    // 最大のMeanを持つアクションを特定する
    //float max_mean_val = -std::numeric_limits<float>::infinity();
    //int max_mean_idx = -1;

    //for (int i = 0; i < new_rows; ++i) {
    //    // 統計データが揃っていない場合のガード
    //    if (i < data.stats.size()) {
    //        if (data.stats[i].mean > max_mean_val) {
    //            max_mean_val = data.stats[i].mean;
    //            max_mean_idx = i;
    //        }
    //    }
    //}

    // 選択されたアクションを特定
    int64_t selected_idx = data.selected_action;

    grid_->BeginBatch();

    // 行構成・アクション名は不変なので、行数が変わったとき(実質初回)だけ構築する
    if (rows_changed) {
        if (new_rows > current_rows) {
            grid_->AppendRows(new_rows - current_rows);
        } else {
            grid_->DeleteRows(new_rows, current_rows - new_rows);
        }
        for (int i = 0; i < new_rows; ++i) {
            grid_->SetRowLabelValue(i, wxString::Format("%d", i));
            grid_->SetCellValue(i, 0, action_names_[i]);
        }
        // 行再構築後のセルattrはデフォルト状態なので、ハイライトは付け直す
        last_selected_row_ = -1;
    }

    // Grid表示用のオフセット値 ★混乱するので無効化
    float grid_offset = 0.0f;// is_adv ? static_cast<float>(global_mean) : 0.0f;

    // 数値セル(1～4列)のみ毎回更新する
    for (int i = 0; i < new_rows; ++i) {
        // AdvantageONなら、平均値を引いた値を表示して確認できるようにする
        // Std(ばらつき)はシフトしても変わらないのでそのまま
        grid_->SetCellValue(i, 1, wxString::Format("%.3f", data.stats[i].mean - grid_offset));
        grid_->SetCellValue(i, 2, wxString::Format("%.3f", data.stats[i].std_dev));
        grid_->SetCellValue(i, 3, wxString::Format("%.3f", data.stats[i].max - grid_offset));
        grid_->SetCellValue(i, 4, wxString::Format("%.3f", data.stats[i].min - grid_offset));
    }

    // 選択行ハイライト (毎回全行を塗り直すとattr更新が支配的になるため、変化した行だけ差分更新)
    const int selected_row = static_cast<int>(selected_idx);
    if (selected_row != last_selected_row_) {
        if (last_selected_row_ >= 0 && last_selected_row_ < new_rows) {
            for (int col = 0; col < 5; ++col) {
                // wxNullColourでリセット（システムデフォルトに戻る）
                grid_->SetCellBackgroundColour(last_selected_row_, col, wxNullColour);
                grid_->SetCellTextColour(last_selected_row_, col, wxNullColour);
            }
        }
        if (selected_row >= 0 && selected_row < new_rows) {
            const wxColour kHighlightBg = wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT);
            const wxColour kHighlightText = wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHTTEXT);
            for (int col = 0; col < 5; ++col) {
                grid_->SetCellBackgroundColour(selected_row, col, kHighlightBg);
                grid_->SetCellTextColour(selected_row, col, kHighlightText); // 文字色も合わせる
            }
        }
        last_selected_row_ = selected_row;
    }

    grid_->EndBatch();

    // 列幅・パネル幅・レイアウトの再計算は行構成が変わったときだけ行う
    // (毎Step実行すると AutoSizeColumn のテキスト測定や全子ウィンドウの同期再描画が支配的コストになる)
    if (rows_changed) {
        // アクション名の列(0列目)だけ文字幅にフィットさせる
        grid_->AutoSizeColumn(0, false);

        // 1～4列目(Mean, Std, Max, Min)は固定幅(kColWidth)を維持する
        for (int i = 1; i < 5; ++i) {
            grid_->SetColSize(i, kColWidth);
        }

        int total_width = grid_->GetRowLabelSize();
        for (int i = 0; i < grid_->GetNumberCols(); ++i) {
            total_width += grid_->GetColSize(i);
        }

        // Gridの中身の仮想高さを計算（列ヘッダー ＋ 全行の高さ）
        int virtual_height = grid_->GetColLabelSize();
        for (int i = 0; i < grid_->GetNumberRows(); ++i) {
            virtual_height += grid_->GetRowSize(i);
        }

        // Gridの実際の描画高さを取得
        int current_height = grid_->GetSize().GetHeight();
        if (current_height < 50) {
            // UI初期化直後でまだSizerが計算されていない場合は親パネルの高さを参照
            current_height = GetClientSize().GetHeight();
        }

        // 中身が実際の高さを超えて「縦スクロールバーが出現する」場合のみ幅を加算
        if (virtual_height > current_height) {
            total_width += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X);
        }
        total_width += 2; // 境界線のマージン（左右1px）

        // Gridの横幅を厳密にロックする（縦方向はSizer任せで拡張させる）
        grid_->SetMinSize(wxSize(total_width, -1));
        grid_->SetMaxSize(wxSize(total_width, -1));

        // Sizer再レイアウト
        Layout();
        RefreshLayoutChildren();
    }

    // --------------------------------------------------------
    // HeatMapデータ加工プロセス
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
        // 現在のフレームでの有効範囲を計算 (Mean ± Xσ)
        // ----------------------------------------------------
        float range_k = config_->hist_range_k;// 1.0f;
        float current_min = static_cast<float>(global_mean - range_k * global_stdev);
        float current_max = static_cast<float>(global_mean + range_k * global_stdev);

        // Min/Maxも考慮してデータが存在する範囲を含める
        auto mm = std::minmax_element(data.vv.begin(), data.vv.end());
        if (current_min > *mm.first) current_min = *mm.first;
        if (current_max < *mm.second) current_max = *mm.second;

        bool is_auto_range = auto_range_check_->GetValue();
        float shrink_rate = is_auto_range ? 0.05f : 0.0f;

        // ----------------------------------------------------
        // 累積範囲 (カメラ) の更新
        // ----------------------------------------------------

        // --- Max側の更新 ---
        if (current_max > accumulated_max_) {
            // 広がる方向は常に即時反映 (データが見切れるのを防ぐため)
            accumulated_max_ = current_max;
        } else {
            // 狭まる方向は shrink_rate に従う (AutoRangeOFFなら狭まらない)
            accumulated_max_ = accumulated_max_ + (current_max - accumulated_max_) * shrink_rate;
        }

        // --- Min側の更新 ---
        if (current_min < accumulated_min_) {
            // 広がる方向は常に即時反映 (データが見切れるのを防ぐため)
            accumulated_min_ = current_min;
        } else {
            // 狭まる方向は shrink_rate に従う (AutoRangeOFFなら狭まらない)
            accumulated_min_ = accumulated_min_ + (current_min - accumulated_min_) * shrink_rate;
        }

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
        // センターライン (0の位置) の計算
        // ----------------------------------------------------
        // Advantageなら 0.0 は平均値の位置。Rawなら絶対値の0。
        float zero_point = 0.0f;
        if (zero_point >= hist_min && zero_point <= hist_max) {
            guide_line_ratio = (zero_point - hist_min) / (hist_max - hist_min);
        }

        // ----------------------------------------------------
        // 投票 (Voting)
        // ----------------------------------------------------
        int n_actions = data.height;

        std::vector<float> new_vv;
        int total_points = n_actions * n_bins;
        new_vv.assign(total_points, 0.0f);

        float inv_range = 1.0f / (hist_max - hist_min);

        for (size_t i = 0; i < data.vv.size(); ++i) {
            int action_idx = static_cast<int>(i / data.width); // vvは行優先格子なので行番号=Action
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

        data.vv = std::move(new_vv);
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

    // 描画 (vvは行優先格子なのでSetGridValuesの一括設定パスを使う)
    uint32_t flags = anet::HeatMapFlags::HM_SumMode | anet::HeatMapFlags::HM_FlipY;
    anet::HeatMap heat_map(data.width, data.height, 0, data.width, 0, data.height, 0, flags);
    heat_map.SetGridValues(data.vv.data(), data.width, data.height);
    auto heatmap_image = heat_map.Render();

    int total_rows = static_cast<int>(action_names_.size());
    int target_height = total_rows * config_->row_height;
    heatmap_panel_->UpdateHeatMap(heatmap_image, 0, target_height, guide_line_ratio);
    SyncHeatMapScroll(false);
}

void QValuePanel::ResetRange() {
    accumulated_min_ = std::numeric_limits<float>::max();
    accumulated_max_ = std::numeric_limits<float>::lowest();
}

int QValuePanel::GetPreferredDockWidth() const
{
    static constexpr int kPreferredHeatMapWidth = 180;

    if (!grid_) {
        return 640;
    }

    int grid_width = grid_->GetMinSize().GetWidth();
    if (grid_width <= 0) {
        grid_width = grid_->GetRowLabelSize();
        for (int i = 0; i < grid_->GetNumberCols(); ++i) {
            grid_width += grid_->GetColSize(i);
        }
        grid_width += wxSystemSettings::GetMetric(wxSYS_VSCROLL_X) + 2;
    }

    return grid_width + kPreferredHeatMapWidth;
}

void QValuePanel::SyncHeatMapScroll(bool refresh_grid)
{
    if (!grid_ || !heatmap_panel_) return;

    int scroll_x = 0;
    int scroll_y = 0;
    grid_->CalcUnscrolledPosition(0, 0, &scroll_x, &scroll_y);
    heatmap_panel_->SetScrollOffsetY(scroll_y);

    if (refresh_grid) {
        grid_->Refresh(false);
        grid_->Update();
        if (auto* grid_window = grid_->GetGridWindow()) {
            grid_window->Refresh(false);
            grid_window->Update();
        }
    }
}

void QValuePanel::RefreshLayoutChildren()
{
    // wxGrid とネイティブコントロールはレイアウト直後に古い描画が残ることがあるため、
    // QValuePanel 内の子ウィンドウへ背景消去付きの再描画を明示的に要求する。
    Refresh(true);
    if (header_panel_) header_panel_->Refresh(true);
    if (hist_check_) hist_check_->Refresh(true);
    if (adv_check_) adv_check_->Refresh(true);
    if (log_scale_check_) log_scale_check_->Refresh(true);
    if (auto_range_check_) auto_range_check_->Refresh(true);
    if (reset_range_button_) reset_range_button_->Refresh(true);

    if (grid_) {
        grid_->Refresh(true);
        if (auto* grid_window = grid_->GetGridWindow()) {
            grid_window->Refresh(true);
        }
    }
    if (heatmap_panel_) heatmap_panel_->Refresh(true);

    wxPanel::Update();
    if (header_panel_) header_panel_->Update();
    if (hist_check_) hist_check_->Update();
    if (adv_check_) adv_check_->Update();
    if (log_scale_check_) log_scale_check_->Update();
    if (auto_range_check_) auto_range_check_->Update();
    if (reset_range_button_) reset_range_button_->Update();
    if (grid_) {
        grid_->Update();
        if (auto* grid_window = grid_->GetGridWindow()) {
            grid_window->Update();
        }
    }
    if (heatmap_panel_) heatmap_panel_->Update();
}

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

void QValuePanel::OnResetRangeClick(wxCommandEvent& event)
{
    // 蓄積された最大・最小範囲をクリア（無限大/無限小に戻す）
    ResetRange();

    // 画面を更新
    Update();
}

void QValuePanel::OnGridActionClick(wxGridEvent& event)
{
    int row = event.GetRow();
    if (!runner_ || row < 0 || row >= static_cast<int>(action_names_.size())) return;

    int64_t action = static_cast<int64_t>(row);
    if (action_handler_) {
        action_handler_(action);
    } else {
        runner_->DoStep(action);
    }
    grid_->ClearSelection();
}

void QValuePanel::OnGridScroll(wxScrollWinEvent& event)
{
    event.Skip();
    CallAfter(&QValuePanel::SyncHeatMapScroll, true);
}

void QValuePanel::OnGridMouseWheel(wxMouseEvent& event)
{
    event.Skip();
    CallAfter(&QValuePanel::SyncHeatMapScroll, true);
}

void QValuePanel::OnGridKeyUp(wxKeyEvent& event)
{
    event.Skip();
    CallAfter(&QValuePanel::SyncHeatMapScroll, true);
}

void QValuePanel::OnSize(wxSizeEvent& event)
{
    Layout();
    RefreshLayoutChildren();
    CallAfter(&QValuePanel::SyncHeatMapScroll, true);
    event.Skip();
}

void QValuePanel::OnCloseWindow(wxCloseEvent& event)
{
    if (notifier_ && observer_) {
        notifier_->Detach(observer_);
    }
}
