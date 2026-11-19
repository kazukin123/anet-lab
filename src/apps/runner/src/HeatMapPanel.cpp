// HeatMapPanel.cpp

#include "HeatMapPanel.hpp"
#include <algorithm>
#include <wx/numformatter.h>
#include "anet/heat_map.hpp"

// Windowsの標準ヘッダーに足りない定義を追加
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

namespace {

wxString DefaultOutputKeyForFunction(const wxString& function_key)
{
    if (function_key == "forward.dist") return "q_dist";
    if (function_key == "forward.v") return "v_dist";
    if (function_key == "forward.a") return "a_dist";
    return "q";
}

wxArrayString OutputKeyChoicesForFunction(const wxString& function_key)
{
    wxArrayString choices;
    if (function_key == "forward.dist") {
        choices.Add("q_dist");
    } else if (function_key == "forward.v") {
        choices.Add("v_dist");
        choices.Add("v");
    } else if (function_key == "forward.a") {
        choices.Add("a_dist");
        choices.Add("a");
    } else {
        choices.Add("q");
    }
    return choices;
}

bool ContainsChoice(const wxArrayString& choices, const wxString& value)
{
    return choices.Index(value) != wxNOT_FOUND;
}

bool IsOutputKeyCompatibleWithFunction(const wxString& function_key, const wxString& output_key)
{
    return ContainsChoice(OutputKeyChoicesForFunction(function_key), output_key);
}

int ClampInt(int value, int min_value, int max_value)
{
    return std::max(min_value, std::min(value, max_value));
}

} // namespace

SweepHeatMapDialog::SweepHeatMapDialog(
    wxWindow* parent, const anet::rl::EnvSpec& env_spec,int default_x, int default_y)
    : wxDialog(parent, wxID_ANY, "Sweep HeatMap Settings"
        , wxDefaultPosition, wxSize(620, 430)
        , wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER) // リサイズ可
{
    for (const auto& kv : env_spec.state_spec.obs_spec) {
        const auto& key = kv.first;
        const auto& spec = kv.second;
        if (spec.type != anet::SpaceType::Vector) continue;

        const int64_t flatten_dim = spec.CalcFlattenDim();
        if (flatten_dim < 1) continue;
        vector_obs_choices_.push_back({
            wxString::FromUTF8(key.c_str()),
            flatten_dim,
        });
    }
    std::sort(vector_obs_choices_.begin(), vector_obs_choices_.end(),
        [](const VectorObsChoice& lhs, const VectorObsChoice& rhs) {
            if (lhs.key == anet::rl::ObsKeys::kVector) return true;
            if (rhs.key == anet::rl::ObsKeys::kVector) return false;
            return lhs.key < rhs.key;
        });

    auto n_actions = env_spec.action_spec.GetNumActions();

    wxArrayString obs_key_choices;
    for (const auto& choice : vector_obs_choices_) {
        obs_key_choices.Add(choice.key);
    }
    wxArrayString network_side_choices {
        "policy-net",
        "target-net",
    };
    wxArrayString function_choices {
        "forward",
        "forward.q",
        "forward.dist",
        "forward.v",
        "forward.a",
    };
    wxArrayString extractor_choices{
        "mean",
        "max",
        "min",
        "std",
        "argmax"
    };

    /// @todo StateとActionの名前でコンボボックス化
    /// @todo メトリクス同時出力の有無（および出力間隔）を指定可能とする

    // メインの縦方向サイザー
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);

    // 入力フォーム用のグリッドサイザー (2列: ラベル, 入力欄)
    // 縦横の隙間(gap)を 10px に設定
    wxFlexGridSizer* grid_sizer = new wxFlexGridSizer(2, 10, 10);

    // 2列目（入力欄）が横幅いっぱいに広がるように設定 (Column index 1)
    grid_sizer->AddGrowableCol(1, 1);

    // --- ObsKey (ComboBox) ---
    grid_sizer->Add(new wxStaticText(this, wxID_ANY, "Obs key:"), 0, wxALIGN_CENTER_VERTICAL);
    obs_key_combo_ = new wxComboBox(this, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, obs_key_choices, wxCB_READONLY);
    if (!vector_obs_choices_.empty()) {
        obs_key_combo_->SetSelection(0);
    }
    grid_sizer->Add(obs_key_combo_, 1, wxEXPAND);

    // --- X (0 ~ max_x) ---
    grid_sizer->Add(new wxStaticText(this, wxID_ANY, "X:"), 0, wxALIGN_CENTER_VERTICAL);
    spin_x_ = new wxSpinCtrl(this, wxID_ANY);
    grid_sizer->Add(spin_x_, 1, wxEXPAND);

    // --- Y (0 ~ max_y) ---
    grid_sizer->Add(new wxStaticText(this, wxID_ANY, "Y:"), 0, wxALIGN_CENTER_VERTICAL);
    spin_y_ = new wxSpinCtrl(this, wxID_ANY);
    grid_sizer->Add(spin_y_, 1, wxEXPAND);

    // --- Network side (ComboBox) ---
    grid_sizer->Add(new wxStaticText(this, wxID_ANY, "Network:"), 0, wxALIGN_CENTER_VERTICAL);
    network_side_combo_ = new wxComboBox(this, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, network_side_choices, wxCB_READONLY);
    network_side_combo_->SetSelection(0);
    grid_sizer->Add(network_side_combo_, 1, wxEXPAND);

    // --- Function key (ComboBox) ---
    grid_sizer->Add(new wxStaticText(this, wxID_ANY, "Function:"), 0, wxALIGN_CENTER_VERTICAL);
    function_combo_ = new wxComboBox(this, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, function_choices, wxCB_READONLY);
    function_combo_->SetSelection(0);
    grid_sizer->Add(function_combo_, 1, wxEXPAND);

    // --- Output key (ComboBox) ---
    grid_sizer->Add(new wxStaticText(this, wxID_ANY, "Output key:"), 0, wxALIGN_CENTER_VERTICAL);
    output_key_combo_ = new wxComboBox(this, wxID_ANY, DefaultOutputKeyForFunction(function_combo_->GetValue()),
        wxDefaultPosition, wxDefaultSize, OutputKeyChoicesForFunction(function_combo_->GetValue()), wxCB_READONLY);
    grid_sizer->Add(output_key_combo_, 1, wxEXPAND);

    // --- ValueIndex (数値入力) ---
    grid_sizer->Add(new wxStaticText(this, wxID_ANY, "Extractor:"), 0, wxALIGN_CENTER_VERTICAL);
    extractor_combo_ = new wxComboBox(this, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, extractor_choices, wxCB_READONLY);
    extractor_combo_->SetSelection(0); // デフォルトで先頭を選択
    grid_sizer->Add(extractor_combo_, 1, wxEXPAND);

    // --- ValueIndex (数値入力) ---
    grid_sizer->Add(new wxStaticText(this, wxID_ANY, "Extractor index:"), 0, wxALIGN_CENTER_VERTICAL);
    extractor_idx_ = new wxSpinCtrl(this, wxID_ANY);
    extractor_idx_->SetRange(-1, std::max<int64_t>(-1, n_actions - 1));
    extractor_idx_->SetValue(-1);
    grid_sizer->Add(extractor_idx_, 1, wxEXPAND);

    // --- Tag (TextCtrl) ---
    grid_sizer->Add(new wxStaticText(this, wxID_ANY, "Tag:"), 0, wxALIGN_CENTER_VERTICAL);
    tag_text_ = new wxTextCtrl(this, wxID_ANY);
    grid_sizer->Add(tag_text_, 1, wxEXPAND);

    // グリッドサイザーをメインサイザーに追加 (余白を持たせる)
    main_sizer->Add(grid_sizer, 1, wxEXPAND | wxALL, 15);

    validation_text_ = new wxStaticText(this, wxID_ANY, "");
    main_sizer->Add(validation_text_, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 15);

    // --- 最下部: OK / Cancel ボタン ---
    wxSizer* button_sizer = CreateButtonSizer(wxOK | wxCANCEL);
    main_sizer->Add(button_sizer, 0, wxALIGN_RIGHT | wxBOTTOM | wxRIGHT, 15);

    // サイザーをセットし、サイズをコンテンツに合わせる
    //SetSizerAndFit(main_sizer);
    SetSizer(main_sizer);

    // 最小サイズを現在のサイズに設定（これ以上小さく潰れないようにする）
    SetMinSize(GetSize());

    UpdateAxisRange();
    spin_x_->SetValue(ClampInt(default_x, 0, std::max(0, static_cast<int>(GetSelectedObsDim()) - 1)));
    spin_y_->SetValue(ClampInt(default_y, 0, std::max(0, static_cast<int>(GetSelectedObsDim()) - 1)));
    UpdateControlsEnabled();

    // デフォルトtag生成
    UpdateTag();

    // 入力値変更で随時タグ再生成、イベントハンドラ
    auto update_tag_func = [this](wxEvent& event) {
        this->UpdateTag();
         event.Skip();
        };
    auto update_obs_func = [this](wxEvent& event) {
        this->UpdateAxisRange();
        this->UpdateTag();
        event.Skip();
        };
    auto update_function_func = [this](wxEvent& event) {
        this->UpdateOutputKeyFromFunction();
        this->UpdateTag();
        event.Skip();
        };
    
    // 入力値変更で随時タグ再生成、Bind
    obs_key_combo_->Bind(wxEVT_COMBOBOX, update_obs_func);
    spin_x_->Bind(wxEVT_SPINCTRL, update_tag_func); // SpinCtrlは「矢印クリック(SPINCTRL)」と「手入力(TEXT)」の両方に反応させる
    spin_x_->Bind(wxEVT_TEXT, update_tag_func);
    spin_y_->Bind(wxEVT_SPINCTRL, update_tag_func);
    spin_y_->Bind(wxEVT_TEXT, update_tag_func);
    network_side_combo_->Bind(wxEVT_COMBOBOX, update_tag_func);
    function_combo_->Bind(wxEVT_COMBOBOX, update_function_func);
    output_key_combo_->Bind(wxEVT_COMBOBOX, update_tag_func);
    output_key_combo_->Bind(wxEVT_TEXT, update_tag_func);
    extractor_combo_->Bind(wxEVT_COMBOBOX, update_tag_func);
    extractor_idx_->Bind(wxEVT_SPINCTRL, update_tag_func);
    extractor_idx_->Bind(wxEVT_TEXT, update_tag_func);

    // 画面中央に配置
    Centre();
}

int64_t SweepHeatMapDialog::GetSelectedObsDim() const
{
    if (!obs_key_combo_) return 0;
    const int selection = obs_key_combo_->GetSelection();
    if (selection == wxNOT_FOUND) return 0;
    if (selection < 0 || static_cast<size_t>(selection) >= vector_obs_choices_.size()) return 0;
    return vector_obs_choices_[static_cast<size_t>(selection)].flatten_dim;
}

wxString SweepHeatMapDialog::BuildNetworkKey() const
{
    return wxString::Format("%s.%s", network_side_combo_->GetValue(), function_combo_->GetValue());
}

void SweepHeatMapDialog::UpdateAxisRange()
{
    const int64_t dim = GetSelectedObsDim();
    const int max_index = std::max(0, static_cast<int>(dim) - 1);

    spin_x_->SetRange(0, max_index);
    spin_y_->SetRange(0, max_index);
    spin_x_->SetValue(ClampInt(spin_x_->GetValue(), 0, max_index));
    spin_y_->SetValue(ClampInt(spin_y_->GetValue(), 0, max_index));
}

void SweepHeatMapDialog::UpdateOutputKeyFromFunction()
{
    const wxString current_output_key = output_key_combo_->GetValue();
    const wxArrayString choices = OutputKeyChoicesForFunction(function_combo_->GetValue());

    output_key_combo_->Clear();
    for (const auto& choice : choices) {
        output_key_combo_->Append(choice);
    }

    const wxString output_key = ContainsChoice(choices, current_output_key)
        ? current_output_key
        : DefaultOutputKeyForFunction(function_combo_->GetValue());
    output_key_combo_->SetStringSelection(output_key);
}

void SweepHeatMapDialog::UpdateControlsEnabled()
{
    const bool has_vector_obs = !vector_obs_choices_.empty();
    obs_key_combo_->Enable(has_vector_obs);
    spin_x_->Enable(has_vector_obs);
    spin_y_->Enable(has_vector_obs);
    network_side_combo_->Enable(has_vector_obs);
    function_combo_->Enable(has_vector_obs);
    output_key_combo_->Enable(has_vector_obs);
    extractor_combo_->Enable(has_vector_obs);
    extractor_idx_->Enable(has_vector_obs);
    tag_text_->Enable(has_vector_obs);

    wxWindow* ok_button = FindWindow(wxID_OK);
    if (ok_button) {
        ok_button->Enable(has_vector_obs);
    }

    validation_text_->SetLabel(has_vector_obs
        ? ""
        : "No vector-type observation key is available for state sweep.");
}

void SweepHeatMapDialog::UpdateTag()
{
    // 各コントロールから現在の値を取得
    wxString obs_key = obs_key_combo_->GetValue();
    int x = spin_x_->GetValue();
    int y = spin_y_->GetValue();
    wxString network_name = BuildNetworkKey();
    wxString output_key = output_key_combo_->GetValue();
    wxString extractor_name = extractor_combo_->GetValue();
    int extractor_idx = extractor_idx_->GetValue();

    // tag生成
    wxString tagIndexStr;
    if (extractor_idx != -1) {
        tagIndexStr = wxString::Format("_%d", extractor_idx);
    }
    wxString newTag = wxString::Format("%s_%s_%s_%s_%d-%d%s",
        obs_key, network_name, output_key, extractor_name, x, y, tagIndexStr);

    // テキストボックスを更新
    tag_text_->SetValue(newTag);
}

SweepHeatMapSettings SweepHeatMapDialog::GetSettings() const
{
    SweepHeatMapSettings settings;
    settings.x = spin_x_->GetValue();
    settings.y = spin_y_->GetValue();
    settings.obs_key = obs_key_combo_->GetValue();
    settings.network_side = network_side_combo_->GetValue();
    settings.function_key = function_combo_->GetValue();
    settings.network_key = BuildNetworkKey();
    settings.output_key = output_key_combo_->GetValue();
    settings.extractor_name = extractor_combo_->GetValue();
    settings.extractor_index = extractor_idx_->GetValue();
    settings.tag = tag_text_->GetValue();

    return settings;
}

SweepHeatMapPanel::SweepHeatMapPanel(wxWindow* parent, const wxString& title,
    const SweepHeatMapSettings& settings, std::shared_ptr<anet::rl::Runner> runner)
    : anet::rl::gui::Panel(parent, wxID_ANY, wxDefaultPosition, wxSize(600, 600))
{
    // メインのレイアウト
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);

    // ==== コントロールエリア ====
    wxBoxSizer* ctrl_sizer = new wxBoxSizer(wxHORIZONTAL);

    // StepText
    wxStaticText* step_label = new wxStaticText(this, wxID_ANY, "STEP:");
    ctrl_sizer->Add(step_label, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    step_text_ = new wxTextCtrl(this, wxID_ANY);
    step_text_->Enable(false);
    step_text_->SetForegroundColour(*wxBLACK);
    step_text_->SetSize(wxSize(60, -1));
    ctrl_sizer->Add(step_text_, 1, wxALIGN_CENTER_VERTICAL | wxALL, 5);

    // ラベル
    wxStaticText* label = new wxStaticText(this, wxID_ANY, "AutoRefresh:");
    ctrl_sizer->Add(label, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);

    // ComboBox
    wxArrayString choices;
    choices.Add("OFF");
    choices.Add("100ms");
    choices.Add("1000ms");
    choices.Add("3000ms");
    refresh_combo_ = new wxComboBox(this, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, choices, wxCB_READONLY);
    refresh_combo_->SetSelection(2);
    ctrl_sizer->Add(refresh_combo_, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);

    // RefreshButton
    wxButton* refresh_button = new wxButton(this, wxID_ANY, "Refresh");
    ctrl_sizer->Add(refresh_button, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);

    // メインに追加 (比率0 = 最小サイズ)
    main_sizer->Add(ctrl_sizer, 0, wxEXPAND | wxALL, 5);

    // GLCanvas
    wxGLAttributes dispAttrs;
    dispAttrs.PlatformDefaults().DoubleBuffer().EndList();

    canvas_ = new wxGLCanvas(this, dispAttrs, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxFULL_REPAINT_ON_RESIZE);
    context_ = new wxGLContext(canvas_);

    // 幅:最大、高さ:最大、比率1 (ウィンドウサイズ変更に追従)
    main_sizer->Add(canvas_, 1, wxEXPAND | wxALL, 0);

    // Sizer設定
    SetSizer(main_sizer);

    // --- イベントバインド ---
    timer_.Bind(wxEVT_TIMER, &SweepHeatMapPanel::OnTimer, this);
    refresh_combo_->Bind(wxEVT_COMBOBOX, &SweepHeatMapPanel::OnComboSelect, this);
    canvas_->Bind(wxEVT_PAINT, &SweepHeatMapPanel::OnPaint, this);
    canvas_->Bind(wxEVT_SIZE, &SweepHeatMapPanel::OnSize, this);
    refresh_button->Bind(wxEVT_BUTTON, &SweepHeatMapPanel::OnRefreshButton, this);

    // 初期設定に基づいてタイマー開始
    UpdateRefreshRate();

    // 画面中央へ
    Centre();

    // Observer生成
    CreateObserver(settings, runner, 10);
}

void SweepHeatMapPanel::CreateObserver(const SweepHeatMapSettings& settings,
    std::shared_ptr<anet::rl::Runner> runner, int log_interval)
{
    auto agent = runner->GetAgent();
    auto notifier = runner->GetNotifier();
    auto env_spec = runner->GetBatchEnv()->GetSpec();

    // flags
    auto flags =
        anet::HeatMapFlags::HM_AutoNormValue
        | anet::HeatMapFlags::HM_AutoScaleAxis
        | anet::HeatMapFlags::HM_SumMode;

    anet::rl::SweepedHeatMapObserverConfig q_sweep_obs_config{
        log_interval,
        flags | anet::HeatMapFlags::HM_AutoScaleAxis,  // flags
        128,    // grid_width
        128,    // grid_height
        -1,     // image_width
        -1,     // image_height
        settings.output_key.ToStdString(),  // output_key
    };

    anet::rl::ValueExtractFunction extractor;
    if (settings.extractor_name == "max") {
        extractor = [settings](
            const torch::Tensor& t, const std::unordered_set<std::string>& req)
            {
                auto ret = anet::rl::extractor::MaxIdxExtractor(t, req, settings.extractor_index);
                return ret;
            };
    } else if (settings.extractor_name == "std") {
        extractor = [settings](
            const torch::Tensor& t, const std::unordered_set<std::string>& req)
            {
                auto ret = anet::rl::extractor::StdIdxExtractor(t, req, settings.extractor_index);
                return ret;
            };
    } else if (settings.extractor_name == "min") {
        extractor = [settings](
            const torch::Tensor& t, const std::unordered_set<std::string>& req)
            {
                auto ret = anet::rl::extractor::MinIdxExtractor(t, req, settings.extractor_index);
                return ret;
            };
    } else if (settings.extractor_name == "argmax") {
        extractor = [settings](
            const torch::Tensor& t, const std::unordered_set<std::string>& req)
            {
                auto ret = anet::rl::extractor::ArgmaxIdxExtractor(t, req, settings.extractor_index);
                return ret;
            };
    } else {
        extractor = [settings](
            const torch::Tensor& t, const std::unordered_set<std::string>& req)
            {
                auto ret = anet::rl::extractor::MeanIdxExtractor(t, req, settings.extractor_index);
                return ret;
            };
    }

    auto processor = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        settings.x,  // x_index = x
        settings.y,  // y_index = y
        extractor,
        settings.obs_key.ToStdString(),
        agent->GetDevice()
    );

    ANET_CHECK_MSG(IsOutputKeyCompatibleWithFunction(settings.function_key, settings.output_key),
        "SweepHeatMapPanel incompatible function/output_key. function_key=" << settings.function_key.ToStdString()
        << " output_key=" << settings.output_key.ToStdString());

    std::optional<anet::TensorDictFunction> policy_forward = agent->GetTensorDictFunction(settings.network_key.ToStdString());
    ANET_CHECK_MSG(policy_forward.has_value(),
        "SweepHeatMapPanel requires TensorDictFunction. key=" << settings.network_key.ToStdString());
    this->observer_ = notifier->Attach<anet::rl::SweepedHeatMapObserver>(
        settings.tag.ToStdString() , q_sweep_obs_config, processor, *policy_forward, processor);
}

SweepHeatMapPanel::~SweepHeatMapPanel()
{
    // タイマー停止
    if (timer_.IsRunning()) {
        timer_.Stop();
    }
    // コンテキストの破棄
    delete context_;
}

void SweepHeatMapPanel::OnComboSelect(wxCommandEvent& event)
{
    UpdateRefreshRate();
}

void SweepHeatMapPanel::UpdateRefreshRate()
{
    int sel = refresh_combo_->GetSelection();

    // タイマーを一旦停止
    timer_.Stop();

    int interval = 0;
    switch (sel) {
    case 0: // OFF
        interval = 0;
        break;
    case 1: // 100ms
        interval = 100;
        break;
    case 2: // 1000ms
        interval = 1000;
        break;
    case 3: // 3000ms
        interval = 3000;
        break;
    default:
        interval = 1000;
        break;
    }

    if (interval > 0) {
        timer_.Start(interval);
    }
}

void SweepHeatMapPanel::OnRefreshButton(wxCommandEvent& event)
{
    captured_ = observer_->GetImageData();
    step_text_->SetValue(anet::FormatWithCommas(captured_.step));
    canvas_->Refresh();
}

void SweepHeatMapPanel::OnTimer(wxTimerEvent& event)
{
    captured_ = observer_->GetImageData();
    step_text_->SetValue(anet::FormatWithCommas(captured_.step));
    canvas_->Refresh();
}

void SweepHeatMapPanel::OnPaint(wxPaintEvent& event)
{
    // 描画イベントハンドラには必ず wxPaintDC が必要
    wxPaintDC dc(canvas_);
    Render();
}

void SweepHeatMapPanel::OnSize(wxSizeEvent& event)
{
    // GLViewportの更新などが必要な場合はここに記述
    // 今回は Render() 内で処理するため Refresh のみ
    canvas_->Refresh();
    event.Skip();
}

void SweepHeatMapPanel::Render()
{
    if (!context_ || !canvas_) return;

    // 現在のコンテキストを設定
    canvas_->SetCurrent(*context_);

    // ビューポートの設定
    const wxSize size = canvas_->GetSize() * canvas_->GetContentScaleFactor();
    glViewport(0, 0, size.x, size.y);

    // 画面クリア
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (observer_ && captured_.image.IsOk()) {
        int width = captured_.image.GetWidth();
        int height = captured_.image.GetHeight();
        if (width > 0 && height > 0) {
            unsigned char* rgb = captured_.image.GetData();

            // ---------------------------------------------------------
            // テクスチャの準備と転送
            // ---------------------------------------------------------
            GLuint textureID;
            glGenTextures(1, &textureID);          // テクスチャID生成
            glBindTexture(GL_TEXTURE_2D, textureID); // バインド

            // フィルタ設定
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            // 端の処理（クランプ）
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            // パッキングアライメントの設定
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);   // データは1バイト詰め（パディングなし）
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);  // 行の長さは画像幅と同じ
            glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0); // 先頭スキップなし
            glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);   // 行スキップなし

            // 転送
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb);

            // ---------------------------------------------------------
            // 描画 (四角形に貼り付け)
            // ---------------------------------------------------------
            glEnable(GL_TEXTURE_2D); // テクスチャ有効化

            // 座標変換行列をリセット（画面全体に描くため）
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            // 白をセット（テクスチャ本来の色を出すため）
            glColor3f(1.0f, 1.0f, 1.0f);

            // 四角形を描画 (-1.0 ~ 1.0 が画面の端から端)
            // MEMO：上下反転して表示される場合は、glTexCoord2fのY座標(第2引数)を 0と1 入れ替え
            glBegin(GL_QUADS);
            // 左下
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
            // 右下
            glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
            // 右上
            glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
            // 左上
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
            glEnd();

            glDisable(GL_TEXTURE_2D);

            // 使い終わったテクスチャは削除（毎フレーム生成している場合）
            /// @todo 本来はメンバ変数にIDを持って使い回すほうが高速
            glDeleteTextures(1, &textureID);
        }
    }

    glFlush();
    canvas_->SwapBuffers();
}
