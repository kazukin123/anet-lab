// HeatMapPanel.cpp

#include "Conv2dPanel.hpp"
#include <format>
#include <wx/numformatter.h>
#include "anet/log.hpp"

// Windowsの標準ヘッダーに足りない定義を追加
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

namespace LOG = anet::log;

struct Conv2dPanel::Config : public anet::Config {
    anet::rl::Conv2dVisualizerConfig conv2d;

    Conv2dPanel::Config(const anet::ConfigData& config_data) : anet::Config("Conv2dPanel")
    {
        ANET_READ_CONFIG(config_data, conv2d.margin_x);
        ANET_READ_CONFIG(config_data, conv2d.margin_y);
        ANET_READ_CONFIG(config_data, conv2d.channels_per_row);
        ANET_READ_CONFIG(config_data, conv2d.flip_vertical);
        ANET_READ_CONFIG(config_data, conv2d.network_key);
        ANET_READ_CONFIG(config_data, conv2d.colormap);
        ANET_READ_CONFIG(config_data, conv2d.scale_factor);
        ANET_READ_CONFIG(config_data, conv2d.min_block_size);
    }
};

Conv2dPanel::Conv2dPanel(
    wxWindow* parent, const wxString& title, anet::rl::RunManager& run_manager, std::shared_ptr<anet::rl::Runner> runner,
    const anet::ConfigData& config_data)
    : anet::rl::gui::Panel(parent, wxID_ANY, wxDefaultPosition, wxSize(600, 600))
{
    // Config
    config_ = std::make_unique<Conv2dPanel::Config>(config_data);

    // メインのレイアウト
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);

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
    canvas_->Bind(wxEVT_PAINT, &Conv2dPanel::OnPaint, this);
    canvas_->Bind(wxEVT_SIZE, &Conv2dPanel::OnSize, this);

    // 画面中央へ
    Centre();

    // Vsualizer生成
    CreateVisualizer(runner);

    // Observer生成&登録
    CreateObserver(run_manager, runner);

}

void Conv2dPanel::CreateVisualizer(std::shared_ptr<anet::rl::Runner> runner)
{
    auto agent = runner->GetAgent();
    auto notifier = runner->GetNotifier();
    auto env_spec = runner->GetBatchEnv()->GetSpec();

    // TensorDictFnを取得
    auto dict_fn = agent->GetTensorDictFunction(config_->conv2d.network_key);
    ANET_CHECK(dict_fn.has_value());
    vis_dict_fn_ = *dict_fn;

    // Observer生成＆登録
    this->visualizer_ = std::make_unique<anet::rl::Conv2dVisualizer>(config_->conv2d);
}

void Conv2dPanel::CreateObserver(anet::rl::RunManager& run_manager, std::shared_ptr<anet::rl::Runner> runner)
{
    // Observer生成&登録
    auto notifier = run_manager.GetNotifier();
    observer_ = notifier->AttachScoped<anet::rl::FunctionTrainObserver>(
        runner,
        [this](const anet::rl::TrainEvent& event)
        {
            if (closed_) return;

            // State取得
            const auto& state = event.experience.next_state;    // Action実施後の現在の状態で評価
            if (!state.obs.defined() || state.obs.size(0) <= 0) {
                LOG::warn() << "Conv2dPanel: failed to get observation.";
                return;
            }

            // TensorDict取得
            torch::Tensor single_obs = state.obs.slice(0, 0, 1);
            auto dict = vis_dict_fn_(single_obs);

            // 画像生成
            auto vis_result = visualizer_->Visualize(0, dict);
            wxImage new_image = vis_result.first;

            //  GUI描画用にスレッドセーフに保存
            if (new_image.IsOk()) {
                std::lock_guard<std::mutex> lock(image_mutex_);
                current_image_ = new_image;
            }

            // 画面リフレッシュ
            Refresh();
        },
        "Conv2dPanel");

    // Detach用にNotifierを保持
    notifier_ = notifier;
}


Conv2dPanel::~Conv2dPanel()
{
    // タイマー停止
    if (observer_ && notifier_) {
        notifier_->Detach(observer_);
    }
    closed_ = true;

}

void Conv2dPanel::OnSize(wxSizeEvent& event)
{
    // GLViewportの更新などが必要な場合はここに記述
    // 今回は Render() 内で処理するため Refresh のみ
    canvas_->Refresh();
    event.Skip();
}

void Conv2dPanel::OnPaint(wxPaintEvent& event)
{
    // 描画イベントハンドラには必ず wxPaintDC が必要
    wxPaintDC dc(canvas_);
    Render();
}

void Conv2dPanel::Render()
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

    if (current_image_.IsOk()) {
        int width = current_image_.GetWidth();
        int height = current_image_.GetHeight();
        if (width > 0 && height > 0) {
            unsigned char* rgb = current_image_.GetData();

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
